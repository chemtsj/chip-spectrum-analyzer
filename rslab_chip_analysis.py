"""Mapping and independent-chip statistics for the RSLAB Streamlit app.

All intensity matrices have shape (spectral positions, mapping points). Mapping
SD describes spatial heterogeneity. It is never relabelled as the measured
uncertainty of a sum. Optional sqrt(N) * SD and SD / sqrt(N) columns assume
independent, identically distributed observations and are estimates only.
"""
from __future__ import annotations

import csv
import re
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import spsolve


@dataclass
class ChipData:
    """One chip: x[shift], y[shift, point], coordinates[point, dimension]."""

    name: str
    x: np.ndarray
    y: np.ndarray
    coordinates: np.ndarray | None = None
    format: str = "mapping"

    def __post_init__(self):
        self.x = np.asarray(self.x, dtype=float)
        self.y = np.asarray(self.y, dtype=float)
        if self.x.ndim != 1 or len(self.x) < 2:
            raise ValueError("Raman shift 값이 2개 이상 필요합니다.")
        if self.y.ndim != 2 or self.y.shape[0] != len(self.x) or self.y.shape[1] < 1:
            raise ValueError("Intensity 배열은 [Raman shift 수, mapping point 수]여야 합니다.")
        if not np.isfinite(self.x).all() or not np.isfinite(self.y).all():
            raise ValueError("Raman shift와 intensity에 빈 값, NaN, 무한대가 있으면 처리할 수 없습니다.")
        if np.any(np.diff(self.x) <= 0):
            raise ValueError("Raman shift는 중복 없이 오름차순이어야 합니다.")
        if self.coordinates is not None:
            self.coordinates = np.asarray(self.coordinates, dtype=float)
            if (self.coordinates.ndim != 2 or
                    self.coordinates.shape[0] != self.y.shape[1] or
                    not np.isfinite(self.coordinates).all()):
                raise ValueError("Mapping 좌표의 개수 또는 숫자 형식이 올바르지 않습니다.")


@dataclass(frozen=True)
class ProcessOptions:
    """Per-point baseline settings. independent only controls reported estimates."""

    baseline: str = "없음"
    degree: int = 2
    lam: float = 1e5
    p: float = 1e-4
    iterations: int = 10
    independent: bool = False

    def __post_init__(self):
        if self.baseline not in ("없음", "AsLS", "Polynomial"):
            raise ValueError("Baseline은 없음, AsLS, Polynomial 중 하나여야 합니다.")
        if (not isinstance(self.degree, (int, np.integer)) or self.degree < 0 or
                not isinstance(self.iterations, (int, np.integer)) or self.iterations < 1 or
                not np.isfinite(self.lam) or self.lam <= 0 or
                not np.isfinite(self.p) or not 0 < self.p < 1):
            raise ValueError("Baseline 차수, λ, p, 반복 횟수를 확인해 주세요.")


def _decode(data: bytes) -> str:
    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ValueError("파일의 문자 인코딩을 읽을 수 없습니다. UTF-8 또는 CP949로 저장해 주세요.")


def _number(cell: str, row: int, col: int) -> float:
    try:
        value = float(cell.strip())
    except ValueError as exc:
        raise ValueError(f"{row}행 {col}열이 숫자가 아니거나 비어 있습니다: {cell[:30]!r}") from exc
    if not np.isfinite(value):
        raise ValueError(f"{row}행 {col}열에 NaN 또는 무한대가 있습니다.")
    return value


def _is_numeric(row: list[str]) -> bool:
    try:
        return bool(row) and all(cell.strip() and np.isfinite(float(cell)) for cell in row)
    except ValueError:
        return False


def _numeric_first_cell(row: list[str]) -> bool:
    """Recognise a data row even when later cells contain invalid values."""
    try:
        float(row[0])
        return True
    except (ValueError, IndexError):
        return False


def _matrix(rows, row_numbers, expected_width):
    result = []
    for cells, line_no in zip(rows, row_numbers):
        if len(cells) != expected_width:
            raise ValueError(f"{line_no}행의 열 수({len(cells)})가 예상 열 수({expected_width})와 다릅니다.")
        result.append([_number(cell, line_no, j + 1) for j, cell in enumerate(cells)])
    if not result:
        raise ValueError("숫자 데이터 행이 없습니다.")
    return np.asarray(result, dtype=float)


def _check_mapping_headers(header):
    summary = re.compile(r"(?:^|[^a-z])(?:mean|sum|sd|std|sem|error|average)(?:$|[^a-z])", re.I)
    for cell in header[1:]:
        if summary.search(cell) or any(word in cell for word in ("평균", "표준편차", "합계", "오차")):
            raise ValueError("이미 평균·합·오차로 요약한 파일입니다. 한 chip의 각 mapping point가 들어 있는 원본 파일을 올려 주세요.")


def _from_arrays(name, x, y, coordinates, source_format):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if len(np.unique(x)) != len(x):
        raise ValueError("중복된 Raman shift가 있습니다. 원본 파일을 확인해 주세요.")
    order = np.argsort(x, kind="stable")
    return ChipData(name, x[order], y[order, :], coordinates, source_format)


def parse_mapping_bytes(data: bytes, name: str) -> ChipData:
    """Strictly read HORIBA row-wise maps or transposed point-column tables.

    Delimiters: tab, comma or semicolon. HORIBA maps start with one/two empty
    coordinate headers and numeric shifts; each following row is one mapping
    point. A transposed table has shift in column A and points in B onward,
    optionally preceded by the app's x coordinate / y coordinate rows. Data
    errors raise an exception; no invalid rows or intensity cells are dropped.
    Already aggregated MEAN/SUM/error exports are rejected.
    """
    text = _decode(data)
    selected = [(i + 1, line) for i, line in enumerate(text.splitlines())
                if line.strip() and not line.lstrip().startswith("#")]
    if not selected:
        raise ValueError("파일이 비어 있습니다.")
    first_line = selected[0][1]
    delimiter = "\t" if "\t" in first_line else ";" if ";" in first_line else ","
    try:
        rows = [next(csv.reader([line], delimiter=delimiter, strict=True)) for _, line in selected]
    except csv.Error as exc:
        raise ValueError(f"구분자 또는 따옴표 형식을 읽을 수 없습니다: {exc}") from exc
    numbers = [i for i, _ in selected]
    header = rows[0]
    if len(header) < 2:
        raise ValueError("최소한 Raman shift 열과 intensity 열이 필요합니다. 탭/쉼표/세미콜론 구분 파일인지 확인해 주세요.")
    leading = 0
    while leading < len(header) and not header[leading].strip():
        leading += 1
    if leading in (1, 2) and _is_numeric(header[leading:]):
        if len(header) - leading < 2:
            raise ValueError("Raman shift 값이 2개 이상 필요합니다.")
        axis = np.asarray([float(cell) for cell in header[leading:]])
        matrix = _matrix(rows[1:], numbers[1:], len(header))
        return _from_arrays(name, axis, matrix[:, leading:].T, matrix[:, :leading], "HORIBA row-wise mapping")

    coordinates = None
    if header[0].strip().lower() == "x coordinate":
        if (len(rows) < 4 or rows[1][0].strip().lower() != "y coordinate" or
                len(rows[1]) != len(header)):
            raise ValueError("x coordinate / y coordinate 행의 형식이 올바르지 않습니다.")
        coordinates = np.asarray([
            [_number(cell, numbers[0], j + 2) for j, cell in enumerate(rows[0][1:])],
            [_number(cell, numbers[1], j + 2) for j, cell in enumerate(rows[1][1:])],
        ], dtype=float).T
        rows, numbers = rows[2:], numbers[2:]
        header = rows[0]
        if len(header) != coordinates.shape[0] + 1:
            raise ValueError("좌표 개수와 mapping intensity 열 수가 다릅니다.")
    if _numeric_first_cell(header):
        matrix = _matrix(rows, numbers, len(header))
    else:
        _check_mapping_headers(header)
        matrix = _matrix(rows[1:], numbers[1:], len(header))
    return _from_arrays(name, matrix[:, 0], matrix[:, 1:], coordinates, "point-column mapping")


def process_chip(chip: ChipData, options: ProcessOptions) -> ChipData:
    """Apply baseline to each full point spectrum, then return an independent copy."""
    y = chip.y.copy()
    if options.baseline == "Polynomial":
        if options.degree >= len(chip.x):
            raise ValueError("Polynomial 차수는 스펙트럼의 데이터 점 수보다 작아야 합니다.")
        index = np.arange(len(chip.x), dtype=float)
        for j in range(y.shape[1]):
            y[:, j] -= np.polyval(np.polyfit(index, y[:, j], options.degree), index)
    elif options.baseline == "AsLS":
        n = len(chip.x)
        if n < 3:
            raise ValueError("AsLS baseline에는 Raman shift 값이 3개 이상 필요합니다.")
        difference = sparse.diags([1., -2., 1.], [0, 1, 2], shape=(n - 2, n), format="csc")
        penalty = options.lam * (difference.T @ difference)
        for j in range(y.shape[1]):
            point = y[:, j].copy()
            if np.all(point == point[0]):
                y[:, j] = 0.0
                continue
            weights = np.ones(n)
            for _ in range(options.iterations):
                w = sparse.spdiags(weights, 0, n, n)
                z = spsolve(w + penalty, weights * point)
                # Preserve the original algorithm except exact ties, where
                # retaining a positive weight prevents an all-zero matrix.
                weights = np.where(point > z, options.p, 1.0 - options.p)
            y[:, j] = point - z
    if not np.isfinite(y).all():
        raise ValueError("Baseline 계산 결과에 유효하지 않은 숫자가 있습니다. 설정을 확인해 주세요.")
    return replace(chip, x=chip.x.copy(), y=y,
                   coordinates=None if chip.coordinates is None else chip.coordinates.copy())


def _sample_sd(y):
    return np.std(y, axis=1, ddof=1) if y.shape[1] > 1 else np.full(y.shape[0], np.nan)


def summary_frame(chip: ChipData, independent: bool = False) -> pd.DataFrame:
    """A shift, B mean, C spatial sample SD, D sum, E optional sum-SD estimate.

    Two extra columns record point count and the optional mean-SEM estimate.
    Undefined SD (N=1) and disabled independence estimates are stored as NaN.
    """
    count = chip.y.shape[1]
    sd = _sample_sd(chip.y)
    unknown = np.full(len(chip.x), np.nan)
    return pd.DataFrame({
        "Raman_shift_cm-1": chip.x,
        "MEAN": np.mean(chip.y, axis=1),
        "SD_within_chip": sd,
        "SUM": np.sum(chip.y, axis=1),
        "SUM_SD_independent_estimate": np.sqrt(count) * sd if independent else unknown,
        "N_points": count,
        "MEAN_SEM_independent_estimate": sd / np.sqrt(count) if independent else unknown,
    })


def _representative(chip: ChipData, mode: str) -> np.ndarray:
    if mode == "MEAN":
        return np.mean(chip.y, axis=1)
    if mode == "SUM":
        return np.sum(chip.y, axis=1)
    raise ValueError("통계 모드는 MEAN 또는 SUM이어야 합니다.")


def peak_record(chip: ChipData, mode: str, lo: float, hi: float) -> dict:
    """Maximum of the representative curve in an inclusive sampled-X window.

    Return keys: chip_name, statistic, window_min/max, peak_x, peak_intensity,
    tie_count, peak_at_boundary, N_points. Exact ties use the lowest sampled X.
    Boundary means first/last included sampled position, not equality to bounds.
    """
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError("피크 구간의 최소/최대값을 확인해 주세요.")
    indices = np.flatnonzero((chip.x >= lo) & (chip.x <= hi))
    if not len(indices):
        raise ValueError(f"{chip.name}: 지정한 구간에 Raman shift 데이터가 없습니다.")
    y = _representative(chip, mode)
    maximum = float(np.max(y[indices]))
    tied = indices[y[indices] == maximum]
    chosen = int(tied[0])
    return {
        "chip_name": chip.name, "statistic": mode,
        "window_min": float(lo), "window_max": float(hi),
        "peak_x": float(chip.x[chosen]), "peak_intensity": maximum,
        "tie_count": int(len(tied)),
        "peak_at_boundary": bool(chosen in (indices[0], indices[-1])),
        "N_points": int(chip.y.shape[1]),
    }


def group_peak_summary(peaks: pd.DataFrame, independent: bool = False) -> pd.DataFrame:
    """Summarise independent chip maxima, grouped by group_label + statistic.

    Input requires chip_name, group_label, statistic, peak_intensity. If window
    bounds exist, all rows within each group must use the same bounds. Each chip
    counts once per group/statistic. Sample SD is undefined for a single chip.
    """
    required = ["chip_name", "group_label", "statistic", "peak_intensity"]
    if any(column not in peaks for column in required) or peaks.empty:
        raise ValueError("칩 간 요약에는 chip_name, group_label, statistic, peak_intensity가 필요합니다.")
    if peaks[required].isna().any().any():
        raise ValueError("칩 이름·그룹·통계 모드·피크 intensity에 누락된 값이 있습니다.")
    for column in ("chip_name", "group_label"):
        if peaks[column].astype(str).str.strip().eq("").any():
            raise ValueError("칩 이름과 그룹 이름은 비워 둘 수 없습니다.")
    if not peaks["statistic"].isin(["MEAN", "SUM"]).all():
        raise ValueError("통계 모드는 MEAN 또는 SUM이어야 합니다.")
    values = pd.to_numeric(peaks["peak_intensity"], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("피크 intensity에는 유한한 숫자만 사용할 수 있습니다.")
    work = peaks.copy()
    work["peak_intensity"] = values
    if work.duplicated(["group_label", "statistic", "chip_name"]).any():
        raise ValueError("한 그룹에 동일한 칩이 중복 포함되어 있습니다.")
    output = []
    for (group, mode), frame in work.groupby(["group_label", "statistic"], sort=False, dropna=False):
        for column in ("window_min", "window_max"):
            if column in frame:
                numeric = pd.to_numeric(frame[column], errors="coerce")
                if not np.isfinite(numeric).all() or numeric.nunique() != 1:
                    raise ValueError("같은 그룹의 모든 칩에 동일한 피크 구간을 적용해 주세요.")
        values = frame["peak_intensity"].to_numpy(float)
        count = len(values)
        sd = float(np.std(values, ddof=1)) if count > 1 else np.nan
        item = {
            "group_label": group, "statistic": mode, "N_chips": count,
            "peak_mean": float(np.mean(values)), "peak_sd": sd,
            "peak_sum": float(np.sum(values)),
            "SUM_SD_independent_estimate": float(np.sqrt(count) * sd) if independent else np.nan,
        }
        for column in ("window_min", "window_max"):
            if column in frame:
                item[column] = float(frame[column].iloc[0])
        output.append(item)
    return pd.DataFrame(output)


def group_spectrum_frame(chips: list[ChipData], mode: str = "MEAN", independent: bool = False) -> pd.DataFrame:
    """Equal-weight spectra across chips on exactly identical X grids.

    Columns: Raman_shift_cm-1, MEAN_between_chips, SD_between_chips,
    SUM_between_chips, SUM_SD_independent_estimate, N_chips,
    MEAN_SEM_independent_estimate. This function never pools mapping points or
    interpolates grids. Separate maxima can still be compared on unequal grids.
    """
    if not chips:
        raise ValueError("한 개 이상의 chip이 필요합니다.")
    reference = chips[0].x
    if any(not np.array_equal(reference, chip.x) for chip in chips[1:]):
        raise ValueError("칩 간 Raman shift 격자가 달라 스펙트럼을 직접 평균낼 수 없습니다. 각 칩의 피크값 비교는 가능합니다.")
    y = np.column_stack([_representative(chip, mode) for chip in chips])
    count = len(chips)
    sd = _sample_sd(y)
    unknown = np.full(len(reference), np.nan)
    return pd.DataFrame({
        "Raman_shift_cm-1": reference,
        "MEAN_between_chips": np.mean(y, axis=1),
        "SD_between_chips": sd,
        "SUM_between_chips": np.sum(y, axis=1),
        "SUM_SD_independent_estimate": np.sqrt(count) * sd if independent else unknown,
        "N_chips": count,
        "MEAN_SEM_independent_estimate": sd / np.sqrt(count) if independent else unknown,
    })


def overview_frame(chips: list[ChipData], groups: list[str], mode: str = "MEAN",
                   lo: float = 900., hi: float = 950., method: str = "chip_peak",
                   reference_x: float | None = None,
                   independent: bool = False) -> pd.DataFrame:
    """Return one bar per chip with an explicitly defined within-chip error.

    ``chip_peak`` chooses each chip's representative-spectrum maximum in the
    inclusive ``lo``/``hi`` window, matching :func:`peak_record`. The spatial
    SD is evaluated across that chip's mapping points at the selected sampled
    shift. It is not the SD of separate point maxima and it is not SEM.

    ``fixed_x`` chooses the sampled shift nearest ``reference_x`` separately
    for every chip; ties select the lower shift. A reference outside any chip's
    measured range is rejected. No interpolation or extrapolation is used.

    ``value`` is MEAN or SUM at the chosen shift. MEAN ``error`` is spatial
    sample SD. SUM ``error`` is NaN unless ``independent`` enables the estimate
    sqrt(N_points)*spatial SD. At N_points=1 all SD/error fields remain NaN.
    ``requested_x``/``x_offset`` are NaN for chip_peak because no fixed shift
    was requested. Group summaries must be computed from these same heights.
    """
    if not chips or len(chips) != len(groups):
        raise ValueError("각 chip에 대응하는 조건 이름이 하나씩 필요합니다.")
    names = [str(chip.name).strip() for chip in chips]
    if any(not name for name in names) or len(set(names)) != len(names):
        raise ValueError("칩 이름은 비어 있거나 중복될 수 없습니다.")
    group_names = []
    for group in groups:
        if group is None or pd.isna(group) or not str(group).strip():
            raise ValueError("조건 이름을 모두 입력해 주세요.")
        group_names.append(str(group).strip())
    if mode not in ("MEAN", "SUM"):
        raise ValueError("통계 모드는 MEAN 또는 SUM이어야 합니다.")
    if method not in ("chip_peak", "fixed_x"):
        raise ValueError("막대 값 계산 기준은 chip_peak 또는 fixed_x여야 합니다.")
    if not np.isfinite(lo) or not np.isfinite(hi) or lo > hi:
        raise ValueError("피크 구간의 최소/최대값을 확인해 주세요.")
    if reference_x is not None and not np.isfinite(reference_x):
        raise ValueError("지정 파수는 유한한 숫자여야 합니다.")
    if method == "fixed_x" and reference_x is None:
        raise ValueError("고정 파수 비교에는 지정 파수를 입력해 주세요.")

    output = []
    for chip, group in zip(chips, group_names):
        peak = None
        if method == "chip_peak":
            peak = peak_record(chip, mode, lo, hi)
            selected = int(np.searchsorted(chip.x, peak["peak_x"]))
            requested, offset = np.nan, np.nan
        else:
            requested = float(reference_x)
            if not chip.x[0] <= requested <= chip.x[-1]:
                raise ValueError(f"{chip.name}: 지정 파수 {requested:g} cm⁻¹가 측정 범위 "
                                 f"{chip.x[0]:g}–{chip.x[-1]:g} cm⁻¹ 밖입니다.")
            selected = int(np.argmin(np.abs(chip.x - requested)))
            offset = float(chip.x[selected] - requested)
        point_values = chip.y[selected, :]
        count = len(point_values)
        mean = float(np.mean(point_values))
        sd = float(np.std(point_values, ddof=1)) if count > 1 else np.nan
        total = float(np.sum(point_values))
        sum_sd = float(np.sqrt(count) * sd) if independent else np.nan
        item = {
            "chip_name": chip.name, "group_label": group,
            "statistic": mode, "method": method,
            "requested_x": requested,
            "Raman_shift_cm-1": float(chip.x[selected]),
            "x_offset": offset,
            "value": mean if mode == "MEAN" else total,
            "error": sd if mode == "MEAN" else sum_sd,
            "error_basis": ("within_chip_spatial_SD" if mode == "MEAN" else
                            "independent_sum_SD_estimate" if independent else "not_estimated"),
            "MEAN": mean, "SD_within_chip": sd, "SUM": total,
            "SUM_SD_independent_estimate": sum_sd, "N_points": int(count),
            "window_min": float(lo), "window_max": float(hi),
            "tie_count": peak["tie_count"] if peak else np.nan,
            "peak_at_boundary": peak["peak_at_boundary"] if peak else False,
        }
        output.append(item)
    return pd.DataFrame(output)


def overview_group_summary(overview: pd.DataFrame,
                           independent: bool = False) -> pd.DataFrame:
    """Equal-chip group statistics of the exact heights in overview_frame.

    The resulting ``value_mean`` and ``value_sd`` are suitable for overlays on
    the well-bar overview. This must replace maximum-peak group statistics when
    the overview uses a fixed shift. Mapping-point counts never weight a chip.
    """
    required = ["chip_name", "group_label", "statistic", "method", "value", "requested_x"]
    if overview.empty or any(column not in overview for column in required):
        raise ValueError("막대 요약에 필요한 칩별 값과 계산 기준이 없습니다.")
    if overview["method"].isna().any() or not overview["method"].isin(["chip_peak", "fixed_x"]).all():
        raise ValueError("막대 값 계산 기준을 확인해 주세요.")
    for _, group in overview.groupby(["group_label", "statistic"], sort=False, dropna=False):
        if group["method"].nunique() != 1:
            raise ValueError("같은 조건의 막대는 같은 계산 기준을 사용해야 합니다.")
        if group["method"].iloc[0] == "fixed_x":
            reference = pd.to_numeric(group["requested_x"], errors="coerce")
            if not np.isfinite(reference).all() or reference.nunique() != 1:
                raise ValueError("같은 조건의 모든 칩에 동일한 지정 파수를 적용해 주세요.")
    adapter = overview.rename(columns={"value": "peak_intensity"})
    grouped = group_peak_summary(adapter, independent=independent).rename(columns={
        "peak_mean": "value_mean", "peak_sd": "value_sd", "peak_sum": "value_sum",
    })
    metadata = overview.groupby(["group_label", "statistic"], sort=False, dropna=False).agg(
        method=("method", "first"), requested_x=("requested_x", "first"),
    ).reset_index()
    return grouped.merge(metadata, on=["group_label", "statistic"], how="left", sort=False,
                         validate="one_to_one")
