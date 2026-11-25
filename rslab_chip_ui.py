"""Streamlit screens for one-chip and between-chip Raman mapping analysis."""
from __future__ import annotations

import hashlib
import html
import io
import json
import re
import zipfile
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from rslab_chip_analysis import (
    ProcessOptions, parse_mapping_bytes, process_chip, summary_frame,
    peak_record, group_peak_summary, group_spectrum_frame, overview_frame, overview_group_summary,
)

from rslab_chip_plots import (
    graph_editor, draw_spectra, draw_group_peaks, draw_chip_overview, figure_bytes,
)


@st.cache_data(show_spinner=False, max_entries=16)
def _chip_cached_process(raw_bytes, filename, options):
    parsed = parse_mapping_bytes(raw_bytes, filename)
    return process_chip(parsed, options)


def _chip_csv(frame):
    return frame.to_csv(index=False, float_format="%.12g").encode("utf-8-sig")


def _chip_safe_name(name):
    return re.sub(r'[\\/:*?"<>|\x00-\x1f]', "_", str(name)).strip(" .") or "chip"


def _chip_default_name(filename):
    """Read A1–Z12 labels, preferring the sample's parenthesized chip code."""
    stem = Path(filename).stem
    token = r"[A-Z](?:1[0-2]|[1-9])"
    match = re.search(rf"\(\s*({token})\s*\)", stem, re.I)
    if not match:
        # Acquisition suffixes such as _785nm_10mW_x10_3s are not chip codes.
        sample_name = re.split(r"[_\s-]\d+(?:\.\d+)?\s*nm(?=[_\s-]|$)",
                               stem, maxsplit=1, flags=re.I)[0]
        match = re.search(rf"(?<![A-Za-z0-9])({token})(?![A-Za-z0-9])", sample_name, re.I)
    return match.group(1).upper() if match else stem


@st.fragment
def _chip_download_button(*args, **kwargs):
    """Keep downloads from rerunning analysis, including on Streamlit 1.37–1.43."""
    version = re.match(r"(\d+)\.(\d+)", getattr(st, "__version__", ""))
    if version and tuple(map(int, version.groups())) >= (1, 44):
        kwargs["on_click"] = "ignore"
    return st.download_button(*args, **kwargs)


def _chip_upload_location(file_id):
    match = re.fullmatch(r"condition_(\d+)_file_(\d+)", str(file_id))
    return (f"조건 {int(match[1])} · 파일 {int(match[2])}"
            if match else str(file_id))


def _chip_metadata_issues(frame):
    """Check the edited names, marking every occurrence of a duplicate name."""
    edited = frame.copy()
    for column in ("chip_name", "group_label"):
        edited[column] = edited[column].fillna("").astype(str).str.strip()
    duplicates = edited.chip_name.ne("") & edited.chip_name.duplicated(keep=False)
    issues = []
    for position, (_, row) in enumerate(edited.iterrows(), 1):
        reasons = []
        if duplicates.iloc[position - 1]:
            reasons.append("칩 이름 중복")
        if not row.chip_name:
            reasons.append("칩 이름 없음")
        if not row.group_label:
            reasons.append("조건 이름 없음")
        if reasons:
            issues.append({"문제": " · ".join(reasons),
                           "칩 이름": row.chip_name or "(빈칸)",
                           "조건명": row.group_label or "(빈칸)",
                           "업로드 위치": _chip_upload_location(row.get("file_id", position)),
                           "원본 파일명": row.get("source_file", "")})
    return edited, pd.DataFrame(issues, columns=["문제", "칩 이름", "조건명", "업로드 위치", "원본 파일명"])


def _chip_duplicate_data_rows(chips, sources):
    """Report separate groups of identical processed X/Y arrays, not mean curves."""
    groups = {}
    for index, chip in enumerate(chips):
        digest = hashlib.sha256()
        for array in (chip.x, chip.y):
            values = np.asarray(array, dtype="<f8").copy(order="C")
            # Numeric equality treats +0 and -0 as the same value.
            values[values == 0] = 0
            digest.update(str(values.shape).encode("ascii"))
            digest.update(values.tobytes())
        groups.setdefault(digest.hexdigest(), []).append(index)
    duplicates = [members for members in groups.values() if len(members) > 1]
    rows = []
    for number, members in enumerate(duplicates, 1):
        for index in members:
            source = sources[index]
            rows.append({"동일 데이터 묶음": f"묶음 {number} ({len(members)}개 파일)",
                         "칩 이름": source["chip_name"], "조건명": source["group_label"],
                         "업로드 위치": _chip_upload_location(source["file_id"]),
                         "원본 파일명": source["source_file"]})
    return pd.DataFrame(rows, columns=["동일 데이터 묶음", "칩 이름", "조건명", "업로드 위치", "원본 파일명"])


def _chip_warning_table(frame, kind):
    """Render an escaped diagnostic table; editable cells do not support colors.

    Keeping this table separate from data_editor also avoids resetting the
    editor's input or showing checks from the previous edit on a rerun.
    """
    if frame.empty:
        return
    color, background, border = (("#991b1b", "#fff1f2", "#fda4af") if kind == "name"
                                 else ("#78350f", "#fffbeb", "#fbbf24"))
    label = "칩 이름 오류 파일 목록" if kind == "name" else "동일한 mapping 데이터 파일 목록"
    head = "".join(f'<th scope="col" style="padding:10px 12px;text-align:left;white-space:nowrap;'
                   f'border-bottom:2px solid {border};">{html.escape(str(column))}</th>'
                   for column in frame.columns)
    body = []
    for values in frame.itertuples(index=False, name=None):
        cells = "".join(f'<td style="padding:10px 12px;border-bottom:1px solid {border};'
                        f'white-space:nowrap;">{html.escape(str(value))}</td>' for value in values)
        body.append(f"<tr>{cells}</tr>")
    st.markdown(
        f'<div class="rslab-chip-diagnostic" style="overflow-x:auto;margin:10px 0 14px;'
        f'border:1px solid {border};border-radius:10px;background:{background};color:{color};">'
        f'<table aria-label="{label}" style="width:100%;border-collapse:collapse;'
        f'font-size:.9rem;color:{color};background:{background};">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>',
        unsafe_allow_html=True,
    )


def _chip_section(title, number):
    """Separate analysis stages with a divider and a shaded, static heading."""
    if number > 1:
        st.divider()
    st.markdown(
        '<div class="rslab-chip-section" style="display:flex;align-items:center;gap:12px;'
        'padding:15px 18px;margin:18px 0 16px;border:1px solid rgba(99,102,241,.23);'
        'border-left:5px solid #6366f1;border-radius:10px;'
        'background:linear-gradient(110deg,rgba(99,102,241,.12),rgba(99,102,241,.035));">'
        f'<span aria-hidden="true" style="display:inline-flex;align-items:center;justify-content:center;'
        'min-width:30px;height:30px;padding:0 4px;border-radius:8px;background:#6366f1;'
        f'color:white;font-size:14px;font-weight:700;">{int(number):02d}</span>'
        f'<h2 style="margin:0;padding:0;font-size:1.2rem;line-height:1.4;'
        f'font-weight:700;color:inherit;">{html.escape(str(title))}</h2></div>',
        unsafe_allow_html=True,
    )


def _chip_spectrum_tables(chips, mode, settings, scope="visible", independent=False):
    """Export per-chip mapping mean/sum spectra without display transformations.

    Shared X grids use one X column; differing grids use separate X/Y pairs,
    padded with blanks. Display order is preserved without interpolation.
    Individual files also include the corresponding spatial SD (MEAN) or the
    optional independent sum-SD estimate (SUM), plus mapping-point counts.
    """
    if mode not in ("MEAN", "SUM") or scope not in ("visible", "all"):
        raise ValueError("스펙트럼 저장 방식과 저장할 곡선을 확인하세요.")
    lookup = {chip.name: chip for chip in chips}
    if len(lookup) != len(chips):
        raise ValueError("저장할 칩 이름이 중복되어 있습니다.")
    order = list(dict.fromkeys(name for name in settings.get("order", []) if name in lookup))
    order.extend(name for name in lookup if name not in order)
    visible = set(settings.get("visible", list(lookup)))
    selected = [lookup[name] for name in order if scope == "all" or name in visible]
    if not selected:
        raise ValueError("저장할 칩을 한 개 이상 선택하세요.")
    common_grid = all(np.array_equal(selected[0].x, chip.x) for chip in selected[1:])
    combined, individual = {}, {}
    if common_grid:
        combined["Raman_shift_cm-1"] = selected[0].x
    for i, chip in enumerate(selected, 1):
        stats = summary_frame(chip, independent=independent)
        if not common_grid:
            combined[f"{chip.name} | Raman_shift_cm-1"] = pd.Series(chip.x)
        combined[f"{chip.name} | {mode}"] = pd.Series(stats[mode].to_numpy())
        error_col = "SD_within_chip" if mode == "MEAN" else "SUM_SD_independent_estimate"
        individual[f"spectra/{i:03d}_{_chip_safe_name(chip.name)}_{mode.lower()}.csv"] = stats[
            ["Raman_shift_cm-1", mode, error_col, "N_points"]
        ].copy()
    return pd.DataFrame(combined), individual, selected, common_grid


def _chip_settings(prefix):
    mode_label = st.radio("칩 내 mapping 신호 요약 방식", ["평균 (MEAN)", "합 (SUM)"], horizontal=True,
                          key=f"{prefix}_mode")
    mode = "MEAN" if mode_label.startswith("평균") else "SUM"
    with st.expander("Baseline 및 통계 설정", expanded=False):
        method = st.selectbox("각 측정점의 baseline", ["없음", "AsLS", "Polynomial"],
                              key=f"{prefix}_baseline")
        st.caption("각 측정점에 baseline을 먼저 적용하고 평균·합을 계산합니다. 이미 보정한 CSV는 '없음'을 선택하세요.")
        degree, lam, p, iterations = 2, 100000.0, 0.0001, 10
        if method == "AsLS":
            a, b, c = st.columns(3)
            lam = a.number_input("AsLS λ", min_value=1.0, value=100000.0,
                                 format="%.6g", key=f"{prefix}_lam")
            p = b.number_input("AsLS p", min_value=0.000001, max_value=0.499999,
                               value=0.0001, format="%.6f", key=f"{prefix}_p")
            iterations = c.number_input("반복 횟수", min_value=1, max_value=200,
                                        value=10, step=1, key=f"{prefix}_iterations")
        elif method == "Polynomial":
            degree = st.number_input("Polynomial 차수", min_value=0, max_value=10,
                                      value=2, step=1, key=f"{prefix}_degree")
        independent = st.checkbox("독립·동일분포 가정으로 SUM의 SD와 MEAN의 SEM 추정", value=False,
                                  key=f"{prefix}_independent")
        st.caption("SD는 표본 표준편차(ddof=1)입니다. 칩 내 SD는 측정 위치 간 편차, 칩 간 SD는 칩별 값의 편차입니다.")
        if independent:
            st.info("추정식: SUM SD = √n × SD, MEAN SEM = SD / √n. 칩 내 계산의 n은 측정점 수, 칩 간 계산의 n은 칩 수입니다. 위치 간 공간 상관은 반영하지 않습니다.")
        else:
            st.caption("독립성 가정이 꺼져 있으면 SUM_SD_independent_estimate와 MEAN_SEM_independent_estimate는 빈칸입니다.")
    a, b = st.columns(2)
    lo = a.number_input("피크 검색 시작 (cm⁻¹)", value=900.0, format="%.6g", key=f"{prefix}_lo")
    hi = b.number_input("피크 검색 끝 (cm⁻¹)", value=950.0, format="%.6g", key=f"{prefix}_hi")
    st.caption("검색 구간의 양 끝을 포함하며, 실제 측정된 파수에서 최대값을 찾습니다. 평활화와 정규화는 적용하지 않습니다.")
    return mode, lo, hi, ProcessOptions(baseline=method, degree=int(degree), lam=float(lam),
                                       p=float(p), iterations=int(iterations), independent=independent)


def _chip_figure_download(fig, stem, prefix, settings):
    st.pyplot(fig)
    png = figure_bytes(fig, "png", dpi=settings.get("dpi", 180))
    choice = st.selectbox("그래프 저장 형식", ["PNG", "PDF", "SVG"], key=f"{prefix}_format")
    extension = choice.lower()
    data = png if extension == "png" else figure_bytes(fig, extension, dpi=settings.get("dpi", 180))
    mime = {"png": "image/png", "pdf": "application/pdf", "svg": "image/svg+xml"}[extension]
    _chip_download_button(f"{choice} 저장", data, f"{stem}.{extension}", mime, key=f"{prefix}_download")
    plt.close(fig)
    return png


def _chip_uploads(prefix, between):
    """Return (files, assigned condition names, stable upload slot identifiers)."""
    if not between:
        uploaded = st.file_uploader("한 칩의 mapping 파일 업로드", type=["txt", "csv", "tsv"],
                                    key=f"{prefix}_upload")
        return ([uploaded], ["단일 칩"], ["single"]) if uploaded else ([], [], [])
    st.caption("같은 조건의 well 파일들을 한 업로드 창에 넣으세요. 조건명은 각 창에서 바꿀 수 있습니다.")
    st.caption("여러 업로드 창에 같은 조건명을 입력하면 하나의 조건으로 묶습니다.")
    count = int(st.number_input("조건 수", min_value=1, max_value=24, value=3, step=1,
                                key=f"{prefix}_condition_count"))
    files, groups, slots = [], [], []
    for row_start in range(0, count, 2):
        columns = st.columns(2)
        for offset, col in enumerate(columns):
            i = row_start + offset
            if i >= count:
                continue
            with col:
                with st.container(border=True):
                    label = st.text_input(f"조건 {i+1} 이름", value=f"조건 {i+1}",
                                          key=f"{prefix}_condition_label_{i}").strip()
                    selected = st.file_uploader(f"조건 {i+1}의 well 파일들", type=["txt", "csv", "tsv"],
                                                accept_multiple_files=True, key=f"{prefix}_condition_upload_{i}")
                    st.caption(f"{len(selected or [])}개 well")
                    for j, uploaded in enumerate(selected or []):
                        files.append(uploaded)
                        groups.append(label)
                        slots.append(f"condition_{i+1:02d}_file_{j+1:03d}")
    return files, groups, slots


def _chip_zip(tables, manifest, spectrum_png=None, graphs=None):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for filename, table in tables.items():
            archive.writestr(filename, _chip_csv(table))
        archive.writestr("processing_manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("column_definitions.txt", _CHIP_COLUMN_HELP)
        if spectrum_png:
            archive.writestr("spectra.png", spectrum_png)
        for filename, data in (graphs or {}).items():
            archive.writestr(f"graphs/{filename}", data)
    return buffer.getvalue()


_CHIP_COLUMN_HELP = """RSLAB chip analysis column definitions
Raman_shift_cm-1: measured spectral axis, sorted ascending.
MEAN / SUM: mean / sum of processed mapping points at each Raman shift.
SD_within_chip: sample SD across spatial mapping points, ddof=1. This is not SEM.
SUM_SD_independent_estimate: sqrt(n)*sample SD, only when independence and identical-distribution approximation is enabled. Otherwise empty.
MEAN_SEM_independent_estimate: sample SD/sqrt(n), under the same assumption. Otherwise empty.
N_points: number of mapping points in a chip.
peak_x / peak_intensity: sampled maximum in inclusive user window. Each chip contributes one value.
tie_count: count of exact tied maxima. Lowest Raman shift is selected.
peak_at_boundary: selected peak is at the first or last sampled point inside the window.
N_chips: number of chips contributing to a group, not total mapping point count.
peak_mean / peak_sd / peak_sum: mean / sample SD / sum of separate chip peak intensities.
MEAN_between_chips / SD_between_chips / SUM_between_chips: per-shift statistics of separate representative chip spectra with equal chip weights.
Overview method=chip_peak: one bar per chip, evaluated at that chip's representative-spectrum maximum. Different chips can use different shifts.
Overview method=fixed_x: one bar per chip at the sampled shift nearest requested_x. No interpolation; actual shift and x_offset are recorded.
value: MEAN or SUM at that selected shift. For MEAN it matches the chip peak only in chip_peak mode.
error: within-chip spatial sample SD at the selected shift for MEAN; optional independent SUM SD estimate for SUM. It is not the SD of separate mapping-point maxima.
error_basis: within_chip_spatial_SD / independent_sum_SD_estimate / not_estimated.
requested_x: explicit reference shift for fixed_x, empty for chip_peak.
x_offset: actual selected Raman shift minus requested_x, empty for chip_peak.
value_mean / value_sd / value_sum: equal-chip condition statistics of the exact overview bar heights. Fixed-x results are separate from maximum-peak statistics.
Display stacking, colors, curve/legend order, labels and axis limits never alter analytical CSV values.
chip_mean_spectra.csv / chip_sum_spectra.csv: one mapping-aggregate spectrum per chip. These are not single mapping-point spectra or an average across chips.
Spectrum export uses a common Raman shift column when X grids match; otherwise each chip has its own X/Y pair, padded with empty cells. No interpolation is applied.
spectra/*_mean.csv / spectra/*_sum.csv: individual chip spectrum with corresponding SD and point count. Exports use full spectral ranges and exclude display offsets.
All SDs require at least two observations. Undefined values are exported empty.
Peak of an averaged spectrum is not the mean of individual chip peaks.
No smoothing, normalization, spectral interpolation, or automatic outlier removal.
"""


def _chip_manifest(options, mode, lo, hi, sources):
    from dataclasses import asdict
    return {"module_version": "2.1", "statistic": mode, "peak_window_cm-1": [lo, hi],
            "options": asdict(options), "sd_ddof": 1,
            "processing_order": "parse -> baseline each mapping point -> within-chip mean/sum -> each-chip maximum -> group statistics",
            "peak_tie_rule": "lowest sampled Raman shift", "grid_policy": "strict; no interpolation",
            "group_weighting": "one chip, one equal weight", "sources": sources}


def render_chip_tools(between=False):
    """Render one analysis page. Plot settings never modify analytical arrays."""
    prefix = "chip_between" if between else "chip_within"
    section_number = 0

    def section(title):
        nonlocal section_number
        section_number += 1
        _chip_section(title, section_number)

    st.header("칩 간 데이터 처리" if between else "칩 내 데이터 처리")
    st.write("파일 1개를 칩 1개로 처리합니다. HORIBA mapping TXT와 전치된 mapping CSV를 읽을 수 있습니다.")
    section("조건별 파일 업로드" if between else "mapping 파일 업로드")
    files, assigned_groups, slots = _chip_uploads(prefix, between)
    section("분석 설정")
    mode, lo, hi, options = _chip_settings(prefix)
    if lo >= hi or not np.isfinite([lo, hi]).all():
        st.error("피크 검색 시작값은 끝값보다 작아야 하며 두 값 모두 유한해야 합니다.")
        return
    if not files:
        st.info("mapping 파일을 올리면 측정점 수를 자동으로 확인하고 결과를 계산합니다.")
        return
    if any(not str(label).strip() for label in assigned_groups):
        st.error("파일을 올린 조건의 이름을 모두 입력하세요.")
        return
    signature = hashlib.sha256(b"".join(slot.encode()+f.name.encode()+hashlib.sha256(f.getvalue()).digest()
                                       for slot, f in zip(slots, files))).hexdigest()[:16]
    chips, sources, errors = [], [], []
    with st.spinner("칩별 mapping 데이터를 계산하는 중입니다."):
        for index, source in enumerate(files):
            try:
                chip = _chip_cached_process(source.getvalue(), source.name, options)
                chips.append(chip)
                sources.append({"file_id": slots[index], "source_file": source.name,
                                "upload_group": assigned_groups[index],
                                "sha256": hashlib.sha256(source.getvalue()).hexdigest(),
                                "N_points": int(chip.y.shape[1]), "N_shifts": int(len(chip.x)),
                                "detected_format": chip.format})
            except (ValueError, TypeError, UnicodeError) as exc:
                errors.append(f"{source.name}: {exc}")
    if errors:
        for message in errors:
            st.error(message)
        st.info("모든 파일을 읽을 수 있도록 수정한 후 다시 업로드하세요. 누락된 칩이 있는 상태로 그룹 통계를 계산하지 않습니다.")
        return
    if between:
        section("칩 이름과 조건 확인")
        st.caption("조건은 업로드 창에 따라 배정됩니다. 조건명은 위쪽 각 업로드 창에서, 칩 이름은 아래 표에서 변경하세요.")
        defaults = pd.DataFrame([{"file_id": s["file_id"], "source_file": s["source_file"],
                                  "chip_name": _chip_default_name(s["source_file"]),
                                  "group_label": s["upload_group"], "N_points": s["N_points"]} for s in sources])
        edited = st.data_editor(defaults, hide_index=True, use_container_width=True,
                                disabled=["file_id", "source_file", "group_label", "N_points"],
                                key=f"{prefix}_groups_{signature}").copy()
        # Condition names belong to their upload slots, not the editor's stale state.
        edited["group_label"] = assigned_groups
        edited, name_issues = _chip_metadata_issues(edited)
        if not name_issues.empty:
            st.error(f"칩 이름 중복 또는 빈칸이 있는 파일 {len(name_issues)}개를 아래 빨간 표에 표시했습니다. 이름을 수정하면 다시 검사합니다.")
            _chip_warning_table(name_issues, "name")
            st.caption("위 편집표의 chip_name 칸에서 이름을 고친 뒤 Enter를 누르거나 다른 칸을 클릭하세요. 조건명은 해당 업로드 창에서 수정하세요. 원본 파일명은 바뀌지 않습니다.")
            return
    else:
        section("칩 이름 확인")
        name = st.text_input("칩 이름", value=_chip_default_name(files[0].name), key=f"{prefix}_name_{signature}").strip()
        if not name:
            st.error("칩 이름을 입력하세요.")
            return
        edited = pd.DataFrame([{"chip_name": name, "group_label": "단일 칩"}])
    named_chips = [replace(chip, name=str(edited.iloc[i].chip_name)) for i, chip in enumerate(chips)]
    for i, source in enumerate(sources):
        source.update(chip_name=named_chips[i].name, group_label=str(edited.iloc[i].group_label))
    if between:
        duplicate_data = _chip_duplicate_data_rows(chips, sources)
        if not duplicate_data.empty:
            st.warning("분석에 사용하는 파수·intensity 값이 완전히 같은 파일이 있습니다. 아래 표에서 같은 묶음에 속한 파일들을 확인하세요.")
            _chip_warning_table(duplicate_data, "data")
            st.caption("현재 baseline 설정을 적용한 전체 mapping 배열을 비교합니다. 평균 스펙트럼이나 파일명만으로 판단하지 않습니다. 잘못 올린 파일은 해당 업로드 창에서 삭제·교체하세요. 칩 이름을 바꿔도 이 경고는 사라지지 않습니다.")
            if not st.checkbox("동일한 데이터가 서로 다른 칩에서 측정된 것이므로 포함", key=f"{prefix}_duplicates_{signature}"):
                return
        if mode == "SUM" and len({c.y.shape[1] for c in chips}) > 1:
            st.warning("칩별 측정점 수가 다릅니다. SUM은 측정점 수의 영향을 받습니다.")
    try:
        peaks = pd.DataFrame([dict(peak_record(c, mode, lo, hi),
                                   group_label=sources[i]["group_label"], source_file=sources[i]["source_file"])
                              for i, c in enumerate(named_chips)])
    except ValueError as exc:
        st.error(str(exc))
        return
    if peaks.peak_at_boundary.any():
        boundary_names = ", ".join(peaks.loc[peaks.peak_at_boundary, "chip_name"].astype(str))
        st.warning(f"{boundary_names}: 최대값이 검색 구간의 첫 또는 마지막 측정점입니다. 피크 전체가 구간에 들어오는지 확인하세요.")
    if (peaks.tie_count > 1).any():
        st.caption("같은 최대값이 여러 곳에 있으면 가장 낮은 파수를 보고합니다. tie_count에 개수를 기록합니다.")

    spectrum_title = "칩별 평균 스펙트럼" if mode == "MEAN" else "칩별 합산 스펙트럼"
    section(spectrum_title)
    point_counts = sorted({c.y.shape[1] for c in named_chips})
    count_description = f"{point_counts[0]}개" if len(point_counts) == 1 else "전체"
    operation = "평균" if mode == "MEAN" else "합산"
    st.info(f"선 하나는 칩(well) 하나의 {count_description} mapping 측정점 intensity를 같은 Raman shift마다 {operation}한 결과입니다.")
    st.caption("각 칩의 전체 mapping을 요약한 곡선입니다. 조건별 칩 간 평균은 아래의 조건별 스펙트럼 통계에서 확인할 수 있습니다." if between
               else "이 곡선은 업로드한 칩의 전체 mapping을 요약한 결과입니다.")
    if options.baseline != "없음":
        st.caption(f"각 측정점에 {options.baseline} baseline 보정을 먼저 적용한 뒤 {operation}합니다.")
    st.caption("그래프 설정의 곡선 순서와 범례 순서를 각각 바꿀 수 있습니다. 수직 스택은 화면과 그래프 파일에만 적용됩니다.")
    show_sd = st.checkbox("칩 내 편차 범위 표시", value=(not between and mode == "MEAN"),
                           key=f"{prefix}_show_sd_{mode}", disabled=mode == "SUM" and not options.independent)
    plot_settings, graphs = {}, {}
    try:
        settings = graph_editor([c.name for c in named_chips], f"{prefix}_spectra", spectrum=True)
        plot_settings["spectra"] = settings
        fig = draw_spectra(named_chips, mode, lo, hi, settings, show_sd=show_sd, independent=options.independent)
        png = _chip_figure_download(fig, "chip_spectra", f"{prefix}_spectra", settings)
    except ValueError as exc:
        st.error(f"스펙트럼 그래프 설정: {exc}")
        return
    st.caption("파수–intensity 수치 데이터 다운로드")
    scope_label = st.radio("저장할 곡선", ["현재 그래프에 표시한 칩", "업로드한 모든 칩"],
                           horizontal=True, key=f"{prefix}_spectrum_export_scope")
    spectrum_scope = "visible" if scope_label == "현재 그래프에 표시한 칩" else "all"
    spectrum_data, spectrum_individual, export_chips, common_grid = _chip_spectrum_tables(
        named_chips, mode, plot_settings["spectra"], scope=spectrum_scope, independent=options.independent)
    spectrum_filename = f"chip_{mode.lower()}_spectra.csv"
    spectrum_sources = [next(source for source in sources if source["chip_name"] == chip.name) for chip in export_chips]
    spectrum_tables = {spectrum_filename: spectrum_data, **spectrum_individual,
                       "spectrum_export_chips.csv": pd.DataFrame(spectrum_sources)}
    spectrum_export = {"statistic": mode, "scope": spectrum_scope,
                       "chip_names": [c.name for c in export_chips], "common_x_grid": common_grid,
                       "mapping_aggregation": "all mapping points within each chip",
                       "spectral_range": "full", "display_offsets_applied": False,
                       "interpolation": False}
    spectrum_manifest = _chip_manifest(options, mode, lo, hi, spectrum_sources)
    spectrum_manifest.update(spectrum_export=spectrum_export)
    st.caption("수치 파일은 전체 파수의 분석 intensity를 저장합니다. 그래프의 수직 스택·개별 오프셋과 축 표시 범위는 수치에 적용하지 않습니다.")
    if not common_grid:
        st.caption("칩별 파수축이 달라 통합 CSV에는 각 칩의 X–Y 쌍을 따로 저장합니다. 길이가 짧은 열은 빈칸으로 두며 보간하지 않습니다.")
    csv_col, zip_col = st.columns(2)
    with csv_col:
        _chip_download_button(f"칩별 {operation} 스펙트럼 CSV 저장", _chip_csv(spectrum_data), spectrum_filename,
                              "text/csv", key=f"{prefix}_spectrum_csv", use_container_width=True)
    with zip_col:
        _chip_download_button("칩별 스펙트럼 CSV 묶음 저장", _chip_zip(spectrum_tables, spectrum_manifest),
                              f"chip_{mode.lower()}_spectra.zip", "application/zip", key=f"{prefix}_spectrum_zip",
                              use_container_width=True)
    section("칩별 구간 최대값")
    st.dataframe(peaks, use_container_width=True, hide_index=True)
    _chip_download_button("칩별 최대값 CSV 저장", _chip_csv(peaks), "chip_peak_values.csv", "text/csv", key=f"{prefix}_peaks")
    tables = {"chip_peak_values.csv": peaks, "chip_conditions.csv": pd.DataFrame(sources), **spectrum_tables}
    for i, chip in enumerate(named_chips, 1):
        stem = f"{i:03d}_{_chip_safe_name(chip.name)}"
        summary = summary_frame(chip, independent=options.independent)
        tables[f"{stem}_statistics.csv"] = summary
        points = pd.DataFrame(chip.y, columns=[f"point_{j+1:03d}" for j in range(chip.y.shape[1])])
        points.insert(0, "Raman_shift_cm-1", chip.x)
        tables[f"{stem}_mapping_processed.csv"] = points
        if chip.coordinates is not None:
            coord_table = pd.DataFrame(chip.coordinates)
            coord_table.insert(0, "point_id", [f"point_{j+1:03d}" for j in range(chip.y.shape[1])])
            tables[f"{stem}_coordinates.csv"] = coord_table
        if not between:
            section("각 파수의 칩 내 통계")
            st.caption(f"{chip.y.shape[1]}개 측정점, {len(chip.x)}개 파수. MEAN 옆 SD는 위치 간 표준편차입니다.")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            _chip_download_button("평균·SD·합 CSV 저장", _chip_csv(summary), f"{stem}_statistics.csv", "text/csv", key=f"{prefix}_stats")

    skipped_spectrum_groups = []
    overview_config = None
    if between:
        grouped = group_peak_summary(peaks, independent=options.independent)
        group_names = list(edited.group_label.drop_duplicates())
        section("조건별 칩 간 통계")
        st.caption("색깔 점은 각 well, 검은 점과 오차막대는 조건별 well 평균±칩 간 SD입니다. 같은 조건의 점은 같은 세로줄에 놓입니다.")
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        _chip_download_button("조건별 통계 CSV 저장", _chip_csv(grouped), "group_peak_statistics.csv", "text/csv", key=f"{prefix}_group_stats")
        tables["group_peak_statistics.csv"] = grouped
        try:
            settings = graph_editor(group_names, f"{prefix}_groups", spectrum=False)
            plot_settings["group_peaks"] = settings
            fig = draw_group_peaks(peaks, grouped, settings)
            graphs["group_peak_statistics.png"] = _chip_figure_download(fig, "group_peak_statistics", f"{prefix}_groups", settings)
        except ValueError as exc:
            st.error(f"조건별 그래프 설정: {exc}")
            return

        section("칩 내·칩 간 편차 한눈에 보기")
        basis_label = st.radio("well 막대 높이의 기준", ["각 칩의 구간 최대값", "동일 파수에서 비교"],
                               horizontal=True, key=f"{prefix}_overview_basis")
        method = "chip_peak" if basis_label == "각 칩의 구간 최대값" else "fixed_x"
        reference = None
        if method == "fixed_x":
            reference = st.number_input("비교할 Raman shift (cm⁻¹)", value=float(np.median(peaks.peak_x)),
                                         format="%.6g", key=f"{prefix}_overview_x_{signature}")
        show_group = st.checkbox("조건별 well 평균과 칩 간 SD 범위 표시", value=True, key=f"{prefix}_overview_group_sd")
        if mode == "MEAN":
            st.caption("막대 1개는 well 1개의 평균 intensity이며, 오차막대는 그 파수에서 mapping 위치 간 SD입니다. 같은 색은 같은 조건입니다.")
        elif options.independent:
            st.caption("막대는 well의 SUM이며, 오차막대는 독립·동일분포 가정의 SUM SD 추정값입니다.")
        else:
            st.caption("막대는 well의 SUM입니다. SUM의 칩 내 SD 추정은 독립성 가정을 켰을 때만 표시합니다.")
        st.caption("조건별 가로선은 막대 높이들의 평균, 음영은 그 well 값들 사이의 칩 간 ±SD입니다.")
        if method == "chip_peak":
            st.caption(f"각 well은 자기 {operation} 스펙트럼의 최대값 위치에서 비교합니다. 실제 비교 파수는 well마다 다를 수 있습니다.")
        else:
            st.caption("입력 파수에 가장 가까운 실제 측정점을 사용합니다. 요청 파수·실제 파수·차이를 표에 기록합니다.")
        try:
            overview = overview_frame(named_chips, edited.group_label.tolist(), mode=mode, lo=lo, hi=hi,
                                      method=method, reference_x=reference, independent=options.independent)
            overview_groups = overview_group_summary(overview, independent=options.independent)
            settings = graph_editor(group_names, f"{prefix}_overview", spectrum=False)
            with st.expander("well 표시 순서", expanded=False):
                order_table = pd.DataFrame({"chip_name": [c.name for c in named_chips],
                                            "display_order": range(1, len(named_chips)+1)})
                order_table = st.data_editor(order_table, disabled=["chip_name"], hide_index=True,
                                             use_container_width=True, key=f"{prefix}_well_order_{signature}")
                ranks = pd.to_numeric(order_table["display_order"], errors="coerce")
                if not np.isfinite(ranks).all():
                    raise ValueError("well 표시 순서에는 유한한 숫자를 입력하세요.")
                settings["well_order"] = order_table.assign(_rank=ranks).sort_values("_rank", kind="stable")["chip_name"].tolist()
                st.caption("작은 번호부터 표시하며 같은 조건의 well은 함께 배치됩니다.")
            plot_settings["overview"] = settings
            fig = draw_chip_overview(overview, mode=mode, settings=settings, show_group_mean=show_group)
            graphs["chip_overview.png"] = _chip_figure_download(fig, "chip_overview", f"{prefix}_overview", settings)
            tables["chip_overview_values.csv"] = overview
            tables["chip_overview_group_statistics.csv"] = overview_groups
            st.dataframe(overview, use_container_width=True, hide_index=True)
            _chip_download_button("well별 막대·오차값 CSV 저장", _chip_csv(overview), "chip_overview_values.csv", "text/csv", key=f"{prefix}_overview_values")
            st.dataframe(overview_groups, use_container_width=True, hide_index=True)
            _chip_download_button("막대 기준 조건별 통계 CSV 저장", _chip_csv(overview_groups), "chip_overview_group_statistics.csv", "text/csv", key=f"{prefix}_overview_group_values")
            overview_config = {"method": method, "reference_x": reference, "show_group_mean_sd": show_group}
        except ValueError as exc:
            st.error(f"well 비교 그래프: {exc}")
            return

        section("조건별 스펙트럼 통계")
        with st.expander("조건별 평균·SD·합 스펙트럼과 칩별 모음"):
            st.caption("같은 Raman shift 격자의 칩만 한 표로 합칩니다. 다른 격자도 각 칩의 피크값 통계는 계산할 수 있습니다.")
            for group_index, group in enumerate(group_names, 1):
                selected = [c for i, c in enumerate(named_chips) if sources[i]["group_label"] == group]
                try:
                    frame = group_spectrum_frame(selected, mode=mode, independent=options.independent)
                    group_stem = f"group_{group_index:03d}_{_chip_safe_name(group)}"
                    tables[f"{group_stem}_spectra_statistics.csv"] = frame
                    collection = pd.DataFrame({"Raman_shift_cm-1": selected[0].x})
                    for chip in selected:
                        collection[f"{chip.name} | {mode}"] = summary_frame(chip, independent=options.independent)[mode].to_numpy()
                    tables[f"{group_stem}_chip_spectra.csv"] = collection
                    st.write(str(group))
                    st.dataframe(frame, use_container_width=True, hide_index=True)
                    _chip_download_button(f"{group} 스펙트럼 통계 CSV 저장", _chip_csv(frame), f"{group_stem}_spectra_statistics.csv", "text/csv", key=f"{prefix}_spectrum_{group_index}")
                except ValueError as exc:
                    st.warning(f"{group}: {exc}")
                    skipped_spectrum_groups.append({"group_label": str(group), "reason": str(exc)})
    manifest = _chip_manifest(options, mode, lo, hi, sources)
    manifest.update(skipped_group_spectra=skipped_spectrum_groups, graph_settings=plot_settings,
                    overview=overview_config, spectrum_export=spectrum_export, app_version="2.1")
    section("전체 결과 저장")
    _chip_download_button("전체 결과 ZIP 저장", _chip_zip(tables, manifest, png, graphs=graphs),
                       "rslab_between_chip_results.zip" if between else "rslab_within_chip_results.zip",
                       "application/zip", key=f"{prefix}_zip", use_container_width=True)
    st.caption("ZIP에는 통계 CSV, 처리된 mapping, 좌표, 칩 조건, 처리·그래프 설정, 그래프가 포함됩니다.")
