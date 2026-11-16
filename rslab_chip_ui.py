"""Streamlit screens for one-chip and between-chip Raman mapping analysis."""
from __future__ import annotations

import hashlib
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
    match = re.search(r"(?<![A-Za-z0-9])([A-H](?:1[0-2]|[1-9]))(?!\d)", filename, re.I)
    return match.group(1).upper() if match else Path(filename).stem


def _chip_settings(prefix):
    mode_label = st.radio("칩 대표 스펙트럼", ["평균 (MEAN)", "합 (SUM)"], horizontal=True,
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
    st.download_button(f"{choice} 저장", data, f"{stem}.{extension}", mime, key=f"{prefix}_download")
    plt.close(fig)
    return png


def _chip_uploads(prefix, between):
    """Return (files, assigned condition names, stable upload slot identifiers)."""
    if not between:
        uploaded = st.file_uploader("한 칩의 mapping 파일 업로드", type=["txt", "csv", "tsv"],
                                    key=f"{prefix}_upload")
        return ([uploaded], ["단일 칩"], ["single"]) if uploaded else ([], [], [])
    st.subheader("조건별 파일 업로드")
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
All SDs require at least two observations. Undefined values are exported empty.
Peak of an averaged spectrum is not the mean of individual chip peaks.
No smoothing, normalization, spectral interpolation, or automatic outlier removal.
"""


def _chip_manifest(options, mode, lo, hi, sources):
    from dataclasses import asdict
    return {"module_version": "2.0", "statistic": mode, "peak_window_cm-1": [lo, hi],
            "options": asdict(options), "sd_ddof": 1,
            "processing_order": "parse -> baseline each mapping point -> within-chip mean/sum -> each-chip maximum -> group statistics",
            "peak_tie_rule": "lowest sampled Raman shift", "grid_policy": "strict; no interpolation",
            "group_weighting": "one chip, one equal weight", "sources": sources}


def render_chip_tools(between=False):
    """Render one analysis page. Plot settings never modify analytical arrays."""
    prefix = "chip_between" if between else "chip_within"
    st.header("칩 간 데이터 처리" if between else "칩 내 데이터 처리")
    st.write("파일 1개를 칩 1개로 처리합니다. HORIBA mapping TXT와 전치된 mapping CSV를 읽을 수 있습니다.")
    files, assigned_groups, slots = _chip_uploads(prefix, between)
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
        st.subheader("칩 이름과 조건 확인")
        st.caption("조건은 업로드 창에 따라 배정됩니다. 조건명은 위쪽 각 업로드 창에서, 칩 이름은 아래 표에서 변경하세요.")
        defaults = pd.DataFrame([{"file_id": s["file_id"], "source_file": s["source_file"],
                                  "chip_name": _chip_default_name(s["source_file"]),
                                  "group_label": s["upload_group"], "N_points": s["N_points"]} for s in sources])
        edited = st.data_editor(defaults, hide_index=True, use_container_width=True,
                                disabled=["file_id", "source_file", "group_label", "N_points"],
                                key=f"{prefix}_groups_{signature}").copy()
        # Condition names belong to their upload slots, not the editor's stale state.
        edited["group_label"] = assigned_groups
        if edited[["chip_name", "group_label"]].isna().any().any():
            st.error("칩 이름과 조건 이름을 모두 입력하세요.")
            return
        for col in ["chip_name", "group_label"]:
            edited[col] = edited[col].astype(str).str.strip()
        if (edited[["chip_name", "group_label"]] == "").any().any() or edited.chip_name.duplicated().any():
            st.error("칩 이름은 중복 없이 입력하고, 칩 이름과 조건 이름을 빈칸으로 두지 마세요.")
            return
    else:
        name = st.text_input("칩 이름", value=_chip_default_name(files[0].name), key=f"{prefix}_name_{signature}").strip()
        if not name:
            st.error("칩 이름을 입력하세요.")
            return
        edited = pd.DataFrame([{"chip_name": name, "group_label": "단일 칩"}])
    named_chips = [replace(chip, name=str(edited.iloc[i].chip_name)) for i, chip in enumerate(chips)]
    for i, source in enumerate(sources):
        source.update(chip_name=named_chips[i].name, group_label=str(edited.iloc[i].group_label))
    if between:
        hashes = [hashlib.sha256(c.x.tobytes()+c.y.tobytes()).hexdigest() for c in chips]
        if len(set(hashes)) < len(hashes):
            st.warning("내용이 완전히 같은 mapping 데이터가 있습니다. 같은 파일을 반복 칩으로 올리지 않았는지 확인하세요.")
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

    st.subheader("대표 스펙트럼")
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
    st.subheader("칩별 구간 최대값")
    st.dataframe(peaks, use_container_width=True, hide_index=True)
    st.download_button("칩별 최대값 CSV 저장", _chip_csv(peaks), "chip_peak_values.csv", "text/csv", key=f"{prefix}_peaks")
    tables = {"chip_peak_values.csv": peaks, "chip_conditions.csv": pd.DataFrame(sources)}
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
            st.subheader("각 파수의 칩 내 통계")
            st.caption(f"{chip.y.shape[1]}개 측정점, {len(chip.x)}개 파수. MEAN 옆 SD는 위치 간 표준편차입니다.")
            st.dataframe(summary, use_container_width=True, hide_index=True)
            st.download_button("평균·SD·합 CSV 저장", _chip_csv(summary), f"{stem}_statistics.csv", "text/csv", key=f"{prefix}_stats")

    skipped_spectrum_groups = []
    overview_config = None
    if between:
        grouped = group_peak_summary(peaks, independent=options.independent)
        group_names = list(edited.group_label.drop_duplicates())
        st.subheader("조건별 칩 간 통계")
        st.caption("색깔 점은 각 well, 검은 점과 오차막대는 조건별 well 평균±칩 간 SD입니다. 같은 조건의 점은 같은 세로줄에 놓입니다.")
        st.dataframe(grouped, use_container_width=True, hide_index=True)
        st.download_button("조건별 통계 CSV 저장", _chip_csv(grouped), "group_peak_statistics.csv", "text/csv", key=f"{prefix}_group_stats")
        tables["group_peak_statistics.csv"] = grouped
        try:
            settings = graph_editor(group_names, f"{prefix}_groups", spectrum=False)
            plot_settings["group_peaks"] = settings
            fig = draw_group_peaks(peaks, grouped, settings)
            graphs["group_peak_statistics.png"] = _chip_figure_download(fig, "group_peak_statistics", f"{prefix}_groups", settings)
        except ValueError as exc:
            st.error(f"조건별 그래프 설정: {exc}")
            return

        st.subheader("칩 내·칩 간 편차 한눈에 보기")
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
            st.caption("각 well은 자기 대표 스펙트럼의 최대값 위치에서 비교합니다. 실제 비교 파수는 well마다 다를 수 있습니다.")
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
            st.download_button("well별 막대·오차값 CSV 저장", _chip_csv(overview), "chip_overview_values.csv", "text/csv", key=f"{prefix}_overview_values")
            st.dataframe(overview_groups, use_container_width=True, hide_index=True)
            st.download_button("막대 기준 조건별 통계 CSV 저장", _chip_csv(overview_groups), "chip_overview_group_statistics.csv", "text/csv", key=f"{prefix}_overview_group_values")
            overview_config = {"method": method, "reference_x": reference, "show_group_mean_sd": show_group}
        except ValueError as exc:
            st.error(f"well 비교 그래프: {exc}")
            return

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
                    st.download_button(f"{group} 스펙트럼 통계 CSV 저장", _chip_csv(frame), f"{group_stem}_spectra_statistics.csv", "text/csv", key=f"{prefix}_spectrum_{group_index}")
                except ValueError as exc:
                    st.warning(f"{group}: {exc}")
                    skipped_spectrum_groups.append({"group_label": str(group), "reason": str(exc)})
    manifest = _chip_manifest(options, mode, lo, hi, sources)
    manifest.update(skipped_group_spectra=skipped_spectrum_groups, graph_settings=plot_settings,
                    overview=overview_config, app_version="2.0")
    st.download_button("전체 결과 ZIP 저장", _chip_zip(tables, manifest, png, graphs=graphs),
                       "rslab_between_chip_results.zip" if between else "rslab_within_chip_results.zip",
                       "application/zip", key=f"{prefix}_zip", use_container_width=True)
    st.caption("ZIP에는 통계 CSV, 처리된 mapping, 좌표, 칩 조건, 처리·그래프 설정, 그래프가 포함됩니다.")
