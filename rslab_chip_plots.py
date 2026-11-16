"""Editable Matplotlib charts for RSLAB chip mapping analysis.

The editor follows the original RSLAB app's size/font/axis/stack/style controls.
It does not import that app or transform analytical data. Every offset belongs
only to the displayed figure. All drawing functions can run without Streamlit.
"""
from __future__ import annotations

import hashlib
import io
import json

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator
import numpy as np
import pandas as pd


_ORIGIN = ["#4E79A7", "#F28E2B", "#59A14F", "#E15759", "#B07AA1", "#76B7B2", "#EDC948", "#FF9DA7"]
_PASTEL = ["#efafd0", "#c6a0ed", "#a6dbea", "#d7edaa", "#f2c295", "#aacec0", "#c2c7ee", "#dfc6ab"]
_LEGEND_LOCATIONS = {
    "자동": "best", "오른쪽 위": "upper right", "왼쪽 위": "upper left",
    "오른쪽 아래": "lower right", "왼쪽 아래": "lower left",
    "위 가운데": "upper center", "아래 가운데": "lower center",
    "오른쪽 가운데": "center right", "왼쪽 가운데": "center left", "가운데": "center",
}
_MAX_AXIS_TICKS = 2000


def palette_colors(count, palette="Origin"):
    """Return deterministic colors, including arbitrarily many conditions."""
    if count < 1:
        return []
    if palette in ("Origin", "Pastel"):
        choices = _ORIGIN if palette == "Origin" else _PASTEL
        return [choices[i % len(choices)] for i in range(count)]
    cmap = mpl.colormaps.get_cmap(palette)
    return [mcolors.to_hex(cmap(i / max(1, count - 1))) for i in range(count)]


def default_settings(curve_names=(), spectrum=True):
    names = list(dict.fromkeys(str(name) for name in curve_names))
    return {
        "width_px": 1100, "height_px": 500, "dpi": 150,
        "font_family": str(plt.rcParams["font.family"][0]),
        "axis_font_size": 12, "tick_font_size": 10, "title_font_size": 14,
        "legend_font_size": 9, "title": "", "xlabel": "", "ylabel": "",
        "xlim": None, "ylim": None, "x_major": None, "x_minor": None,
        "y_major": None, "y_minor": None, "grid": False, "frame": True,
        "x_log": False, "y_log": False, "plot_style": "line", "marker": "o",
        "frame_width": 1.0, "linewidth": 1.5, "palette": "Origin",
        "colors": dict(zip(names, palette_colors(len(names)))), "linewidths": {},
        "labels": {name: name for name in names}, "visible": names.copy(),
        "order": names.copy(), "legend_order": names.copy(), "offsets": {},
        "stack": False, "stack_auto": True, "stack_step": 0.0,
        "reverse_stack": False, "legend": True, "legend_loc": "best",
        "legend_outside": False, "legend_bbox": (1.02, 1.0), "legend_ncol": 1,
        "peak_window": bool(spectrum), "peak_window_legend": False,
        "peak_markers": bool(spectrum), "transparent": False,
        "x_tick_rotation": 0 if spectrum else 20, "error_capsize": 4.0,
        "error_linewidth": 1.0, "marker_size": 32.0, "bar_width": 0.8,
    }


def _clean_rank_table(frame, names):
    required = {"curve", "plot_rank", "legend_rank", "label", "color", "linewidth", "offset", "visible"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise ValueError("곡선 설정 표의 형식이 올바르지 않습니다.")
    if frame["curve"].tolist() != list(names):
        raise ValueError("곡선 설정 표의 곡선 ID를 변경할 수 없습니다.")
    out = frame.copy()
    for column in ("plot_rank", "legend_rank", "linewidth", "offset"):
        out[column] = pd.to_numeric(out[column], errors="coerce")
        if not np.isfinite(out[column].to_numpy(float)).all():
            raise ValueError(f"곡선 설정의 {column}에는 유한한 숫자를 입력하세요.")
    if (out.linewidth <= 0).any():
        raise ValueError("곡선 선 굵기는 0보다 커야 합니다.")
    for color in out.color:
        if not isinstance(color, str) or not mcolors.is_color_like(color):
            raise ValueError("색상은 #4E79A7 같은 HEX 코드 또는 유효한 색상 이름으로 입력하세요.")
    if out.label.isna().any():
        raise ValueError("범례 이름을 입력하세요. 빈 문자열을 사용하면 이름을 감출 수 있습니다.")
    out["label"] = out.label.astype(str)
    return out


def graph_editor(curve_names: list[str], prefix: str, *, spectrum=True) -> dict:
    """Render an isolated, stable-key graph editor and return serializable settings.

    Rank columns are independent: plot_rank changes stack/curve order while
    legend_rank changes only legend order. Equal ranks retain input order.
    For categorical graphs, curve_names normally contains condition names.
    """
    import streamlit as st

    names = list(dict.fromkeys(str(name) for name in curve_names))
    defaults = default_settings(names, spectrum=spectrum)
    key = lambda item: f"{prefix}_{item}"
    result = defaults.copy()
    with st.expander("그래프 편집", expanded=False):
        size_tab, axis_tab, stack_tab, style_tab, legend_tab = st.tabs(
            ["크기·글꼴", "축 설정", "수직 스택", "곡선·색상·순서", "범례"]
        )
        with size_tab:
            a, b, c = st.columns(3)
            result["width_px"] = int(a.number_input("그림 너비 (px)", min_value=300, max_value=5000,
                value=defaults["width_px"], step=50, key=key("width")))
            result["height_px"] = int(b.number_input("그림 높이 (px)", min_value=250, max_value=4000,
                value=defaults["height_px"], step=50, key=key("height")))
            result["dpi"] = int(c.number_input("PNG DPI", min_value=72, max_value=600,
                value=defaults["dpi"], step=1, key=key("dpi")))
            installed = {font.name for font in mpl.font_manager.fontManager.ttflist}
            fonts = list(dict.fromkeys([defaults["font_family"], "DejaVu Sans", "Liberation Sans",
                *[font for font in ("NanumGothic", "Noto Sans CJK JP", "Noto Sans CJK KR", "Arial", "Times New Roman") if font in installed]]))
            result["font_family"] = st.selectbox("글꼴", fonts, key=key("font"))
            a, b, c, d = st.columns(4)
            result["axis_font_size"] = a.number_input("축 제목 크기", min_value=5, max_value=40, value=12, key=key("axis_font"))
            result["tick_font_size"] = b.number_input("눈금 크기", min_value=5, max_value=32, value=10, key=key("tick_font"))
            result["title_font_size"] = c.number_input("제목 크기", min_value=5, max_value=48, value=14, key=key("title_font"))
            result["legend_font_size"] = d.number_input("범례 크기", min_value=5, max_value=32, value=9, key=key("legend_font"))
            result["transparent"] = st.checkbox("배경 투명하게", value=False, key=key("transparent"))
        with axis_tab:
            result["title"] = st.text_input("그래프 제목", value="", key=key("title"))
            a, b = st.columns(2)
            result["xlabel"] = a.text_input("X축 제목 (빈칸이면 자동)", value="", key=key("xlabel"))
            result["ylabel"] = b.text_input("Y축 제목 (빈칸이면 자동)", value="", key=key("ylabel"))
            for axis in ("x", "y"):
                a, b, c = st.columns(3)
                auto = a.checkbox(f"{axis.upper()}축 범위 자동", value=True, key=key(f"{axis}_auto"))
                low = b.number_input(f"{axis.upper()} 최소", value=0.0, format="%.6g", disabled=auto, key=key(f"{axis}_min"))
                high = c.number_input(f"{axis.upper()} 최대", value=1500.0 if axis == "x" else 100.0,
                    format="%.6g", disabled=auto, key=key(f"{axis}_max"))
                result[f"{axis}lim"] = None if auto else (float(low), float(high))
                result[f"{axis}_log"] = st.checkbox(f"{axis.upper()}축 로그 스케일", value=False,
                    disabled=(not spectrum and axis == "x"), key=key(f"{axis}_log"))
                if spectrum or axis == "y":
                    a, b, c = st.columns(3)
                    ticks_auto = a.checkbox(f"{axis.upper()} 눈금 간격 자동", value=True, key=key(f"{axis}_ticks_auto"))
                    major = b.number_input(f"{axis.upper()} 주 눈금 간격", min_value=0.000001, value=100.0 if axis == "x" else 20.0,
                        format="%.6g", disabled=ticks_auto, key=key(f"{axis}_major"))
                    minor = c.number_input(f"{axis.upper()} 보조 눈금 간격 (0=없음)", min_value=0.0,
                        value=0.0, format="%.6g", disabled=ticks_auto, key=key(f"{axis}_minor"))
                    result[f"{axis}_major"] = None if ticks_auto or result[f"{axis}_log"] else float(major)
                    result[f"{axis}_minor"] = None if ticks_auto or minor == 0 or result[f"{axis}_log"] else float(minor)
            result["x_tick_rotation"] = st.number_input("X 눈금 글자 회전 (°)", min_value=-90, max_value=90,
                value=defaults["x_tick_rotation"], step=5, key=key("rotation"))
            result["grid"] = st.checkbox("격자 표시", value=False, key=key("grid"))
            result["frame"] = st.checkbox("위쪽·오른쪽 테두리 표시", value=True, key=key("frame"))
            result["frame_width"] = st.number_input("축 테두리 굵기", min_value=0.1, max_value=5.0,
                value=1.0, step=0.1, key=key("frame_width"))
        with stack_tab:
            if spectrum:
                result["stack"] = st.checkbox("수직 스택 사용", value=False, key=key("stack"))
                result["stack_auto"] = st.checkbox("곡선 범위로 자동 간격 계산", value=True,
                    disabled=not result["stack"], key=key("stack_auto"))
                result["stack_step"] = st.number_input("곡선 간 추가 간격", value=0.0,
                    format="%.6g", disabled=not result["stack"], key=key("stack_step"))
                result["reverse_stack"] = st.checkbox("스택 위아래 뒤집기", value=False,
                    disabled=not result["stack"], key=key("reverse_stack"))
                st.caption("기본은 표시 순서 1번이 위쪽입니다. 자동 간격을 끄면 입력한 간격만 적용합니다. 오프셋은 그림에만 적용되고 피크·통계·CSV 값은 바뀌지 않습니다.")
            else:
                st.caption("수직 스택은 Raman 스펙트럼에 적용합니다. 막대와 조건별 통계 그래프는 실제 intensity 위치를 유지합니다.")
        with style_tab:
            if spectrum:
                styles = {"라인": "line", "산점도": "scatter", "라인 + 점": "line+marker"}
                result["plot_style"] = styles[st.selectbox("스펙트럼 표시", list(styles), key=key("plot_style"))]
            marker_options = {"원": "o", "사각형": "s", "삼각형": "^", "마름모": "D"}
            result["marker"] = marker_options[st.selectbox("점 모양", list(marker_options), key=key("marker"))]
            result["marker_size"] = st.number_input("점 크기 (pt²)", min_value=1.0, max_value=300.0,
                value=32.0, step=1.0, key=key("marker_size"))
            result["palette"] = st.selectbox("색상 팔레트", ["Origin", "Pastel", "viridis", "plasma", "coolwarm", "tab10"], key=key("palette"))
            result["linewidth"] = st.number_input("기본 선 굵기", min_value=0.1, max_value=8.0,
                value=1.5, step=0.1, key=key("linewidth"))
            colors = palette_colors(len(names), result["palette"])
            signature = hashlib.sha256(json.dumps([names, result["palette"]], ensure_ascii=False).encode()).hexdigest()[:12]
            base = pd.DataFrame([{"curve": name, "visible": True, "plot_rank": i+1, "legend_rank": i+1,
                "label": name, "color": colors[i], "linewidth": result["linewidth"], "offset": 0.0}
                for i, name in enumerate(names)], columns=["curve", "visible", "plot_rank", "legend_rank", "label", "color", "linewidth", "offset"])
            st.caption("plot_rank=표시/스택 순서, legend_rank=범례 순서입니다. 숫자를 바꿔 원하는 순서를 지정하세요. color는 HEX 색상, offset은 개별 곡선의 표시 오프셋입니다.")
            if names:
                edited = st.data_editor(base, disabled=["curve"], hide_index=True, use_container_width=True,
                    column_config={"curve": "곡선 / 조건", "visible": "표시", "plot_rank": "표시 순서",
                        "legend_rank": "범례 순서", "label": "범례 이름", "color": "색상 (HEX)",
                        "linewidth": "선 굵기", "offset": "개별 오프셋"}, key=key(f"curves_{signature}"))
                table = _clean_rank_table(edited, names)
                result["order"] = table.sort_values("plot_rank", kind="stable").curve.tolist()
                result["legend_order"] = table.sort_values("legend_rank", kind="stable").curve.tolist()
                result["visible"] = table.loc[table.visible.fillna(False).astype(bool), "curve"].tolist()
                for field, column in (("labels", "label"), ("colors", "color"), ("linewidths", "linewidth"), ("offsets", "offset")):
                    result[field] = dict(zip(table.curve, table[column]))
            if spectrum:
                result["peak_window"] = st.checkbox("피크 검색 구간 음영 표시", value=True, key=key("peak_window"))
                result["peak_window_legend"] = st.checkbox("검색 구간도 범례에 표시", value=False,
                    disabled=not result["peak_window"], key=key("peak_window_legend"))
                result["peak_markers"] = st.checkbox("구간 최대값 점 표시", value=True, key=key("peak_markers"))
            else:
                a, b = st.columns(2)
                result["error_capsize"] = a.number_input("오차막대 머리 길이", min_value=0.0, max_value=15.0,
                    value=4.0, step=0.5, key=key("capsize"))
                result["error_linewidth"] = b.number_input("오차막대 굵기", min_value=0.1, max_value=5.0,
                    value=1.0, step=0.1, key=key("error_width"))
                result["bar_width"] = st.number_input("막대 너비", min_value=0.1, max_value=1.0,
                    value=.8, step=.05, key=key("bar_width"))
        with legend_tab:
            result["legend"] = st.checkbox("범례 표시", value=True, key=key("legend"))
            label = st.selectbox("범례 위치", list(_LEGEND_LOCATIONS), disabled=not result["legend"], key=key("legend_loc"))
            result["legend_loc"] = _LEGEND_LOCATIONS[label]
            result["legend_outside"] = st.checkbox("범례를 그래프 바깥에 배치", value=False,
                disabled=not result["legend"], key=key("legend_outside"))
            a, b, c = st.columns(3)
            bx = a.number_input("범례 X 위치", value=1.02, step=0.05, disabled=not result["legend_outside"], key=key("legend_x"))
            by = b.number_input("범례 Y 위치", value=1.0, step=0.05, disabled=not result["legend_outside"], key=key("legend_y"))
            result["legend_ncol"] = int(c.number_input("범례 열 수", min_value=1, max_value=12, value=1, step=1, key=key("legend_ncol")))
            result["legend_bbox"] = (float(bx), float(by))
            st.caption("범례 순서는 '곡선·색상·순서'의 legend_rank 열에서 따로 바꿀 수 있습니다.")
    _validate_settings(result)
    return result


def _validate_settings(settings):
    for name in ("width_px", "height_px", "dpi"):
        value = float(settings.get(name, 1))
        if not np.isfinite(value) or value <= 0:
            raise ValueError("그림 크기와 DPI는 0보다 큰 숫자여야 합니다.")
    for axis in ("x", "y"):
        bounds = settings.get(f"{axis}lim")
        if bounds is not None and (len(bounds) != 2 or not np.isfinite(bounds).all() or bounds[0] >= bounds[1]):
            raise ValueError(f"{axis.upper()}축 최소값은 최대값보다 작아야 합니다.")
        if settings.get(f"{axis}_log") and bounds is not None and bounds[0] <= 0:
            raise ValueError(f"{axis.upper()}축 로그 스케일의 범위는 0보다 커야 합니다.")
        for tick in ("major", "minor"):
            value = settings.get(f"{axis}_{tick}")
            if value is not None and (not np.isfinite(value) or value <= 0):
                raise ValueError("눈금 간격은 0보다 커야 합니다.")
    for value in settings.get("colors", {}).values():
        if not mcolors.is_color_like(value):
            raise ValueError("색상 설정을 확인해 주세요.")


def _settings(settings, names, spectrum=True):
    result = default_settings(names, spectrum)
    if settings:
        result.update(settings)
    _validate_settings(result)
    return result


def _ordered(names, settings):
    available = list(dict.fromkeys(str(name) for name in names))
    order = [name for name in settings.get("order", []) if name in available]
    order.extend(name for name in available if name not in order)
    visible = set(settings.get("visible", available))
    return [name for name in order if name in visible]


def _figure(settings):
    fig, ax = plt.subplots(figsize=(settings["width_px"]/settings["dpi"], settings["height_px"]/settings["dpi"]),
        dpi=settings["dpi"], layout="constrained")
    if settings.get("transparent"):
        fig.patch.set_alpha(0)
        ax.patch.set_alpha(0)
    return fig, ax


def _style_axes(ax, settings, xlabel, ylabel, categorical=False):
    # Validate BEFORE constructing a MultipleLocator or requesting tick labels.
    # Tiny user-entered spacing can otherwise make Matplotlib allocate billions
    # of tick locations while drawing an otherwise small figure.
    for axis_name in ("x", "y"):
        if (categorical and axis_name == "x") or settings.get(f"{axis_name}_log"):
            continue
        bounds = settings.get(f"{axis_name}lim")
        if bounds is None:
            bounds = getattr(ax, f"get_{axis_name}lim")()
        span = abs(float(bounds[1]) - float(bounds[0]))
        for kind in ("major", "minor"):
            step = settings.get(f"{axis_name}_{kind}")
            if step is not None:
                estimated = span / float(step)
                # MultipleLocator can include a tick outside either limit.
                if not np.isfinite(estimated) or estimated + 3 > _MAX_AXIS_TICKS:
                    plt.close(ax.figure)
                    raise ValueError(
                        f"{axis_name.upper()}축 {'주' if kind == 'major' else '보조'} 눈금이 "
                        f"{_MAX_AXIS_TICKS}개를 넘습니다. 눈금 간격을 늘리거나 축 범위를 줄여 주세요."
                    )
    font = settings["font_family"]
    ax.set_xlabel(settings.get("xlabel") or xlabel, fontsize=settings["axis_font_size"], fontfamily=font)
    ax.set_ylabel(settings.get("ylabel") or ylabel, fontsize=settings["axis_font_size"], fontfamily=font)
    ax.set_title(settings.get("title", ""), fontsize=settings["title_font_size"], fontfamily=font)
    ax.tick_params(axis="both", labelsize=settings["tick_font_size"])
    for label in [*ax.get_xticklabels(), *ax.get_yticklabels()]:
        label.set_fontfamily(font)
    for label in ax.get_xticklabels():
        label.set_rotation(settings["x_tick_rotation"])
        if settings["x_tick_rotation"]:
            label.set_ha("right" if settings["x_tick_rotation"] > 0 else "left")
    for axis_name in ("x", "y"):
        if settings.get(f"{axis_name}_log") and not (categorical and axis_name == "x"):
            getattr(ax, f"set_{axis_name}scale")("log")
        bounds = settings.get(f"{axis_name}lim")
        if bounds is not None:
            getattr(ax, f"set_{axis_name}lim")(*bounds)
        axis = getattr(ax, f"{axis_name}axis")
        if not (categorical and axis_name == "x") and not settings.get(f"{axis_name}_log"):
            for kind in ("major", "minor"):
                step = settings.get(f"{axis_name}_{kind}")
                if step is not None:
                    getattr(axis, f"set_{kind}_locator")(MultipleLocator(step))
    for name, spine in ax.spines.items():
        spine.set_linewidth(settings["frame_width"])
        if name in ("right", "top"):
            spine.set_visible(settings["frame"])
    if settings["grid"]:
        ax.grid(True, alpha=.2)
    else:
        ax.grid(False)
    ax.set_axisbelow(True)


def _legend(ax, handles, settings, extra=()):
    if not settings.get("legend"):
        return
    order = [name for name in settings.get("legend_order", []) if name in handles]
    order.extend(name for name in handles if name not in order)
    pairs = [(handles[name], settings.get("labels", {}).get(name, name)) for name in order]
    pairs.extend(extra)
    pairs = [(handle, label) for handle, label in pairs if str(label)]
    if not pairs:
        return
    kwargs = {"loc": settings["legend_loc"], "ncol": settings["legend_ncol"],
        "prop": {"size": settings["legend_font_size"], "family": settings["font_family"]}}
    if settings.get("legend_outside"):
        kwargs.update(loc="upper left" if settings["legend_loc"] == "best" else settings["legend_loc"],
            bbox_to_anchor=settings["legend_bbox"])
    ax.legend([pair[0] for pair in pairs], [pair[1] for pair in pairs], **kwargs)


def _curve_values(chip, mode):
    if mode == "MEAN":
        return np.mean(chip.y, axis=1)
    if mode == "SUM":
        return np.sum(chip.y, axis=1)
    raise ValueError("대표 스펙트럼은 MEAN 또는 SUM이어야 합니다.")


def spectrum_offsets(chips, mode, settings):
    """Calculate additive DISPLAY offsets with first curve on top by default."""
    lookup = {str(chip.name): chip for chip in chips}
    names = _ordered(lookup, settings)
    offsets = {name: float(settings.get("offsets", {}).get(name, 0.0)) for name in names}
    if not settings.get("stack") or len(names) < 2:
        return offsets
    bottom_to_top = names if settings.get("reverse_stack") else list(reversed(names))
    previous_upper = None
    ranges = {}
    for name in names:
        chip, y = lookup[name], _curve_values(lookup[name], mode)
        mask = np.isfinite(y)
        if settings.get("xlim") is not None:
            lo, hi = settings["xlim"]
            mask &= (chip.x >= lo) & (chip.x <= hi)
        values = y[mask] if mask.any() else y[np.isfinite(y)]
        ranges[name] = (float(np.min(values)), float(np.max(values)))
    for index, name in enumerate(bottom_to_top):
        low, high = ranges[name]
        if settings.get("stack_auto"):
            gap = .10 * max(high-low, abs(high)*.05, 1e-12)
            base = 0.0 if previous_upper is None else previous_upper + gap - low
            offsets[name] += base + float(settings["stack_step"]) * index
            previous_upper = high + base
        else:
            offsets[name] += float(settings["stack_step"]) * index
    return offsets


def draw_spectra(chips, mode, lo, hi, settings=None, show_sd=False, independent=False):
    """Draw chip representatives; stacking, legend order and crop are view-only."""
    if not chips or not np.isfinite([lo, hi]).all() or lo >= hi:
        raise ValueError("스펙트럼과 올바른 피크 검색 범위가 필요합니다.")
    names = [str(chip.name) for chip in chips]
    settings = _settings(settings, names)
    ordered = _ordered(names, settings)
    if not ordered:
        raise ValueError("표시할 곡선을 한 개 이상 선택하세요.")
    lookup = {str(chip.name): chip for chip in chips}
    offsets = spectrum_offsets(chips, mode, settings)
    fig, ax = _figure(settings)
    handles = {}
    for i, name in enumerate(ordered):
        chip = lookup[name]
        x, raw = np.asarray(chip.x), _curve_values(chip, mode)
        y = raw + offsets[name]
        color = settings["colors"].get(name, palette_colors(len(ordered), settings["palette"])[i])
        width = settings["linewidths"].get(name, settings["linewidth"])
        if settings.get("x_log") and np.any(x <= 0):
            plt.close(fig)
            raise ValueError("X축 로그 스케일에는 0보다 큰 파수가 필요합니다.")
        if settings.get("y_log") and np.any(y <= 0):
            plt.close(fig)
            raise ValueError("Y축 로그 스케일에는 표시값이 모두 0보다 커야 합니다. 선형 축을 사용하거나 표시 오프셋을 조절하세요.")
        style = settings.get("plot_style", "line")
        line, = ax.plot(x, y, color=color, linewidth=width,
            linestyle="None" if style == "scatter" else "-",
            marker=settings.get("marker", "o") if style != "line" else None,
            markersize=np.sqrt(settings["marker_size"]))
        handles[name] = line
        if show_sd:
            sd = np.std(chip.y, axis=1, ddof=1) if chip.y.shape[1] > 1 else np.full(len(x), np.nan)
            if mode == "SUM":
                sd = sd * np.sqrt(chip.y.shape[1]) if independent else np.full(len(x), np.nan)
            if np.isfinite(sd).any():
                if settings.get("y_log") and np.any((y-sd)[np.isfinite(sd)] <= 0):
                    plt.close(fig)
                    raise ValueError("SD 범위가 0 이하까지 내려가므로 로그 Y축으로 표시할 수 없습니다.")
                ax.fill_between(x, y-sd, y+sd, color=color, alpha=.16)
        mask = (x >= lo) & (x <= hi)
        if settings["peak_markers"] and mask.any():
            ids = np.flatnonzero(mask)
            peak = ids[np.argmax(raw[mask])]
            ax.scatter([x[peak]], [y[peak]], color=color, s=settings["marker_size"], zorder=4)
    extra = []
    if settings["peak_window"]:
        span = ax.axvspan(lo, hi, color="gray", alpha=.10)
        if settings["peak_window_legend"]:
            extra.append((span, "Peak window"))
    ylabel = f"{mode} intensity (a.u.)" + (" + display offset" if settings["stack"] or any(offsets.values()) else "")
    _style_axes(ax, settings, "Raman shift (cm$^{-1}$)", ylabel)
    _legend(ax, handles, settings, extra)
    return fig


def draw_group_peaks(peaks, grouped, settings=None):
    """Plot every chip at its condition's EXACT X; black points show mean ± SD."""
    required = {"group_label", "peak_intensity"}
    if not required.issubset(peaks.columns) or not {"group_label", "peak_mean", "peak_sd"}.issubset(grouped.columns):
        raise ValueError("조건별 그래프에 필요한 피크 통계 열이 없습니다.")
    groups = grouped.group_label.astype(str).drop_duplicates().tolist()
    settings = _settings(settings, groups, spectrum=False)
    order = _ordered(groups, settings)
    if not order:
        raise ValueError("표시할 조건을 한 개 이상 선택하세요.")
    fig, ax = _figure(settings)
    handles = {}
    for pos, name in enumerate(order):
        values = pd.to_numeric(peaks.loc[peaks.group_label.astype(str) == name, "peak_intensity"], errors="coerce").to_numpy(float)
        if not len(values) or not np.isfinite(values).all():
            plt.close(fig)
            raise ValueError("조건별 피크값에 누락되거나 잘못된 숫자가 있습니다.")
        row = grouped.loc[grouped.group_label.astype(str) == name].iloc[0]
        sd = float(row.peak_sd)
        if settings.get("y_log") and (np.any(values <= 0) or float(row.peak_mean) <= 0 or
                (np.isfinite(sd) and float(row.peak_mean)-sd <= 0)):
            plt.close(fig)
            raise ValueError("조건별 값 또는 SD 범위가 0 이하이므로 로그 Y축으로 표시할 수 없습니다.")
        color = settings["colors"].get(name, palette_colors(len(order), settings["palette"])[pos])
        handles[name] = ax.scatter(np.full(len(values), float(pos)), values, color=color,
            s=settings["marker_size"]+6, marker=settings.get("marker", "o"), alpha=.8, zorder=3, edgecolors="white", linewidths=.45)
        ax.errorbar(pos, float(row.peak_mean), yerr=sd if np.isfinite(sd) else None, color="black",
            fmt="o", markersize=5, capsize=settings["error_capsize"], linewidth=settings["error_linewidth"], zorder=4)
    ax.set_xticks(np.arange(len(order)), [settings["labels"].get(name, name) for name in order])
    ax.set_xlim(-.5, len(order)-.5)
    mode = str(peaks.statistic.iloc[0]) if "statistic" in peaks else "representative"
    _style_axes(ax, settings, "Condition", f"Peak intensity of {mode} spectrum (a.u.)", categorical=True)
    extra = [(Line2D([], [], color="black", marker="o", linestyle="-", markersize=4), "Condition mean ± SD")]
    _legend(ax, handles, settings, extra)
    return fig


def _overview_order(overview, settings):
    """Keep condition blocks contiguous even when the supplied order lists wells."""
    frame = overview.copy()
    frame["chip_name"] = frame.chip_name.astype(str)
    frame["group_label"] = frame.group_label.astype(str)
    groups = frame.group_label.drop_duplicates().tolist()
    requested = [str(item) for item in settings.get("order", [])]
    visible = set(str(item) for item in settings.get("visible", [*groups, *frame.chip_name.tolist()]))
    group_mode = bool(set(requested) & set(groups))
    group_order = [item for item in settings.get("group_order", []) if item in groups]
    group_order.extend(item for item in requested if item in groups and item not in group_order)
    if not group_mode:
        for chip_name in requested:
            match = frame.loc[frame.chip_name == chip_name, "group_label"]
            if len(match) and match.iloc[0] not in group_order:
                group_order.append(match.iloc[0])
    group_order.extend(group for group in groups if group not in group_order)
    parts = []
    for group in group_order:
        part = frame.loc[frame.group_label == group].copy()
        if group_mode:
            if group not in visible:
                continue
        else:
            part = part[part.chip_name.isin(visible)]
            rank = {name: i for i, name in enumerate(requested)}
            part["_rank"] = part.chip_name.map(rank).fillna(len(rank))
            part = part.sort_values("_rank", kind="stable").drop(columns="_rank")
        if settings.get("well_order"):
            rank = {str(name): i for i, name in enumerate(settings["well_order"])}
            part["_well_rank"] = part.chip_name.map(rank).fillna(len(rank))
            part = part.sort_values("_well_rank", kind="stable").drop(columns="_well_rank")
        parts.append(part)
    return pd.concat(parts, ignore_index=True) if parts else frame.iloc[:0]


def draw_chip_overview(overview, mode="MEAN", settings=None, show_group_mean=True):
    """One well per bar, spatial error bars, grouped color, group mean ± SD bands.

    Required columns: chip_name, group_label, value, error. The caller defines
    which sampled Raman shift supplies each well value and records it in the
    exported table. Between-chip bands use these exact bar heights, so they
    remain consistent for chip-specific peaks or a common selected shift.
    """
    if not {"chip_name", "group_label", "value", "error"}.issubset(overview.columns) or overview.empty:
        raise ValueError("막대그래프에는 chip_name, group_label, value, error 열이 필요합니다.")
    groups = overview.group_label.astype(str).drop_duplicates().tolist()
    settings = _settings(settings, groups, spectrum=False)
    frame = _overview_order(overview, settings)
    if frame.empty:
        raise ValueError("표시할 조건 또는 칩을 한 개 이상 선택하세요.")
    frame["value"] = pd.to_numeric(frame.value, errors="coerce")
    parsed_error = pd.to_numeric(frame.error, errors="coerce")
    if (frame.error.notna() & parsed_error.isna()).any():
        raise ValueError("오차에는 숫자 또는 빈칸을 입력하세요.")
    frame["error"] = parsed_error
    if not np.isfinite(frame.value).all() or (frame.error.dropna() < 0).any() or np.isinf(frame.error).any():
        raise ValueError("막대 높이는 유한한 숫자, 오차는 0 이상의 숫자 또는 빈칸이어야 합니다.")
    fig, ax = _figure(settings)
    color_lookup = dict(zip(groups, palette_colors(len(groups), "Pastel")))
    color_lookup.update({key: value for key, value in settings.get("colors", {}).items() if key in groups})
    color_lookup.update(settings.get("group_colors", {}))
    positions = np.arange(len(frame), dtype=float)
    handles, extra = {}, []
    band_drawn = False
    for group in frame.group_label.drop_duplicates():
        indexes = np.flatnonzero((frame.group_label == group).to_numpy())
        values = frame.iloc[indexes].value.to_numpy(float)
        color = color_lookup[group]
        ax.bar(positions[indexes], values, width=settings["bar_width"], color=color, edgecolor="black", linewidth=1.1, zorder=2)
        errors = frame.iloc[indexes].error.to_numpy(float)
        good = np.isfinite(errors)
        if settings.get("y_log") and (np.any(values <= 0) or np.any(values[good]-errors[good] <= 0)):
            plt.close(fig)
            raise ValueError("칩 값 또는 오차 범위가 0 이하이므로 로그 Y축으로 표시할 수 없습니다.")
        if good.any():
            ax.errorbar(positions[indexes][good], values[good], yerr=errors[good], fmt="none", color="black",
                capsize=settings["error_capsize"], linewidth=settings["error_linewidth"], zorder=4)
        handles[group] = Patch(facecolor=color, edgecolor="black")
        if show_group_mean:
            center = float(np.mean(values))
            left, right = positions[indexes[0]]-.48, positions[indexes[-1]]+.48
            ax.hlines(center, left, right, color="black", linestyle="--", linewidth=1.3, zorder=5)
            if len(values) > 1:
                between_sd = float(np.std(values, ddof=1))
                if settings.get("y_log") and center-between_sd <= 0:
                    plt.close(fig)
                    raise ValueError("칩 간 SD 범위가 0 이하이므로 로그 Y축으로 표시할 수 없습니다.")
                ax.fill_between([left, right], [center-between_sd]*2, [center+between_sd]*2,
                    color=color, alpha=.19, zorder=1)
                band_drawn = True
        if indexes[-1] < len(frame)-1:
            ax.axvline(positions[indexes[-1]]+.5, color="gray", linestyle=":", linewidth=.7, alpha=.5)
    labels = [settings["labels"].get(name, name) for name in frame.chip_name]
    ax.set_xticks(positions, labels)
    ax.set_xlim(-.6, len(frame)-.4)
    if show_group_mean:
        extra.append((Line2D([], [], color="black", linestyle="--"), "Condition mean"))
        if band_drawn:
            extra.append((Patch(facecolor="gray", alpha=.18), "Between-chip ± SD"))
    _style_axes(ax, settings, "Well / chip", f"{mode} intensity (a.u.)", categorical=True)
    _legend(ax, handles, settings, extra)
    return fig


def figure_bytes(fig, format="png", dpi=None):
    """Export the exact displayed figure to PNG/PDF/SVG without tight recropping."""
    format = str(format).lower()
    if format not in ("png", "pdf", "svg"):
        raise ValueError("저장 형식은 PNG, PDF 또는 SVG여야 합니다.")
    buffer = io.BytesIO()
    fig.savefig(buffer, format=format, dpi=dpi or fig.dpi, transparent=fig.patch.get_alpha() == 0)
    return buffer.getvalue()
