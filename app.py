"""Standalone Streamlit entrypoint for chip mapping analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="Chip Spectrum Analyzer", layout="wide")

# packages.txt installs Nanum Gothic on Community Cloud. System Noto is a fallback.
for font_path in (
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
):
    if font_path.is_file():
        family = font_manager.FontProperties(fname=font_path).get_name()
        if not any(item.name == family for item in font_manager.fontManager.ttflist):
            font_manager.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = [family, "DejaVu Sans"]
        break
plt.rcParams["axes.unicode_minus"] = False

from rslab_chip_ui import render_chip_tools

st.title("Chip Spectrum Analyzer")
st.caption("Raman mapping 데이터를 칩 내 통계와 조건별 칩 간 통계로 정리합니다.")
page = st.sidebar.radio(
    "분석 선택", ["칩 내 데이터 처리", "칩 간 데이터 처리"], key="chip_app_page"
)
render_chip_tools(between=page == "칩 간 데이터 처리")
