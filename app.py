"""Standalone Streamlit entrypoint for chip mapping analysis."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(
    page_title="RSLAB Chip Spectrum Analyzer", page_icon="💻", layout="wide"
)

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

st.markdown(
    """
    <style>
    [data-testid="stSidebar"] div.stButton > button {
        min-height: 3.25rem;
        border-radius: 0.85rem;
        font-weight: 600;
        text-align: left;
        justify-content: flex-start;
        padding-left: 1rem;
        transition: background-color 120ms ease, border-color 120ms ease;
    }
    .rslab-chip-title {
        display: flex;
        align-items: center;
        gap: 0.7rem;
        margin: 0.25rem 0 0.45rem;
        flex-wrap: wrap;
    }
    .rslab-chip-title h1 {
        font-size: 2rem;
        line-height: 1.3;
        font-weight: 750;
        padding: 0;
        margin: 0;
    }
    </style>
    <div class="rslab-chip-title">
      <h1>RSLAB Chip Spectrum Analyzer</h1>
      <svg width="40" height="40" viewBox="0 0 40 40" fill="none"
           xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Microchip">
        <g stroke="#4f46e5" stroke-width="2.2" stroke-linecap="round" stroke-linejoin="round">
          <rect x="10" y="10" width="20" height="20" rx="4"/>
          <rect x="15" y="15" width="10" height="10" rx="1.5"/>
          <path d="M15 5v5m10-5v5M15 30v5m10-5v5M5 15h5m-5 10h5m20-10h5m-5 10h5"/>
        </g>
      </svg>
    </div>
    """,
    unsafe_allow_html=True,
)
st.caption("Raman mapping 데이터를 칩 내 통계와 조건별 칩 간 통계로 정리합니다.")

PAGES = ("칩 내 데이터 처리", "칩 간 데이터 처리")
# Keep navigation separate from the old radio widget's key. Streamlit may
# remove the old widget-owned key when that radio is no longer rendered.
PAGE_STATE_KEY = "rslab_active_page_v21"
selected_page = st.session_state.get(PAGE_STATE_KEY, st.session_state.get("chip_app_page"))
st.session_state[PAGE_STATE_KEY] = selected_page if selected_page in PAGES else PAGES[0]


def select_chip_page(page):
    """Navigation callbacks update the page before Streamlit reruns the script."""
    st.session_state[PAGE_STATE_KEY] = page


with st.sidebar:
    st.markdown("### 칩 데이터 분석")
    st.caption("분석할 항목을 선택하세요.")
    for page_index, page_label in enumerate(PAGES):
        st.button(
            page_label,
            key=f"chip_app_navigation_{page_index}",
            type="primary" if st.session_state[PAGE_STATE_KEY] == page_label else "secondary",
            use_container_width=True,
            on_click=select_chip_page,
            args=(page_label,),
        )
    st.divider()
    st.caption("RSLAB Chip Spectrum Analyzer · v2.1")

page = st.session_state[PAGE_STATE_KEY]
render_chip_tools(between=page == "칩 간 데이터 처리")
