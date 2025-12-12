import streamlit as st


def hidden_page_top_margin() -> None:
    hide_deploy = """
    <style>
    [data-testid="stHeader"] {
        display: none !important;
    }
    [data-testid="stToolbar"] {
        display: none !important;
    }
    .stAppHeader {
        display: none !important;
    }
    .stMainBlockContainer {
        padding-top: 0 !important;
    }
    .stVerticalBlock:first-of-type {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    .st-emotion-cache-6c7yup {
        margin-top: 0 !important;
        padding-top: 0 !important;
    }
    h1, h2, h3, h4, h5, h6, p, div {
        margin-bottom: 0 !important;
        margin-top: 0 !important;
        padding-bottom: 0 !important;
        padding-top: 0 !important;
    }
    hr.compact {
        display: none !important;
        margin: 0 !important;       
        padding: 0 !important;
    }
    </style>
    <hr class="compact">
    """
    st.markdown(hide_deploy, unsafe_allow_html=True)


def minimal_divider() -> None:
    """여백 최소화 가로선을 렌더링합니다.

    기본 st.markdown("---")의 과도한 여백을 제거합니다.
    """
    st.markdown(
        '<div style="height: 1px; background-color: #ddd; margin: 0; padding: 0;"></div>',
        unsafe_allow_html=True,
    )
