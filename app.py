import streamlit as st

st.set_page_config(
    page_title="AIDEOM-VN Dashboard",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded"
)

pg = st.navigation([
    st.Page("pages/01_🏠_Trang_chủ.py", title="Trang chủ"),
    st.Page("pages/02_📈Bài 1 — Cobb-Douglas + AI.py", title="Bài 1 — Cobb-Douglas + AI"),
    st.Page("pages/03_💰Bài 2 — LP ngân sách số.py", title="Bài 2 — LP ngân sách số"),
    st.Page("pages/04_🏭_Bài 3 — Priority 10 ngành.py", title="Bài 3 — Priority 10 ngành"),
    st.Page("pages/05_🗺️_Bài 4 — LP ngành-vùng.py", title="Bài 4 — LP ngành-vùng"),
    st.Page("pages/06_📦_Bài 5 — MIP 15 dự án.py", title="Bài 5 — MIP 15 dự án"),
    st.Page("pages/07_📊_Bài 6 — TOPSIS 6 vùng.py", title="Bài 6 — TOPSIS 6 vùng"),
    st.Page("pages/08_🎯_Bài 7 — NSGA-II Pareto.py", title="Bài 7 — NSGA-II Pareto"),
    st.Page("pages/09_⏳_Bài 8 — Động 2026-2035.py", title="Bài 8 — Động 2026-2035"),
    st.Page("pages/10_👷_Bài 9 — Lao động & AI.py", title="Bài 9 — Lao động & AI"),
    st.Page("pages/11_🎲_Bài 10 — Stochastic SP.py", title="Bài 10 — Stochastic SP"),
    st.Page("pages/12_🤖_Bài 11 — Q-learning RL.py", title="Bài 11 — Q-learning RL"),
    st.Page("pages/13_🇻🇳_Bài 12 — AIDEOM tích hợp.py", title="🇻Bài 12 — AIDEOM tích hợp")
    ])

pg.run()