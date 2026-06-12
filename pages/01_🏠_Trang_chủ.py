import streamlit as st
import pandas as pd
st.set_page_config(
    page_title="AIDEOM-VN",
    page_icon="🇻🇳",
    layout="wide"
)

# ─────────────────────────────────────────────
# CSS DARK THEME (đồng nhất Bài 11)
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Nền tổng thể */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2d2f3e;
}
[data-testid="stSidebar"] * {
    color: #c0c4d0 !important;
}

/* Card metric */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8b8fa8;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #6ee7b7;
    margin-top: 2px;
}

/* Badge cấp độ */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 6px;
}
.badge-hard   { background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed55; }
.badge-rl     { background:#0ea5e922; color:#38bdf8; border:1px solid #0ea5e955; }
.badge-info   { background:#10b98122; color:#6ee7b7; border:1px solid #10b98155; }

/* Thông báo kết quả */
.result-box {
    background: #14532d33;
    border: 1px solid #16a34a66;
    border-radius: 8px;
    padding: 10px 16px;
    color: #86efac;
    font-size: 0.9rem;
}
.warn-box {
    background: #78350f33;
    border: 1px solid #d9770666;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fcd34d;
    font-size: 0.9rem;
}

/* Bảng */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Nút */
.stButton > button {
    background: linear-gradient(135deg,#7c3aed,#6d28d9);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 22px;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Slider label */
[data-testid="stSlider"] label { color: #a0a4b8 !important; font-size:0.85rem; }

h2, h3 { color: #e2e8f0 !important; }
hr { border-color: #2d2f3e; }
</style>
""", unsafe_allow_html=True)
   
st.title("🇻🇳 AIDEOM-VN")

st.subheader(
    "Hệ thống hỗ trợ ra quyết định phát triển kinh tế Việt Nam trong kỷ nguyên AI"
)

st.markdown("""
### Giới thiệu

Bài tập cuối kỳ môn Các mô hình ra quyết định.

Các module:

- Bài 1: Cobb-Douglas + AI
- Bài 2: LP ngân sách số
- Bài 3: Priority 10 ngành
- Bài 4: LP ngành-vùng
- Bài 5: MIP 15 dự án
- Bài 6: TOPSIS 6 vùng
- Bài 7: NSGA-II Pareto
- Bài 8: Động 2026-2035                      
- Bài 9: Lao động & AI   
- Bài 10: Stochastic SP     
- Bài 11: Q-learning RL
- Bài 12: AIDEOM tích hợp          

""")

with st.sidebar:
    st.markdown("### 🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    
    st.divider()
    st.caption("📂 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("⚙️ Tools: Python, Streamlit, PuLP,..")
    st.caption("📘 Dựa trên giáo trình AIDEOM-VN 2026")
