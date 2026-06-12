"""
Bài 12 — AIDEOM-VN Dashboard tích hợp (6 module M1–M6)
Streamlit · Dark theme đồng nhất với Bài 11
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.optimize import minimize, linprog
from pulp import (LpProblem, LpVariable, LpMaximize, lpSum,
                  PULP_CBC_CMD, value as lp_value, LpStatus)

# ─────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bài 12 — AIDEOM-VN | Dashboard tích hợp",
    page_icon="🇻🇳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS DARK THEME (đồng nhất Bài 11)
# ─────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stAppViewContainer"] { background-color:#0f1117; color:#e0e0e0; }
[data-testid="stSidebar"]          { background-color:#1a1d27; border-right:1px solid #2d2f3e; }
[data-testid="stSidebar"] *        { color:#c0c4d0 !important; }
.metric-card {
    background:#1e2130; border:1px solid #2d3250;
    border-radius:10px; padding:16px 20px; text-align:center;
}
.metric-card .label  { font-size:.78rem; color:#8b8fa8; margin-bottom:4px;
                        text-transform:uppercase; letter-spacing:.05em; }
.metric-card .value  { font-size:1.5rem; font-weight:700; color:#a78bfa; }
.metric-card .delta  { font-size:.8rem; color:#6ee7b7; margin-top:2px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:.72rem; font-weight:600; letter-spacing:.04em; margin-right:6px; }
.badge-main  { background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed55; }
.badge-mod   { background:#0ea5e922; color:#38bdf8; border:1px solid #0ea5e955; }
.badge-scen  { background:#10b98122; color:#6ee7b7; border:1px solid #10b98155; }
.result-box  { background:#14532d33; border:1px solid #16a34a66; border-radius:8px;
               padding:10px 16px; color:#86efac; font-size:.9rem; }
.warn-box    { background:#78350f33; border:1px solid #d9770666; border-radius:8px;
               padding:10px 16px; color:#fcd34d; font-size:.9rem; }
.module-box  { background:#1e2130; border:1px solid #2d3250; border-radius:10px;
               padding:14px 18px; margin-bottom:8px; }
.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#6d28d9);
    color:white; border:none; border-radius:8px;
    padding:8px 22px; font-weight:600; font-size:.95rem;
}
.stButton > button:hover { opacity:.88; }
[data-testid="stSlider"] label { color:#a0a4b8 !important; font-size:.85rem; }
h2,h3 { color:#e2e8f0 !important; }
hr { border-color:#2d2f3e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# SIDEBAR (đồng nhất Bài 11)
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.divider()
    st.caption("📂 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("⚙️ Tools: Python, Streamlit, PuLP,..")
    st.caption("📘 Dựa trên giáo trình AIDEOM-VN 2026")


# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("## 🇻🇳 Bài 12 — AIDEOM-VN Dashboard tích hợp")
st.markdown("""
<span class='badge badge-main'>ĐỒ ÁN TÍCH HỢP</span>
<span class='badge badge-mod'>6 module M1-M6</span>
<span class='badge badge-scen'>5 kịch bản chính sách</span>
""", unsafe_allow_html=True)

st.markdown("""
Mô hình AIDEOM-VN tích hợp **6 module**: Dự báo (M1) → Sẵn sàng số (M2) → Phân bổ (M3)
→ Lao động (M4) → Đánh giá rủi ro (M5) → Dashboard (M6).

So sánh **5 kịch bản chính sách** trong Mục 15 của bài báo nguồn.
""")
st.divider()

# ─────────────────────────────────────────────
# DỮ LIỆU DÙNG CHUNG (inline — không cần CSV)
# ─────────────────────────────────────────────
YEARS = np.array([2020,2021,2022,2023,2024,2025])
Y_GDP = np.array([8044.4,8487.5,9513.3,10221.8,11511.9,12847.6])
K_arr = np.array([16500,17800,19600,21300,23500,25900],dtype=float)
L_arr = np.array([53.6,50.5,51.7,52.4,52.9,53.4])
D_arr = np.array([12.0,12.7,14.3,16.5,18.3,19.5])
AI_arr= np.array([55.6,60.2,65.4,67.0,73.8,80.1])
H_arr = np.array([24.1,26.1,26.2,27.0,28.4,29.2])

ALPHA=0.33; BETA_L=0.42; GAMMA_D=0.10; DELTA_AI=0.08; THETA_H=0.07

REGIONS = ["TD Miền núi PB","Đồng bằng sông Hồng","Bắc TB + DH TB","Tây Nguyên","Đông Nam Bộ","ĐB Sông CL"]
ITEMS   = ["I","D","AI","H"]
BETA_MATRIX = np.array([
    [1.15,0.85,0.55,1.30],
    [0.95,1.25,1.40,1.05],
    [1.05,0.95,0.85,1.15],
    [1.20,0.75,0.45,1.35],
    [0.90,1.30,1.55,1.00],
    [1.10,0.85,0.65,1.25],
])
D0 = np.array([38,78,55,32,82,48],dtype=float)

SECTORS = ["Nông-Lâm-TS","CN CBCT","Xây dựng","Bán buôn-lẻ","Tài chính-NH","Logistics","CNTT-TT","GD-ĐT"]
LABOR   = np.array([13.20,11.50,4.80,7.80,0.55,1.95,0.62,2.15])
RISK    = np.array([18,42,25,38,52,35,28,22])/100
A1_LABOR= np.array([8.5,32.5,12.8,22.4,45.8,28.5,62.5,18.5])
B1_LABOR= np.array([45,28,35,32,22,30,20,55])
C1_LABOR= np.array([5.2,62.4,18.5,48.2,72.5,42.8,32.5,12.5])
D1_LABOR= np.array([50,32,42,38,26,36,24,62])

SCENARIOS = {
    "S1 · Truyền thống":  np.array([0.70,0.10,0.10,0.10]),
    "S2 · Số hóa nhanh":  np.array([0.25,0.45,0.15,0.15]),
    "S3 · AI dẫn dắt":    np.array([0.20,0.20,0.45,0.15]),
    "S4 · Bao trùm số":   np.array([0.30,0.20,0.10,0.40]),
    "S5 · Tối ưu cân bằng": None,   # chạy LP
}
SCEN_COLORS = ["#6366f1","#22d3ee","#a78bfa","#34d399","#f59e0b"]

# ─────────────────────────────────────────────
# HÀM TIỆN ÍCH
# ─────────────────────────────────────────────
def compute_tfp(Y,K,L,D,AI,H):
    return Y / (K**ALPHA * L**BETA_L * D**GAMMA_D * AI**DELTA_AI * H**THETA_H)

def forecast_gdp_2030(D30, AI30, H30, K0=25900, L0=53.4, A0=None, tfp_growth=0.012, years=5):
    if A0 is None:
        A0 = compute_tfp(Y_GDP[-1],K_arr[-1],L_arr[-1],D_arr[-1],AI_arr[-1],H_arr[-1])
    K,L = K0, L0
    results = []
    for t in range(1, years+1):
        A = A0 * (1+tfp_growth)**t
        K = K * 1.06
        L = L * 1.01
        D  = D30 * (t/years)
        AI = AI30 * (t/years)
        H  = H30  * (t/years)
        Yt = A * K**ALPHA * L**BETA_L * D**GAMMA_D * AI**DELTA_AI * H**THETA_H
        results.append({"Năm":2025+t,"GDP dự báo":round(Yt,1),"K":round(K,0),"D":round(D,2)})
    return pd.DataFrame(results)

def run_topsis(X, weights, is_benefit):
    R = X / np.sqrt((X**2).sum(axis=0)+1e-12)
    V = R * weights
    A_star = np.where(is_benefit, V.max(0), V.min(0))
    A_neg  = np.where(is_benefit, V.min(0), V.max(0))
    S_star = np.sqrt(((V-A_star)**2).sum(1))
    S_neg  = np.sqrt(((V-A_neg )**2).sum(1))
    return S_neg / (S_star + S_neg + 1e-12)

def run_lp_optimal():
    """M3 — LP tối ưu S5, trả về alloc tỷ trọng [I,D,AI,H]"""
    m = LpProblem("S5_optimal", LpMaximize)
    x = {(r,j): LpVariable(f"x_{r}_{j}", lowBound=0)
         for r in range(6) for j in range(4)}
    m += lpSum(BETA_MATRIX[r,j]*x[r,j] for r in range(6) for j in range(4))
    m += lpSum(x[r,j] for r in range(6) for j in range(4)) <= 50000
    for r in range(6):
        m += lpSum(x[r,j] for j in range(4)) >= 5000
        m += lpSum(x[r,j] for j in range(4)) <= 12000
    m += lpSum(x[r,3] for r in range(6)) >= 12000
    m.solve(PULP_CBC_CMD(msg=False))
    if LpStatus[m.status] == "Optimal":
        mat = np.array([[lp_value(x[r,j]) for j in range(4)] for r in range(6)])
        total = mat.sum()
        alloc = mat.sum(axis=0) / total if total>0 else np.ones(4)/4
        return alloc, lp_value(m.objective), mat
    return np.array([0.25,0.25,0.25,0.25]), 0, np.zeros((6,4))

def simulate_scenario(alloc, budget_annual=1000, T=5):
    """Mô phỏng 5 năm với một kịch bản phân bổ, trả về KPIs"""
    K,D,AI,H = 25900.0, 19.5, 80.1, 29.2
    L = 53.4; A = compute_tfp(Y_GDP[-1],K_arr[-1],L_arr[-1],D_arr[-1],AI_arr[-1],H_arr[-1])
    gdp_series=[Y_GDP[-1]]; netjob_total=0
    for _ in range(T):
        K  = 0.95*K  + alloc[0]*budget_annual
        D  = 0.88*D  + alloc[1]*budget_annual/500
        AI = 0.85*AI + alloc[2]*budget_annual/20
        H  = 0.98*H  + 0.8*alloc[3]*budget_annual/200
        A  = A*(1+0.003*D/100+0.002*AI/100+0.004*H/100)
        Y  = A * K**ALPHA * L**BETA_L * D**GAMMA_D * AI**DELTA_AI * H**THETA_H
        gdp_series.append(round(Y,1))
        # NetJob đơn giản
        nj = sum((A1_LABOR[i]*alloc[2]*budget_annual
                  + B1_LABOR[i]*alloc[3]*budget_annual/10
                  - C1_LABOR[i]*RISK[i]*alloc[2]*budget_annual)
                 for i in range(8))
        netjob_total += nj
    growth = (gdp_series[-1]-gdp_series[0])/gdp_series[0]*100
    gini_proxy = 0.38 - alloc[3]*0.05 + alloc[2]*0.02
    emission   = 0.5*(alloc[0]+alloc[2])
    cyber_risk = alloc[2]*0.4 - alloc[3]*0.15
    return {
        "GDP 2030 (nghìn tỷ)": round(gdp_series[-1],1),
        "Tăng trưởng GDP 5Y (%)": round(growth,2),
        "NetJob ròng (nghìn)": round(netjob_total/1000,1),
        "Gini proxy": round(max(gini_proxy,0.30),3),
        "Phát thải (index)": round(emission,3),
        "Cyber Risk (index)": round(max(cyber_risk,0),3),
        "gdp_series": gdp_series,
    }


# ─────────────────────────────────────────────
# TABS CHÍNH
# ─────────────────────────────────────────────
tab_overview, tab_m3, tab_scen, tab_risk = st.tabs([
    "🏠 Tổng quan (M1-M2)",
    "💰 Phân bổ (M3)",
    "📊 5 Kịch bản (M6)",
    "⚠️ Cảnh báo rủi ro (M4-M5)",
])


# ════════════════════════════════════════════
# TAB 1 — TỔNG QUAN M1 + M2
# ════════════════════════════════════════════
with tab_overview:
    # ── M1 ───────────────────────────────────
    st.markdown("### M1 — Dự báo kinh tế (Cobb-Douglas)")

    A_hist = compute_tfp(Y_GDP, K_arr, L_arr, D_arr, AI_arr, H_arr)
    A_mean = A_hist.mean()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>GDP 2025</div>
            <div class='value'>12.848</div>
            <div class='delta'>nghìn tỷ VND</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>TFP trung bình</div>
            <div class='value'>{A_mean:.4f}</div>
            <div class='delta'>2020–2025</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Tăng trưởng GDP</div>
            <div class='value'>8,02%</div>
            <div class='delta'>2025 (NSO)</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown(f"""<div class='metric-card'>
            <div class='label'>Kinh tế số / GDP</div>
            <div class='value'>19,5%</div>
            <div class='delta'>2025 ước tính</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Nút chạy M1
    col_btn, col_param = st.columns([1, 3])
    with col_btn:
        run_m1 = st.button("▶ Chạy M1", key="run_m1")
    with col_param:
        with st.expander("⚙️ Tham số dự báo 2030"):
            d30  = st.slider("D 2030 (%)", 20.0, 40.0, 30.0, 1.0)
            ai30 = st.slider("AI 2030 (nghìn DN)", 80.0, 150.0, 100.0, 5.0)
            h30  = st.slider("H 2030 (%)", 25.0, 45.0, 35.0, 1.0)
            tfp_g= st.slider("TFP growth (%/năm)", 0.5, 3.0, 1.2, 0.1)

    if run_m1:
        df_fc = forecast_gdp_2030(d30, ai30, h30, tfp_growth=tfp_g/100)
        st.session_state["m1_forecast"] = df_fc

    # TFP chart luôn hiển thị
    fig_tfp = go.Figure()
    fig_tfp.add_trace(go.Scatter(x=YEARS, y=A_hist, mode="lines+markers",
        line=dict(color="#a78bfa", width=2), marker=dict(size=7), name="TFP A_t"))
    fig_tfp.add_hline(y=A_mean, line_dash="dash", line_color="#f87171",
                      annotation_text=f"Mean={A_mean:.4f}")
    fig_tfp.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
        font_color="#c0c4d0", height=260,
        xaxis=dict(title="Năm", gridcolor="#2d2f3e"),
        yaxis=dict(title="TFP (A_t)", gridcolor="#2d2f3e"),
        margin=dict(l=50,r=20,t=30,b=40), title="TFP Việt Nam 2020–2025")
    st.plotly_chart(fig_tfp, use_container_width=True)

    if "m1_forecast" in st.session_state:
        df_fc = st.session_state["m1_forecast"]
        st.markdown(f"""<div class='result-box'>✅ M1 done — GDP dự báo 2030: 
        <b>{df_fc['GDP dự báo'].iloc[-1]:,.1f}</b> nghìn tỷ VND</div>""",
        unsafe_allow_html=True)
        fig_fc = go.Figure()
        all_y = list(YEARS) + list(df_fc["Năm"])
        all_g = list(Y_GDP) + list(df_fc["GDP dự báo"])
        fig_fc.add_trace(go.Scatter(x=YEARS, y=Y_GDP, mode="lines+markers",
            name="Thực tế", line=dict(color="#38bdf8",width=2)))
        fig_fc.add_trace(go.Scatter(x=df_fc["Năm"], y=df_fc["GDP dự báo"],
            mode="lines+markers", name="Dự báo",
            line=dict(color="#f59e0b",width=2,dash="dot")))
        fig_fc.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=270,
            xaxis=dict(title="Năm",gridcolor="#2d2f3e"),
            yaxis=dict(title="GDP (nghìn tỷ VND)",gridcolor="#2d2f3e"),
            legend=dict(bgcolor="#1e2130"), margin=dict(l=50,r=20,t=30,b=40),
            title="Dự báo GDP 2026–2030")
        st.plotly_chart(fig_fc, use_container_width=True)

    st.divider()

    # ── M2 ───────────────────────────────────
    st.markdown("### M2 — Đánh giá sẵn sàng số (TOPSIS)")

    col_btn2, _ = st.columns([1,3])
    with col_btn2:
        run_m2 = st.button("▶ Chạy M2", key="run_m2")

    REGION_DATA = np.array([
        [57.0,  3.5, 38, 22, 21.5, 0.18, 72, 0.405],
        [152.3, 20.0,78, 68, 36.8, 0.85, 92, 0.358],
        [87.5,  8.2, 55, 40, 27.5, 0.32, 84, 0.372],
        [68.9,  0.8, 32, 18, 18.2, 0.15, 68, 0.412],
        [158.9, 18.5,82, 75, 42.5, 0.78, 94, 0.385],
        [80.5,  2.1, 48, 30, 16.8, 0.22, 78, 0.392],
    ], dtype=float)
    IS_BENEFIT = [True,True,True,True,True,True,True,False]
    W_EXPERT   = np.array([0.10,0.10,0.15,0.20,0.15,0.15,0.05,0.10])

    if run_m2:
        scores = run_topsis(REGION_DATA, W_EXPERT, IS_BENEFIT)
        df_topsis = pd.DataFrame({
            "Vùng": REGIONS,
            "TOPSIS Score": np.round(scores,4),
            "Xếp hạng": pd.Series(scores).rank(ascending=False).astype(int)
        }).sort_values("TOPSIS Score", ascending=False).reset_index(drop=True)
        st.session_state["m2_topsis"] = df_topsis

    if "m2_topsis" in st.session_state:
        df_t = st.session_state["m2_topsis"]
        fig_t = px.bar(df_t, x="Vùng", y="TOPSIS Score", color="TOPSIS Score",
            color_continuous_scale="Viridis", title="Xếp hạng AI Readiness các vùng (TOPSIS)")
        fig_t.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=300, showlegend=False,
            xaxis=dict(gridcolor="#2d2f3e"), yaxis=dict(gridcolor="#2d2f3e"),
            margin=dict(l=50,r=20,t=40,b=60))
        st.plotly_chart(fig_t, use_container_width=True)
        top1 = df_t.iloc[0]["Vùng"]
        st.markdown(f"""<div class='result-box'>✅ M2 done — Vùng sẵn sàng AI nhất: 
        <b>{top1}</b> (Score={df_t.iloc[0]['TOPSIS Score']:.4f})</div>""",
        unsafe_allow_html=True)
    else:
        st.markdown("<div class='warn-box'>⚠️ Nhấn <b>Chạy M2</b> để xem kết quả TOPSIS.</div>",
                    unsafe_allow_html=True)

    st.divider()
    # Bảng tổng kết thiết kế hệ thống
    st.markdown("### 📋 Tóm tắt thiết kế hệ thống")
    df_sys = pd.DataFrame({
        "Module": ["M1","M2","M3","M4","M5","M6"],
        "Tên": ["Dự báo kinh tế","Sẵn sàng số","Tối ưu phân bổ","Lao động","Rủi ro","Dashboard"],
        "Đầu vào": ["Macro 2020-2025","Sectors, Regions","Budget, β-matrix","x_AI, x_H","Risk params","Outputs M1-M5"],
        "Đầu ra": ["GDP, TFP 2030","Digital + AI Index","Phân bổ ngành-vùng","NetJob từng ngành","Cyber, Env, Dependency","Trực quan kịch bản"],
        "Kỹ thuật": ["Cobb-Douglas (Bài 1)","TOPSIS (Bài 6)","LP (Bài 4) + Dynamic (Bài 8)","LP (Bài 9)","NSGA-II (Bài 7) + SP (Bài 10)","Streamlit + Plotly"],
    })
    st.dataframe(df_sys, use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 2 — PHÂN BỔ M3
# ════════════════════════════════════════════
with tab_m3:
    st.markdown("### M3 — Tối ưu phân bổ ngân sách (LP)")
    st.caption("Giải LP 24 biến, 6 vùng × 4 hạng mục, ngân sách 50.000 tỷ VND — tìm phân bổ S5 tối ưu.")

    run_m3 = st.button("▶ Chạy M3 — Giải LP tối ưu", key="run_m3")

    if run_m3:
        with st.spinner("Đang giải LP..."):
            alloc_s5, z_star, mat_opt = run_lp_optimal()
        st.session_state["m3_result"] = {
            "alloc": alloc_s5, "z_star": z_star, "matrix": mat_opt
        }

    if "m3_result" in st.session_state:
        res = st.session_state["m3_result"]
        alloc_s5 = res["alloc"]
        z_star   = res["z_star"]
        mat_opt  = res["matrix"]
        SCENARIOS["S5 · Tối ưu cân bằng"] = alloc_s5

        st.markdown(f"""<div class='result-box'>✅ M3 done — 
        Z* = <b>{z_star:,.1f}</b> tỷ VND GDP gain &nbsp;|&nbsp; 
        Phân bổ tối ưu: I={alloc_s5[0]*100:.1f}% · D={alloc_s5[1]*100:.1f}% · 
        AI={alloc_s5[2]*100:.1f}% · H={alloc_s5[3]*100:.1f}%</div>""",
        unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Heatmap phân bổ tối ưu
        st.markdown("#### 🔥 Heatmap phân bổ tối ưu (6 vùng × 4 hạng mục)")
        fig_hm = go.Figure(go.Heatmap(
            z=mat_opt, x=ITEMS, y=REGIONS,
            colorscale="Viridis",
            text=[[f"{mat_opt[r,j]:,.0f}" for j in range(4)] for r in range(6)],
            texttemplate="%{text}"
        ))
        fig_hm.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=320,
            xaxis_title="Hạng mục đầu tư", yaxis_title="Vùng",
            margin=dict(l=160,r=20,t=30,b=50))
        st.plotly_chart(fig_hm, use_container_width=True)

        # Bảng chi tiết
        df_alloc = pd.DataFrame(mat_opt, index=REGIONS, columns=ITEMS)
        df_alloc["Tổng"] = df_alloc.sum(axis=1)
        st.dataframe(df_alloc.round(1), use_container_width=True)

        # Biểu đồ tỷ trọng phân bổ S5
        st.markdown("#### 📊 Tỷ trọng phân bổ S5 · Tối ưu cân bằng")
        fig_pie = go.Figure(go.Pie(
            labels=["I · Hạ tầng","D · Số hóa","AI","H · Nhân lực"],
            values=alloc_s5*100, hole=0.4,
            marker_colors=["#6366f1","#22d3ee","#a78bfa","#34d399"]
        ))
        fig_pie.update_layout(paper_bgcolor="#0f1117", font_color="#c0c4d0",
            height=280, margin=dict(l=20,r=20,t=20,b=20),
            legend=dict(bgcolor="#1e2130"))
        st.plotly_chart(fig_pie, use_container_width=True)
    else:
        st.markdown("<div class='warn-box'>⚠️ Nhấn <b>Chạy M3</b> để giải LP và tạo phân bổ S5.</div>",
                    unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 3 — 5 KỊCH BẢN M6
# ════════════════════════════════════════════
with tab_scen:
    st.markdown("### 📊 So sánh 5 kịch bản chính sách (M6 Dashboard)")

    # Kiểm tra S5
    s5_ready = SCENARIOS["S5 · Tối ưu cân bằng"] is not None
    if not s5_ready:
        st.markdown(
            "<div class='warn-box'>💡 Gợi ý: Chạy M3 ở tab <b>Phân bổ</b> để có kịch bản S5 tối ưu.</div>",
            unsafe_allow_html=True)
        SCENARIOS["S5 · Tối ưu cân bằng"] = np.array([0.35,0.25,0.20,0.20])

    run_scen = st.button("▶ Chạy tất cả 5 kịch bản", key="run_scen")

    if run_scen:
        results = {}
        for name, alloc in SCENARIOS.items():
            results[name] = simulate_scenario(alloc)
        st.session_state["scen_results"] = results

    if "scen_results" in st.session_state:
        results = st.session_state["scen_results"]
        names   = list(results.keys())

        # KPI cards
        st.markdown("#### 🏆 KPI tổng hợp năm 2030")
        cols = st.columns(5)
        kpi_keys = ["GDP 2030 (nghìn tỷ)","Tăng trưởng GDP 5Y (%)","NetJob ròng (nghìn)"]
        for i,(name,res) in enumerate(results.items()):
            with cols[i]:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label' style='color:{SCEN_COLORS[i]}'>{name}</div>
                    <div class='value' style='color:{SCEN_COLORS[i]};font-size:1.2rem'>
                    {res['GDP 2030 (nghìn tỷ)']:,.1f}</div>
                    <div class='delta'>GDP 2030 (nghìn tỷ VND)</div>
                    <div style='font-size:.78rem;color:#8b8fa8;margin-top:6px'>
                    Tăng trưởng: {res['Tăng trưởng GDP 5Y (%)']:.1f}%<br>
                    NetJob: {res['NetJob ròng (nghìn)']:,.0f}k</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Biểu đồ GDP trajectory
        st.markdown("#### 📈 Quỹ đạo GDP 2025–2030 theo kịch bản")
        fig_traj = go.Figure()
        yr = list(range(2025, 2031))
        for i,(name,res) in enumerate(results.items()):
            fig_traj.add_trace(go.Scatter(
                x=yr, y=res["gdp_series"], mode="lines+markers",
                name=name, line=dict(color=SCEN_COLORS[i], width=2),
                marker=dict(size=6)))
        fig_traj.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=360,
            xaxis=dict(title="Năm",gridcolor="#2d2f3e"),
            yaxis=dict(title="GDP (nghìn tỷ VND)",gridcolor="#2d2f3e"),
            legend=dict(bgcolor="#1e2130"), margin=dict(l=60,r=20,t=30,b=50))
        st.plotly_chart(fig_traj, use_container_width=True)

        # Radar chart đa chiều
        st.markdown("#### 🕸️ Radar chart — So sánh đa chiều")
        radar_metrics = ["Tăng trưởng GDP 5Y (%)","NetJob ròng (nghìn)"]
        radar_cats = ["GDP Growth","NetJob","Bình đẳng\n(1-Gini)","Xanh\n(1-Emission)","An ninh\n(1-Cyber)"]

        fig_radar = go.Figure()
        for i,(name,res) in enumerate(results.items()):
            vals = [
                res["Tăng trưởng GDP 5Y (%)"]/10,
                min(res["NetJob ròng (nghìn)"]/500, 1),
                1 - (res["Gini proxy"]-0.3)/0.15,
                1 - res["Phát thải (index)"],
                max(0, 1 - res["Cyber Risk (index)"]),
            ]
            vals_c = vals + [vals[0]]
            cats_c = radar_cats + [radar_cats[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=vals_c, theta=cats_c, fill="toself",
                name=name, line=dict(color=SCEN_COLORS[i])))
        fig_radar.update_layout(
            polar=dict(bgcolor="#1a1d27",
                       radialaxis=dict(visible=True, range=[0,1], gridcolor="#2d2f3e"),
                       angularaxis=dict(gridcolor="#2d2f3e")),
            paper_bgcolor="#0f1117", font_color="#c0c4d0",
            legend=dict(bgcolor="#1e2130"), height=380,
            margin=dict(l=40,r=40,t=40,b=40))
        st.plotly_chart(fig_radar, use_container_width=True)

        # Bảng tổng hợp KPI
        st.markdown("#### 📋 Bảng tổng hợp KPI 5 kịch bản")
        kpi_cols = ["GDP 2030 (nghìn tỷ)","Tăng trưởng GDP 5Y (%)","NetJob ròng (nghìn)",
                    "Gini proxy","Phát thải (index)","Cyber Risk (index)"]
        df_kpi = pd.DataFrame(
            {name: {k: res[k] for k in kpi_cols} for name, res in results.items()}
        ).T.reset_index().rename(columns={"index":"Kịch bản"})
        st.dataframe(df_kpi, use_container_width=True, hide_index=True)

        # Phân bổ hạng mục theo kịch bản
        st.markdown("#### 💰 Phân bổ hạng mục đầu tư theo kịch bản")
        alloc_data = []
        for name, alloc in SCENARIOS.items():
            for j,(item) in enumerate(["I · Hạ tầng","D · Số hóa","AI","H · Nhân lực"]):
                alloc_data.append({"Kịch bản":name,"Hạng mục":item,"Tỷ trọng (%)":alloc[j]*100})
        df_alloc_scen = pd.DataFrame(alloc_data)
        fig_alloc = px.bar(df_alloc_scen, x="Kịch bản", y="Tỷ trọng (%)", color="Hạng mục",
            barmode="stack", color_discrete_sequence=["#6366f1","#22d3ee","#a78bfa","#34d399"],
            title="Cơ cấu đầu tư theo kịch bản")
        fig_alloc.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=320,
            xaxis=dict(gridcolor="#2d2f3e"), yaxis=dict(gridcolor="#2d2f3e"),
            legend=dict(bgcolor="#1e2130"), margin=dict(l=50,r=20,t=40,b=80))
        st.plotly_chart(fig_alloc, use_container_width=True)

    else:
        st.markdown("<div class='warn-box'>⚠️ Nhấn <b>Chạy tất cả 5 kịch bản</b> để xem so sánh.</div>",
                    unsafe_allow_html=True)


# ════════════════════════════════════════════
# TAB 4 — CẢNH BÁO RỦI RO M4 + M5
# ════════════════════════════════════════════
with tab_risk:
    st.markdown("### ⚠️ Cảnh báo rủi ro (M4 — Lao động & M5 — Đa mục tiêu)")

    # ── M4 — Lao động ────────────────────────
    st.markdown("#### 👷 M4 — Mô phỏng thị trường lao động")

    col_s, col_b = st.columns([2,1])
    with col_s:
        sel_scen = st.selectbox("Chọn kịch bản phân tích",
            list(SCENARIOS.keys()), key="m4_scen")
    with col_b:
        run_m4 = st.button("▶ Chạy M4", key="run_m4")

    if run_m4:
        alloc_sel = SCENARIOS[sel_scen]
        if alloc_sel is None:
            alloc_sel = np.array([0.35,0.25,0.20,0.20])
        budget = 30000
        x_ai = alloc_sel[2] * budget / 8
        x_h  = alloc_sel[3] * budget / 8
        NewJob   = A1_LABOR * x_ai
        Upgrade  = B1_LABOR * x_h
        Displaced= C1_LABOR * RISK * x_ai
        NetJob   = NewJob + Upgrade - Displaced

        df_labor = pd.DataFrame({
            "Ngành": SECTORS,
            "Lao động (tr)": LABOR,
            "Rủi ro TĐH (%)": (RISK*100).round(1),
            "NewJob (nghìn)": (NewJob/1000).round(1),
            "Upgrade (nghìn)": (Upgrade/1000).round(1),
            "Displaced (nghìn)": (Displaced/1000).round(1),
            "NetJob (nghìn)": (NetJob/1000).round(1),
        })
        st.session_state["m4_result"] = df_labor

    if "m4_result" in st.session_state:
        df_l = st.session_state["m4_result"]
        fig_nj = go.Figure()
        colors_nj = ["#34d399" if v>=0 else "#f87171" for v in df_l["NetJob (nghìn)"]]
        fig_nj.add_trace(go.Bar(
            x=df_l["Ngành"], y=df_l["NetJob (nghìn)"],
            marker_color=colors_nj,
            text=df_l["NetJob (nghìn)"].round(1), textposition="outside"
        ))
        fig_nj.add_hline(y=0, line_dash="dash", line_color="#f87171")
        fig_nj.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=300, title="NetJob ròng theo ngành (nghìn việc làm)",
            xaxis=dict(gridcolor="#2d2f3e"), yaxis=dict(title="NetJob (nghìn)",gridcolor="#2d2f3e"),
            margin=dict(l=50,r=20,t=40,b=80))
        st.plotly_chart(fig_nj, use_container_width=True)

        total_nj = df_l["NetJob (nghìn)"].sum()
        neg_sectors = df_l[df_l["NetJob (nghìn)"]<0]["Ngành"].tolist()
        if neg_sectors:
            st.markdown(f"<div class='warn-box'>⚠️ Ngành có NetJob âm: <b>{', '.join(neg_sectors)}</b> — cần tăng đầu tư đào tạo H.</div>",
                        unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-box'>✅ Tất cả ngành có NetJob ≥ 0 | Tổng: <b>{total_nj:,.1f} nghìn</b> việc làm ròng.</div>",
                        unsafe_allow_html=True)
        st.dataframe(df_l, use_container_width=True, hide_index=True)
    else:
        st.markdown("<div class='warn-box'>⚠️ Nhấn <b>Chạy M4</b> để xem phân tích lao động.</div>",
                    unsafe_allow_html=True)

    st.divider()

    # ── M5 — Rủi ro ──────────────────────────
    st.markdown("#### 🛡️ M5 — Đánh giá rủi ro tổng hợp")

    if "scen_results" in st.session_state:
        results = st.session_state["scen_results"]

        # Gauge charts cho kịch bản đang chọn
        sel_risk = st.selectbox("Kịch bản xem rủi ro", list(results.keys()), key="m5_sel")
        res_r = results[sel_risk]

        col1,col2,col3 = st.columns(3)
        risk_items = [
            ("Phát thải (index)", "🌿 Rủi ro môi trường", "#34d399", 0, 1),
            ("Cyber Risk (index)","🔐 Rủi ro an ninh mạng","#f87171", 0, 1),
            ("Gini proxy",        "⚖️ Rủi ro bất bình đẳng","#f59e0b", 0.30, 0.45),
        ]
        for col, (key, label, color, vmin, vmax) in zip([col1,col2,col3], risk_items):
            val = res_r[key]
            pct = (val - vmin)/(vmax - vmin + 1e-9)
            level = "🟢 Thấp" if pct<0.4 else ("🟡 Trung bình" if pct<0.7 else "🔴 Cao")
            with col:
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>{label}</div>
                    <div class='value' style='color:{color};font-size:1.3rem'>{val:.3f}</div>
                    <div class='delta'>{level}</div>
                </div>""", unsafe_allow_html=True)

        # Bảng rủi ro tất cả kịch bản
        st.markdown("<br>**📋 Ma trận rủi ro 5 kịch bản**", unsafe_allow_html=True)
        df_risk = pd.DataFrame({
            name: {
                "Phát thải": res["Phát thải (index)"],
                "Cyber Risk": res["Cyber Risk (index)"],
                "Gini": res["Gini proxy"],
            } for name, res in results.items()
        }).T.reset_index().rename(columns={"index":"Kịch bản"})
        st.dataframe(df_risk, use_container_width=True, hide_index=True)

        # Heatmap rủi ro
        risk_matrix = np.array([[results[n]["Phát thải (index)"],
                                  results[n]["Cyber Risk (index)"],
                                  results[n]["Gini proxy"]] for n in results])
        fig_risk_hm = go.Figure(go.Heatmap(
            z=risk_matrix,
            x=["Phát thải","Cyber Risk","Gini"],
            y=list(results.keys()),
            colorscale="RdYlGn_r",
            text=[[f"{v:.3f}" for v in row] for row in risk_matrix],
            texttemplate="%{text}"
        ))
        fig_risk_hm.update_layout(paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0", height=280,
            title="Heatmap rủi ro theo kịch bản",
            margin=dict(l=160,r=20,t=40,b=50))
        st.plotly_chart(fig_risk_hm, use_container_width=True)

    else:
        st.markdown(
            "<div class='warn-box'>⚠️ Hãy chạy <b>5 kịch bản</b> ở tab Kịch bản trước.</div>",
            unsafe_allow_html=True)

    st.divider()
    # Khuyến nghị chính sách tổng hợp
    st.markdown("#### 📌 Khuyến nghị chính sách tổng hợp")
    st.markdown("""
| Mục tiêu | Kịch bản tốt nhất | Lý do |
|---------|------------------|-------|
| **Tối đa GDP 2030** | S3 · AI dẫn dắt | Hệ số AI cao nhất, tác động năng suất mạnh |
| **Tối đa việc làm** | S4 · Bao trùm số | H cao → đào tạo lại lao động, giảm displaced |
| **Tối thiểu phát thải** | S4 · Bao trùm số | AI thấp → tiêu thụ điện năng thấp hơn |
| **Tối thiểu bất bình đẳng** | S4 · Bao trùm số | Ưu tiên vùng yếu, SME, nông nghiệp |
| **Cân bằng tổng thể** | S5 · Tối ưu (LP) | Giải LP tối ưu hóa đa ràng buộc |

> **Kết luận AIDEOM-VN:** Không có kịch bản nào tối ưu trên mọi chiều.  
> Lựa chọn phụ thuộc vào ưu tiên chính sách — tăng trưởng, bao trùm hay môi trường.  
> Đây chính xác là lý do cần mô hình AIDEOM-VN: **cung cấp bằng chứng định lượng**  
> cho quá trình thảo luận chính sách dân chủ.

> 📚 *Tham chiếu: Nghị quyết 57-NQ/TW (2024) · QĐ 749/QĐ-TTg (2020) · QĐ 127/QĐ-TTg (2021) · COP26 Net-Zero 2050*
""")