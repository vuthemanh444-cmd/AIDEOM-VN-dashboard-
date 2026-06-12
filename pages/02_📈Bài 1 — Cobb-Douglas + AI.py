import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Bài 1 - Cobb Douglas",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)
with st.sidebar:
    st.markdown("### 🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    st.divider()
    st.caption("📂 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("⚙️ Tools: Python, Streamlit, PuLP,..")
    st.caption("📘 Dựa trên giáo trình AIDEOM-VN 2026")

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
.metric-card .value  { font-size:1.6rem; font-weight:700; color:#a78bfa; }
.metric-card .delta  { font-size:.8rem; color:#6ee7b7; margin-top:2px; }
.badge { display:inline-block; padding:2px 10px; border-radius:999px;
         font-size:.72rem; font-weight:600; letter-spacing:.04em; margin-right:6px; }
.badge-hard  { background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed55; }
.badge-rl    { background:#0ea5e922; color:#38bdf8; border:1px solid #0ea5e955; }
.badge-info  { background:#10b98122; color:#6ee7b7; border:1px solid #10b98155; }
.badge-mod   { background:#f59e0b22; color:#fcd34d; border:1px solid #f59e0b55; }
.result-box  { background:#14532d33; border:1px solid #16a34a66; border-radius:8px;
               padding:10px 16px; color:#86efac; font-size:.9rem; }
.warn-box    { background:#78350f33; border:1px solid #d9770666; border-radius:8px;
               padding:10px 16px; color:#fcd34d; font-size:.9rem; }
.stDataFrame { border-radius:8px; overflow:hidden; }
.stButton > button {
    background:linear-gradient(135deg,#7c3aed,#6d28d9);
    color:white; border:none; border-radius:8px;
    padding:8px 22px; font-weight:600; font-size:.95rem;
}
.stButton > button:hover { opacity:.88; }
[data-testid="stSlider"] label { color:#a0a4b8 !important; font-size:.85rem; }
h2, h3 { color:#e2e8f0 !important; }
hr { border-color:#2d2f3e; }
</style>
""", unsafe_allow_html=True)

# =========================
# TIÊU ĐỀ
# =========================
st.markdown("## 📈 Bài 1 - Mô hình Cobb-Douglas mở rộng")
st.markdown(
    "<span class='badge badge-info'>DỄ</span>"
    "<span class='badge badge-info'>Cobb-Douglas</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Phân tích tăng trưởng GDP Việt Nam giai đoạn 2020-2025

Mô hình sử dụng các yếu tố:

- K: Vốn tích lũy
- L: Lao động
- D: Chuyển đổi số
- AI: Doanh nghiệp AI
- H: Nhân lực số
""")

st.divider()

# =========================
# CÔNG THỨC
# =========================
st.markdown('### 📚 Hàm sản xuất')

st.latex(
    r'''
    Y_t
    =
    A_t
    \cdot
    K_t^{\alpha}
    \cdot
    L_t^{\beta}
    \cdot
    D_t^{\gamma}
    \cdot
    AI_t^{\delta}
    \cdot
    H_t^{\theta}
    '''
)

st.info(
    """
    Trong đó:
    A_t là năng suất nhân tố tổng hợp (TFP),
    α = 0.33,
    β = 0.42,
    γ = 0.10,
    δ = 0.08,
    θ = 0.07
    """
)

with st.expander("📖 Giải thích các hệ số trong mô hình"):

    st.write("""
    Các hệ số được giả định theo đề bài và phản ánh mức độ
    đóng góp của từng yếu tố vào tăng trưởng GDP.

    • α = 0.33 : đóng góp của vốn (K)

    • β = 0.42 : đóng góp của lao động (L)

    • γ = 0.10 : đóng góp của chuyển đổi số (D)

    • δ = 0.08 : đóng góp của AI (AI)

    • θ = 0.07 : đóng góp của nhân lực số (H)

    Tổng các hệ số bằng 1, thể hiện giả định hiệu suất không đổi theo quy mô.
    """)
# =========================
# ĐỌC DỮ LIỆU
# =========================
    df = pd.read_csv("data/vietnam_macro_2020_2025.csv")

# =========================
# BIẾN DỮ LIỆU
# =========================
Y = df["GDP_trillion_VND"].values
K = df["K"].values
L = df["L"].values
D = df["D"].values
AI = df["AI"].values
H = df["H"].values

# =========================
# THAM SỐ MÔ HÌNH
# =========================
alpha = 0.33
beta = 0.42
gamma = 0.10
delta = 0.08
theta = 0.07

# =========================
# TÍNH TFP
# =========================
A = Y / (
    K**alpha *
    L**beta *
    D**gamma *
    AI**delta *
    H**theta
)

result = pd.DataFrame({
    "Year": df["Year"],
    "TFP_A": A
})

# =========================
# KPI CARDS
# =========================
st.markdown('### 📊 Tổng quan TFP')

col1, col2, col3 = st.columns(3)

col1.metric(
    "TFP trung bình",
    f"{A.mean():.2f}"
)

col2.metric(
    "TFP cao nhất",
    f"{A.max():.2f}"
)

growth = ((A[-1] - A[0]) / A[0]) * 100

col3.metric(
    "Tăng trưởng TFP",
    f"{growth:.2f}%"
)

st.divider()

# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "📈 TFP",
    "🎯 Dự báo & MAPE",
    "📊 Phân rã tăng trưởng",
    "🚀 Kịch bản 2030",
    "📝 Thảo luận chính sách"
])

# =========================
# TAB 1
# =========================
with tab1:

    st.markdown('### Bảng dữ liệu đầu vào')

    st.dataframe(
    df,
    use_container_width=True,
    hide_index=True
    )

    csv = df.to_csv(index=False)
    # Tải dữ liệu CSV tab1
    st.download_button(
    "📥 Tải dữ liệu CSV",
    csv,
    "vietnam_macro_2020_2025.csv",
    "text/csv"
)
# =========================
# TAB 2
# =========================
with tab2:

    st.markdown('### 📈 Bảng TFP theo năm')

    st.dataframe(
    result,
    use_container_width=True,
    hide_index=True
)

    fig = px.line(
        result,
        x="Year",
        y="TFP_A",
        markers=True,
        title="Xu hướng TFP giai đoạn 2020-2025"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        xaxis_title="Năm",
        yaxis_title="TFP (Aₜ)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

# =========================
# TAB 3 - CÂU 1.4.2
# =========================
with tab3:

    st.markdown('### 🎯 Dự báo GDP và đánh giá MAPE')

    A_mean = np.mean(A)

    Y_hat = (
        A_mean
        * (K**alpha)
        * (L**beta)
        * (D**gamma)
        * (AI**delta)
        * (H**theta)
    )

    mape = np.mean(
        np.abs((Y - Y_hat) / Y)
    ) * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "TFP trung bình",
        f"{A_mean:.2f}"
    )

    col2.metric(
        "MAPE",
        f"{mape:.2f}%"
    )

    forecast_df = pd.DataFrame({
        "Year": df["Year"],
        "GDP thực tế": Y,
        "GDP dự báo": np.round(Y_hat, 2)
    })

    st.dataframe(
    forecast_df,
    use_container_width=True,
    hide_index=True
)

    fig = px.line(
        forecast_df,
        x="Year",
        y=["GDP thực tế", "GDP dự báo"],
        markers=True,
        title="So sánh GDP thực tế và GDP dự báo"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    if mape < 5:
        st.success(
            "MAPE nhỏ hơn 5% → Mô hình có độ chính xác rất cao."
        )
    elif mape < 10:
        st.info(
            "MAPE nhỏ hơn 10% → Mô hình dự báo tốt."
        )
    else:
        st.warning(
            "MAPE tương đối cao, cần xem xét mô hình."
        )
        # =========================
# TAB 4 - CÂU 1.4.3
# =========================
with tab4:

    st.markdown('### 📊 Phân rã tăng trưởng GDP')

    # --------------------------------
    # Tốc độ tăng trưởng bình quân (%)
    # --------------------------------
    gK = np.mean(np.diff(np.log(K))) * 100
    gL = np.mean(np.diff(np.log(L))) * 100
    gD = np.mean(np.diff(np.log(D))) * 100
    gAI = np.mean(np.diff(np.log(AI))) * 100
    gH = np.mean(np.diff(np.log(H))) * 100
    gA = np.mean(np.diff(np.log(A))) * 100

    # --------------------------------
    # Đóng góp tuyệt đối
    # --------------------------------
    contri_K = alpha * gK
    contri_L = beta * gL
    contri_D = gamma * gD
    contri_AI = delta * gAI
    contri_H = theta * gH
    contri_TFP = gA

    # --------------------------------
    # Tổng tăng trưởng
    # --------------------------------
    total_growth = (
        contri_K
        + contri_L
        + contri_D
        + contri_AI
        + contri_H
        + contri_TFP
    )

    # --------------------------------
    # Tỷ trọng đóng góp (%)
    # --------------------------------
    pct_K = contri_K / total_growth * 100
    pct_L = contri_L / total_growth * 100
    pct_D = contri_D / total_growth * 100
    pct_AI = contri_AI / total_growth * 100
    pct_H = contri_H / total_growth * 100
    pct_TFP = contri_TFP / total_growth * 100

    growth_df = pd.DataFrame({
        "Yếu tố": [
            "K",
            "L",
            "D",
            "AI",
            "H",
            "TFP"
        ],
        "Tỷ trọng đóng góp (%)": [
            pct_K,
            pct_L,
            pct_D,
            pct_AI,
            pct_H,
            pct_TFP
        ]
    })

    # --------------------------------
    # KPI
    # --------------------------------
    max_factor = growth_df.loc[
        growth_df["Tỷ trọng đóng góp (%)"].idxmax(),
        "Yếu tố"
    ]

    max_value = growth_df["Tỷ trọng đóng góp (%)"].max()

    col1, col2 = st.columns(2)

    col1.metric(
        "Yếu tố đóng góp lớn nhất",
        max_factor
    )

    col2.metric(
        "Tỷ trọng lớn nhất",
        f"{max_value:.2f}%"
    )

    st.divider()

    # --------------------------------
    # Bảng kết quả
    # --------------------------------
    st.markdown('### 📋 Bảng phân rã tăng trưởng')

    st.dataframe(
    growth_df,
    use_container_width=True,
    hide_index=True
)

    # --------------------------------
    # Biểu đồ
    # --------------------------------
    fig = px.bar(
        growth_df,
        x="Yếu tố",
        y="Tỷ trọng đóng góp (%)",
        text_auto=".2f",
        title="Tỷ trọng đóng góp của các yếu tố vào tăng trưởng GDP"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        xaxis_title="Yếu tố",
        yaxis_title="Tỷ trọng (%)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # --------------------------------
    # Nhận xét
    # --------------------------------
    st.success(
        f"Yếu tố đóng góp lớn nhất vào tăng trưởng GDP giai đoạn 2020-2025 là {max_factor}, chiếm khoảng {max_value:.2f}% tổng tăng trưởng."
    )
    # =========================
# TAB 5 - CÂU 1.4.4
# =========================
with tab5:

    st.markdown('### 🚀 Dự báo GDP Việt Nam năm 2030')

    # -------------------------
    # DỮ LIỆU NĂM 2025
    # -------------------------
    K_2025 = K[-1]
    L_2025 = L[-1]
    A_2025 = A[-1]
    GDP_2025 = Y[-1]

    # -------------------------
    # KỊCH BẢN 2030
    # -------------------------
    K_2030 = K_2025 * (1.06)**5
    L_2030 = L_2025 * (1.06)**5

    D_2030 = 30
    AI_2030 = 100
    H_2030 = 35

    A_2030 = A_2025 * (1.012)**5

    GDP_2030 = (
        A_2030
        * (K_2030**alpha)
        * (L_2030**beta)
        * (D_2030**gamma)
        * (AI_2030**delta)
        * (H_2030**theta)
    )

    # -------------------------
    # KPI CARDS
    # -------------------------
    growth_2030 = ((GDP_2030 - GDP_2025) / GDP_2025) * 100

    col1, col2 = st.columns(2)

    col1.metric(
        "GDP dự báo năm 2030",
        f"{GDP_2030:,.2f}"
    )

    col2.metric(
        "Tăng trưởng so với 2025",
        f"{growth_2030:.2f}%"
    )

    st.divider()

    # -------------------------
    # BẢNG KỊCH BẢN
    # -------------------------
    st.markdown('### 📋 Thông số giả định năm 2030')

    scenario_df = pd.DataFrame({
        "Biến": [
            "K",
            "L",
            "D",
            "AI",
            "H",
            "TFP"
        ],
        "Giá trị năm 2030": [
            round(K_2030, 2),
            round(L_2030, 2),
            D_2030,
            AI_2030,
            H_2030,
            round(A_2030, 2)
        ]
    })

    st.dataframe(
    scenario_df,
    use_container_width=True,
    hide_index=True
)

    # -------------------------
    # BIỂU ĐỒ SO SÁNH GDP
    # -------------------------
    compare_df = pd.DataFrame({
        "Năm": ["2025", "2030"],
        "GDP": [GDP_2025, GDP_2030]
    })

    fig = px.bar(
        compare_df,
        x="Năm",
        y="GDP",
        text_auto=".2f",
        title="So sánh GDP năm 2025 và GDP dự báo năm 2030"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # -------------------------
    # THÔNG TIN KỊCH BẢN
    # -------------------------
    st.info(
        """
        Kịch bản giả định:

        • K tăng 6%/năm

        • L tăng 6%/năm

        • D = 30%

        • AI = 100 nghìn doanh nghiệp

        • H = 35%

        • TFP tăng 1,2%/năm
        """
    )

    # -------------------------
    # KẾT LUẬN
    # -------------------------
    st.markdown('### 📌 Kết luận quản trị')

    st.success(f"""
    Theo mô hình Cobb-Douglas mở rộng, GDP Việt Nam năm 2030
    được dự báo đạt khoảng {GDP_2030:,.2f} nghìn tỷ VND.

    So với năm 2025, quy mô GDP tăng khoảng
    {growth_2030:.2f}%.

    Kết quả cho thấy việc:

    • Đẩy mạnh chuyển đổi số

    • Thúc đẩy ứng dụng AI trong doanh nghiệp

    • Nâng cao chất lượng nguồn nhân lực số

    • Duy trì tích lũy vốn và tăng năng suất

    có thể tạo ra tác động tích cực tới tăng trưởng kinh tế
    trong giai đoạn đến năm 2030.
    """)

# =========================
# TAB 6 - CÂU 1.5
# =========================
with tab6:

    st.header("📝 Thảo luận chính sách")

    # ==================================
    # Câu a
    # ==================================
    st.markdown('### a) TFP của Việt Nam có xu hướng tăng hay giảm trong giai đoạn 2020-2025?')

    tfp_growth = ((A[-1] - A[0]) / A[0]) * 100

    if tfp_growth > 0:

        st.write(
            f"""
            TFP tăng khoảng {tfp_growth:.2f}% trong giai đoạn 2020-2025.

            Điều này cho thấy chất lượng tăng trưởng kinh tế
            đang được cải thiện. Nền kinh tế không chỉ tăng trưởng
            nhờ mở rộng vốn và lao động mà còn nhờ nâng cao
            hiệu quả sử dụng các nguồn lực sản xuất.
            """
        )

    else:

        st.warning(
            f"""
            TFP giảm khoảng {abs(tfp_growth):.2f}% trong giai đoạn 2020-2025.

            Điều này cho thấy tăng trưởng chủ yếu dựa vào
            mở rộng đầu vào sản xuất thay vì nâng cao hiệu quả.
            """
        )

    st.divider()

    # ==================================
    # Câu b
    # ==================================
    st.markdown('### b) Trong các yếu tố D, AI và H, yếu tố nào đóng góp nhiều nhất?')

    digital_df = growth_df[
        growth_df["Yếu tố"].isin(["D", "AI", "H"])
    ]

    top_factor = digital_df.loc[
        digital_df["Tỷ trọng đóng góp (%)"].idxmax(),
        "Yếu tố"
    ]

    top_value = digital_df["Tỷ trọng đóng góp (%)"].max()

    factor_name = {
        "D": "Chuyển đổi số",
        "AI": "Doanh nghiệp AI",
        "H": "Nhân lực số"
    }

    st.markdown(
        f"""
        Yếu tố đóng góp lớn nhất là {factor_name[top_factor]}
        ({top_factor}), chiếm khoảng {top_value:.2f}% tổng tăng trưởng.

        Kết quả cho thấy đây là động lực quan trọng nhất
        trong nhóm các yếu tố kinh tế số giai đoạn vừa qua.
        """
    )

    st.divider()

    # ==================================
    # Câu c
    # ==================================
    st.markdown('### c) Mục tiêu 30% kinh tế số/GDP vào năm 2030 có khả thi không?')

    st.markdown(
        f"""
        Theo kịch bản mô phỏng đến năm 2030,
        GDP Việt Nam được dự báo đạt khoảng
        {GDP_2030:,.2f} nghìn tỷ VND.

        Kết quả mô hình cho thấy việc gia tăng
        mức độ chuyển đổi số (D),
        mở rộng số lượng doanh nghiệp AI,
        nâng cao chất lượng nhân lực số
        và cải thiện TFP đều tạo tác động tích cực
        tới tăng trưởng kinh tế.

        Do đó mục tiêu đạt 30% kinh tế số/GDP
        được đánh giá là có tính khả thi.

        Tuy nhiên cần các điều kiện hỗ trợ:

        • Tiếp tục đầu tư hạ tầng số.

        • Nâng cao kỹ năng số cho lực lượng lao động.

        • Khuyến khích doanh nghiệp ứng dụng AI.

        • Duy trì tăng trưởng TFP bền vững.

        • Hoàn thiện khung pháp lý cho kinh tế số.
        """
    )

    st.divider()

    # ==================================
    # TỔNG KẾT
    # ==================================
    st.markdown('### 📌 Kết luận chung')

    st.success(
        """
        Kết quả mô hình Cobb-Douglas mở rộng cho thấy
        các yếu tố công nghệ, chuyển đổi số và năng suất
        đang ngày càng đóng vai trò quan trọng đối với
        tăng trưởng kinh tế Việt Nam.

        Điều này hàm ý rằng các chính sách phát triển
        kinh tế số và đổi mới sáng tạo sẽ là động lực
        then chốt cho tăng trưởng dài hạn.
        """
    )

st.divider()
st.caption("""
Nguồn dữ liệu:

• Tổng cục Thống kê Việt Nam (GSO)

• World Bank

• Dữ liệu mô phỏng phục vụ học tập và nghiên cứu
""")