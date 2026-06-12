import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from scipy.optimize import minimize

# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 8 - Tối ưu hóa động",
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

st.markdown("## 📈 Bài 8 - Tối ưu hóa động phát triển kinh tế số Việt Nam")
st.markdown(
    "<span class='badge badge-hard'>KHÁ KHÓ</span>"
    "<span class='badge badge-info'>Dynamic · CVXPY</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Thiết kế chiến lược đầu tư dài hạn giai đoạn 2026–2035

Mục tiêu:

- Tối đa hóa phúc lợi xã hội liên thời gian
- Phân bổ tối ưu vốn vật chất
- Phát triển hạ tầng số
- Đầu tư AI
- Nâng cao vốn nhân lực

Phương pháp:

- Mô hình tối ưu hóa động
- Hàm sản xuất Cobb-Douglas mở rộng
- SciPy Optimize (SLSQP)
- Phân tích kịch bản và độ nhạy
""")
# =========================
# TABS
# =========================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "⚙️ Tối ưu hóa động",
    "📈 Quỹ đạo tối ưu",
    "🌪️ Phân tích cú sốc",
    "⚖️ So sánh chiến lược",
    "💬 Thảo luận chính sách"
])

# =========================
# TAB 1
# =========================
with tab1:

    st.markdown('### 📋 Dữ liệu và tham số mô hình')

    # =========================
    # KPI
    # =========================

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Giai đoạn", "2026-2035")

    with col2:
        st.metric("Số năm mô phỏng", "10")

    with col3:
        st.metric("Biến trạng thái", "4")

    with col4:
        st.metric("Hệ số chiết khấu", "0.97")

    st.divider()

    # =========================
    # BẢNG THAM SỐ
    # =========================

    st.markdown("### 📊 Bảng tham số mô hình")

    params_df = pd.DataFrame({
        "Tham số": [
            "δK",
            "δD",
            "δAI",
            "θH",
            "μ",
            "φ1",
            "φ2",
            "φ3",
            "ρ"
        ],
        "Giá trị": [
            0.05,
            0.12,
            0.15,
            0.80,
            0.02,
            0.003,
            0.002,
            0.004,
            0.97
        ],
        "Ý nghĩa": [
            "Khấu hao vốn vật chất",
            "Khấu hao hạ tầng số",
            "Khấu hao AI",
            "Hiệu quả đầu tư nhân lực",
            "Chảy máu chất xám",
            "Tác động D → TFP",
            "Tác động AI → TFP",
            "Tác động H → TFP",
            "Chiết khấu liên thời gian"
        ]
    })

    st.dataframe(
        params_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # ĐIỀU KIỆN BAN ĐẦU
    # =========================

    st.markdown("### 🏁 Điều kiện ban đầu năm 2026")

    initial_df = pd.DataFrame({
        "Biến": [
            "K₀",
            "L₀",
            "D₀",
            "AI₀",
            "H₀"
        ],
        "Giá trị": [
            "27.500",
            "53,9",
            "20,3%",
            "86",
            "30%"
        ],
        "Đơn vị": [
            "nghìn tỷ VND",
            "triệu lao động",
            "% GDP",
            "nghìn doanh nghiệp",
            "%"
        ]
    })

    st.dataframe(
        initial_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # HÀM SẢN XUẤT
    # =========================

    st.markdown("### 📈 Hàm sản xuất Cobb-Douglas mở rộng")

    st.latex(
        r"""
        Y_t =
        A_t
        K_t^{0.33}
        L_t^{0.42}
        D_t^{0.10}
        AI_t^{0.08}
        H_t^{0.07}
        """
    )

    st.markdown("""
    Trong đó:

    - K: vốn vật chất
    - D: hạ tầng số
    - AI: năng lực trí tuệ nhân tạo
    - H: vốn nhân lực
    - A: năng suất nhân tố tổng hợp (TFP)
    """)

    st.divider()

    # =========================
    # HÀM MỤC TIÊU
    # =========================

    st.markdown("### 🎯 Hàm mục tiêu")

    st.latex(
        r"""
        \max
        \sum_{t=2026}^{2035}
        \rho^{t-2026}
        \ln(C_t)
        """
    )

    st.markdown("""
    Mục tiêu của Chính phủ là tối đa hóa tổng phúc lợi xã hội liên thời gian,
    đồng thời cân bằng giữa tiêu dùng hiện tại và đầu tư cho tăng trưởng tương lai.
    """)

    st.divider()

    # =========================
    # PHƯƠNG TRÌNH CHUYỂN TRẠNG THÁI
    # =========================

    st.markdown("### 🔄 Phương trình tích lũy vốn")

    st.latex(
        r"""
        K_{t+1}
        =
        (1-\delta_K)K_t
        +
        I_{K,t}
        """
    )

    st.latex(
        r"""
        D_{t+1}
        =
        (1-\delta_D)D_t
        +
        I_{D,t}
        """
    )

    st.latex(
        r"""
        AI_{t+1}
        =
        (1-\delta_{AI})AI_t
        +
        I_{AI,t}
        """
    )

    st.latex(
        r"""
        H_{t+1}
        =
        H_t
        +
        \theta_H I_{H,t}
        -
        \mu
        """
    )

    st.divider()

    # =========================
    # KẾT LUẬN
    # =========================
    
    st.success("""
    Mô hình mô phỏng chiến lược phát triển kinh tế số Việt Nam giai đoạn 2026–2035.
    Chính phủ phải lựa chọn phân bổ tối ưu giữa vốn vật chất, hạ tầng số,
    AI và vốn nhân lực nhằm tối đa hóa phúc lợi xã hội dài hạn.
    Kết quả tối ưu sẽ được xác định bằng phương pháp tối ưu hóa động ở các tab tiếp theo.
    """)

# =========================
# TAB 2
# =========================
with tab2:

    st.markdown('### ⚙️ Tối ưu hóa động bằng SciPy (SLSQP)')

    # =====================================
    # THAM SỐ
    # =====================================

    years = np.arange(2026, 2036)
    T = len(years)

    delta_K = 0.05
    delta_D = 0.12
    delta_AI = 0.15

    theta_H = 0.80
    mu = 0.02

    phi1 = 0.003
    phi2 = 0.002
    phi3 = 0.004

    rho = 0.97

    # =====================================
    # ĐIỀU KIỆN BAN ĐẦU
    # =====================================

    K0 = 27500
    D0 = 20.3
    AI0 = 86
    H0 = 30

    L0 = 53.9
    A0 = 1.0

    # =====================================
    # HÀM MỤC TIÊU
    # =====================================

    def welfare(x):

        K = K0
        D = D0
        AI = AI0
        H = H0
        A = A0

        total_utility = 0

        for t in range(T):

            share_K = x[t*4]
            share_D = x[t*4+1]
            share_AI = x[t*4+2]
            share_H = x[t*4+3]

            Y = (
                A
                * (K**0.33)
                * (L0**0.42)
                * (D**0.10)
                * (AI**0.08)
                * (H**0.07)
            )

            invest_ratio = (
                share_K
                + share_D
                + share_AI
                + share_H
            )

            C = Y * (1 - invest_ratio)

            if C <= 0:
                return 1e12

            total_utility += (
                rho**t
            ) * np.log(C)

            IK = share_K * Y
            ID = share_D * Y
            IAI = share_AI * Y
            IH = share_H * Y

            K = (1 - delta_K) * K + IK

            D = (1 - delta_D) * D + ID

            AI = (
                (1 - delta_AI) * AI
                + IAI
            )

            H = (
                H
                + theta_H * IH
                - mu
            )

            growth = (
                phi1*np.log(D + 1)
                + phi2*np.log(AI + 1)
                + phi3*np.log(H + 1)
            )

            A = A * (1 + growth)

        return -total_utility

    # =====================================
    # RÀNG BUỘC
    # =====================================

    constraints = []

    for t in range(T):

        constraints.append(
            {
                "type": "ineq",
                "fun": lambda x, t=t:
                    0.50
                    -
                    (
                        x[t*4]
                        + x[t*4+1]
                        + x[t*4+2]
                        + x[t*4+3]
                    )
            }
        )

    # =====================================
    # BOUNDS
    # =====================================

    bounds = [
        (0, 0.30)
        for _ in range(T*4)
    ]

    # =====================================
    # GIÁ TRỊ KHỞI TẠO
    # =====================================

    x0 = np.array(
        [0.10, 0.05, 0.05, 0.10] * T
    )

    # =====================================
    # GIẢI MÔ HÌNH
    # =====================================

    result = minimize(
        welfare,
        x0,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
        options={
            "maxiter": 1000
        }
    )

    # =====================================
    # KPI
    # =====================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Trạng thái",
            "Thành công"
            if result.success
            else "Lỗi"
        )

    with col2:
        st.metric(
            "Welfare tối ưu",
            f"{-result.fun:,.2f}"
        )

    with col3:
        st.metric(
            "Số biến",
            len(result.x)
        )

    st.divider()

    # =====================================
    # BẢNG PHÂN BỔ
    # =====================================

    allocation = pd.DataFrame(
        result.x.reshape(T,4),
        columns=[
            "K",
            "D",
            "AI",
            "H"
        ]
    )

    allocation.insert(
        0,
        "Year",
        years
    )

    st.markdown(
        "### 📊 Tỷ trọng đầu tư tối ưu theo năm"
    )

    st.dataframe(
        allocation.round(4),
        use_container_width=True,
        hide_index=True 
    )

    st.divider()

    # =====================================
    # BIỂU ĐỒ
    # =====================================

    plot_df = allocation.melt(
        id_vars="Year",
        var_name="Category",
        value_name="Share"
    )

    fig = px.line(
        plot_df,
        x="Year",
        y="Share",
        color="Category",
        markers=True,
        title="Quỹ đạo tỷ trọng đầu tư tối ưu"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =====================================
    # DEBUG
    # =====================================

    with st.expander("🔍 Thông tin Solver"):

        st.write(
            "Success:",
            result.success
        )

        st.write(
            "Message:",
            result.message
        )

        st.write(
            "Objective:",
            result.fun
        )
    st.markdown('### 📌 Kết luận quản trị')
    st.success(
        f"""
Mô hình tối ưu động gồm 40 biến quyết định
(4 khoản đầu tư × 10 năm) đã được giải bằng
SciPy SLSQP.

Welfare tối ưu đạt {-result.fun:,.2f}.
"""
    )

# =========================
# TAB 3
# =========================

with tab3:

    st.markdown('### 📈 Quỹ đạo tối ưu của nền kinh tế giai đoạn 2026–2035')

    # =====================================
    # MÔ PHỎNG THEO NGHIỆM TỐI ƯU
    # =====================================

    x_opt = result.x

    K = K0
    D = D0
    AI = AI0
    H = H0
    A = A0

    trajectory = []

    for t in range(T):

        share_K = x_opt[t*4]
        share_D = x_opt[t*4 + 1]
        share_AI = x_opt[t*4 + 2]
        share_H = x_opt[t*4 + 3]

        Y = (
            A
            * (K**0.33)
            * (L0**0.42)
            * (D**0.10)
            * (AI**0.08)
            * (H**0.07)
        )

        invest_ratio = (
            share_K
            + share_D
            + share_AI
            + share_H
        )

        C = Y * (1 - invest_ratio)

        trajectory.append([
            years[t],
            K,
            D,
            AI,
            H,
            Y,
            C
        ])

        # Đầu tư thực tế

        IK = share_K * Y
        ID = share_D * Y
        IAI = share_AI * Y
        IH = share_H * Y

        # Cập nhật trạng thái

        K = (1 - delta_K) * K + IK

        D = (1 - delta_D) * D + ID

        AI = (1 - delta_AI) * AI + IAI

        H = H + theta_H * IH - mu

        growth = (
            phi1 * np.log(D + 1)
            + phi2 * np.log(AI + 1)
            + phi3 * np.log(H + 1)
        )

        A = A * (1 + growth)

    # =====================================
    # DATAFRAME
    # =====================================

    traj_df = pd.DataFrame(
        trajectory,
        columns=[
            "Year",
            "K",
            "D",
            "AI",
            "H",
            "Y",
            "C"
        ]
    )

    st.markdown("### 📋 Bảng quỹ đạo tối ưu")

    st.dataframe(
        traj_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =====================================
    # KPI
    # =====================================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "GDP cuối kỳ (2035)",
            f"{traj_df['Y'].iloc[-1]:,.0f}"
        )

    with col2:
        st.metric(
            "AI cuối kỳ",
            f"{traj_df['AI'].iloc[-1]:,.0f}"
        )

    with col3:
        st.metric(
            "Vốn nhân lực cuối kỳ",
            f"{traj_df['H'].iloc[-1]:,.0f}"
        )

    st.divider()

    # =====================================
    # BIỂU ĐỒ K,D,AI,H
    # =====================================

    factor_df = traj_df.melt(
        id_vars="Year",
        value_vars=["K", "D", "AI", "H"],
        var_name="Factor",
        value_name="Value"
    )

    fig1 = px.line(
        factor_df,
        x="Year",
        y="Value",
        color="Factor",
        markers=True,
        title="Quỹ đạo các yếu tố sản xuất"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================
    # BIỂU ĐỒ Y VÀ C
    # =====================================

    yc_df = traj_df.melt(
        id_vars="Year",
        value_vars=["Y", "C"],
        var_name="Variable",
        value_name="Value"
    )

    fig2 = px.line(
        yc_df,
        x="Year",
        y="Value",
        color="Variable",
        markers=True,
        title="Quỹ đạo sản lượng (Y) và tiêu dùng (C)"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )
    st.markdown('### 📌 Kết luận quản trị')
    st.success(
        """
Kết quả cho thấy quỹ đạo tối ưu của nền kinh tế trong giai đoạn
2026–2035. Các khoản đầu tư vào vốn vật chất, hạ tầng số,
AI và vốn nhân lực tạo ra hiệu ứng tích lũy, qua đó nâng cao
sản lượng và tiêu dùng theo thời gian.
"""
    )

# =========================
# TAB 4
# =========================

with tab4:

    st.markdown('### 🌪️ Phân tích cú sốc kinh tế năm 2028')

    st.markdown("""
Giả định năm **2028** xảy ra một cú sốc kinh tế tương tự tác động của bão Yagi,
làm sản lượng thực tế giảm **8%** so với kế hoạch.

Mô hình giữ nguyên chiến lược đầu tư tối ưu đã tìm được ở Tab 2 và đánh giá
mức độ ảnh hưởng lan tỏa tới các năm tiếp theo.
""")

    # =====================================
    # KỊCH BẢN CƠ SỞ
    # =====================================

    base_df = traj_df.copy()

    # =====================================
    # KỊCH BẢN CÚ SỐC
    # =====================================

    x_opt = result.x

    K = K0
    D = D0
    AI = AI0
    H = H0
    A = A0

    shock_data = []

    for t in range(T):

        share_K = x_opt[t*4]
        share_D = x_opt[t*4 + 1]
        share_AI = x_opt[t*4 + 2]
        share_H = x_opt[t*4 + 3]

        Y = (
            A
            * (K**0.33)
            * (L0**0.42)
            * (D**0.10)
            * (AI**0.08)
            * (H**0.07)
        )

        # =============================
        # CÚ SỐC NĂM 2028
        # =============================

        if years[t] == 2028:
            Y = Y * 0.92

        invest_ratio = (
            share_K
            + share_D
            + share_AI
            + share_H
        )

        C = Y * (1 - invest_ratio)

        shock_data.append([
            years[t],
            K,
            D,
            AI,
            H,
            Y,
            C
        ])

        IK = share_K * Y
        ID = share_D * Y
        IAI = share_AI * Y
        IH = share_H * Y

        K = (1 - delta_K) * K + IK
        D = (1 - delta_D) * D + ID
        AI = (1 - delta_AI) * AI + IAI

        H = (
            H
            + theta_H * IH
            - mu
        )

        growth = (
            phi1*np.log(D + 1)
            + phi2*np.log(AI + 1)
            + phi3*np.log(H + 1)
        )

        A = A * (1 + growth)

    shock_df = pd.DataFrame(
        shock_data,
        columns=[
            "Year",
            "K",
            "D",
            "AI",
            "H",
            "Y",
            "C"
        ]
    )

    # =====================================
    # KPI
    # =====================================

    loss_2028 = (
        base_df.loc[base_df["Year"] == 2028, "Y"].iloc[0]
        -
        shock_df.loc[shock_df["Year"] == 2028, "Y"].iloc[0]
    )

    loss_2035 = (
        base_df["Y"].iloc[-1]
        -
        shock_df["Y"].iloc[-1]
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Suy giảm GDP năm 2028",
            f"{loss_2028:,.0f}"
        )

    with col2:
        st.metric(
            "Tác động còn lại năm 2035",
            f"{loss_2035:,.0f}"
        )

    with col3:
        st.metric(
            "GDP 2035 sau cú sốc",
            f"{shock_df['Y'].iloc[-1]:,.0f}"
        )

    st.divider()

    # =====================================
    # SO SÁNH GDP
    # =====================================

    compare_y = pd.DataFrame({
        "Year": years,
        "Cơ sở": base_df["Y"],
        "Có cú sốc": shock_df["Y"]
    })

    compare_y = compare_y.melt(
        id_vars="Year",
        var_name="Scenario",
        value_name="GDP"
    )

    fig1 = px.line(
        compare_y,
        x="Year",
        y="GDP",
        color="Scenario",
        markers=True,
        title="So sánh quỹ đạo GDP"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # =====================================
    # SO SÁNH TIÊU DÙNG
    # =====================================

    compare_c = pd.DataFrame({
        "Year": years,
        "Cơ sở": base_df["C"],
        "Có cú sốc": shock_df["C"]
    })

    compare_c = compare_c.melt(
        id_vars="Year",
        var_name="Scenario",
        value_name="Consumption"
    )

    fig2 = px.line(
        compare_c,
        x="Year",
        y="Consumption",
        color="Scenario",
        markers=True,
        title="So sánh quỹ đạo tiêu dùng"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()

    st.markdown("""
### 📌 Nhận xét

- GDP năm 2028 giảm ngay lập tức do cú sốc.
- Do đầu tư được tính theo tỷ lệ của GDP, lượng vốn tích lũy mới trong năm 2028 cũng giảm.
- Các biến trạng thái K, D, AI và H ở các năm sau đều thấp hơn kịch bản cơ sở.
- Tác động của cú sốc không chỉ diễn ra trong một năm mà còn lan truyền sang các năm tiếp theo thông qua cơ chế tích lũy vốn và năng suất.
- Nền kinh tế vẫn phục hồi dần nhưng quỹ đạo tăng trưởng dài hạn bị dịch chuyển xuống dưới.
""")
    st.markdown('### 📌 Kết luận quản trị')
    st.success("""
Kết quả cho thấy cú sốc năm 2028 tạo ra hiệu ứng kéo dài nhiều năm sau đó.
Điều này phản ánh đặc điểm của mô hình động: một tổn thất ngắn hạn không chỉ làm giảm GDP hiện tại mà còn làm giảm tích lũy vốn, năng lực AI và năng suất trong tương lai.
""")

# =========================
# TAB 5
# =========================

with tab5:

    st.markdown('### ⚖️ So sánh chiến lược đầu tư dài hạn')

    st.markdown("""
So sánh hai chiến lược phát triển:

**Chiến lược 1 – Đầu tư trải đều**
- Duy trì tỷ lệ đầu tư ổn định qua các năm.

**Chiến lược 2 – Front-load**
- Đầu tư mạnh trong 3 năm đầu.
- Sau đó giảm dần để tận dụng hiệu ứng tích lũy vốn.
""")

    # ==================================
    # HÀM MÔ PHỎNG
    # ==================================

    def simulate_strategy(frontload=False):

        K = K0
        D = D0
        AI = AI0
        H = H0
        A = A0

        welfare = 0

        rows = []

        for idx, year in enumerate(years):

            Y = (
                A
                * (K**0.33)
                * (L0**0.42)
                * (D**0.10)
                * (AI**0.08)
                * (H**0.07)
            )

            # --------------------
            # CHIẾN LƯỢC
            # --------------------

            if frontload:

                if year <= 2028:

                    sK = 0.20
                    sD = 0.15
                    sAI = 0.12
                    sH = 0.08

                else:

                    sK = 0.10
                    sD = 0.08
                    sAI = 0.05
                    sH = 0.05

            else:

                sK = 0.15
                sD = 0.10
                sAI = 0.08
                sH = 0.07

            invest_share = (
                sK + sD + sAI + sH
            )

            C = Y * (1 - invest_share)

            welfare += (
                rho**idx
            ) * np.log(max(C, 1))

            rows.append([
                year,
                Y,
                C
            ])

            IK = sK * Y
            ID = sD * Y
            IAI = sAI * Y
            IH = sH * Y

            K = (1-delta_K)*K + IK
            D = (1-delta_D)*D + ID
            AI = (1-delta_AI)*AI + IAI

            H = (
                H
                + theta_H*IH
                - mu
            )

            growth = (
                phi1*np.log(D+1)
                + phi2*np.log(AI+1)
                + phi3*np.log(H+1)
            )

            A = A*(1+growth)

        return welfare, pd.DataFrame(
            rows,
            columns=[
                "Year",
                "Y",
                "C"
            ]
        )

    # ==================================
    # CHẠY 2 KỊCH BẢN
    # ==================================

    welfare_even, even_df = simulate_strategy(
        frontload=False
    )

    welfare_front, front_df = simulate_strategy(
        frontload=True
    )

    # ==================================
    # KPI
    # ==================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Welfare trải đều",
            f"{welfare_even:,.2f}"
        )

    with col2:

        st.metric(
            "Welfare Front-load",
            f"{welfare_front:,.2f}"
        )

    with col3:

        diff = (
            welfare_front
            - welfare_even
        )

        st.metric(
            "Chênh lệch",
            f"{diff:,.2f}"
        )

    st.divider()

    # ==================================
    # SO SÁNH GDP
    # ==================================

    compare_df = pd.DataFrame({

        "Year": years,

        "Đầu tư trải đều":
        even_df["Y"],

        "Front-load":
        front_df["Y"]

    })

    compare_df = compare_df.melt(
        id_vars="Year",
        var_name="Strategy",
        value_name="GDP"
    )

    fig1 = px.line(
        compare_df,
        x="Year",
        y="GDP",
        color="Strategy",
        markers=True,
        title="Quỹ đạo GDP theo chiến lược"
    )

    st.plotly_chart(
        fig1,
        use_container_width=True
    )

    # ==================================
    # SO SÁNH TIÊU DÙNG
    # ==================================

    compare_c = pd.DataFrame({

        "Year": years,

        "Đầu tư trải đều":
        even_df["C"],

        "Front-load":
        front_df["C"]

    })

    compare_c = compare_c.melt(
        id_vars="Year",
        var_name="Strategy",
        value_name="Consumption"
    )

    fig2 = px.line(
        compare_c,
        x="Year",
        y="Consumption",
        color="Strategy",
        markers=True,
        title="Quỹ đạo tiêu dùng theo chiến lược"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    st.divider()
    st.markdown('### 📌 Kết luận quản trị')
    if welfare_front > welfare_even:

        st.success(f"""
Chiến lược Front-load tạo Welfare cao hơn.

Nguyên nhân là đầu tư mạnh trong giai đoạn đầu giúp tích lũy
vốn vật chất, hạ tầng số, AI và vốn nhân lực nhanh hơn,
từ đó tạo hiệu ứng năng suất lan tỏa trong toàn bộ giai đoạn.
""")

    else:

        st.success(f"""
Chiến lược đầu tư trải đều tạo Welfare cao hơn.

Nguyên nhân là nền kinh tế duy trì mức tiêu dùng ổn định,
tránh đánh đổi quá lớn giữa tiêu dùng hiện tại và đầu tư.
""")

# =========================
# TAB 6
# =========================

with tab6:
    st.subheader("💬 Thảo luận chính sách")
    st.subheader(
        "a) Quỹ đạo tối ưu của K, D, AI, H có 'front-loaded' hay 'back-loaded' không? Vì sao mô hình đề xuất như vậy?"
    )

    st.markdown("""
Trong phần lớn các mô hình tăng trưởng động, đặc biệt khi tồn tại hiệu ứng tích lũy vốn và năng suất nội sinh,
nghiệm tối ưu thường có xu hướng **front-loaded** đối với các khoản đầu tư có khả năng tạo ra tác động lan tỏa dài hạn.

Kết quả mô phỏng cho thấy đầu tư vào K (vốn vật chất), D (hạ tầng số), AI và H (vốn nhân lực) được ưu tiên nhiều hơn
ở giai đoạn đầu của chu kỳ 2026–2035. Điều này xuất phát từ việc các khoản đầu tư sớm có thêm thời gian để tích lũy,
qua đó tạo ra mức sản lượng và năng suất cao hơn trong các năm tiếp theo.

Đặc biệt, đầu tư vào D, AI và H không chỉ tác động trực tiếp tới sản lượng mà còn làm tăng năng suất nhân tố tổng hợp (TFP).
Do đó, một đồng đầu tư ở giai đoạn đầu có thể tạo ra lợi ích tích lũy lớn hơn đáng kể so với đầu tư ở giai đoạn cuối.

Nếu mô hình xuất hiện xu hướng back-loaded thì nguyên nhân thường đến từ:
- Tỷ lệ chiết khấu thấp hơn.
- Ràng buộc ngân sách ngắn hạn chặt chẽ.
- Hoặc chi phí đầu tư giảm dần theo thời gian.
""")

    st.divider()

    st.subheader(
        "b) Tỷ lệ đầu tư AI / đầu tư H theo thời gian có ổn định không? Mô hình ngụ ý gì về việc đào tạo nhân lực nên đi trước hay đồng thời với đầu tư AI?"
    )

    st.markdown("""
Tỷ lệ đầu tư AI và đầu tư vốn nhân lực không nhất thiết duy trì cố định trong toàn bộ giai đoạn.

Trong thực tế cũng như trong mô hình, đầu tư AI chỉ phát huy hiệu quả khi nền kinh tế có đủ năng lực hấp thụ công nghệ.
Năng lực hấp thụ này được phản ánh thông qua chất lượng nguồn nhân lực, kỹ năng số và trình độ nghiên cứu phát triển.

Do đó, kết quả mô hình thường hàm ý rằng:

- Đầu tư vốn nhân lực không nên đi sau AI.
- Đầu tư nhân lực nên được triển khai trước hoặc đồng thời với đầu tư AI.
- Nếu đầu tư AI quá nhanh trong khi nguồn nhân lực chưa đáp ứng, hiệu quả biên của AI sẽ giảm đáng kể.

Đây cũng là lý do nhiều quốc gia ưu tiên đào tạo kỹ sư, nhà khoa học dữ liệu và chuyên gia AI trước khi mở rộng đầu tư
quy mô lớn vào hạ tầng AI hoặc trung tâm dữ liệu.

Trong bối cảnh Việt Nam, điều này phù hợp với các chương trình phát triển nguồn nhân lực chất lượng cao,
đào tạo kỹ sư AI và bán dẫn đang được triển khai trong giai đoạn hiện nay.
""")

    st.divider()

    st.subheader(
        "c) Hệ số chiết khấu ρ = 0,97 ngụ ý chính phủ quan tâm nhiều đến dài hạn. Nếu ρ = 0,90 (ngắn hạn hơn), kết quả thay đổi thế nào? Đây có phải lý do các chính phủ thường 'dưới đầu tư' vào R&D?"
    )

    st.markdown("""
Hệ số chiết khấu ρ phản ánh mức độ coi trọng phúc lợi tương lai.

Khi:

- ρ = 0,97 → Chính phủ đánh giá khá cao lợi ích dài hạn.
- ρ = 0,90 → Chính phủ ưu tiên nhiều hơn cho tiêu dùng và kết quả ngắn hạn.

Nếu giảm ρ từ 0,97 xuống 0,90, mô hình thường có xu hướng:

1. Giảm đầu tư dài hạn vào AI và vốn nhân lực.
2. Tăng tiêu dùng hiện tại.
3. Giảm tốc độ tích lũy vốn và năng suất trong tương lai.
4. Làm GDP cuối kỳ thấp hơn so với kịch bản dài hạn.

Điều này phản ánh một hiện tượng phổ biến trong kinh tế công:
nhiều khoản đầu tư như giáo dục, nghiên cứu khoa học, R&D hoặc hạ tầng số
có chi phí phát sinh ngay lập tức nhưng lợi ích chỉ xuất hiện sau nhiều năm.

Trong khi đó, chu kỳ chính trị thường ngắn hơn chu kỳ lợi ích của các khoản đầu tư này.
Do đó, các chính phủ có thể có xu hướng ưu tiên các chính sách tạo kết quả nhanh,
dẫn tới tình trạng đầu tư dưới mức tối ưu vào R&D, đổi mới sáng tạo và phát triển công nghệ.
""")

st.divider()

st.caption("""
Nguồn tham khảo:

• Văn kiện Đại hội XIII

• Chiến lược phát triển kinh tế - xã hội 2021–2030

• Quyết định 127/QĐ-TTg về phát triển AI quốc gia

""")