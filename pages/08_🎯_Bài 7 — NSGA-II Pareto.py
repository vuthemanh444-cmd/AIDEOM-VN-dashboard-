import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
from pymoo.termination import get_termination

# ==================================
# CẤU HÌNH TRANG
# ==================================

st.set_page_config(
    page_title="Bài 7 - Tối ưu đa mục tiêu",
    page_icon="🎯",
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
# ==================================
# TIÊU ĐỀ
# ==================================

st.markdown("## 🎯 Bài 7 - Tối ưu hóa đa mục tiêu cho phát triển kinh tế số và AI")
st.markdown(
    "<span class='badge badge-hard'>KHÁ KHÓ</span>"
    "<span class='badge badge-info'>NSGA-II · Pareto</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Lựa chọn phương án phân bổ ngân sách trong điều kiện nhiều mục tiêu chính sách

Bài toán xem xét đồng thời:

- Tăng trưởng GDP
- Bao trùm xã hội
- Môi trường và phát thải
- An ninh dữ liệu

Phương pháp:

- Tối ưu hóa đa mục tiêu
- NSGA-II
- Pareto Front
- TOPSIS lựa chọn nghiệm thỏa hiệp
""")

# =========================
# ĐỌC DỮ LIỆU
# =========================

regions_df = pd.read_csv(
    "data/marginal_impact_coefficient.csv"
)

params_df = pd.read_csv(
    "data/vietnam_regions_3.csv"
)

# ==================================
# TABS
# ==================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "⚙️ NSGA-II",
    "📊 Pareto Front",
    "🎯 TOPSIS",
    "⚖️ Chi phí cơ hội",
    "💬 Thảo luận chính sách"
])

# ==================================
# TAB 1
# ==================================
with tab1:

    st.markdown('### 📋 Dữ liệu vùng kinh tế')

    st.dataframe(
        regions_df,
        use_container_width=True,
        hide_index=True
    )

    csv_regions = regions_df.to_csv(index=False)

    st.download_button(
        "📥 Tải dữ liệu vùng",
        csv_regions,
        "vietnam_regions_2024.csv",
        "text/csv"
    )

    st.divider()

    st.markdown('### 📋 Tham số môi trường và an ninh dữ liệu')

    st.dataframe(
        params_df,
        use_container_width=True,
        hide_index=True
    )

    csv_params = params_df.to_csv(index=False)

    st.download_button(
        "📥 Tải dữ liệu tham số",
        csv_params,
        "vietnam_regions_3.csv",
        "text/csv"
    )
     
# ==================================
# TAB 2
# ==================================
with tab2:

    st.markdown('### ⚙️ Tối ưu đa mục tiêu bằng NSGA-II')

    # =========================
    # CHUẨN BỊ DỮ LIỆU
    # =========================

    regions = regions_df["Region"].tolist()

    investment_types = [
        "I",
        "D",
        "AI",
        "H"
    ]

    beta = regions_df[
        ["I", "D", "AI", "H"]
    ].values

    emission = params_df["eᵣ"].values
    rho = params_df["ρᵣ"].values
    sigma = params_df["σᵣ"].values

    n_regions = len(regions)

    # =========================
    # NSGA-II PROBLEM
    # =========================

    class VietnamDigitalProblem(
        ElementwiseProblem
    ):

        def __init__(self):

            super().__init__(
                n_var=24,
                n_obj=4,
                n_ieq_constr=1,
                xl=0,
                xu=12000
            )

        def _evaluate(
            self,
            x,
            out,
            *args,
            **kwargs
        ):

            X = x.reshape(
                n_regions,
                4
            )

            # ---------------------
            # f1 GDP Gain
            # maximize
            # ---------------------

            f1 = -np.sum(
                X * beta
            )

            # ---------------------
            # f2 Equality
            # ---------------------

            region_budget = X.sum(
                axis=1
            )

            f2 = np.mean(
                np.abs(
                    region_budget
                    -
                    region_budget.mean()
                )
            )

            # ---------------------
            # f3 Emission
            # ---------------------

            f3 = np.sum(
                emission
                *
                (
                    X[:,0]
                    +
                    X[:,2]
                )
            )

            # ---------------------
            # f4 Data Security
            # ---------------------

            f4 = np.sum(
                rho * X[:,2]
            ) - np.sum(
                sigma * X[:,3]
            )

            # ---------------------
            # Budget Constraint
            # ---------------------

            g1 = (
                X.sum()
                -
                50000
            )

            out["F"] = [
                f1,
                f2,
                f3,
                f4
            ]

            out["G"] = [
                g1
            ]

    # =========================
    # GIẢI NSGA-II
    # =========================

    problem = VietnamDigitalProblem()

    algorithm = NSGA2(
        pop_size=100
    )

    result = minimize(
        problem,
        algorithm,
        get_termination(
            "n_gen",
            200
        ),
        seed=42,
        verbose=False
    )

    F = result.F

    pareto_df = pd.DataFrame({
        "GDP Gain": -F[:,0],
        "Inequality": F[:,1],
        "Emission": F[:,2],
        "Security Risk": F[:,3]
    })

    # =========================
    # KPI
    # =========================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Biến quyết định",
        24
    )

    col2.metric(
        "Mục tiêu",
        4
    )

    col3.metric(
        "Population",
        100
    )

    col4.metric(
        "Pareto Solutions",
        len(pareto_df)
    )

    st.divider()

    # =========================
    # BẢNG PARETO
    # =========================

    st.markdown('### 📋 Tập nghiệm Pareto')

    st.dataframe(
        pareto_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.scatter(
        pareto_df,
        x="GDP Gain",
        y="Inequality",
        color="Emission",
        title="Pareto Front"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD
    # =========================

    csv = pareto_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải Pareto Front",
        csv,
        "pareto_front.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.info(f"""
    Thuật toán NSGA-II đã tìm được
    {len(pareto_df)} nghiệm Pareto.

    Không tồn tại một phương án tối ưu
    tuyệt đối cho cả bốn mục tiêu.

    Mỗi nghiệm Pareto đại diện cho một
    sự đánh đổi giữa:

    • Tăng trưởng GDP

    • Bao trùm xã hội

    • Phát thải môi trường

    • An ninh dữ liệu

    Đây là cơ sở để nhà hoạch định
    chính sách lựa chọn phương án phù hợp
    với ưu tiên phát triển của Việt Nam.
    """)

# ==================================
# TABS 3
# ==================================

with tab3:

    st.markdown('### 📊 Quần thể Pareto cuối cùng')

    # =========================
    # TRÍCH XUẤT PARETO FRONT
    # =========================

    pareto_df = pd.DataFrame(
        result.F,
        columns=[
            "f1_GDP",
            "f2_Gini",
            "f3_Emission",
            "f4_Risk"
        ]
    )

    # Vì pymoo mặc định MIN tất cả mục tiêu,
    # ta đổi dấu lại f1 để dễ diễn giải

    pareto_df["f1_GDP"] = -pareto_df["f1_GDP"]

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số nghiệm Pareto",
        len(pareto_df)
    )

    col2.metric(
        "GDP cao nhất",
        f"{pareto_df['f1_GDP'].max():,.0f}"
    )

    col3.metric(
        "GDP thấp nhất",
        f"{pareto_df['f1_GDP'].min():,.0f}"
    )

    st.divider()

    # =========================
    # BẢNG DỮ LIỆU
    # =========================

    st.markdown('### 📋 Tập nghiệm Pareto')

    st.dataframe(
        pareto_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # SCATTER 3D
    # =========================

    st.markdown('### 🌐 Pareto Scatter Plot 3D')

    fig_3d = px.scatter_3d(
        pareto_df,
        x="f1_GDP",
        y="f2_Gini",
        z="f3_Emission",
        color="f4_Risk",
        title="Pareto Scatter Plot 3D",
        height=700
    )

    st.plotly_chart(
        fig_3d,
        use_container_width=True
    )

    st.divider()

    # =========================
    # PARALLEL COORDINATES
    # =========================

    st.markdown('### 📈 Parallel Coordinates')

    fig_parallel = px.parallel_coordinates(
        pareto_df,
        dimensions=[
            "f1_GDP",
            "f2_Gini",
            "f3_Emission",
            "f4_Risk"
        ],
        color="f1_GDP"
    )

    fig_parallel.update_layout(
        height=700
    )

    st.plotly_chart(
        fig_parallel,
        use_container_width=True
    )

    st.divider()

    # =========================
    # DOWNLOAD
    # =========================

    csv = pareto_df.to_csv(index=False)

    st.download_button(
        "📥 Tải tập nghiệm Pareto",
        csv,
        "pareto_front.csv",
        "text/csv"
    )

    st.divider()

    # =========================
    # KẾT LUẬN QUẢN TRỊ
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(f"""
    NSGA-II tìm được {len(pareto_df)} nghiệm Pareto.

    Không tồn tại một phương án tối ưu tuyệt đối.

    Mỗi nghiệm Pareto phản ánh một mức đánh đổi khác nhau giữa:

    • Tăng trưởng GDP

    • Công bằng vùng miền

    • Môi trường

    • An ninh dữ liệu

    Đây chính là cơ sở định lượng để Chính phủ lựa chọn
    phương án phù hợp với ưu tiên phát triển trong từng giai đoạn.
    """)

# ==================================
# TABS 4
# ==================================

with tab4:

    st.markdown('### 🏆 Lựa chọn nghiệm thỏa hiệp bằng TOPSIS')

    # =========================
    # DỮ LIỆU PARETO
    # =========================

    pareto_df = pd.DataFrame(
        result.F,
        columns=[
            "GDP",
            "Gini",
            "Emission",
            "Risk"
        ]
    )

    # đổi dấu GDP về dạng maximize

    pareto_df["GDP"] = -pareto_df["GDP"]

    # =========================
    # TRỌNG SỐ CHÍNH SÁCH
    # =========================

    weights = np.array([
        0.40,   # GDP
        0.25,   # Bao trùm
        0.20,   # Môi trường
        0.15    # An ninh
    ])

    # =========================
    # TOPSIS
    # =========================

    X = pareto_df.values

    # Vector normalization

    norm = X / np.sqrt((X**2).sum(axis=0))

    # Weighted matrix

    V = norm * weights

    # GDP = benefit
    # các mục tiêu còn lại = cost

    ideal_best = np.array([
        V[:,0].max(),
        V[:,1].min(),
        V[:,2].min(),
        V[:,3].min()
    ])

    ideal_worst = np.array([
        V[:,0].min(),
        V[:,1].max(),
        V[:,2].max(),
        V[:,3].max()
    ])

    S_plus = np.sqrt(
        ((V - ideal_best)**2).sum(axis=1)
    )

    S_minus = np.sqrt(
        ((V - ideal_worst)**2).sum(axis=1)
    )

    C = S_minus / (S_plus + S_minus)

    pareto_df["TOPSIS Score"] = C

    pareto_df = pareto_df.sort_values(
        "TOPSIS Score",
        ascending=False
    ).reset_index(drop=True)

    best_solution = pareto_df.iloc[0]

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số nghiệm Pareto",
        len(pareto_df)
    )

    col2.metric(
        "TOPSIS cao nhất",
        f"{best_solution['TOPSIS Score']:.4f}"
    )

    col3.metric(
        "Nghiệm được chọn",
        "#1"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    st.markdown('### 📋 Xếp hạng TOPSIS')

    st.dataframe(
        pareto_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ
    # =========================

    st.markdown('### 📊 TOPSIS Score')

    top10 = pareto_df.head(10)

    fig = px.bar(
        top10,
        x=top10.index.astype(str),
        y="TOPSIS Score",
        text_auto=".3f",
        title="TOPSIS Score của các nghiệm Pareto tốt nhất"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =========================
    # DOWNLOAD
    # =========================

    csv = pareto_df.to_csv(index=False)

    st.download_button(
        "📥 Tải kết quả TOPSIS",
        csv,
        "pareto_topsis.csv",
        "text/csv"
    )

    st.divider()

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(f"""
    TOPSIS đã lựa chọn một nghiệm thỏa hiệp từ
    {len(pareto_df)} nghiệm Pareto.

    Nghiệm này đạt TOPSIS Score =
    {best_solution['TOPSIS Score']:.4f}.

    Kết quả phản ánh ưu tiên chính sách:

    • 40% cho tăng trưởng GDP

    • 25% cho bao trùm xã hội

    • 20% cho môi trường

    • 15% cho an ninh dữ liệu

    Đây là phương án cân bằng nhất giữa
    bốn mục tiêu phát triển của Việt Nam.
    """)

# ==================================
# TABS 5
# ==================================

with tab5:

    st.markdown('### ⚖️ Phân tích chi phí cơ hội')

    # =========================
    # DỮ LIỆU PARETO
    # =========================

    pareto_df = pd.DataFrame(
        result.F,
        columns=[
            "GDP",
            "Gini",
            "Emission",
            "Risk"
        ]
    )

    pareto_df["GDP"] = -pareto_df["GDP"]

    # =========================
    # TOPSIS (giống Tab 4)
    # =========================

    weights = np.array([
        0.40,
        0.25,
        0.20,
        0.15
    ])

    X = pareto_df.values

    norm = X / np.sqrt(
        (X**2).sum(axis=0)
    )

    V = norm * weights

    ideal_best = np.array([
        V[:,0].max(),
        V[:,1].min(),
        V[:,2].min(),
        V[:,3].min()
    ])

    ideal_worst = np.array([
        V[:,0].min(),
        V[:,1].max(),
        V[:,2].max(),
        V[:,3].max()
    ])

    S_plus = np.sqrt(
        ((V - ideal_best)**2).sum(axis=1)
    )

    S_minus = np.sqrt(
        ((V - ideal_worst)**2).sum(axis=1)
    )

    pareto_df["TOPSIS"] = (
        S_minus /
        (S_plus + S_minus)
    )

    # =========================
    # NGHIỆM TĂNG TRƯỞNG CAO NHẤT
    # =========================

    growth_solution = pareto_df.loc[
        pareto_df["GDP"].idxmax()
    ]

    # =========================
    # NGHIỆM THỎA HIỆP
    # =========================

    compromise_solution = pareto_df.loc[
        pareto_df["TOPSIS"].idxmax()
    ]

    # =========================
    # CHI PHÍ CƠ HỘI
    # =========================

    gini_cost = (
        (
            growth_solution["Gini"]
            - compromise_solution["Gini"]
        )
        /
        compromise_solution["Gini"]
    ) * 100

    emission_cost = (
        (
            growth_solution["Emission"]
            - compromise_solution["Emission"]
        )
        /
        compromise_solution["Emission"]
    ) * 100

    # =========================
    # CHÊNH LỆCH TUYỆT ĐỐI
    # =========================

    gini_gap = (
    growth_solution["Gini"]
    - compromise_solution["Gini"]
    )

    emission_gap = (
    growth_solution["Emission"]
    - compromise_solution["Emission"]
    )

    gdp_gain = (
    growth_solution["GDP"]
    - compromise_solution["GDP"]
    )

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
    "GDP tăng thêm",
    f"{gdp_gain:,.0f}"
    )

    col2.metric(
    "Tăng bất bình đẳng",
    f"{gini_gap:,.2f}"
    )

    col3.metric(
    "Tăng phát thải",
    f"{emission_gap:,.2f}"
    )

    # =========================
    # BẢNG SO SÁNH
    # =========================

    compare_df = pd.DataFrame({

        "Chỉ tiêu": [
            "GDP",
            "Gini",
            "Emission",
            "Risk"
        ],

        "Nghiệm tăng trưởng tối đa": [
            growth_solution["GDP"],
            growth_solution["Gini"],
            growth_solution["Emission"],
            growth_solution["Risk"]
        ],

        "Nghiệm thỏa hiệp": [
            compromise_solution["GDP"],
            compromise_solution["Gini"],
            compromise_solution["Emission"],
            compromise_solution["Risk"]
        ]
    })

    st.dataframe(
        compare_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ
    # =========================

    chart_df = pd.DataFrame({

        "Mục tiêu": [
            "Bao trùm (Gini)",
            "Môi trường"
        ],

        "Chi phí cơ hội (%)": [
            gini_cost,
            emission_cost
        ]
    })

    fig = px.bar(
        chart_df,
        x="Mục tiêu",
        y="Chi phí cơ hội (%)",
        text_auto=".2f",
        title="Chi phí cơ hội của nghiệm tăng trưởng tối đa"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(f"""
    Nghiệm tăng trưởng tối đa giúp GDP tăng thêm
    {gdp_gain:,.0f} đơn vị so với nghiệm thỏa hiệp.

    Tuy nhiên:

    • Chỉ số bất bình đẳng tăng thêm {gini_gap:,.2f}

    • Phát thải tăng thêm {emission_gap:,.2f}

    Điều này cho thấy việc tối đa hóa tăng trưởng
    tạo ra chi phí cơ hội đáng kể về bao trùm xã hội
    và môi trường.

    Nghiệm thỏa hiệp được TOPSIS lựa chọn phản ánh
    sự cân bằng tốt hơn giữa bốn mục tiêu chính sách.
    """)

# ==================================
# TABS 6
# ==================================
with tab6:

    st.markdown('### 💬 Thảo luận chính sách')

    # =========================
    # CÂU A
    # =========================

    st.markdown("""
    ### a) Khi quan sát đường biên Pareto, em thấy đánh đổi giữa tăng trưởng và bao trùm có rõ ràng không? Mức đánh đổi đó nói lên điều gì về cơ cấu kinh tế Việt Nam?
    """)

    st.write("""
    Kết quả Pareto cho thấy tồn tại sự đánh đổi tương đối rõ ràng giữa mục tiêu tăng trưởng GDP và mục tiêu bao trùm xã hội. 
    Các nghiệm đạt mức GDP rất cao thường đi kèm với mức độ bất bình đẳng lớn hơn và phân bổ nguồn lực tập trung vào các vùng có năng suất cao như Đồng bằng sông Hồng và Đông Nam Bộ.

    Điều này phản ánh đặc điểm của cơ cấu kinh tế Việt Nam hiện nay: tăng trưởng kinh tế vẫn chủ yếu được dẫn dắt bởi các cực tăng trưởng lớn, trong khi nhiều vùng còn khó khăn cần thêm nguồn lực để thu hẹp khoảng cách phát triển. 
    Vì vậy, nếu chỉ tối đa hóa tăng trưởng, chênh lệch vùng miền có thể gia tăng.
    """)

    st.divider()

    # =========================
    # CÂU B
    # =========================

    st.markdown("""
    ### b) Trọng số (0,40; 0,25; 0,20; 0,15) có phản ánh đúng ưu tiên hiện tại của Việt Nam không? Em sẽ điều chỉnh thế nào để phù hợp với COP26 và Quyết định 127/QĐ-TTg?
    """)

    st.write("""
    Bộ trọng số hiện tại tương đối phù hợp với định hướng phát triển của Việt Nam trong giai đoạn chuyển đổi số, khi mục tiêu tăng trưởng kinh tế vẫn được ưu tiên cao nhất.

    Tuy nhiên, sau cam kết phát thải ròng bằng 0 tại COP26 và định hướng phát triển AI bền vững trong Quyết định 127/QĐ-TTg, có thể cân nhắc tăng trọng số cho mục tiêu môi trường và an ninh dữ liệu.

    Một phương án điều chỉnh có thể là:

    • Tăng trưởng: 35%

    • Bao trùm: 25%

    • Môi trường: 25%

    • An ninh dữ liệu: 15%

    Cách phân bổ này giúp cân bằng hơn giữa tăng trưởng kinh tế và các mục tiêu phát triển bền vững dài hạn.
    """)

    st.divider()

    # =========================
    # CÂU C
    # =========================

    st.markdown("""
    ### c) Vai trò của NSGA-II ở đây có gì khác so với LP đơn mục tiêu? Nó có thay thế được quyết định chính trị không?
    """)

    st.write("""
    LP đơn mục tiêu chỉ tạo ra một nghiệm tối ưu duy nhất dựa trên một tiêu chí cụ thể, chẳng hạn tối đa hóa GDP. 
    Trong khi đó, NSGA-II tạo ra cả một tập nghiệm Pareto, cho phép quan sát nhiều phương án đánh đổi giữa tăng trưởng, công bằng, môi trường và an ninh dữ liệu.

    Vì vậy, NSGA-II đóng vai trò là công cụ hỗ trợ ra quyết định, giúp nhà hoạch định chính sách nhìn thấy hậu quả của từng lựa chọn và lượng hóa các đánh đổi giữa các mục tiêu.

    Tuy nhiên, NSGA-II không thể thay thế quyết định chính trị. 
    Việc lựa chọn nghiệm nào trên đường biên Pareto cuối cùng vẫn phụ thuộc vào ưu tiên phát triển của quốc gia, các mục tiêu chiến lược, điều kiện xã hội và các cân nhắc thể chế. 
    Thuật toán chỉ cung cấp bằng chứng định lượng, còn quyết định cuối cùng thuộc về quá trình hoạch định chính sách.
    """)

st.divider()

st.caption("""
Nguồn dữ liệu:
           
• Quyết định 411/QĐ-TTg về Chiến lược quốc gia phát triển kinh tế số và xã hội số
           
• Dữ liệu vùng kinh tế xã hội Việt Nam (Bài 4)
""")