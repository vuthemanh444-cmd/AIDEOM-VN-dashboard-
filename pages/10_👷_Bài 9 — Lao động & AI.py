import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pulp import *

# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 9 - AI và thị trường lao động",
    page_icon="👷",
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

st.markdown("## 👷 Bài 9 - Tối ưu hóa đào tạo lại lao động trong kỷ nguyên AI")
st.markdown(
    "<span class='badge badge-hard'>KHÁ KHÓ</span>"
    "<span class='badge badge-info'>LP · NetJob</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Bài toán phân bổ ngân sách đào tạo và AI cho thị trường lao động Việt Nam

Mục tiêu:

- Tối đa hóa tổng số việc làm ròng (NetJob)
- Giảm tác động mất việc do tự động hóa
- Bảo đảm mọi ngành đều có NetJob không âm
- Hỗ trợ quá trình chuyển đổi số và AI bền vững

Phương pháp:

- Linear Programming (PuLP / CBC)
- Phân tích ngưỡng đào tạo tối thiểu
- Sankey Diagram mô phỏng dịch chuyển lao động
- Kiểm tra tính khả thi của các ràng buộc xã hội
""")

# =========================
# ĐỌC DỮ LIỆU
# =========================

df = pd.read_csv("data/vietnam_sectors_2.csv")

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "⚙️ Tối ưu NetJob",
    "🎯 Ngưỡng đào tạo tối thiểu",
    "🌊 Dịch chuyển lao động",
    "🔒 Ràng buộc mất việc ≤ 5%",
    "💬 Thảo luận chính sách"
])

# =========================
# TAB 1
# =========================
with tab1:

    st.markdown('### 📋 Dữ liệu đầu vào')

    st.markdown("""
Bộ dữ liệu gồm 8 ngành kinh tế lớn của Việt Nam.

Các tham số:

- Labor: quy mô lao động (triệu người)
- Risk: mức độ rủi ro tự động hóa (%)
- a₁, a₂: việc làm mới tạo ra từ AI và dữ liệu
- b₁: việc làm nâng cấp từ đào tạo
- c₁: việc làm bị thay thế do tự động hóa
- d₁: năng lực đào tạo lại lao động
""")

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Số ngành",
            len(df)
        )

    with col2:
        st.metric(
            "Tổng lao động",
            f"{df['Labor'].sum():.2f} triệu"
        )

    with col3:
        st.metric(
            "Rủi ro TB",
            f"{df['Risk'].mean():.1f}%"
        )

    st.divider()

    fig = px.bar(
        df,
        x="Sectors",
        y="Labor",
        color="Risk",
        title="Quy mô lao động và rủi ro tự động hóa"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    csv = df.to_csv(index=False)

    st.download_button(
        "📥 Tải dữ liệu",
        csv,
        "data/vietnam_sectors_2.csv",
        "text/csv"
    )

# =========================
# TAB 2
# =========================

with tab2:

    st.markdown('### ⚙️ Tối ưu hóa NetJob bằng PuLP')

    st.markdown("""
Mục tiêu:

- Tối đa hóa tổng số việc làm ròng (NetJob)
- Ngân sách tối đa: 30.000 tỷ đồng
- Mọi ngành phải có NetJob không âm
""")

    # ==========================
    # MÔ HÌNH LP
    # ==========================

    model = LpProblem(
        "Vietnam_AI_Labor",
        LpMaximize
    )

    sectors = df["Sectors"].tolist()

    x_AI = LpVariable.dicts(
        "AI",
        sectors,
        lowBound=0
    )

    x_H = LpVariable.dicts(
        "Human",
        sectors,
        lowBound=0
    )

    # ==========================
    # HÀM MỤC TIÊU
    # ==========================

    netjob_expr = []

    for i, row in df.iterrows():

        s = row["Sectors"]

        risk = row["Risk"] / 100

        netjob = (
            row["a₁"] * x_AI[s]
            + row["b₁"] * x_H[s]
            - row["c₁"] * risk * x_AI[s]
        )

        netjob_expr.append(netjob)

    model += lpSum(netjob_expr)

    # ==========================
    # NGÂN SÁCH
    # ==========================

    model += (
        lpSum(
            x_AI[s] + x_H[s]
            for s in sectors
        )
        <= 30000
    )

    # ==========================
    # NETJOB >= 0
    # ==========================

    for i, row in df.iterrows():

        s = row["Sectors"]

        risk = row["Risk"] / 100

        model += (
            row["a₁"] * x_AI[s]
            + row["b₁"] * x_H[s]
            - row["c₁"] * risk * x_AI[s]
            >= 0
        )

    # ==========================
    # SOLVE
    # ==========================

    model.solve(PULP_CBC_CMD(msg=False))

    # ==========================
    # KẾT QUẢ
    # ==========================

    results = []

    total_netjob = 0

    for i, row in df.iterrows():

        s = row["Sectors"]

        ai = value(x_AI[s])
        h = value(x_H[s])

        risk = row["Risk"] / 100

        netjob = (
            row["a₁"] * ai
            + row["b₁"] * h
            - row["c₁"] * risk * ai
        )

        total_netjob += netjob

        results.append([
            s,
            ai,
            h,
            netjob
        ])

    result_df = pd.DataFrame(
        results,
        columns=[
            "Ngành",
            "Đầu tư AI",
            "Đầu tư đào tạo",
            "NetJob"
        ]
    )
    
    total_budget = (
        result_df["Đầu tư AI"].sum()
        + result_df["Đầu tư đào tạo"].sum()
    )

    # ==========================
    # KPI
    # ==========================

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Trạng thái",
            LpStatus[model.status]
        )

    with col2:
        st.metric(
            "Ngân sách sử dụng",
            f"{total_budget:,.0f}"
        )

    with col3:
        st.metric(
            "Tổng NetJob",
            f"{total_netjob:,.0f}"
        )

    st.divider()

    # ==========================
    # BẢNG KẾT QUẢ
    # ==========================

    st.markdown('### Kết quả phân bổ tối ưu')

    st.dataframe(
        result_df.style.format({
            "Đầu tư AI":"{:,.2f}",
            "Đầu tư đào tạo":"{:,.2f}",
            "NetJob":"{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================
    # BIỂU ĐỒ
    # ==========================

    plot_df = result_df.melt(
        id_vars="Ngành",
        value_vars=[
            "Đầu tư AI",
            "Đầu tư đào tạo"
        ],
        var_name="Loại đầu tư",
        value_name="Giá trị"
    )

    fig = px.bar(
        plot_df,
        x="Ngành",
        y="Giá trị",
        color="Loại đầu tư",
        barmode="group",
        title="Phân bổ ngân sách tối ưu theo ngành"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()
    st.markdown('### 📌Nhận xét')
    st.success("""
    Mô hình tối ưu dồn toàn bộ ngân sách vào ngành Giáo dục-Đào tạo do hệ số tạo việc làm từ đào tạo (b₁=55) cao nhất. Đây là đặc điểm thường gặp của mô hình LP tuyến tính khi không có các ràng buộc cân bằng ngành hoặc giới hạn hấp thụ vốn

    """)

    st.markdown('### 🔍 Kiểm chứng bằng CVXPY')
    import cvxpy as cp

    n = len(df)

    # Biến quyết định
    x_ai = cp.Variable(n, nonneg=True)
    x_h = cp.Variable(n, nonneg=True)

    # Hệ số
    risk = df["Risk"].values / 100
    a1 = df["a₁"].values
    b1 = df["b₁"].values
    c1 = df["c₁"].values

    # NetJob
    netjob = (
    (a1 - c1 * risk) * x_ai
    + b1 * x_h
   )

   # Hàm mục tiêu
    objective = cp.Maximize(cp.sum(netjob))

   # Ràng buộc
    constraints = [
    cp.sum(x_ai + x_h) <= 30000,
    netjob >= 0
   ]

   # Giải
    problem = cp.Problem(
    objective,
    constraints
   )

    problem.solve()

   # Kết quả
    cvx_result = pd.DataFrame({
    "Ngành": df["Sectors"],
    "x_AI_CVXPY": x_ai.value,
    "x_H_CVXPY": x_h.value,
    "NetJob_CVXPY": netjob.value
    })

    st.dataframe(
    cvx_result.round(2),
    use_container_width=True,
    hide_index=True
    )

    st.metric(
    "Tổng NetJob (CVXPY)",
    f"{problem.value:,.0f}"
    )

    st.markdown('### 📌 Kết luận quản trị')
    st.success("""
    Kết quả kiểm chứng bằng CVXPY cho nghiệm gần như trùng khớp với PuLP/CBC. Cả hai phương pháp đều phân bổ gần như toàn bộ ngân sách vào ngành Giáo dục - Đào tạo do ngành này có hệ số tạo việc làm từ đào tạo cao nhất (b₁ = 55). Sai khác rất nhỏ (0,04 tỷ đồng) xuất phát từ sai số số học của thuật toán tối ưu.

    """)
# =========================
# TAB 3
# =========================

with tab3:

    st.markdown('### 🎓 Ngưỡng đào tạo tối thiểu cho ngành CN chế biến chế tạo')

    # Thông số ngành 2
    sector = df[df["Sectors"] == "CN chế biến chế tạo"].iloc[0]

    a1 = sector["a₁"]
    b1 = sector["b₁"]
    c1 = sector["c₁"]
    risk = sector["Risk"] / 100

    # Giả định đầu tư AI tối đa
    x_ai_max = 30000

    ai_coef = a1 - c1 * risk

    st.markdown(f"""
    **Thông số ngành CN chế biến chế tạo**

    - a₁ = {a1}
    - b₁ = {b1}
    - c₁ = {c1}
    - Risk = {risk:.0%}
    """)
    
    st.markdown("Hệ số việc làm ròng từ AI:")
    st.markdown("[ 32.5 − 62.4 × 0.42 = 6.292 ]")
    
    if ai_coef >= 0:

        x_h_min = 0

        netjob_ai_only = ai_coef * x_ai_max

        st.success(
            "AI vẫn tạo việc làm ròng dương, do đó không cần đầu tư đào tạo tối thiểu để giữ NetJob ≥ 0."
        )

    else:

        x_h_min = (-ai_coef * x_ai_max) / b1

        netjob_ai_only = ai_coef * x_ai_max

        st.warning(
            "AI làm giảm việc làm ròng, cần đầu tư đào tạo để bù đắp."
        )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "AI tối đa",
        f"{x_ai_max:,.0f} tỷ"
    )

    col2.metric(
        "NetJob nếu chỉ đầu tư AI",
        f"{netjob_ai_only:,.0f}"
    )

    col3.metric(
        "Ngưỡng x_H tối thiểu",
        f"{x_h_min:,.2f} tỷ"
    )

    st.markdown("### 📌 Phân tích kết quả")

    st.markdown(f"""
    Với ngành **CN chế biến chế tạo**, hệ số tạo việc làm ròng từ AI là:

    \[
    {ai_coef:.3f}
    \]

    Vì hệ số này **dương**, nên ngay cả khi toàn bộ ngân sách
    30.000 tỷ đồng được đầu tư vào AI thì số việc làm ròng vẫn tăng.

    Do đó:

    \[
    x_H^{{min}} = 0
    \]

    Điều này cho thấy trong bộ dữ liệu giả định, AI không thay thế lao động
    nhanh hơn tốc độ tạo việc làm mới trong ngành chế biến chế tạo.
    """)

# =========================
# TAB 4
# =========================

with tab4:

    st.markdown('### 🌊 Mô phỏng luồng dịch chuyển lao động')

    vulnerable = [
        "Nông-Lâm-Thủy sản",
        "Xây dựng",
        "Bán buôn-bán lẻ"
    ]


    df_vul = df[df["Sectors"].isin(vulnerable)].copy()

    # Giả định đầu tư tối ưu từ kết quả Tab 2
    # Nếu bạn đã có result_df thì dùng trực tiếp
    total_budget = 30000

    # Phân bổ minh họa
    # Giả sử 10% lao động bị thay thế
    df_vul["Displaced"] = (
        df_vul["Labor"] * 1_000_000 * 0.10
    )

    # 70% được đào tạo lại
    df_vul["Retrained"] = (
        df_vul["Displaced"] * 0.70
    )

    # 30% thất nghiệp tạm thời
    df_vul["Unemployed"] = (
        df_vul["Displaced"] * 0.30
    )

    st.dataframe(
        df_vul[
            [
                "Sectors",
                "Labor",
                "Displaced",
                "Retrained",
                "Unemployed"
            ]
        ].round(0),
        use_container_width=True,
        hide_index=True
    )

    # ==========================
    # Sankey
    # ==========================

    labels = [
        "Nông-Lâm-Thủy sản",
        "Xây dựng",
        "Bán buôn-bán lẻ",
        "Đào tạo lại",
        "Thất nghiệp tạm thời"
    ]

    source = []
    target = []
    value = []

    for i, row in enumerate(df_vul.itertuples()):

        source.append(i)
        target.append(3)
        value.append(row.Retrained)

        source.append(i)
        target.append(4)
        value.append(row.Unemployed)

    fig = go.Figure(
        go.Sankey(
            node=dict(
                pad=20,
                thickness=20,
                label=labels
            ),
            link=dict(
                source=source,
                target=target,
                value=value
            )
        )
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        title="Luồng dịch chuyển lao động do AI",
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    st.divider()

    st.markdown('### Lưu ý')
    st.warning("""Do dữ liệu đầu vào không cung cấp luồng dịch chuyển lao động chi tiết, mô hình minh họa giả định 10% lao động trong các ngành dễ tổn thương bị thay thế bởi tự động hóa; trong đó 70% được đào tạo lại và 30% thất nghiệp tạm thời""")
    
    st.divider()
    st.markdown("""
### 📌 Diễn giải

- Lao động phổ thông trong các ngành dễ bị tự động hóa có nguy cơ bị thay thế bởi AI.
- Một phần lao động được đào tạo lại để chuyển sang các công việc mới.
- Phần còn lại đối mặt với thất nghiệp tạm thời trong giai đoạn chuyển đổi.

Biểu đồ Sankey cho thấy luồng dịch chuyển lao động từ các ngành truyền thống sang nhóm lao động được tái đào tạo hoặc thất nghiệp ngắn hạn.
""")

# =========================
# TAB 5
# =========================

with tab5:

    st.markdown('### ⚖️ Kiểm tra ràng buộc mất việc tối đa 5%')

    # ==========================
    # Mô hình LP mới
    # ==========================

    model = LpProblem(
        "NetJob_Constraint_5Percent",
        LpMaximize
    )

    sectors = df["Sectors"].tolist()

    x_AI = LpVariable.dicts(
        "AI",
        sectors,
        lowBound=0
    )

    x_H = LpVariable.dicts(
        "H",
        sectors,
        lowBound=0
    )

    netjob_expr = []

    for _, row in df.iterrows():

        s = row["Sectors"]

        risk = row["Risk"] / 100

        netjob = (
            row["a₁"] * x_AI[s]
            + row["b₁"] * x_H[s]
            - row["c₁"] * risk * x_AI[s]
        )

        netjob_expr.append(netjob)

        # NetJob >= 0
        model += netjob >= 0

        # --------------------------
        # NEW CONSTRAINT
        # Displaced <= 5% Labor
        # --------------------------

        labor_people = row["Labor"] * 1_000_000

        model += (
            row["c₁"] * risk * x_AI[s]
            <= 0.05 * labor_people
        )

    # Budget
    model += lpSum(
        x_AI[s] + x_H[s]
        for s in sectors
    ) <= 30000

    # Objective
    model += lpSum(netjob_expr)

    model.solve()

    status = LpStatus[model.status]

    st.metric(
        "Trạng thái mô hình",
        status
    )

    # ==========================
    # Kết quả
    # ==========================

    result = []

    for _, row in df.iterrows():

        s = row["Sectors"]

        risk = row["Risk"] / 100

        displaced = (
            row["c₁"]
            * risk
            * x_AI[s].varValue
        )

        limit = (
            0.05
            * row["Labor"]
            * 1_000_000
        )

        result.append({
            "Ngành": s,
            "DisplacedJob": displaced,
            "Giới hạn 5%": limit
        })

    result_df = pd.DataFrame(result)

    st.dataframe(
        result_df.round(0),
        use_container_width=True,
        hide_index=True
    )

    st.markdown('### 📌Nhận xét')
    st.markdown("""Ràng buộc "không ngành nào mất quá 5% lao động" không làm thay đổi nghiệm tối ưu. Điều này cho thấy chiến lược tối ưu của mô hình vốn đã thiên về đào tạo lại lao động thay vì đẩy mạnh tự động hóa bằng AI. Do đó tác động thay thế lao động của AI ở nghiệm tối ưu là rất nhỏ, khiến ràng buộc mới không trở thành ràng buộc hoạt động (non-binding constraint)""")
    st.divider()
    st.markdown("### 📌 Kết luận quản trị")

    if status == "Optimal":

        st.success("""
Bài toán vẫn khả thi sau khi bổ sung ràng buộc
'không ngành nào mất quá 5% lao động'.

Nguyên nhân là nghiệm tối ưu của mô hình gần như
không đầu tư vào AI mà tập trung vào đào tạo lao động.
Do đó số lao động bị thay thế rất thấp và luôn nằm
dưới ngưỡng 5%.
""")

    else:

        st.error("""
Bài toán trở nên không khả thi sau khi thêm
ràng buộc mất việc tối đa 5%.
""")

# =========================
# TAB 6
# =========================

with tab6:

    st.header("💬 Thảo luận chính sách")

    # =====================================================
    # a
    # =====================================================

    st.subheader(
        "a) Ngành nào cần đầu tư đào tạo lại nhiều nhất theo kết quả tối ưu? Có khớp với cảm nhận thực tế ở Việt Nam không?"
    )

    st.markdown("""
Theo nghiệm tối ưu của mô hình, gần như toàn bộ ngân sách 30.000 tỷ đồng được phân bổ cho **Giáo dục - Đào tạo** thông qua biến đầu tư đào tạo lại lao động (x_H).

Nguyên nhân là ngành này có hệ số tạo việc làm từ đào tạo cao nhất trong bộ dữ liệu:

- b₁ = 55 việc làm/tỷ đồng
- d₁ = 62 việc làm đào tạo lại/tỷ đồng

Điều này khiến mỗi đồng vốn đầu tư vào đào tạo trong ngành Giáo dục tạo ra nhiều việc làm hơn so với các ngành khác.

Về mặt thực tế, kết quả này phản ánh khá đúng định hướng phát triển của Việt Nam trong bối cảnh chuyển đổi số và AI. Khi công nghệ thay đổi nhanh, việc đầu tư vào đào tạo lại lao động và nâng cấp kỹ năng được xem là giải pháp bền vững để giảm tác động tiêu cực của tự động hóa.
""")

    st.success("""
Kết quả cho thấy đào tạo nguồn nhân lực là công cụ hiệu quả nhất để duy trì việc làm ròng dương trong quá trình chuyển đổi số.
""")

    # =====================================================
    # b
    # =====================================================

    st.subheader(
        "b) Ngành Tài chính - Ngân hàng có nguy cơ thay thế 52% nhưng cũng có hệ số tạo việc làm mới rất cao. Mô hình khuyến nghị chiến lược gì cho ngành này?"
    )

    st.markdown("""
Ngành Tài chính - Ngân hàng có:

- Risk = 52% (cao nhất trong các ngành)
- a₁ = 45.8 (khả năng tạo việc làm mới từ AI rất cao)

Điều này cho thấy AI vừa là cơ hội vừa là rủi ro đối với ngành.

Các công việc mang tính lặp lại như:
- giao dịch viên,
- nhập liệu,
- xử lý hồ sơ,
- chăm sóc khách hàng cơ bản

có nguy cơ bị tự động hóa mạnh.

Ngược lại, AI lại tạo ra nhu cầu lớn đối với:

- chuyên gia dữ liệu,
- quản trị rủi ro,
- kiểm toán số,
- phân tích tín dụng bằng AI,
- kỹ sư hệ thống tài chính số.

Do đó mô hình ngụ ý rằng ngành Tài chính - Ngân hàng không nên chống lại tự động hóa mà nên kết hợp:

1. Đầu tư AI để nâng cao năng suất.
2. Đầu tư đào tạo lại lao động hiện hữu.
3. Chuyển lao động từ các vị trí dễ thay thế sang các vị trí có giá trị gia tăng cao hơn.
""")

    # =====================================================
    # c
    # =====================================================

    st.subheader(
        "c) Có nên đầu tư x_AI vào ngành Nông - Lâm - Thủy sản không, vì hệ số tạo việc làm AI thấp nhưng số lao động dịch chuyển rất lớn? Mô hình nói gì?"
    )

    st.markdown("""
Theo dữ liệu:

- a₁ = 8.5 (thấp nhất trong các ngành)
- Lao động = 13.2 triệu người (cao nhất)
- Risk = 18%

Do hiệu quả tạo việc làm từ AI thấp nên mô hình tối ưu gần như không phân bổ ngân sách AI cho ngành này.

Từ góc độ kinh tế học tối ưu, đây là kết quả hợp lý vì nguồn lực có hạn sẽ được chuyển sang các ngành tạo ra nhiều việc làm hơn trên mỗi đơn vị đầu tư.

Tuy nhiên, từ góc độ chính sách công, việc không đầu tư AI hoàn toàn cho nông nghiệp có thể chưa tối ưu.

AI trong nông nghiệp có thể đem lại nhiều lợi ích gián tiếp:

- nông nghiệp chính xác,
- dự báo thời tiết,
- tối ưu tưới tiêu,
- quản lý dịch bệnh,
- truy xuất nguồn gốc.

Những lợi ích này không được phản ánh đầy đủ trong hàm mục tiêu NetJob.
""")

    st.success("""
Mô hình tối ưu việc làm có xu hướng đánh giá thấp giá trị chiến lược dài hạn của AI trong nông nghiệp. Nhà hoạch định chính sách cần xem xét thêm các lợi ích ngoài việc làm.
""")

    # =====================================================
    # d
    # =====================================================

    st.subheader(
        "d) 'Tốc độ tự động hóa không nên vượt quá năng lực đào tạo lại' được biểu diễn bằng ràng buộc nào? Có nên bổ sung ràng buộc nào để bảo đảm an sinh xã hội không?"
    )

    st.markdown("""
Trong mô hình, nguyên tắc này được thể hiện bởi ràng buộc:

DisplacedJobᵢ ≤ RetrainingCapacityᵢ

hay:

c₁ᵢ · x_AIᵢ · riskᵢ ≤ d₁ᵢ · x_Hᵢ

Ý nghĩa của ràng buộc:

- Lao động bị thay thế bởi AI không được vượt quá số lao động có khả năng được đào tạo lại.
- Quá trình tự động hóa phải diễn ra cùng với quá trình nâng cấp kỹ năng lao động.

Để tăng tính thực tiễn, có thể bổ sung thêm các ràng buộc an sinh xã hội như:
    **(1) Giới hạn tỷ lệ mất việc tối đa theo ngành**

DisplacedJobᵢ ≤ 5% × Laborᵢ

Nhằm tránh các cú sốc lao động quá lớn trong thời gian ngắn.

**(2) Bảo đảm ngân sách đào tạo tối thiểu**

x_Hᵢ ≥ H_min

Để mọi ngành đều được hỗ trợ nâng cao kỹ năng lao động.

**(3) Ưu tiên nhóm lao động dễ bị tổn thương**

x_H(ngành 1,3,4) ≥ α × Tổng ngân sách đào tạo

Nhóm lao động phổ thông thường chịu tác động mạnh nhất từ tự động hóa.

**(4) Bảo đảm việc làm ròng tối thiểu**

NetJobᵢ ≥ β

Giúp duy trì ổn định xã hội và hạn chế thất nghiệp kéo dài.

**(5) Bổ sung quỹ hỗ trợ chuyển đổi việc làm**

TransitionSupportᵢ ≥ λ × DisplacedJobᵢ

Khoản hỗ trợ này có thể dùng cho trợ cấp thất nghiệp, đào tạo lại hoặc tư vấn nghề nghiệp cho lao động bị ảnh hưởng.

""")

    st.success("""
AI không nên được triển khai nhanh hơn khả năng đào tạo lại lao động. Đây là nguyên tắc quan trọng để bảo đảm quá trình chuyển đổi số diễn ra bao trùm và bền vững. Để bảo đảm an sinh xã hội, quá trình tự động hóa cần đi kèm các cơ chế đào tạo lại, hỗ trợ chuyển đổi nghề và giới hạn mức mất việc chấp nhận được. Tối ưu kinh tế không nên chỉ tối đa hóa việc làm ròng mà còn phải bảo đảm tính bao trùm của quá trình chuyển đổi số.
""")

st.divider()

st.caption("""
Nguồn dữ liệu:

• ILO Vietnam (2024)

• OECD AI Employment Report (2024)

""")