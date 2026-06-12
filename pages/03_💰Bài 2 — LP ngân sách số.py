import streamlit as st
import pandas as pd
import numpy as np
from scipy.optimize import linprog
import plotly.express as px
from pulp import *

# =========================
# CẤU HÌNH TRANG
# =========================
st.set_page_config(
    page_title="Bài 2 - Quy hoạch tuyến tính",
    page_icon="📊",
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
st.markdown("## 📊 Bài 2 - Phân bổ ngân sách chuyển đổi số")
st.markdown(
    "<span class='badge badge-info'>DỄ</span>"
    "<span class='badge badge-info'>LP đơn giản</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Tối ưu hóa ngân sách chuyển đổi số quốc gia năm 2026

Mục tiêu:

- Tối đa hóa GDP kỳ vọng
- Phân bổ ngân sách 100 nghìn tỷ VND
- Tuân thủ các ràng buộc chính sách của Chính phủ
""")

st.divider()

# =========================
# HÀM MỤC TIÊU
# =========================
st.markdown('### 📚 Mô hình toán học')

st.latex(
    r'''
    \max Z =
    0.85x_1 +
    1.20x_2 +
    0.95x_3 +
    1.35x_4
    '''
)

st.info("""
x₁ : Hạ tầng số

x₂ : AI và dữ liệu

x₃ : Nhân lực số

x₄ : R&D công nghệ
""")

with st.expander("📖 Giải thích ý nghĩa các hệ số"):

    st.write("""
    - 0.85 : Hiệu quả đầu tư hạ tầng số

    - 1.20 : Hiệu quả đầu tư AI

    - 0.95 : Hiệu quả đầu tư nhân lực số

    - 1.35 : Hiệu quả đầu tư R&D

    Hệ số càng lớn thì tác động tới GDP càng mạnh.
    """)

# =========================
# THIẾT LẬP MÔ HÌNH
# =========================

c = np.array([
    -0.85,
    -1.20,
    -0.95,
    -1.35
])

A = [
    [1, 1, 1, 1],
    [-1, 0, 0, 0],
    [0, -1, 0, 0],
    [0, 0, -1, 0],
    [0, 0, 0, -1],
    [0.35, -0.65, 0.35, -0.65]
]

b = [
    100,
    -25,
    -15,
    -20,
    -10,
    0
]

bounds = [
    (0, None),
    (0, None),
    (0, None),
    (0, None)
]

# =========================
# GIẢI BÀI TOÁN
# =========================

result = linprog(
    c,
    A_ub=A,
    b_ub=b,
    bounds=bounds,
    method="highs"
)

x1, x2, x3, x4 = result.x

Z = -result.fun

# =========================
# KPI
# =========================

st.markdown('### 📈 Kết quả tối ưu')

col1, col2, col3 = st.columns(3)

col1.metric(
    "GDP tối đa",
    f"{Z:.2f}"
)

col2.metric(
    "Ngân sách sử dụng",
    f"{x1+x2+x3+x4:.2f}"
)

col3.metric(
    "Trạng thái",
    "Optimal" if result.success else "Failed"
)

st.divider()

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "📋 Mô hình",
    "🎯 Kết quả tối ưu",
    "📊 Biểu đồ",
    "💰 Dual Values",
    "🚀 Phân tích độ nhạy",
    "👨‍💻 Ưu tiên nhân lực số",
    "📝 Thảo luận chính sách"
])

# =========================
# TAB 1
# =========================

with tab1:

    st.markdown('### 📋 Ràng buộc bài toán tối ưu hóa')

    # KPI
    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ngân sách",
        "100",
        "nghìn tỷ VND"
    )

    col2.metric(
        "Biến quyết định",
        "4"
    )

    col3.metric(
        "Số ràng buộc",
        "6"
    )

    st.divider()

    constraints = pd.DataFrame({
        "Ràng buộc": [
            "Ngân sách tổng",
            "Hạ tầng số tối thiểu",
            "AI tối thiểu",
            "Nhân lực số tối thiểu",
            "R&D tối thiểu",
            "Công nghệ chiến lược"
        ],
        "Điều kiện": [
            "x₁ + x₂ + x₃ + x₄ ≤ 100",
            "x₁ ≥ 25",
            "x₂ ≥ 15",
            "x₃ ≥ 20",
            "x₄ ≥ 10",
            "x₂ + x₄ ≥ 35% tổng ngân sách"
        ]
    })

    st.dataframe(
        constraints,
        use_container_width=True,
        hide_index=True
    )

    with st.expander("📖 Giải thích ý nghĩa các ràng buộc"):

        st.write("""
        • x₁: Đầu tư hạ tầng số

        • x₂: Đầu tư AI và dữ liệu

        • x₃: Đầu tư nhân lực số

        • x₄: Đầu tư R&D công nghệ

        ----------------------------

        Ràng buộc ngân sách đảm bảo tổng chi không vượt
        100 nghìn tỷ VND.

        Các ràng buộc tối thiểu phản ánh định hướng
        của Chính phủ trong phát triển chuyển đổi số.

        Điều kiện x₂ + x₄ ≥ 35% nhằm bảo đảm tỷ trọng
        đầu tư cho công nghệ chiến lược (AI và R&D).
        """)

    st.info("""
    Bài toán được mô hình hóa dưới dạng
    Quy hoạch tuyến tính (Linear Programming).

    Mục tiêu là tối đa hóa GDP kỳ vọng từ nguồn
    ngân sách đầu tư công.
    """)

# =========================
# TAB 2
# =========================

with tab2:

    st.markdown('### 🎯 Kết quả tối ưu hóa bằng scipy.optimize.linprog')

    # =========================
    # KPI
    # =========================

    col1, col2 = st.columns(2)

    col1.metric(
        "GDP tối đa (Z*)",
        f"{Z:.2f}"
    )

    budget_used = x1 + x2 + x3 + x4

    col2.metric(
        "Ngân sách sử dụng",
        f"{budget_used:.2f}"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    solution_df = pd.DataFrame({
        "Khoản mục": [
            "Hạ tầng số",
            "AI & Dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Đầu tư tối ưu": [
            round(x1, 2),
            round(x2, 2),
            round(x3, 2),
            round(x4, 2)
        ]
    })

    st.markdown('### 📋 Phân bổ ngân sách tối ưu')

    st.dataframe(
        solution_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.bar(
        solution_df,
        x="Khoản mục",
        y="Đầu tư tối ưu",
        text_auto=".2f",
        title="Phân bổ ngân sách tối ưu"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        xaxis_title="Hạng mục",
        yaxis_title="Ngân sách (nghìn tỷ VND)",
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # NHẬN XÉT TỰ ĐỘNG
    # =========================

    max_item = solution_df.loc[
        solution_df["Đầu tư tối ưu"].idxmax(),
        "Khoản mục"
    ]

    st.success(f"""
    Giá trị tối ưu:

    Z* = {Z:.2f} nghìn tỷ VND

    Khoản mục được ưu tiên đầu tư nhiều nhất là:
    {max_item}.

    Điều này phản ánh hiệu quả tạo GDP cao của hạng mục này
    trong điều kiện ngân sách hiện tại.
    """)

    # =========================
    # GIẢI THÍCH
    # =========================

    with st.expander("📖 Giải thích kết quả tối ưu"):

        st.write("""
        Mô hình sử dụng phương pháp Linear Programming.

        Solver sẽ tự động tìm tổ hợp đầu tư cho:

        • Hạ tầng số

        • AI & Dữ liệu

        • Nhân lực số

        • R&D công nghệ

        sao cho GDP kỳ vọng đạt mức cao nhất nhưng vẫn
        thỏa mãn toàn bộ các ràng buộc của đề bài.
        """)

# =========================
# TAB 3
# =========================

with tab3:

    st.markdown('### 📊 Cơ cấu phân bổ ngân sách tối ưu')

    chart_df = pd.DataFrame({
        "Khoản mục": [
            "Hạ tầng số",
            "AI & Dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Ngân sách": [
            x1,
            x2,
            x3,
            x4
        ]
    })

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Tổng ngân sách",
        f"{sum(chart_df['Ngân sách']):.0f}"
    )

    col2.metric(
        "Khoản mục lớn nhất",
        chart_df.loc[
            chart_df["Ngân sách"].idxmax(),
            "Khoản mục"
        ]
    )

    col3.metric(
        "Tỷ lệ AI + R&D",
        f"{((x2+x4)/100)*100:.1f}%"
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ TRÒN
    # =========================

    fig = px.pie(
        chart_df,
        names="Khoản mục",
        values="Ngân sách",
        hole=0.45,
        title="Cơ cấu phân bổ ngân sách tối ưu"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=550
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # BẢNG CƠ CẤU
    # =========================

    chart_df["Tỷ trọng (%)"] = (
        chart_df["Ngân sách"]
        / chart_df["Ngân sách"].sum()
        * 100
    )

    st.dataframe(
        chart_df.round(2),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # NHẬN XÉT
    # =========================

    max_item = chart_df.loc[
        chart_df["Ngân sách"].idxmax(),
        "Khoản mục"
    ]

    st.success(f"""
    Cơ cấu phân bổ ngân sách cho thấy:

    • {max_item} nhận tỷ trọng đầu tư lớn nhất.

    • AI và R&D chiếm {(x2+x4):.0f}% tổng ngân sách,
      vượt mức tối thiểu 35% theo yêu cầu đề bài.

    • Nguồn lực được tập trung vào các lĩnh vực
      có khả năng tạo giá trị gia tăng cao cho nền kinh tế.
    """)

    with st.expander("📖 Giải thích ý nghĩa cơ cấu đầu tư"):

        st.write("""
        Biểu đồ thể hiện tỷ trọng ngân sách tối ưu
        được mô hình đề xuất.

        Nếu một hạng mục nhận nhiều ngân sách hơn,
        điều đó cho thấy hiệu quả tạo GDP của hạng mục đó
        cao hơn trong điều kiện các ràng buộc hiện hành.

        Đây là cơ sở hỗ trợ ra quyết định phân bổ
        nguồn lực công cho chuyển đổi số quốc gia.
        """)

# =========================
# TAB 4 - CÂU 2.4.2
# =========================
with tab4:

    st.markdown('### 💰 Phân tích Dual Values (Shadow Prices)')

    # -------------------------
    # MÔ HÌNH PULP
    # -------------------------

    model = LpProblem(
        "Digital_Budget_Allocation",
        LpMaximize
    )

    x1_p = LpVariable("Infrastructure", lowBound=0)
    x2_p = LpVariable("AI_Data", lowBound=0)
    x3_p = LpVariable("Digital_HR", lowBound=0)
    x4_p = LpVariable("R&D", lowBound=0)

    # Hàm mục tiêu

    model += (
        0.85 * x1_p
        + 1.20 * x2_p
        + 0.95 * x3_p
        + 1.35 * x4_p
    )

    # Ràng buộc

    c_budget = (
        x1_p + x2_p + x3_p + x4_p <= 100
    )

    c_x1 = (
        x1_p >= 25
    )

    c_x2 = (
        x2_p >= 15
    )

    c_x3 = (
        x3_p >= 20
    )

    c_x4 = (
        x4_p >= 10
    )

    c_strategy = (
        x2_p + x4_p
        >=
        0.35 * (
            x1_p + x2_p + x3_p + x4_p
        )
    )

    model += c_budget, "Total_Budget"
    model += c_x1, "Infrastructure_Min"
    model += c_x2, "AI_Min"
    model += c_x3, "HR_Min"
    model += c_x4, "RD_Min"
    model += c_strategy, "Strategic_Tech"

    # -------------------------
    # GIẢI BÀI TOÁN
    # -------------------------

    model.solve(PULP_CBC_CMD(msg=False))

    # -------------------------
    # LẤY DUAL VALUE
    # -------------------------

    dual_df = pd.DataFrame({
        "Ràng buộc": [
            name
            for name in model.constraints.keys()
        ],
        "Dual Value": [
            round(
                constraint.pi,
                4
            )
            for constraint
            in model.constraints.values()
        ],
        "Slack": [
            round(
                constraint.slack,
                4
            )
            for constraint
            in model.constraints.values()
        ]
    })

    # -------------------------
    # KPI
    # -------------------------

    budget_shadow = dual_df.loc[
        dual_df["Ràng buộc"] == "Total_Budget",
        "Dual Value"
    ].values[0]

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Shadow Price ngân sách",
        f"{budget_shadow:.4f}"
    )

    col2.metric(
        "Số ràng buộc",
        len(dual_df)
    )

    col3.metric(
        "Trạng thái",
        LpStatus[model.status]
    )

    st.divider()

    # -------------------------
    # BẢNG DUAL VALUES
    # -------------------------

    st.markdown('### 📋 Bảng giá đối ngẫu')

    st.dataframe(
    dual_df,
    use_container_width=True,
    hide_index=True
)

    # -------------------------
    # BIỂU ĐỒ
    # -------------------------

    fig_dual = px.bar(
        dual_df,
        x="Ràng buộc",
        y="Dual Value",
        text_auto=".4f",
        title="Shadow Price của các ràng buộc"
    )

    fig_dual.update_layout(
        height=500
    )

    st.plotly_chart(
        fig_dual,
        use_container_width=True
    )

    # -------------------------
    # GIẢI THÍCH CHÍNH SÁCH
    # -------------------------

    st.markdown('### 📝 Ý nghĩa chính sách')

    if budget_shadow > 0:

        st.success(
            f"""
            Shadow Price của ràng buộc ngân sách
            bằng {budget_shadow:.4f}.

            Điều này có nghĩa:

            Nếu Chính phủ tăng thêm
            1 nghìn tỷ VND ngân sách,
            GDP kỳ vọng tối đa sẽ tăng thêm khoảng
            {budget_shadow:.4f} nghìn tỷ VND.

            Đây là thước đo giá trị kinh tế cận biên
            của nguồn vốn đầu tư công.
            """
        )

    else:

        st.warning(
            """
            Shadow Price bằng 0.

            Điều này cho thấy việc tăng thêm ngân sách
            hiện tại chưa tạo ra lợi ích kinh tế bổ sung
            do các ràng buộc khác đang chi phối bài toán.
            """
        )

    # -------------------------
    # NHẬN XÉT QUẢN TRỊ
    # -------------------------

    st.markdown('### 📌 Kết luận quản trị')

    st.info(
        """
        Các Dual Values cho biết mức độ quan trọng
        của từng ràng buộc đối với kết quả tối ưu.

        Những ràng buộc có Shadow Price lớn
        là các ràng buộc đang giới hạn khả năng
        gia tăng GDP của nền kinh tế.

        Vì vậy các nhà hoạch định chính sách
        nên ưu tiên nới lỏng hoặc điều chỉnh
        các ràng buộc này khi xây dựng chiến lược
        đầu tư công trong tương lai.
        """
    )

    # =========================
# TAB 5 - CÂU 2.4.3
# =========================
with tab5:

    st.markdown('### 🚀 Phân tích độ nhạy ngân sách')

    budgets = [100, 120, 140]

    results = []

    for B in budgets:

        c = np.array([
            -0.85,
            -1.20,
            -0.95,
            -1.35
        ])

        A = [
            [1, 1, 1, 1],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, -1, 0],
            [0, 0, 0, -1],
            [0.35, -0.65, 0.35, -0.65]
        ]

        b = [
            B,
            -25,
            -15,
            -20,
            -10,
            0
        ]

        sol = linprog(
            c,
            A_ub=A,
            b_ub=b,
            bounds=[(0, None)] * 4,
            method="highs"
        )

        results.append({
            "Ngân sách": B,
            "GDP tối đa": round(-sol.fun, 2)
        })

    sensitivity_df = pd.DataFrame(results)

    # =========================
    # KPI
    # =========================

    increase_total = (
        sensitivity_df.iloc[-1]["GDP tối đa"]
        -
        sensitivity_df.iloc[0]["GDP tối đa"]
    )

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Z*(100)",
        f"{sensitivity_df.iloc[0]['GDP tối đa']:.2f}"
    )

    col2.metric(
        "Z*(120)",
        f"{sensitivity_df.iloc[1]['GDP tối đa']:.2f}"
    )

    col3.metric(
        "Z*(140)",
        f"{sensitivity_df.iloc[2]['GDP tối đa']:.2f}"
    )

    col4.metric(
        "Tăng thêm",
        f"{increase_total:.2f}"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    st.markdown('### 📋 Kết quả độ nhạy')

    st.dataframe(
        sensitivity_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.line(
        sensitivity_df,
        x="Ngân sách",
        y="GDP tối đa",
        markers=True,
        title="Đường cong giá trị tối ưu Z*(B)"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=500,
        xaxis_title="Ngân sách (nghìn tỷ VND)",
        yaxis_title="GDP tối đa (nghìn tỷ VND)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # PHÂN TÍCH MỨC TĂNG
    # =========================

    growth_100_120 = (
        (
            sensitivity_df.iloc[1]["GDP tối đa"]
            -
            sensitivity_df.iloc[0]["GDP tối đa"]
        )
        /
        sensitivity_df.iloc[0]["GDP tối đa"]
    ) * 100

    growth_120_140 = (
        (
            sensitivity_df.iloc[2]["GDP tối đa"]
            -
            sensitivity_df.iloc[1]["GDP tối đa"]
        )
        /
        sensitivity_df.iloc[1]["GDP tối đa"]
    ) * 100

    st.markdown('### 📈 Nhận xét')

    st.success(f"""
    • Khi ngân sách tăng từ 100 lên 120 nghìn tỷ VND,
    GDP tối đa tăng {growth_100_120:.2f}%.

    • Khi ngân sách tăng từ 120 lên 140 nghìn tỷ VND,
    GDP tối đa tăng {growth_120_140:.2f}%.

    • Tổng mức tăng GDP tối đa khi ngân sách tăng từ
    100 lên 140 nghìn tỷ VND là {increase_total:.2f}
    nghìn tỷ VND.
    """)

    st.info("""
    Đường cong Z*(B) gần như tuyến tính.

    Điều này cho thấy mỗi đơn vị ngân sách bổ sung
    vẫn được phân bổ vào các khoản đầu tư có hiệu quả cao,
    đặc biệt là AI và R&D công nghệ.
    """)

    st.markdown('### 📌 Kết luận quản trị')

    st.success("""
    Kết quả phân tích độ nhạy cho thấy việc mở rộng
    ngân sách chuyển đổi số giúp gia tăng đáng kể
    GDP kỳ vọng.

    Trong phạm vi phân tích từ 100 đến 140 nghìn tỷ VND,
    chưa xuất hiện dấu hiệu lợi ích cận biên giảm mạnh.

    Điều này hàm ý rằng Chính phủ có thể cân nhắc
    tiếp tục tăng quy mô đầu tư số nếu điều kiện
    ngân sách cho phép.
    """)

# =========================
# TAB 6 - CÂU 2.4.4
# =========================
with tab6:

    st.markdown('### 👨‍💻 Kịch bản ưu tiên nhân lực số')

    st.info("""
    Chính phủ giả định thiếu hụt kỹ sư AI nên nâng mức đầu tư tối thiểu
    cho nhân lực số:

    x₃ ≥ 30

    thay cho:

    x₃ ≥ 20
    """)

    # =========================
    # GIẢI LẠI BÀI TOÁN
    # =========================

    c_new = np.array([
        -0.85,
        -1.20,
        -0.95,
        -1.35
    ])

    A_new = [
        [1, 1, 1, 1],
        [-1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, -1],
        [0.35, -0.65, 0.35, -0.65]
    ]

    b_new = [
        100,
        -25,
        -15,
        -30,   # thay đổi từ 20 lên 30
        -10,
        0
    ]

    result_new = linprog(
        c_new,
        A_ub=A_new,
        b_ub=b_new,
        bounds=[(0, None)] * 4,
        method="highs"
    )

    feasible = result_new.success

    x1_new, x2_new, x3_new, x4_new = result_new.x

    Z_new = -result_new.fun

    # Lấy kết quả tối ưu ban đầu từ Tab 2
    Z_old = Z

    change_abs = Z_new - Z_old

    percent_change = (
        change_abs / Z_old
    ) * 100

    # =========================
    # KPI
    # =========================

    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "Khả thi?",
        "✅ Có" if feasible else "❌ Không"
    )

    col2.metric(
        "GDP ban đầu",
        f"{Z_old:.2f}"
    )

    col3.metric(
        "GDP mới",
        f"{Z_new:.2f}"
    )

    col4.metric(
        "Chênh lệch",
        f"{change_abs:.2f}",
        f"{percent_change:.2f}%"
    )

    st.divider()

    # =========================
    # BẢNG SO SÁNH PHÂN BỔ
    # =========================

    st.markdown('### 📋 So sánh phương án phân bổ')

    compare_df = pd.DataFrame({
        "Khoản mục": [
            "Hạ tầng số",
            "AI & Dữ liệu",
            "Nhân lực số",
            "R&D công nghệ"
        ],
        "Ban đầu": [
            25,
            15,
            20,
            40
        ],
        "Kịch bản mới": [
            round(x1_new, 2),
            round(x2_new, 2),
            round(x3_new, 2),
            round(x4_new, 2)
        ]
    })

    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ GDP
    # =========================

    st.markdown('### 📊 So sánh giá trị tối ưu')

    gdp_df = pd.DataFrame({
        "Kịch bản": [
            "Ban đầu",
            "Ưu tiên nhân lực số"
        ],
        "GDP tối đa": [
            Z_old,
            Z_new
        ]
    })

    fig = px.bar(
        gdp_df,
        x="Kịch bản",
        y="GDP tối đa",
        text_auto=".2f",
        title="So sánh giá trị tối ưu giữa hai kịch bản"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=500,
        yaxis_title="GDP tối đa (nghìn tỷ VND)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # NHẬN XÉT
    # =========================

    st.markdown('### 📈 Nhận xét')

    if feasible:

        if Z_new < Z_old:

            st.warning(f"""
            Bài toán vẫn khả thi.

            GDP tối đa giảm từ {Z_old:.2f}
            xuống {Z_new:.2f}.

            Mức giảm là {abs(percent_change):.2f}%.
            """)

        elif Z_new > Z_old:

            st.success(f"""
            Bài toán vẫn khả thi.

            GDP tối đa tăng từ {Z_old:.2f}
            lên {Z_new:.2f}.

            Mức tăng là {percent_change:.2f}%.
            """)

    else:

        st.error("""
        Bài toán không còn khả thi.
        Các ràng buộc hiện tại không thể đồng thời thỏa mãn.
        """)

    st.info("""
    Nguyên nhân là do một phần ngân sách được chuyển từ
    các hạng mục có hiệu quả kinh tế cao như AI và R&D
    sang đầu tư nhân lực số.

    Điều này có thể làm giảm hiệu quả GDP ngắn hạn,
    nhưng lại giúp tăng năng lực công nghệ và nguồn nhân lực
    trong dài hạn.
    """)

    # =========================
    # KẾT LUẬN QUẢN TRỊ
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(f"""
    Việc nâng mức đầu tư tối thiểu cho nhân lực số từ
    20 lên 30 nghìn tỷ VND vẫn đảm bảo tính khả thi
    của bài toán phân bổ ngân sách.

    Tuy nhiên, do ngân sách bị dịch chuyển khỏi các
    hạng mục có hiệu quả kinh tế cao hơn như AI và R&D,
    giá trị GDP tối đa thay đổi {percent_change:.2f}%.

    Điều này phản ánh sự đánh đổi giữa:

    • Hiệu quả kinh tế ngắn hạn

    và

    • Năng lực nguồn nhân lực số dài hạn.

    Trong thực tế, Chính phủ có thể chấp nhận sự đánh đổi này
    để giải quyết tình trạng thiếu hụt kỹ sư AI và hỗ trợ
    chiến lược phát triển kinh tế số bền vững.
    """)

# =========================
# TAB 7 - THẢO LUẬN CHÍNH SÁCH
# =========================
with tab7:

    st.header("📝 Thảo luận chính sách")

    # ==================================
    # CÂU A
    # ==================================
    st.markdown('### a) Khi ngân sách tăng thêm 1 tỷ VND thì GDP kỳ vọng tăng thêm bao nhiêu?')

    shadow_price = 1.35

    st.markdown(f"""
    Shadow Price của ràng buộc ngân sách tổng bằng
    {shadow_price:.2f}.

    Điều này cho thấy nếu ngân sách đầu tư công
    tăng thêm 1 đơn vị (nghìn tỷ VND trong mô hình),
    GDP kỳ vọng tối đa sẽ tăng thêm khoảng
    {shadow_price:.2f} đơn vị.

    Đây chính là lợi ích cận biên của nguồn vốn đầu tư công.
    """)

    st.markdown("""
    Tuy nhiên đây không phải là cận trên tuyệt đối
    của chi phí cơ hội vốn công.

    Shadow Price chỉ đúng trong phạm vi mà cấu trúc
    nghiệm tối ưu chưa thay đổi.

    Khi ngân sách tăng quá lớn, các ràng buộc khác
    có thể trở nên chặt hơn và giá trị này sẽ thay đổi.
    """)

    st.divider()

    # ==================================
    # CÂU B
    # ==================================
    st.markdown('### b) Vì sao R&D có hệ số tác động cao nhất nhưng ràng buộc tối thiểu thấp nhất?')

    st.markdown("""
    Trong mô hình, R&D có hệ số tác động lớn nhất (1.35),
    nghĩa là mỗi đồng đầu tư vào nghiên cứu và phát triển
    tạo ra giá trị GDP kỳ vọng cao nhất.

    Tuy nhiên Chính phủ chỉ yêu cầu mức đầu tư tối thiểu
    tương đối thấp vì:

    • Hiệu quả R&D thường xuất hiện trong trung và dài hạn.

    • Nhiều dự án R&D có mức độ rủi ro cao.

    • Khả năng hấp thụ công nghệ của doanh nghiệp
      còn phụ thuộc vào nhân lực và hạ tầng.

    • Nếu đầu tư quá lớn khi chưa có nền tảng phù hợp
      có thể dẫn tới lãng phí nguồn lực.
    """)

    st.markdown("""
    Vì vậy Nhà nước thường ưu tiên tạo mức nền tối thiểu,
    sau đó để mô hình tối ưu quyết định mức phân bổ bổ sung.
    """)

    st.divider()

    # ==================================
    # CÂU C
    # ==================================
    st.markdown('### c) Tỷ lệ 35% cho AI + R&D có khả thi trong thực tiễn không?')

    st.write("""
    Theo mô hình, yêu cầu:

    AI + R&D ≥ 35% tổng ngân sách

    nhằm thúc đẩy các công nghệ chiến lược
    và tăng tốc chuyển đổi số quốc gia.
    """)

    st.warning("""
    Tuy nhiên trong thực tế việc duy trì tỷ lệ này
    không phải lúc nào cũng dễ dàng.
    """)

    st.markdown("""
    Một số nguyên nhân:

    • Ngân sách nhà nước còn phải ưu tiên
      hạ tầng giao thông.

    • Chi cho y tế, giáo dục và an sinh xã hội
      vẫn chiếm tỷ trọng lớn.

    • Các địa phương có mức độ sẵn sàng số khác nhau.

    • Năng lực triển khai AI và R&D chưa đồng đều.
    """)

    st.markdown("""
    Mặc dù vậy, xét trong dài hạn,
    mục tiêu 35% được đánh giá là khả thi nếu:

    • Tăng cường hợp tác công tư (PPP).

    • Thu hút đầu tư công nghệ cao từ khu vực tư nhân.

    • Ưu tiên các chương trình AI quốc gia.

    • Gắn đầu tư R&D với nhu cầu thực tế của doanh nghiệp.

    • Nâng cao chất lượng nguồn nhân lực số.
    """)

    st.divider()

    # ==================================
    # TỔNG KẾT
    # ==================================
    st.markdown('### 📌 Kết luận chung')

    st.success("""
    Kết quả mô hình cho thấy ngân sách đầu tư số
    có khả năng tạo ra tác động tích cực tới tăng trưởng GDP.

    Trong các hạng mục đầu tư, R&D và AI là những động lực
    có hiệu quả kinh tế cao nhất.

    Tuy nhiên việc ra quyết định chính sách không chỉ dựa
    trên hiệu quả kinh tế ngắn hạn mà còn cần cân nhắc:

    • Năng lực hấp thụ công nghệ.

    • Chất lượng nguồn nhân lực.

    • Tính bền vững của ngân sách nhà nước.

    • Mục tiêu phát triển dài hạn của nền kinh tế số.
    """)
    
# =========================
# NGUỒN
# =========================

st.divider()

st.caption("""
Nguồn tham khảo:

• Quyết định 749/QĐ-TTg

• Quyết định 411/QĐ-TTg

• World Bank Digital Economy Report

• OECD AI Policy Report
""")