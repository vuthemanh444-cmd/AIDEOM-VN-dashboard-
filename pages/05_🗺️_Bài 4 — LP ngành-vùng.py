import streamlit as st
import pandas as pd
import numpy as np
import pulp
import cvxpy as cp
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns 

# =====================================================
# CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="Bài 4 - LP ngân sách số theo ngành vùng",
    page_icon="🌍",
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
# =====================================================
# TIÊU ĐỀ
# =====================================================

st.markdown("## 🌍 Bài 4 — Quy hoạch tuyến tính phân bổ ngân sách số theo ngành - vùng")
st.markdown(
    "<span class='badge badge-mod'>TRUNG BÌNH</span>"
    "<span class='badge badge-info'>LP đầy đủ</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Mục tiêu

Tối ưu hóa phân bổ ngân sách số quốc gia giữa:

- 6 vùng kinh tế - xã hội
- 4 nhóm đầu tư số

gồm:

- I: Hạ tầng số
- D: Chuyển đổi số doanh nghiệp
- AI: Năng lực AI
- H: Nhân lực số

nhằm tối đa hóa GDP gain đồng thời đảm bảo công bằng vùng miền.
""")

st.divider()

# =====================================================
# ĐỌC DỮ LIỆU
# =====================================================

beta_df = pd.read_csv(
    "data/marginal_impact_coefficient.csv"
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "🎯 PuLP Optimization",
    "⚖️ So sánh CVXPY",
    "🔥 Heatmap phân bổ",
    "📊 Chi phí công bằng",
    "📝 Thảo luận chính sách"
])
# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.markdown('### 📊 Dữ liệu đầu vào')

    st.markdown("""
Bài toán sử dụng bộ dữ liệu về hệ số tác động biên của các khoản đầu tư số
đối với từng vùng kinh tế - xã hội của Việt Nam.

Mỗi hệ số phản ánh mức độ đóng góp của 1 tỷ đồng đầu tư vào GDP gain
của vùng tương ứng.

Các nhóm đầu tư gồm:

- I : Hạ tầng số
- D : Chuyển đổi số doanh nghiệp
- AI : Năng lực AI
- H : Nhân lực số
""")

    st.divider()

    # ==========================================
    # DỮ LIỆU GỐC
    # ==========================================

    st.markdown(
        "### Bảng hệ số tác động biên (β)"
    )

    st.dataframe(
        beta_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==========================================
    # THÔNG TIN MÔ HÌNH
    # ==========================================

    st.markdown(
        "### Thông số bài toán"
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Ngân sách tổng",
            "50.000"
        )

    with col2:
        st.metric(
            "Số vùng",
            "6"
        )

    with col3:
        st.metric(
            "Nhóm đầu tư",
            "4"
        )

    st.divider()

    # ==========================================
    # RÀNG BUỘC
    # ==========================================

    st.markdown("""
### Các ràng buộc chính

- Tổng ngân sách ≤ 50.000 tỷ đồng

- Mỗi vùng nhận tối thiểu 5.000 tỷ đồng

- Mỗi vùng nhận tối đa 12.000 tỷ đồng

- Tổng đầu tư cho nhân lực số ≥ 12.000 tỷ đồng

- Chỉ số số hóa của mọi vùng phải đạt ít nhất 70% vùng dẫn đầu
""")

    st.warning("""
Ràng buộc công bằng vùng miền là yếu tố quan trọng nhất của bài toán.

Đây cũng là ràng buộc có thể làm giảm GDP gain nhưng giúp hạn chế
khoảng cách phát triển số giữa các vùng.
""")

# =====================================================
# TAB 2
# ====================================================
with tab2:

    st.markdown('### 🎯 Giải bài toán bằng PuLP')

    st.markdown("""
Mục tiêu:

- Xây dựng mô hình quy hoạch tuyến tính bằng PuLP.
- Giải bằng solver CBC mặc định.
- Tối đa hóa GDP gain.
- Tìm ma trận phân bổ ngân sách tối ưu giữa:
    - 6 vùng kinh tế
    - 4 nhóm đầu tư số
""")

    st.divider()

    # ==================================================
    # DỮ LIỆU
    # ==================================================

    regions = beta_df["Region"].tolist()

    categories = [
        "I",
        "D",
        "AI",
        "H"
    ]

    # ==================================================
    # HỆ SỐ TÁC ĐỘNG BIÊN
    # ==================================================

    beta = {}

    for _, row in beta_df.iterrows():

        r = row["Region"]

        beta[(r, "I")] = float(row["I"])
        beta[(r, "D")] = float(row["D"])
        beta[(r, "AI")] = float(row["AI"])
        beta[(r, "H")] = float(row["H"])

    # ==================================================
    # DIGITAL INDEX BAN ĐẦU
    # ==================================================

    D0 = {

        "Trung du miền núi phía Bắc": 38,

        "Đồng bằng sông Hồng": 78,

        "Bắc Trung Bộ + DH Trung Bộ": 55,

        "Tây Nguyên": 32,

        "Đông Nam Bộ": 82,

        "Đồng bằng sông Cửu Long": 48

    }

    gamma = 0.002

    lambda_fair = 0.7

    # ==================================================
    # MÔ HÌNH PULP
    # ==================================================

    model = pulp.LpProblem(
        "Digital_Budget_Allocation",
        pulp.LpMaximize
    )

    # ==================================================
    # BIẾN QUYẾT ĐỊNH
    # ==================================================

    x = pulp.LpVariable.dicts(
        "x",
        (regions, categories),
        lowBound=0
    )

    # Biến hỗ trợ linear hóa

    M = pulp.LpVariable(
        "Dmax",
        lowBound=0
    )

    # ==================================================
    # HÀM MỤC TIÊU
    # ==================================================

    model += pulp.lpSum(

        beta[(r, j)]
        *
        x[r][j]

        for r in regions
        for j in categories

    )

    # ==================================================
    # C1: NGÂN SÁCH TỔNG
    # ==================================================

    model += pulp.lpSum(

        x[r][j]

        for r in regions
        for j in categories

    ) <= 50000

    # ==================================================
    # C2 + C3
    # ==================================================

    for r in regions:

        model += pulp.lpSum(
            x[r][j]
            for j in categories
        ) >= 5000

        model += pulp.lpSum(
            x[r][j]
            for j in categories
        ) <= 12000

    # ==================================================
    # C4
    # ==================================================

    model += pulp.lpSum(
        x[r]["H"]
        for r in regions
    ) >= 12000

    # ==================================================
    # C5
    # ==================================================

    for r in regions:

        model += (

            D0[r]
            +
            gamma * x[r]["D"]

            >=

            lambda_fair * M

        )

        model += (

            D0[r]
            +
            gamma * x[r]["D"]

            <=

            M

        )

    # ==================================================
    # SOLVE
    # ==================================================

        # ==================================================
    # SOLVE
    # ==================================================

    model.solve(
        pulp.PULP_CBC_CMD(
            msg=False
        )
    )

    pulp_status = pulp.LpStatus[
        model.status
    ]

    st.metric(
        "Solver Status",
        pulp_status
    )

    st.divider()

    # ==================================================
    # INFEASIBLE
    # ==================================================

    if pulp_status != "Optimal":
        pulp_objective = None
        pulp_solution_df = None
    

        st.error(f"""
Mô hình không có nghiệm khả thi.

Solver Status: {pulp_status}
""")

        st.markdown(
    "### Phân tích nguyên nhân vô nghiệm"
)

    st.markdown("""
**Bước 1. Xác định vùng có chỉ số số hóa cao nhất**

Từ dữ liệu ban đầu:

- Đông Nam Bộ: 82 điểm
- Tây Nguyên: 32 điểm

Do đó chỉ số số hóa lớn nhất là:
""")

    st.latex(
    r"M = 82"
)

    st.markdown("""
---

**Bước 2. Áp dụng ràng buộc công bằng vùng miền (C5)**

Theo đề bài:
""")

    st.latex(
    r"D_r + 0.002x_{D,r} \ge 0.7M"
)

    st.markdown("""
Thay giá trị M = 82:
""")

    st.latex(
    r"0.7 \times 82 = 57.4"
)

    st.markdown("""
Điều này có nghĩa là mọi vùng phải đạt tối thiểu 57.4 điểm số hóa sau đầu tư.

---

**Bước 3. Kiểm tra vùng yếu nhất (Tây Nguyên)**

Tây Nguyên có:
""")

    st.latex(
    r"D_4 = 32"
)

    st.markdown("""
Thay vào ràng buộc:
""")

    st.latex(
    r"32 + 0.002x_{D,4} \ge 57.4"
)

    st.markdown("""
Suy ra:
""")

    st.latex(
    r"x_{D,4} \ge 12,700"
)

    st.markdown("""
---

**Bước 4. So sánh với giới hạn ngân sách vùng**
""")

    st.latex(
    r"\sum_j x_{4,j} \le 12,000"
)

    st.markdown("""
Ngay cả khi toàn bộ ngân sách của Tây Nguyên được dành cho hạng mục D
(Chuyển đổi số), vùng này cũng chỉ nhận tối đa:
""")

    st.latex(
    r"x_{D,4} \le 12,000"
)

    st.markdown("""
Trong khi điều kiện công bằng yêu cầu:
""")

    st.latex(
    r"x_{D,4} \ge 12,700"
)

    st.error("""
12.700 > 12.000

⇒ Không tồn tại phương án phân bổ ngân sách nào có thể đồng thời thỏa mãn
ràng buộc ngân sách vùng (C3) và ràng buộc công bằng vùng miền (C5).
""")
    st.markdown('### 📌 Kết luận')
    st.error(""" 
Mô hình PuLP được xây dựng đúng nhưng tập ràng buộc hiện tại là không khả thi
(Infeasible). Nguyên nhân chính đến từ ràng buộc công bằng vùng miền (C5)
quá chặt so với giới hạn ngân sách tối đa của mỗi vùng.
""")
        
# =====================================================
# TABS 3
# =====================================================

with tab3:

    st.markdown('### ⚙️ Giải bằng CVXPY và so sánh với PuLP')

    st.markdown("""
Mục tiêu:

- Cài đặt lại mô hình bằng CVXPY.
- Kiểm tra tính nhất quán giữa hai công cụ tối ưu hóa.
- Đánh giá liệu kết quả vô nghiệm đến từ mô hình hay do solver.
""")

    st.divider()

    # ==================================================
    # CVXPY VARIABLES
    # ==================================================

    n_region = len(regions)
    n_category = len(categories)

    region_idx = {
        r: i
        for i, r in enumerate(regions)
    }

    category_idx = {
        c: j
        for j, c in enumerate(categories)
    }

    X = cp.Variable(
        (n_region, n_category),
        nonneg=True
    )

    M = cp.Variable(
        nonneg=True
    )

    # ==================================================
    # OBJECTIVE
    # ==================================================

    beta_matrix = np.array([

        [
            beta[(r, c)]
            for c in categories
        ]

        for r in regions

    ])

    objective = cp.Maximize(

        cp.sum(
            cp.multiply(
                beta_matrix,
                X
            )
        )

    )

    # ==================================================
    # CONSTRAINTS
    # ==================================================

    constraints = []

    # C1

    constraints.append(

        cp.sum(X)
        <= 50000

    )

    # C2 + C3

    for r in range(n_region):

        constraints.append(

            cp.sum(X[r, :])
            >= 5000

        )

        constraints.append(

            cp.sum(X[r, :])
            <= 12000

        )

    # C4

    H_idx = category_idx["H"]

    constraints.append(

        cp.sum(
            X[:, H_idx]
        )
        >= 12000

    )

    # C5

    D_idx = category_idx["D"]

    for r_name in regions:

        r = region_idx[r_name]

        constraints.append(

            D0[r_name]
            +
            gamma * X[r, D_idx]

            >=

            lambda_fair * M

        )

        constraints.append(

            D0[r_name]
            +
            gamma * X[r, D_idx]

            <=

            M

        )

    # ==================================================
    # SOLVE
    # ==================================================

    problem = cp.Problem(
        objective,
        constraints
    )

    problem.solve()

    cvxpy_status = problem.status

    st.markdown(
        "### Kết quả CVXPY"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "PuLP Status",
            pulp_status
        )

    with col2:

        st.metric(
            "CVXPY Status",
            cvxpy_status
        )

    st.divider()

    # ==================================================
    # SO SÁNH
    # ==================================================

    comparison_df = pd.DataFrame({

        "Phương pháp": [
            "PuLP + CBC",
            "CVXPY"
        ],

        "Trạng thái": [
            pulp_status,
            cvxpy_status
        ]

    })

    st.markdown(
        "### So sánh kết quả"
    )

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==================================================
    # NHẬN XÉT
    # ==================================================

    if (
        pulp_status != "Optimal"
        and
        cvxpy_status != "optimal"
    ):
        st.markdown('### 📌 Kết luận') 
        st.success("""
PuLP và CVXPY đều cho cùng kết quả:

→ Mô hình không có nghiệm khả thi.

Điều này chứng tỏ nguyên nhân xuất phát từ chính cấu trúc mô hình
và dữ liệu đầu vào, không phải do thuật toán hay solver.
""")

        st.warning("""
Ràng buộc công bằng vùng miền (C5) là nguyên nhân chính dẫn tới
tình trạng vô nghiệm.

Nội dung này sẽ tiếp tục được phân tích ở các phần sau.
""")

    else:

        st.success("""
Hai phương pháp cho kết quả nhất quán.
Điều này xác nhận tính đúng đắn của mô hình tối ưu hóa.
""")

# =====================================================
# TABS 4
# =====================================================

with tab4:

    st.markdown('### 🔥 Heatmap phân bổ tối ưu')

    st.markdown("""
Mục tiêu:

- Trực quan hóa phương án phân bổ ngân sách tối ưu.
- Xác định vùng nhận nhiều ngân sách nhất.
- Xác định hạng mục được ưu tiên ở từng vùng.
""")

    st.divider()

    # ==================================================
    # KIỂM TRA NGHIỆM
    # ==================================================

    if pulp_status != "Optimal":

        st.error("""
Không thể tạo heatmap vì mô hình không có nghiệm khả thi.
""")

        st.markdown("""
Do mô hình ở Câu 4.4.1 vô nghiệm (Infeasible),
không tồn tại ma trận phân bổ ngân sách tối ưu.

Vì vậy:

- Không xác định được vùng nhận nhiều ngân sách nhất.
- Không xác định được hạng mục ưu tiên.
- Không thể trực quan hóa bằng heatmap.
""")

        st.warning("""
Heatmap chỉ có thể được tạo sau khi mô hình trở nên khả thi,
ví dụ khi điều chỉnh hoặc loại bỏ ràng buộc công bằng vùng miền (C5).
""")

    else:

        # ==================================================
        # HEATMAP
        # ==================================================

        fig, ax = plt.subplots(
            figsize=(8, 5)
        )

        sns.heatmap(

            pulp_solution_df,

            annot=True,

            fmt=".0f",

            cmap="YlGnBu",

            linewidths=0.5,

            ax=ax

        )

        ax.set_title(
            "Phân bổ ngân sách tối ưu"
        )

        st.pyplot(fig)

        st.divider()

        # ==================================================
        # PHÂN TÍCH
        # ==================================================

        region_total = pulp_solution_df.sum(
            axis=1
        )

        category_max = pulp_solution_df.idxmax(
            axis=1
        )

        largest_region = region_total.idxmax()

        st.markdown(
            "### Phân tích kết quả"
        )

        st.success(f"""
Vùng nhận nhiều ngân sách nhất:

**{largest_region}**
""")

        summary_df = pd.DataFrame({

            "Region": pulp_solution_df.index,

            "Hạng mục ưu tiên": category_max.values

        })

        st.markdown(
            "### Hạng mục được ưu tiên tại từng vùng"
        )

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True
        )

# =====================================================
# TABS 5
# =====================================================
with tab5:

    st.markdown('### 📊 Đánh giá chi phí kinh tế của công bằng vùng miền')

    st.markdown("""
Mục tiêu:

- Loại bỏ ràng buộc công bằng vùng miền (C5).
- Giải lại mô hình tối ưu.
- So sánh với mô hình ban đầu.
- Đánh giá chi phí kinh tế của công bằng vùng miền.
""")

    st.divider()

    # ==================================================
    # MÔ HÌNH KHÔNG CÓ C5
    # ==================================================

    nofair_model = pulp.LpProblem(
        "No_Fairness_Model",
        pulp.LpMaximize
    )

    x_nf = pulp.LpVariable.dicts(
        "x",
        (regions, categories),
        lowBound=0
    )

    # --------------------------------------------------
    # OBJECTIVE
    # --------------------------------------------------

    nofair_model += pulp.lpSum(

        beta[(r, j)]
        *
        x_nf[r][j]

        for r in regions
        for j in categories

    )

    # --------------------------------------------------
    # C1
    # --------------------------------------------------

    nofair_model += pulp.lpSum(

        x_nf[r][j]

        for r in regions
        for j in categories

    ) <= 50000

    # --------------------------------------------------
    # C2 + C3
    # --------------------------------------------------

    for r in regions:

        nofair_model += pulp.lpSum(
            x_nf[r][j]
            for j in categories
        ) >= 5000

        nofair_model += pulp.lpSum(
            x_nf[r][j]
            for j in categories
        ) <= 12000

    # --------------------------------------------------
    # C4
    # --------------------------------------------------

    nofair_model += pulp.lpSum(

        x_nf[r]["H"]

        for r in regions

    ) >= 12000

    # ==================================================
    # SOLVE
    # ==================================================

    nofair_model.solve(
        pulp.PULP_CBC_CMD(
            msg=False
        )
    )

    nofair_status = pulp.LpStatus[
        nofair_model.status
    ]

    st.metric(
        "Solver Status",
        nofair_status
    )

    st.divider()

    # ==================================================
    # OPTIMAL
    # ==================================================

    if nofair_status == "Optimal":

        Z_nofair = pulp.value(
            nofair_model.objective
        )

        allocation_nf = pd.DataFrame(
            index=regions,
            columns=categories
        )

        for r in regions:

            for j in categories:

                allocation_nf.loc[r, j] = (
                    x_nf[r][j].varValue
                )

        allocation_nf = allocation_nf.apply(
            pd.to_numeric
        )

        st.metric(
            "Objective Value",
            f"{Z_nofair:,.2f}"
        )

        st.markdown(
            "### Ma trận phân bổ tối ưu (không có C5)"
        )

        st.dataframe(
            allocation_nf.style.format("{:,.2f}"),
            use_container_width=True
        )

        st.divider()

        # ==============================================
        # COST OF FAIRNESS
        # ==============================================

        st.markdown(
            "### Chi phí kinh tế của công bằng vùng miền"
        )

        if pulp_status == "Optimal":

            cost_fairness = (
                Z_nofair
                -
                pulp_objective
            )

            result_df = pd.DataFrame({

                "Mô hình": [
                    "Có C5",
                    "Không C5"
                ],

                "Objective": [
                    pulp_objective,
                    Z_nofair
                ]

            })

            st.dataframe(
                result_df.style.format(
                    "{:,.2f}"
                ),
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Cost of Fairness",
                f"{cost_fairness:,.2f}"
            )

        else:
            
            st.markdown("""
Mô hình có ràng buộc công bằng vùng miền (C5) vô nghiệm.

Do đó không tồn tại giá trị tối ưu Z_Fairness để tính trực tiếp:

Cost of Fairness = Z_NoFair − Z_Fairness
""")

            st.markdown("""
Trong trường hợp này có thể kết luận:

- Ràng buộc công bằng vùng miền quá chặt.
- Chi phí của công bằng không chỉ làm giảm GDP gain.
- Chi phí thực tế còn lớn hơn vì khiến toàn bộ mô hình mất tính khả thi.
""")

        st.divider()
        st.markdown('### 📌 Kết luận')
        st.success("""
Khi bỏ ràng buộc công bằng vùng miền, mô hình tìm được nghiệm khả thi
và đạt mức GDP gain cao hơn.

Điều này cho thấy việc theo đuổi công bằng tuyệt đối giữa các vùng
có thể làm giảm hiệu quả kinh tế hoặc thậm chí khiến bài toán
không còn khả thi.
""")
        
# =====================================================
# TABS 6
# =====================================================

with tab6:
    st.header("📝 Thảo luận chính sách")
    st.markdown('### a) Nếu bỏ ràng buộc công bằng, vốn sẽ chảy về vùng nào? Tại sao? Hậu quả xã hội dài hạn ra sao?')

    st.markdown("""
Kết quả mô hình không có ràng buộc công bằng (C5) cho thấy ngân sách có xu hướng
được phân bổ nhiều hơn cho các vùng có hệ số tác động biên cao nhất.

Về mặt kinh tế, điều này là hợp lý vì mô hình tối đa hóa GDP gain sẽ ưu tiên
những nơi có khả năng tạo ra lợi ích kinh tế lớn nhất trên mỗi đơn vị vốn đầu tư.

Các vùng như Đông Nam Bộ hoặc Đồng bằng sông Hồng thường có lợi thế về:

- Hạ tầng số đã phát triển.
- Mật độ doanh nghiệp cao.
- Khả năng hấp thụ công nghệ tốt.
- Nhu cầu chuyển đổi số lớn.

Do đó vốn đầu tư có xu hướng tập trung vào các vùng này thay vì các vùng còn khó khăn.

Tuy nhiên, nếu kéo dài trong nhiều năm, sự tập trung vốn quá mức có thể làm gia tăng
chênh lệch phát triển số giữa các vùng.

Những vùng có trình độ số hóa thấp sẽ tiếp tục tụt lại phía sau, dẫn đến:

- Khoảng cách năng suất lao động gia tăng.
- Chênh lệch cơ hội tiếp cận công nghệ.
- Bất bình đẳng về thu nhập và chất lượng dịch vụ công.
- Khó đạt mục tiêu phát triển cân bằng giữa các vùng lãnh thổ.
""")

    st.warning("""
Từ góc độ chính sách công, tối đa hóa hiệu quả kinh tế không phải lúc nào cũng đồng nghĩa
với tối đa hóa phúc lợi xã hội. Vì vậy cần có các cơ chế hỗ trợ vùng khó khăn nhằm hạn chế
sự phân hóa phát triển số trong dài hạn.
""")

    st.divider()

    st.markdown('### b) Ràng buộc trần ngân sách mỗi vùng (C3) có thể coi như một chính sách phân quyền. Nó làm giảm Z* bao nhiêu phần trăm? Mức giảm này có chấp nhận được không?')

    st.markdown("""
Ràng buộc C3 giới hạn tổng ngân sách của mỗi vùng ở mức tối đa 12.000 tỷ đồng.

Về bản chất, đây có thể được xem như một cơ chế phân quyền hoặc phân bổ cân bằng nguồn lực,
nhằm tránh việc toàn bộ ngân sách tập trung vào một số ít vùng có hiệu quả đầu tư cao.

Để đánh giá tác động của C3, cần so sánh:

- Giá trị tối ưu khi có C3.
- Giá trị tối ưu khi bỏ C3.

Tỷ lệ suy giảm hiệu quả kinh tế được tính theo công thức:
""")

    st.latex(
        r"""
        \%\Delta Z
        =
        \frac{Z_{NoC3}-Z_{C3}}
        {Z_{NoC3}}
        \times 100
        """
    )

    st.markdown("""
Nếu tỷ lệ giảm chỉ ở mức nhỏ, có thể coi đây là chi phí hợp lý để đổi lấy mục tiêu
phát triển cân bằng giữa các vùng.

Ngược lại, nếu Z* giảm quá mạnh thì cần cân nhắc nới lỏng giới hạn ngân sách,
vì khi đó chi phí cơ hội của chính sách phân quyền trở nên quá lớn.
""")

    st.warning("""
C3 phản ánh sự đánh đổi giữa hiệu quả kinh tế và công bằng phân bổ nguồn lực.
Đây là một vấn đề thường gặp trong hoạch định chính sách phát triển vùng.
""")

    st.divider()

    st.markdown('### c) Vùng Tây Nguyên có sàn 5.000 tỷ nhưng hệ số AI rất thấp (0,45). Nên đầu tư vào AI tại Tây Nguyên hay tập trung H và I trước? Mô hình trả lời như thế nào?')

    st.markdown("""
Trong dữ liệu đầu vào, hệ số tác động biên của AI tại Tây Nguyên chỉ bằng 0,45,
thấp nhất trong tất cả các vùng.

Điều này có nghĩa là mỗi tỷ đồng đầu tư vào AI tại Tây Nguyên tạo ra mức GDP gain
thấp hơn đáng kể so với các hạng mục khác.

Ngược lại:

- H (vốn nhân lực) có hệ số 1,35.
- I (hạ tầng số) có hệ số 1,20.

Do đó xét riêng về hiệu quả kinh tế ngắn hạn, mô hình có xu hướng ưu tiên:

1. Hạ tầng số (I).
2. Vốn nhân lực số (H).
3. Sau đó mới đến AI.

Kết quả này phản ánh logic phát triển công nghệ:

- Hạ tầng là điều kiện cần.
- Nhân lực là điều kiện đủ.
- AI chỉ phát huy hiệu quả khi hai yếu tố trên đã được hình thành.

Vì vậy đối với Tây Nguyên, chiến lược hợp lý hơn là đầu tư trước vào hạ tầng số
và đào tạo nhân lực số, sau đó mới mở rộng đầu tư AI ở giai đoạn tiếp theo.
""")

    st.success("""
Mô hình cho thấy đầu tư AI không nên được thực hiện đồng đều giữa các vùng.

Đối với các vùng có nền tảng số còn yếu như Tây Nguyên, việc ưu tiên hạ tầng và
nhân lực trước khi đầu tư mạnh vào AI sẽ mang lại hiệu quả kinh tế cao hơn.
""")

st.divider()

st.caption("""
Nguồn dữ liệu:

• Quyết định 411/QĐ-TTg về Chiến lược quốc gia phát triển kinh tế số và xã hội số
""")