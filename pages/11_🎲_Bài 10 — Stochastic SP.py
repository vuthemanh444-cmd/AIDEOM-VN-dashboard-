import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from pyomo.environ import *

# =====================================================
# CẤU HÌNH TRANG
# =====================================================

st.set_page_config(
    page_title="Bài 10 - Quy hoạch ngẫu nhiên hai giai đoạn",
    page_icon="🎲",
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

st.markdown("## 🎲 Bài 10. Quy hoạch ngẫu nhiên hai giai đoạn (Two-Stage Stochastic Programming)")
st.markdown(
    "<span class='badge badge-rl'>KHÓ</span>"
    "<span class='badge badge-info'>Stochastic · SP</span>",
    unsafe_allow_html=True
)
st.divider()
st.markdown("""
**Mục tiêu:** Xây dựng mô hình quy hoạch ngẫu nhiên hai giai đoạn cho bài toán phân bổ ngân sách đầu tư số của Việt Nam trong điều kiện bất định về tăng trưởng toàn cầu, FDI và xuất khẩu.

- Giai đoạn 1: Quyết định ngân sách ban đầu (Here-and-Now)
- Giai đoạn 2: Điều chỉnh theo từng kịch bản (Recourse Decision)
- Phương pháp: Two-Stage Stochastic Programming
- Công cụ: Pyomo + GLPK/CBC
""")

st.divider()

# =====================================================
# ĐỌC DỮ LIỆU
# =====================================================

scenario_df = pd.read_csv(
    "data/Scenario_Tree.csv"
)

beta_df = pd.read_csv(
    "data/β^s_j according to the scenario.csv"
)

# =====================================================
# TABS
# =====================================================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([

    "📋 Dữ liệu",
    "🎯 Mô hình Stochastic Programming",
    "📈 So sánh EV và SP",
    "💰 VSS và EVPI",
    "🛡️ Robust Optimization",
    "📝 Thảo luận chính sách"
])

# =====================================================
# TAB 1
# =====================================================
with tab1:

    st.markdown('### Dữ liệu đầu vào của mô hình')

    st.markdown("""
Mô hình sử dụng hai nhóm dữ liệu:

- Cây kịch bản kinh tế (Scenario Tree)
- Hệ số hiệu quả đầu tư theo từng kịch bản
""")

    st.divider()

    # ==========================================
    # SCENARIO TREE
    # ==========================================

    st.markdown(
        "### Bảng 1. Cây kịch bản kinh tế"
    )

    st.dataframe(
        scenario_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
Ý nghĩa các biến:

- GrowthTG: Tăng trưởng kinh tế thế giới (%)
- FDI VN: Dòng vốn FDI vào Việt Nam (tỷ USD)
- Export VN: Tăng trưởng xuất khẩu (%)
- Probability: Xác suất xảy ra của kịch bản
""")

    st.divider()

    # ==========================================
    # BETA COEFFICIENT
    # ==========================================

    st.markdown(
        "### Bảng 2. Hệ số hiệu quả đầu tư theo kịch bản"
    )

    st.dataframe(
        beta_df,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("""
Ý nghĩa:

- β(cơ bản): hiệu quả đầu tư ở giai đoạn 1.
- βᵢˢ: hiệu quả đầu tư trong từng kịch bản ở giai đoạn 2.
- Hệ số của H (nhân lực) tăng trong kịch bản khủng hoảng vì lao động được đào tạo có khả năng thích ứng tốt hơn trước các cú sốc kinh tế.
""")

    st.divider()

    # ==========================================
    # THÔNG TIN MÔ HÌNH
    # ==========================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Số kịch bản",
            len(scenario_df)
        )

    with col2:

        st.metric(
            "Danh mục đầu tư",
            4
        )

    with col3:

        st.metric(
            "Ngân sách tổng",
            "80.000"
        )

    st.divider()

    st.success("""
Mô hình gồm 4 kịch bản kinh tế và 4 hạng mục đầu tư
(I, D, AI, H). Chính phủ phân bổ tối đa 65.000 tỷ đồng
ở giai đoạn đầu và giữ lại 15.000 tỷ đồng dự phòng để
điều chỉnh theo các kịch bản xảy ra trong tương lai.
""")
    
# =====================================================
# TAB 2
# =====================================================

with tab2:

    st.markdown('### 🎯 Mô hình Stochastic Programming hai giai đoạn bằng Pyomo')
    st.markdown("""

### Cấu trúc mô hình

**First-stage (Here-and-Now Decisions)**

Chính phủ quyết định phân bổ ngân sách ban đầu:

* I: Hạ tầng số
* D: Chuyển đổi số
* AI: Trí tuệ nhân tạo
* H: Vốn nhân lực


**Second-stage (Recourse Decisions)**

Sau khi kịch bản xảy ra, Chính phủ được phép điều chỉnh ngân sách:

* y(I,s)
* y(D,s)
* y(AI,s)
* y(H,s)

cho từng trạng thái kinh tế.
""")
    st.divider()
    st.markdown("""
### Ràng buộc First-stage

Tổng ngân sách phân bổ ban đầu không vượt quá 65.000 tỷ đồng:
""")

    st.latex(
    r"\sum_{j} x_j \le 65\,000"
)

    st.markdown("""
### Ràng buộc Second-stage

Ngân sách điều chỉnh trong mỗi kịch bản không vượt quá 15.000 tỷ đồng:
""")

    st.latex(
    r"\sum_{j} y_j^{s} \le 15\,000,\quad \forall s \in S"
)

    st.markdown("""
Khả năng mở rộng đầu tư AI phụ thuộc vào mức đầu tư vốn nhân lực ở giai đoạn đầu:
""")

    st.latex(
    r"y_{AI}^{s} \le 0.5\,x_H,\quad \forall s \in S"
)

    st.divider()

    # ==================================================
    # CHUẨN BỊ DỮ LIỆU
    # ==================================================

    scenarios = scenario_df["Scenario"].tolist()

    categories = [
        "I",
        "D",
        "AI",
        "H"
    ]

    # ==================================================
    # PARAMETER
    # ==================================================

    probability = dict(
        zip(
            scenario_df["Scenario"],
            scenario_df["Probability"]
        )
    )

    beta_base = {
        "I": 1.00,
        "D": 1.10,
        "AI": 1.25,
        "H": 0.95
    }

    beta_s = {}

    for s in scenarios:

        beta_s[(s, "I")] = beta_df.loc[
            beta_df["Category"].str.contains("I"),
            s
        ].values[0]

        beta_s[(s, "D")] = beta_df.loc[
            beta_df["Category"].str.contains("D"),
            s
        ].values[0]

        beta_s[(s, "AI")] = beta_df.loc[
            beta_df["Category"] == "AI",
            s
        ].values[0]

        beta_s[(s, "H")] = beta_df.loc[
            beta_df["Category"].str.contains("H"),
            s
        ].values[0]

    # ==================================================
    # PYOMO MODEL
    # ==================================================

    model = ConcreteModel()

    # -------------------------
    # SET
    # -------------------------

    model.J = Set(
        initialize=categories
    )

    model.S = Set(
        initialize=scenarios
    )

    # -------------------------
    # PARAM
    # -------------------------

    model.beta = Param(
        model.J,
        initialize=beta_base
    )

    model.p = Param(
        model.S,
        initialize=probability
    )

    model.beta_s = Param(
        model.S,
        model.J,
        initialize=beta_s
    )

    # -------------------------
    # FIRST-STAGE VARIABLES
    # -------------------------

    model.x = Var(
        model.J,
        domain=NonNegativeReals
    )

    # -------------------------
    # SECOND-STAGE VARIABLES
    # -------------------------

    model.y = Var(
        model.S,
        model.J,
        domain=NonNegativeReals
    )

    # ==================================================
    # OBJECTIVE FUNCTION
    # ==================================================

    def objective_rule(m):

        first_stage = sum(
            m.beta[j] * m.x[j]
            for j in m.J
        )

        second_stage = sum(

            m.p[s]
            *
            sum(
                m.beta_s[s, j]
                *
                m.y[s, j]
                for j in m.J
            )

            for s in m.S
        )

        return first_stage + second_stage

    model.objective = Objective(
        rule=objective_rule,
        sense=maximize
    )

    # ==================================================
    # CONSTRAINTS
    # ==================================================

    model.first_budget = Constraint(

        expr=sum(
            model.x[j]
            for j in model.J
        )
        <= 65000

    )

    def second_budget_rule(m, s):

        return (

            sum(
                m.y[s, j]
                for j in m.J
            )

            <= 15000

        )

    model.second_budget = Constraint(
        model.S,
        rule=second_budget_rule
    )

    def ai_expansion_rule(m, s):

        return (

            m.y[s, "AI"]

            <=

            0.5 * m.x["H"]

        )

    model.ai_constraint = Constraint(
        model.S,
        rule=ai_expansion_rule
    )

    # ==================================================
    # SOLVER
    # ==================================================

    solver = SolverFactory('cbc')

    result = solver.solve(
        model,
        tee=False
    )

    # ==================================================
    # LƯU KẾT QUẢ CHO CÁC TAB SAU
    # ==================================================

    SP_objective = value(
        model.objective
    )

    SP_first_stage = {

        j: value(model.x[j])

        for j in categories

    }

    SP_second_stage = pd.DataFrame({

        "Kịch bản": scenarios,

        "I": [
            value(model.y[s, "I"])
            for s in scenarios
        ],

        "D": [
            value(model.y[s, "D"])
            for s in scenarios
        ],

        "AI": [
            value(model.y[s, "AI"])
            for s in scenarios
        ],

        "H": [
            value(model.y[s, "H"])
            for s in scenarios
        ]

    })
# Lưu cho Tab 3 và Tab 4

    st.session_state["SP_objective"] = SP_objective
    st.session_state["SP_first_stage"] = SP_first_stage
    st.session_state["SP_second_stage"] = SP_second_stage
    st.session_state["probability"] = probability
    st.session_state["beta_s"] = beta_s
    st.session_state["scenarios"] = scenarios
    st.session_state["categories"] = categories
    # ==================================================
    # KPI
    # ==================================================

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "First-stage Variables",
            len(categories)
        )

    with col2:

        st.metric(
            "Second-stage Variables",
            len(categories) * len(scenarios)
        )

    with col3:

        st.metric(
            "Objective Value",
            f"{SP_objective:,.2f}"
        )

    st.divider()

    # ==================================================
    # FIRST STAGE DECISION
    # ==================================================

    st.markdown(
        "### Quyết định First-stage tối ưu"
    )

    first_stage_df = pd.DataFrame([{

        "Kịch bản": "First-stage",

        "I": SP_first_stage["I"],
        "D": SP_first_stage["D"],
        "AI": SP_first_stage["AI"],
        "H": SP_first_stage["H"]

    }])

    st.dataframe(
        first_stage_df,
        use_container_width=True,
        hide_index=True
    )

    # ==================================================
    # SECOND STAGE DECISION
    # ==================================================

    st.markdown(
        "### Quyết định điều chỉnh ngân sách (Second-stage)"
    )

    st.dataframe(
        SP_second_stage,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown('### 📌 Kết luận quản trị')

    st.success("""
    Kết quả cho thấy quyết định đầu tư ban đầu (first-stage) được đưa ra trước
khi biết trạng thái tương lai của nền kinh tế, trong khi các quyết định
điều chỉnh (second-stage) cho phép Chính phủ phản ứng linh hoạt đối với
từng kịch bản xảy ra.
""")

# =====================================================
# TAB 3
# =====================================================

with tab3:

    st.markdown('### 📊 So sánh EV và Deterministi')

    st.markdown("""
Mục tiêu của phần này là so sánh:

- EV (Expected Value): sử dụng hệ số trung bình có trọng số của các kịch bản.
- SP (Stochastic Programming): tối ưu trực tiếp trên toàn bộ cây kịch bản.
- Deterministic: giải riêng từng kịch bản.
""")
    
    st.divider()

    # ==================================================
    # LẤY KẾT QUẢ TAB 2
    # ==================================================

    SP_objective = st.session_state["SP_objective"]
    SP_first_stage = st.session_state["SP_first_stage"]

    scenarios = st.session_state["scenarios"]
    categories = st.session_state["categories"]

    probability = st.session_state["probability"]
    beta_s = st.session_state["beta_s"]

    # ==================================================
    # TÍNH HỆ SỐ KỲ VỌNG
    # ==================================================

    beta_EV = {}

    for j in categories:

        beta_EV[j] = sum(
            probability[s] * beta_s[(s, j)]
            for s in scenarios
        )

    beta_ev_df = pd.DataFrame({

        "Hạng mục": categories,

        "Beta EV": [
            beta_EV[j]
            for j in categories
        ]

    })

    st.markdown(
        "### Hệ số kỳ vọng EV"
    )

    st.dataframe(
        beta_ev_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==================================================
    # EV MODEL
    # ==================================================

    EV_model = ConcreteModel()

    EV_model.J = Set(
        initialize=categories
    )

    EV_model.x = Var(
        EV_model.J,
        domain=NonNegativeReals
    )

    EV_model.beta = Param(
        EV_model.J,
        initialize=beta_EV
    )

    EV_model.obj = Objective(

        expr=sum(
            EV_model.beta[j]
            *
            EV_model.x[j]
            for j in EV_model.J
        ),

        sense=maximize

    )

    EV_model.budget = Constraint(

        expr=sum(
            EV_model.x[j]
            for j in EV_model.J
        )
        <= 65000

    )

    solver.solve(
        EV_model,
        tee=False
    )

    EV_objective = value(
        EV_model.obj
    )

    EV_first_stage = {

        j: value(EV_model.x[j])

        for j in categories

    }
    # ==================================================
    # DETERMINISTIC THEO TỪNG KỊCH BẢN
    # ==================================================
    WS_values = {}

    deterministic_results = []

    for s in scenarios:

        det_model = ConcreteModel()

    # --------------------------
    # SET
    # --------------------------

        det_model.J = Set(
        initialize=categories
    )

    # --------------------------
    # PARAM
    # --------------------------

        beta_det = {

        j: beta_s[(s, j)]

        for j in categories

    }

        det_model.beta = Param(
        det_model.J,
        initialize=beta_det
    )

    # --------------------------
    # FIRST-STAGE
    # --------------------------

        det_model.x = Var(
        det_model.J,
        domain=NonNegativeReals
    )

    # --------------------------
    # SECOND-STAGE
    # --------------------------

        det_model.y = Var(
        det_model.J,
        domain=NonNegativeReals
    )

    # --------------------------
    # OBJECTIVE
    # --------------------------

        det_model.obj = Objective(

        expr=

        sum(
            det_model.beta[j]
            *
            det_model.x[j]
            for j in det_model.J
        )

        +

        sum(
            det_model.beta[j]
            *
            det_model.y[j]
            for j in det_model.J
        ),

        sense=maximize

    )

    # --------------------------
    # FIRST-STAGE BUDGET
    # --------------------------

        det_model.first_budget = Constraint(

        expr=

        sum(
            det_model.x[j]
            for j in det_model.J
        )

        <= 65000

    )

    # --------------------------
    # SECOND-STAGE BUDGET
    # --------------------------

        det_model.second_budget = Constraint(

        expr=

        sum(
            det_model.y[j]
            for j in det_model.J
        )

        <= 15000

    )

    # --------------------------
    # AI EXPANSION
    # --------------------------

        det_model.ai_constraint = Constraint(

        expr=

        det_model.y["AI"]

        <=

        0.5
        *
        det_model.x["H"]

    )

    # --------------------------
    # SOLVE
    # --------------------------

        solver.solve(
        det_model,
        tee=False
    )

    # --------------------------
    # SAVE WS
    # --------------------------

        WS_values[s] = value(
        det_model.obj
    )

        deterministic_results.append({

        "Kịch bản": s,

        "I": value(det_model.x["I"]),
        "D": value(det_model.x["D"]),
        "AI": value(det_model.x["AI"]),
        "H": value(det_model.x["H"]),

        "Objective":
        value(det_model.obj)

    })

        deterministic_df = pd.DataFrame(
        deterministic_results
    )
    # ==================================================
    # LƯU CHO TAB 4
    # ==================================================

        st.session_state["EV_objective"] = EV_objective
        st.session_state["WS_values"] = WS_values
    # ==================================================
    # BẢNG SO SÁNH EV VS SP
    # ==================================================

        comparison_df = pd.DataFrame([

        {
            "Mô hình": "EV",

            "I": EV_first_stage["I"],
            "D": EV_first_stage["D"],
            "AI": EV_first_stage["AI"],
            "H": EV_first_stage["H"]
        },

        {
            "Mô hình": "SP",

            "I": SP_first_stage["I"],
            "D": SP_first_stage["D"],
            "AI": SP_first_stage["AI"],
            "H": SP_first_stage["H"]
        }

    ])

    # ==================================================
    # DETERMINISTIC TABLE
    # ==================================================

    col1, col2 = st.columns(2)

    with col1:

         st.metric(
            "EV Objective",
            f"{EV_objective:,.2f}"
        )

    with col2:

         st.metric(
            "SP Objective",
            f"{SP_objective:,.2f}"
        )  
    st.markdown(
        "### So sánh quyết định First-stage"
    )  
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )   
    st.markdown(
        "### Lời giải Wait-and-See theo từng kịch bản"
    )  
    st.dataframe(
        deterministic_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()
    st.markdown('### 📌 Kết luận quản trị')
    st.success("""
Kết quả cho thấy:

• EV và SP đều lựa chọn đầu tư toàn bộ ngân sách ban đầu vào AI.

• Tuy nhiên khi biết trước kịch bản tương lai (Wait-and-See),
quyết định tối ưu có thể thay đổi đáng kể.

• Trong các kịch bản lạc quan và cơ sở,
AI vẫn là lĩnh vực mang lại hiệu quả cao nhất.

• Trong các kịch bản bi quan và khủng hoảng,
đầu tư vào vốn nhân lực trở nên hấp dẫn hơn nhờ khả năng
nâng cao sức chống chịu của nền kinh tế.

Các kết quả này sẽ được sử dụng để tính VSS và EVPI ở phần tiếp theo.
""")
       

# =====================================================
# TAB 4
# =====================================================

with tab4:

    st.markdown('### 📈 Giá trị của thông tin và mô hình ngẫu nhiên')

    st.markdown("""
### Các chỉ tiêu đánh giá

**Value of Stochastic Solution (VSS)**

Đo lường lợi ích thu được khi Chính phủ xem xét bất định trong quá trình ra quyết định.

\\[
VSS = SP - EV
\\]

---

**Expected Value of Perfect Information (EVPI)**

Đo lường giá trị tối đa của thông tin hoàn hảo.

\\[
EVPI = WS - SP
\\]

Trong đó:

- EV: Expected Value
- SP: Stochastic Programming
- WS: Wait-and-See
""")

    st.divider()

    # ==================================================
    # WAIT-AND-SEE
    # ==================================================

    WS = sum(
        probability[s] * WS_values[s]
        for s in scenarios
    )

    # ==================================================
    # VSS
    # ==================================================

    VSS = SP_objective - EV_objective

    # ==================================================
    # EVPI
    # ==================================================

    EVPI = WS - SP_objective

    # ==================================================
    # BẢNG THÀNH PHẦN WS
    # ==================================================

    ws_detail = pd.DataFrame({

        "Kịch bản": scenarios,

        "Xác suất": [
            probability[s]
            for s in scenarios
        ],

        "Objective tối ưu": [
            WS_values[s]
            for s in scenarios
        ],

        "Đóng góp": [

            probability[s] * WS_values[s]

            for s in scenarios

        ]

    })

    st.markdown(
        "### Tính Wait-and-See (WS)"
    )

    st.dataframe(
        ws_detail,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==================================================
    # KẾT QUẢ TỔNG HỢP
    # ==================================================

    result_df = pd.DataFrame({

        "Chỉ tiêu": [

            "EV",
            "SP",
            "WS",
            "VSS",
            "EVPI"

        ],

        "Giá trị": [

            EV_objective,
            SP_objective,
            WS,
            VSS,
            EVPI

        ]

    })

    st.markdown(
        "### Kết quả tổng hợp"
    )

    st.dataframe(
        result_df.style.format({
            "Giá trị": "{:,.2f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # ==================================================
    # KPI
    # ==================================================

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "VSS",
            f"{VSS:,.2f}"
        )

    with col2:

        st.metric(
            "EVPI",
            f"{EVPI:,.2f}"
        )

    st.divider()

    # ==================================================
    # DIỄN GIẢI
    # ==================================================

    st.markdown(f"""
### Diễn giải kết quả

**EV = {EV_objective:,.2f}**

Lợi ích đạt được khi Chính phủ sử dụng một kịch bản trung bình để ra quyết định.


**SP = {SP_objective:,.2f}**

Lợi ích đạt được khi Chính phủ xem xét đồng thời tất cả các trạng thái kinh tế có thể xảy ra.


**WS = {WS:,.2f}**

Lợi ích lý tưởng nếu Chính phủ biết trước tương lai và luôn chọn được quyết định tối ưu cho từng kịch bản.


**VSS = {VSS:,.2f}**

Lợi ích bổ sung do mô hình Stochastic Programming mang lại so với cách tiếp cận Expected Value.


**EVPI = {EVPI:,.2f}**

Giá trị kinh tế tối đa của thông tin hoàn hảo về tương lai.
""")

    st.divider()
    st.markdown('### 📌 Kết luận quản trị')
    st.success(f"""
Kết quả cho thấy:

• VSS = {VSS:,.2f}

→ Đây là phần lợi ích bổ sung khi xem xét bất định thay vì sử dụng một kịch bản trung bình.

• EVPI = {EVPI:,.2f}

→ Đây là giá trị tối đa mà Chính phủ sẵn sàng trả để biết chính xác trạng thái kinh tế tương lai.

Nếu EVPI lớn, việc đầu tư vào hệ thống dự báo kinh tế và phân tích kịch bản có thể mang lại lợi ích đáng kể.
""")

# =====================================================
# TAB 5 
# =====================================================
with tab5:

    st.markdown('### Robust Optimization (Minimax Regret)')

    st.markdown("""
Mục tiêu:

- Không tối đa hóa giá trị kỳ vọng.
- Tối thiểu hóa regret lớn nhất.
- Tìm quyết định an toàn nhất trong trường hợp bất lợi.
""")

    st.divider()

    # ==========================================
    # ROBUST MODEL
    # ==========================================

    robust = ConcreteModel()

    robust.J = Set(
        initialize=categories
    )

    robust.S = Set(
        initialize=scenarios
    )

    robust.x = Var(
        robust.J,
        domain=NonNegativeReals
    )

    robust.R = Var(
        domain=NonNegativeReals
    )

    # ==========================================
    # BUDGET
    # ==========================================

    robust.budget = Constraint(

        expr=sum(
            robust.x[j]
            for j in robust.J
        )
        <= 65000

    )

    # ==========================================
    # REGRET CONSTRAINTS
    # ==========================================

    def regret_rule(m, s):

        scenario_value = sum(
            beta_s[(s, j)]
            * m.x[j]
            for j in m.J
        )

        return (

            m.R

            >=

            WS_values[s]
            -
            scenario_value

        )

    robust.regret = Constraint(
        robust.S,
        rule=regret_rule
    )

    # ==========================================
    # OBJECTIVE
    # ==========================================

    robust.obj = Objective(

        expr=robust.R,

        sense=minimize

    )

    # ==========================================
    # SOLVE
    # ==========================================

    solver.solve(
        robust,
        tee=False
    )

    # ==========================================
    # SAVE RESULTS
    # ==========================================

    robust_decision = {

        j: value(
            robust.x[j]
        )

        for j in categories

    }

    robust_R = value(
        robust.R
    )

    # ==========================================
    # SO SÁNH SP VS ROBUST
    # ==========================================

    compare_df = pd.DataFrame([

        {
            "Mô hình": "SP",

            "I": SP_first_stage["I"],
            "D": SP_first_stage["D"],
            "AI": SP_first_stage["AI"],
            "H": SP_first_stage["H"]
        },

        {
            "Mô hình": "Robust",

            "I": robust_decision["I"],
            "D": robust_decision["D"],
            "AI": robust_decision["AI"],
            "H": robust_decision["H"]
        }

    ])

    st.markdown(
        "### So sánh quyết định First-stage"
    )

    st.dataframe(
        compare_df.style.format({
            "I": "{:,.0f}",
            "D": "{:,.0f}",
            "AI": "{:,.0f}",
            "H": "{:,.0f}"
        }),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.metric(
        "Maximum Regret",
        f"{robust_R:,.2f}"
    )

    st.divider()
    st.markdown('### 📌 Kết luận quản trị')
    st.success(f"""
Kết quả Robust Optimization tìm kiếm phương án đầu tư có
regret lớn nhất nhỏ nhất.

Maximum Regret = {robust_R:,.2f}

Khác với Stochastic Programming (tối đa hóa lợi ích kỳ vọng),
Robust Optimization tập trung vào việc bảo vệ quyết định trước
kịch bản bất lợi nhất.

Do đó lời giải Robust thường thận trọng hơn,
đổi lấy khả năng chống chịu rủi ro cao hơn.
""")
# =====================================================
# TAB 6
# =====================================================

with tab6:
    st.subheader("💬 Thảo luận chính sách")
    st.subheader(
        "a) So với lời giải xác định, lời giải SP có xu hướng đầu tư H nhiều hơn hay ít hơn? Vì sao?"
    )

    st.markdown("""
Kết quả của bài toán cho thấy lời giải Stochastic Programming (SP) vẫn ưu tiên đầu tư mạnh vào AI do đây là lĩnh vực mang lại lợi ích kỳ vọng cao nhất trong các kịch bản lạc quan và cơ sở. Tuy nhiên, khi so sánh với các lời giải Deterministic theo từng kịch bản, có thể thấy rằng trong các trạng thái bất lợi như **Bi quan** và **Khủng hoảng**, mô hình tối ưu lại chuyển nguồn lực sang **vốn nhân lực (H)**.

Điều này phản ánh vai trò của vốn nhân lực như một yếu tố giúp nền kinh tế thích ứng trước các cú sốc. Trong khi AI mang lại hiệu quả tăng trưởng cao khi điều kiện kinh tế thuận lợi, vốn nhân lực giúp duy trì khả năng chuyển đổi việc làm, hấp thụ công nghệ mới và nâng cao sức chống chịu của nền kinh tế khi đối mặt với rủi ro.

Do đó, mặc dù lời giải SP trong bộ dữ liệu hiện tại chưa phân bổ ngân sách ban đầu cho H, kết quả Robust Optimization cho thấy khi nhà hoạch định chính sách quan tâm nhiều hơn đến rủi ro, tỷ trọng đầu tư vào H tăng lên đáng kể.
""")

    st.warning("""
Kết quả này cho thấy vốn nhân lực không phải lúc nào cũng tạo ra lợi ích kỳ vọng cao nhất trong ngắn hạn,
nhưng lại đóng vai trò quan trọng trong việc giảm thiểu tổn thất khi xuất hiện các cú sốc kinh tế.
""")

    st.divider()

    st.subheader(
        "b) VSS dương nói lên điều gì về giá trị của tư duy xác suất trong hoạch định chính sách Việt Nam?"
    )

    st.markdown("""
Kết quả tính toán cho thấy:

- EV = 80.275
- SP = 98.575
- VSS = 18.300

Giá trị VSS dương chứng tỏ việc xem xét đồng thời nhiều kịch bản bất định mang lại lợi ích kinh tế cao hơn đáng kể so với cách tiếp cận chỉ dựa trên một kịch bản trung bình.

Trong thực tiễn hoạch định chính sách, các quyết định đầu tư công thường được đưa ra trong môi trường chứa nhiều bất định liên quan đến tăng trưởng kinh tế, dòng vốn đầu tư nước ngoài, thương mại quốc tế, thiên tai hoặc dịch bệnh. Nếu chỉ dựa trên một dự báo trung bình, nhà hoạch định chính sách có thể đánh giá thấp rủi ro và bỏ lỡ các phương án đầu tư có khả năng thích ứng tốt hơn.

Do đó, VSS dương là bằng chứng cho thấy tư duy xác suất giúp nâng cao chất lượng ra quyết định, đồng thời tăng khả năng chống chịu của nền kinh tế trước các biến động trong tương lai.
""")

    st.warning("""
VSS càng lớn thì chi phí cơ hội của việc bỏ qua yếu tố bất định càng cao.
Điều này nhấn mạnh sự cần thiết của các công cụ phân tích kịch bản và dự báo rủi ro trong hoạch định chính sách công.
""")

    st.divider()

    st.subheader(
        "c) Đại dịch COVID-19 (2020-2022) và bão Yagi (2024) là các cú sốc thực tế. Liệu Việt Nam có đang 'dưới đầu tư' vào nhân lực số như một hàng hóa bảo hiểm?"
    )

    st.markdown("""
Đại dịch COVID-19 và bão Yagi là những ví dụ điển hình về các cú sốc có khả năng làm gián đoạn hoạt động kinh tế - xã hội trên diện rộng. Trong các giai đoạn này, khả năng thích ứng của lực lượng lao động trở thành yếu tố quyết định mức độ phục hồi của nền kinh tế.

Kết quả mô hình cho thấy trong các kịch bản bất lợi, đầu tư vào vốn nhân lực mang lại hiệu quả tương đối cao hơn so với nhiều lĩnh vực khác. Điều này xuất phát từ việc lao động có kỹ năng số và trình độ chuyên môn cao thường:

- Dễ dàng chuyển đổi công việc khi thị trường biến động.
- Nhanh chóng thích nghi với công nghệ mới.
- Duy trì năng suất trong điều kiện làm việc từ xa hoặc môi trường bất ổn.
- Hỗ trợ quá trình chuyển đổi số của doanh nghiệp và khu vực công.

Từ góc độ kinh tế học, vốn nhân lực số có thể được xem như một dạng "hàng hóa bảo hiểm" cho nền kinh tế. Hiệu quả của khoản đầu tư này có thể không được phản ánh đầy đủ trong điều kiện bình thường, nhưng giá trị của nó tăng mạnh khi xuất hiện các cú sốc lớn.

Do đó, có cơ sở để cho rằng Việt Nam vẫn cần tiếp tục tăng cường đầu tư vào đào tạo kỹ năng số, giáo dục STEM, chuyển đổi kỹ năng lao động và phát triển nguồn nhân lực chất lượng cao nhằm nâng cao khả năng chống chịu của nền kinh tế trong dài hạn.
""")

    st.warning("""
Bài học từ COVID-19 và các thiên tai gần đây cho thấy đầu tư vào nhân lực số không chỉ là chính sách tăng trưởng,
mà còn là một công cụ quản trị rủi ro và bảo đảm khả năng phục hồi của nền kinh tế trước các cú sốc trong tương lai.
""")

st.divider()
st.caption("""
Nguồn dữ liệu
           
• Dữ liệu từ bài tập được cung cấp bởi giảng viên.
           
• Khung lý thuyết: Stochastic Programming hai giai đoạn.
""")