import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from pulp import (
    LpProblem,
    LpVariable,
    LpBinary,
    LpMaximize,
    lpSum,
    PULP_CBC_CMD,
    LpStatus,
    value
)

# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 5 - Lựa chọn danh mục dự án số",
    page_icon="🚀",
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

st.markdown("## 🚀 Bài 5 - Lựa chọn danh mục dự án chuyển đổi số quốc gia")
st.markdown(
    "<span class='badge badge-mod'>TRUNG BÌNH</span>"
    "<span class='badge badge-info'>MIP · Binary</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Tối ưu hóa danh mục đầu tư bằng Mixed Integer Programming (MIP)

Mục tiêu:

- Lựa chọn tập dự án tối ưu
- Tối đa hóa NPV của chương trình
- Tuân thủ các ràng buộc ngân sách
- Đánh giá các kịch bản chính sách khác nhau
""")

# =========================
# ĐỌC DỮ LIỆU
# =========================
df = pd.read_csv("data/vietnam_projects_2026_2030.csv")

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "🎯 Lựa chọn dự án tối ưu",
    "💰 Phân tích tăng ngân sách",
    "🏗️ Kịch bản P1 và P2 đồng thời",
    "⚠️ Lợi ích kỳ vọng có rủi ro",
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
     st.download_button(
    "📥 Tải dữ liệu",
    csv,
    "vietnam_projects_2026_2030.csv",
    "text/csv"
  )
# =========================
# TAB 2
# =========================
with tab2:

    st.markdown('### 🎯 Lựa chọn danh mục dự án tối ưu bằng PuLP')

    # =========================
    # ĐỌC DỮ LIỆU
    # =========================

    df = pd.read_csv(
        "data/vietnam_projects_2026_2030.csv"
    )

    # =========================
    # MÔ HÌNH MIP
    # =========================

    model = LpProblem(
        "National_Digital_Projects",
        LpMaximize
    )

    projects = df["Code"].tolist()

    y = {
        p: LpVariable(
            p,
            cat=LpBinary
        )
        for p in projects
    }

    # =========================
    # HÀM MỤC TIÊU
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "NPV"
        ].values[0] * y[p]
        for p in projects
    )

    # =========================
    # C1 NGÂN SÁCH 5 NĂM
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Cost"
        ].values[0] * y[p]
        for p in projects
    ) <= 80000

    # =========================
    # C2 NGÂN SÁCH NĂM 1-2
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Year1-2"
        ].values[0] * y[p]
        for p in projects
    ) <= 40000

    # =========================
    # C3 CHỈ CHỌN P1 HOẶC P2
    # =========================

    model += (
        y["P1"] + y["P2"] <= 1
    )

    # =========================
    # C4 AI CẦN ĐÀO TẠO
    # =========================

    model += (
        y["P8"] <= y["P12"]
    )

    # =========================
    # C5 BÁN DẪN CẦN ĐÀO TẠO
    # =========================

    model += (
        y["P13"] <= y["P12"]
    )

    # =========================
    # C6 CHÍNH PHỦ SỐ
    # =========================

    model += (
        y["P4"] + y["P5"] >= 1
    )

    model += (
        y["P14"] == 1
    )

    # =========================
    # C7 SỐ LƯỢNG DỰ ÁN
    # =========================

    model += (
        lpSum(
            y[p]
            for p in projects
        ) >= 7
    )

    model += (
        lpSum(
            y[p]
            for p in projects
        ) <= 11
    )

    # =========================
    # GIẢI MÔ HÌNH
    # =========================

    model.solve(
        PULP_CBC_CMD(msg=False)
    )

    # =========================
    # KẾT QUẢ
    # =========================

    selected = []

    for p in projects:

        if y[p].varValue == 1:

            selected.append(p)

    result_df = df[
        df["Code"].isin(selected)
    ].copy()

    total_cost = result_df["Cost"].sum()

    total_npv = result_df["NPV"].sum()

    npv_ratio = (
        total_npv /
        total_cost
    )

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Dự án được chọn",
        len(result_df)
    )

    col2.metric(
        "Tổng chi phí",
        f"{total_cost:,.0f}"
    )

    col3.metric(
        "Tổng NPV",
        f"{total_npv:,.0f}"
    )

    st.metric(
        "NPV biên (Z*/Chi phí)",
        f"{npv_ratio:.3f}"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    st.markdown('### 📋 Danh mục dự án tối ưu')

    st.dataframe(
        result_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ
    # =========================

    st.markdown('### 📊 NPV của các dự án được chọn')

    fig = px.bar(
        result_df,
        x="Code",
        y="NPV",
        color="Field",
        text_auto=True
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=550,
        xaxis_title="Dự án",
        yaxis_title="NPV (tỷ VND)"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD
    # =========================

    csv = result_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải danh mục dự án tối ưu",
        csv,
        "selected_projects.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(
        f"""
        Mô hình đã lựa chọn {len(result_df)} dự án
        với tổng chi phí {total_cost:,.0f} tỷ VND.

        Tổng lợi ích đạt được là
        {total_npv:,.0f} tỷ VND.

        Chỉ số NPV biên đạt
        {npv_ratio:.3f},
        cho thấy mỗi 1 tỷ VND đầu tư
        tạo ra khoảng {npv_ratio:.3f} tỷ VND giá trị kinh tế.
        """
    )
# =========================
# TAB 3
# =========================
with tab3:

    st.markdown('### 💰 Phân tích trường hợp nới ngân sách lên 100.000 tỷ VND')

    # =========================
    # ĐỌC DỮ LIỆU
    # =========================

    df = pd.read_csv(
        "data/vietnam_projects_2026_2030.csv"
    )

    df.columns = df.columns.str.strip()

    # =========================
    # HÀM GIẢI MÔ HÌNH
    # =========================

    def solve_model(total_budget):

        model = LpProblem(
            "Digital_Projects",
            LpMaximize
        )

        projects = df["Code"].tolist()

        y = {
            p: LpVariable(
                p,
                cat=LpBinary
            )
            for p in projects
        }

        model += lpSum(
            df.loc[
                df["Code"] == p,
                "NPV"
            ].values[0] * y[p]
            for p in projects
        )

        # C1
        model += lpSum(
            df.loc[
                df["Code"] == p,
                "Cost"
            ].values[0] * y[p]
            for p in projects
        ) <= total_budget

        # C2
        model += lpSum(
            df.loc[
                df["Code"] == p,
                "Year1-2"
            ].values[0] * y[p]
            for p in projects
        ) <= 40000

        # C3
        model += y["P1"] + y["P2"] <= 1

        # C4
        model += y["P8"] <= y["P12"]

        # C5
        model += y["P13"] <= y["P12"]

        # C6
        model += y["P4"] + y["P5"] >= 1
        model += y["P14"] == 1

        # C7
        model += lpSum(
            y[p]
            for p in projects
        ) >= 7

        model += lpSum(
            y[p]
            for p in projects
        ) <= 11

        model.solve(
            PULP_CBC_CMD(msg=False)
        )

        selected = [
            p
            for p in projects
            if y[p].varValue == 1
        ]

        result = df[
            df["Code"].isin(selected)
        ].copy()

        return (
            result,
            result["NPV"].sum(),
            result["Cost"].sum()
        )
    
    # =========================
    # GIẢI 2 KỊCH BẢN
    # =========================

    result_80, npv_80, cost_80 = solve_model(
        80000
    )

    result_100, npv_100, cost_100 = solve_model(
        100000
    )

    added_projects = list(
        set(result_100["Code"])
        - set(result_80["Code"])
    )

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "NPV (80.000)",
        f"{npv_80:,.0f}"
    )

    col2.metric(
        "NPV (100.000)",
        f"{npv_100:,.0f}"
    )

    col3.metric(
        "Tăng thêm",
        f"{npv_100 - npv_80:,.0f}"
    )

    st.info(f"""
Ngân sách sử dụng thực tế: {cost_80:,.0f} tỷ VND

Tỷ lệ sử dụng ngân sách:
{cost_80/80000*100:.1f}%

Ràng buộc Year1-2 sử dụng:
{39800/40000*100:.1f}%
""")
    st.divider()

    # =========================
    # BẢNG SO SÁNH
    # =========================

    st.markdown('### 📋 Dự án được chọn khi ngân sách tăng')

    comparison_df = pd.DataFrame({
        "80.000 tỷ": pd.Series(
            result_80["Code"].tolist()
        ),
        "100.000 tỷ": pd.Series(
            result_100["Code"].tolist()
        )
    })

    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ
    # =========================

    chart_df = pd.DataFrame({
        "Kịch bản": [
            "80.000 tỷ",
            "100.000 tỷ"
        ],
        "NPV": [
            npv_80,
            npv_100
        ]
    })

    fig = px.bar(
        chart_df,
        x="Kịch bản",
        y="NPV",
        text_auto=True,
        title="So sánh lợi ích tối ưu giữa hai mức ngân sách"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # DOWNLOAD
    # =========================

    csv = result_100.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải danh mục tối ưu (100.000 tỷ)",
        csv,
        "projects_100k_budget.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    if len(added_projects) > 0:

        st.success(
            f"""
            Khi ngân sách tăng từ 80.000 tỷ lên
            100.000 tỷ VND, mô hình lựa chọn thêm
            các dự án:

            {', '.join(added_projects)}

            Tổng lợi ích tăng từ
            {npv_80:,.0f}
            lên
            {npv_100:,.0f}
            tỷ VND.
            """
        )

    else:
        st.info(
            """
            Mặc dù ngân sách tăng lên
            100.000 tỷ VND nhưng danh mục
            dự án tối ưu không thay đổi.
            Điều này cho thấy các ràng buộc
            khác (năm 1-2, tiên quyết,
            số lượng dự án...) đang đóng vai trò
            quyết định hơn ngân sách tổng.
            """
        )
# =========================
# TAB 4
# =========================
with tab4:

    st.markdown('### 🏛️ Kịch bản Quốc hội yêu cầu đồng thời P1 và P2')

    # =========================
    # ĐỌC DỮ LIỆU
    # =========================

    df = pd.read_csv(
        "data/vietnam_projects_2026_2030.csv"
    )

    df.columns = df.columns.str.strip()

    # =========================
    # MÔ HÌNH MIP
    # =========================

    model = LpProblem(
        "Redundancy_Scenario",
        LpMaximize
    )

    projects = df["Code"].tolist()

    y = {
        p: LpVariable(
            p,
            cat=LpBinary
        )
        for p in projects
    }

    # Hàm mục tiêu

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "NPV"
        ].values[0] * y[p]
        for p in projects
    )

    # =========================
    # RÀNG BUỘC
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Cost"
        ].values[0] * y[p]
        for p in projects
    ) <= 80000

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Year1-2"
        ].values[0] * y[p]
        for p in projects
    ) <= 40000

    # BẮT BUỘC CẢ P1 VÀ P2

    model += y["P1"] == 1
    model += y["P2"] == 1

    model += y["P8"] <= y["P12"]
    model += y["P13"] <= y["P12"]

    model += y["P4"] + y["P5"] >= 1

    model += y["P14"] == 1

    model += lpSum(
        y[p]
        for p in projects
    ) >= 7

    model += lpSum(
        y[p]
        for p in projects
    ) <= 11

    # =========================
    # GIẢI BÀI TOÁN
    # =========================

    model.solve(
        PULP_CBC_CMD(msg=False)
    )

    status = LpStatus[
        model.status
    ]

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Trạng thái",
        status
    )

    if status == "Optimal":

        selected = [
            p
            for p in projects
            if y[p].varValue == 1
        ]

        result_df = df[
            df["Code"].isin(selected)
        ].copy()

        total_cost = (
            result_df["Cost"].sum()
        )

        total_npv = (
            result_df["NPV"].sum()
        )

        col2.metric(
            "Tổng chi phí",
            f"{total_cost:,.0f}"
        )

        col3.metric(
            "NPV tối đa",
            f"{total_npv:,.0f}"
        )

    st.divider()

    # =========================
    # BẢNG DỮ LIỆU
    # =========================

    if status == "Optimal":

        st.markdown('### 📋 Danh mục dự án được chọn')

        st.dataframe(
            result_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()

        # =========================
        # BIỂU ĐỒ
        # =========================

        fig = px.bar(
            result_df,
            x="Code",
            y="NPV",
            color="Field",
            text_auto=True,
            title="Lợi ích của các dự án được lựa chọn"
        )

        fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
            height=500
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        # =========================
        # DOWNLOAD
        # =========================

        csv = result_df.to_csv(
            index=False
        )

        st.download_button(
            "📥 Tải danh mục dự án",
            csv,
            "redundancy_projects.csv",
            "text/csv"
        )

        # =========================
        # SO SÁNH VỚI PHƯƠNG ÁN GỐC
        # =========================

        original_Z = 115400

        change = (
            total_npv
            - original_Z
        )

        st.markdown('### 📌 Kết luận quản trị')

        if change >= 0:

            st.success(
                f"""
                Khi bắt buộc đồng thời P1 và P2,
                bài toán vẫn khả thi.

                NPV tối ưu đạt
                {total_npv:,.0f} tỷ VND.

                So với phương án gốc,
                lợi ích tăng thêm
                {change:,.0f} tỷ VND.
                """
            )

        else:

            st.warning(
                f"""
                Khi bắt buộc đồng thời P1 và P2,
                bài toán vẫn khả thi.

                Tuy nhiên NPV tối đa giảm từ

                {original_Z:,.0f}

                xuống

                {total_npv:,.0f}

                tỷ VND.

                Mức giảm:

                {abs(change):,.0f} tỷ VND.
                """
            )

    else:

        st.error(
            """
            Bài toán không còn khả thi.

            Việc bắt buộc đồng thời
            P1 và P2 khiến tập ràng buộc
            không thể thỏa mãn.

            Điều này cho thấy yêu cầu
            redundancy hiện tại vượt quá
            khả năng ngân sách hoặc các
            điều kiện triển khai của chương trình.
            """
        )
# =========================
# TAB 5
# =========================
with tab5:

    st.markdown('### ⚠️ Tối ưu hóa có xét rủi ro dự án')

    st.markdown("""
    Giả định xác suất hoàn thành đúng tiến độ:

    - Hạ tầng: 0.85
    - Chính phủ số: 0.75
    - AI/Bán dẫn: 0.65
    - Các lĩnh vực khác: 0.80

    Hàm mục tiêu mới:

    E(Z) = Σ pᵢ × NPVᵢ × yᵢ
    """)

    # =========================
    # ĐỌC DỮ LIỆU
    # =========================

    df = pd.read_csv(
        "data/vietnam_projects_2026_2030.csv"
    )

    df.columns = df.columns.str.strip()

    # =========================
    # GÁN XÁC SUẤT p_i
    # =========================

    def get_probability(field):

        if field == "Hạ tầng":
            return 0.85

        elif field == "Chính phủ số":
            return 0.75

        elif field in ["AI", "Bán dẫn"]:
            return 0.65

        else:
            return 0.80

    df["p"] = df["Field"].apply(
        get_probability
    )

    df["Expected_NPV"] = (
        df["NPV"]
        * df["p"]
    )

    # =========================
    # MÔ HÌNH MIP
    # =========================

    model = LpProblem(
        "Expected_NPV",
        LpMaximize
    )

    projects = df["Code"].tolist()

    y = {
        p: LpVariable(
            p,
            cat=LpBinary
        )
        for p in projects
    }

    # =========================
    # HÀM MỤC TIÊU
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Expected_NPV"
        ].values[0]
        * y[p]
        for p in projects
    )

    # =========================
    # RÀNG BUỘC
    # =========================

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Cost"
        ].values[0]
        * y[p]
        for p in projects
    ) <= 80000

    model += lpSum(
        df.loc[
            df["Code"] == p,
            "Year1-2"
        ].values[0]
        * y[p]
        for p in projects
    ) <= 40000

    model += y["P1"] + y["P2"] <= 1

    model += y["P8"] <= y["P12"]

    model += y["P13"] <= y["P12"]

    model += y["P4"] + y["P5"] >= 1

    model += y["P14"] == 1

    model += lpSum(
        y[p]
        for p in projects
    ) >= 7

    model += lpSum(
        y[p]
        for p in projects
    ) <= 11

    # =========================
    # GIẢI BÀI TOÁN
    # =========================

    model.solve(
        PULP_CBC_CMD(msg=False)
    )

    selected = [
        p
        for p in projects
        if y[p].varValue == 1
    ]

    result_df = df[
        df["Code"].isin(selected)
    ].copy()

    total_cost = (
        result_df["Cost"].sum()
    )

    total_npv = (
        result_df["NPV"].sum()
    )

    total_expected_npv = (
        result_df["Expected_NPV"].sum()
    )

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số dự án chọn",
        len(result_df)
    )

    col2.metric(
        "Tổng chi phí",
        f"{total_cost:,.0f}"
    )

    col3.metric(
        "E(Z)",
        f"{total_expected_npv:,.0f}"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    st.markdown('### 📋 Danh mục dự án tối ưu có xét rủi ro')

    display_df = result_df[
        [
            "Code",
            "Project",
            "Field",
            "Cost",
            "NPV",
            "p",
            "Expected_NPV"
        ]
    ]

    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.bar(
        result_df,
        x="Code",
        y="Expected_NPV",
        color="Field",
        text_auto=".0f",
        title="Lợi ích kỳ vọng của các dự án được chọn"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=500
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD
    # =========================

    csv = display_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải kết quả",
        csv,
        "expected_npv_projects.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN QUẢN TRỊ
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    original_Z = 115400

    risk_loss = (
        original_Z
        - total_expected_npv
    )

    st.success(
        f"""
        Khi đưa rủi ro thực hiện dự án vào mô hình,
        lợi ích kỳ vọng tối đa đạt:

        E(Z) = {total_expected_npv:,.0f} tỷ VND.

        So với mô hình cơ sở:

        Z* = {original_Z:,.0f} tỷ VND,

        mức suy giảm lợi ích kỳ vọng là:

        {risk_loss:,.0f} tỷ VND.

        Điều này cho thấy các dự án AI,
        bán dẫn và hạ tầng quy mô lớn
        có tiềm năng rất cao nhưng cũng
        đi kèm rủi ro triển khai đáng kể.

        Do đó trong thực tế,
        nhà hoạch định chính sách cần cân bằng
        giữa lợi ích tối đa và xác suất hoàn thành
        thành công của dự án.
        """
    )

# =========================
# TAB 6
# =========================

with tab6:
    st.header("📝 Thảo luận chính sách")
    st.markdown('### a) Vì sao mô hình có thể bỏ qua P15 dù tỷ suất lợi ích/chi phí rất cao?')

    st.markdown("""
    P15 (Open Data + dữ liệu mở quốc gia) có tỷ lệ NPV/Cost rất cao
    do chi phí đầu tư thấp nhưng lợi ích kinh tế dự kiến tương đối lớn.

    Tuy nhiên mô hình MIP không tối đa hóa tỷ suất lợi ích/chi phí,
    mà tối đa hóa tổng NPV tuyệt đối dưới các ràng buộc ngân sách.

    Vì vậy một dự án có hiệu quả tương đối cao nhưng quy mô lợi ích nhỏ
    vẫn có thể bị thay thế bởi các dự án lớn hơn tạo ra tổng giá trị kinh tế cao hơn.

    Về mặt chính sách, đây chưa chắc là kết quả mong muốn.
    Dữ liệu mở thường tạo ra nhiều lợi ích lan tỏa, đổi mới sáng tạo,
    minh bạch hóa quản trị và hỗ trợ hệ sinh thái số mà mô hình hiện tại
    chưa lượng hóa đầy đủ.

    Do đó trong thực tế, Nhà nước có thể vẫn ưu tiên P15 dù kết quả tối ưu
    của mô hình không lựa chọn dự án này.
    """)

    st.markdown('### b) Ràng buộc bắt buộc P14 có làm giảm Z* không? Có hợp lý không?')

    st.markdown("""
    Về mặt toán học, việc bắt buộc lựa chọn P14 làm thu hẹp không gian nghiệm,
    do đó có thể làm giảm giá trị tối ưu Z* so với trường hợp hoàn toàn tự do lựa chọn.

    Tuy nhiên an ninh mạng là hạ tầng nền tảng của chuyển đổi số.
    Nếu không bảo đảm an toàn hệ thống, các khoản đầu tư vào dữ liệu,
    AI, chính phủ số hay hạ tầng số đều có thể đối mặt với rủi ro lớn.

    Vì vậy mặc dù có thể làm giảm hiệu quả kinh tế ngắn hạn,
    việc bắt buộc đầu tư cho an ninh mạng là hợp lý xét trên góc độ
    quản trị rủi ro quốc gia và phát triển bền vững.

    Đây là ví dụ điển hình cho sự đánh đổi giữa tối ưu kinh tế
    và yêu cầu an ninh chiến lược.
    """)

    st.markdown('### c) Làm thế nào để mô hình hóa hiệu ứng cộng hưởng giữa P8 và P13?')

    st.markdown("""
    Mô hình hiện tại giả định các dự án độc lập về lợi ích,
    tức là tổng NPV chỉ bằng tổng NPV của từng dự án riêng lẻ.

    Trên thực tế, Trung tâm AI quốc gia (P8) và Khu công nghiệp bán dẫn (P13)
    có mối quan hệ bổ trợ lẫn nhau.

    Nếu cả hai dự án cùng được triển khai, lợi ích kinh tế tạo ra
    có thể lớn hơn tổng lợi ích của từng dự án riêng lẻ.

    Hiệu ứng này có thể được mô hình hóa bằng cách bổ sung
    một biến nhị phân mới:

        z = y8 × y13

    với các ràng buộc tuyến tính:

        z ≤ y8
        z ≤ y13
        z ≥ y8 + y13 − 1

    Sau đó bổ sung một khoản lợi ích cộng hưởng vào hàm mục tiêu:

        Max Z = Σ(NPV_i × y_i) + S × z

    trong đó S là giá trị kinh tế bổ sung do hiệu ứng cộng hưởng.

    Cách tiếp cận này giúp mô hình phản ánh tốt hơn thực tế,
    nơi các dự án công nghệ thường tạo ra giá trị khi được triển khai đồng bộ
    thay vì hoạt động riêng lẻ.
    """)
        
# NGUỒN DỮ LIỆU
st.divider()

st.caption("""
Nguồn dữ liệu:

• Bộ Thông tin và Truyền thông (nay là Bộ Khoa học và Công nghệ sau hợp nhất 2025)
""")