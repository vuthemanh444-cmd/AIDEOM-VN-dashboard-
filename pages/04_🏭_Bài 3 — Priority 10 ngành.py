import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 3 - Priority Index",
    page_icon="🏭",
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

st.markdown("## 🏭 Bài 3 - Chỉ số ưu tiên chuyển đổi số ngành kinh tế")
st.markdown(
    "<span class='badge badge-info'>DỄ</span>"
    "<span class='badge badge-info'>MCDM cơ bản</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Xác định ngành nên ưu tiên chuyển đổi số và AI

Phương pháp:

- Chuẩn hóa Min-Max
- Xây dựng Priority Index
- Xếp hạng ngành
- Đề xuất ưu tiên đầu tư
""")


# =========================
# ĐỌC DỮ LIỆU
# =========================

df = pd.read_csv(
    "data/vietnam_sectors_2024.csv"
)

# =========================
# CHUẨN HÓA MIN-MAX
# =========================

norm_df = df.copy()

positive_cols = [
    "Growth",
    "Productivity",
    "Spillover",
    "Export",
    "Employment",
    "AIReadiness"
]

for col in positive_cols:

    norm_df[col] = (
        norm_df[col] - norm_df[col].min()
    ) / (
        norm_df[col].max()
        - norm_df[col].min()
    )

# =========================
# ĐẢO DẤU RISK
# =========================

norm_df["Risk"] = (
    norm_df["Risk"].max()
    - norm_df["Risk"]
) / (
    norm_df["Risk"].max()
    - norm_df["Risk"].min()
)

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5 = st.tabs([

    "⚙️ Chuẩn hóa",
    "🏆 Xếp hạng Priority",
    "🎛️ Độ nhạy AI",
    "⚖️ So sánh kịch bản",
    "📝 Thảo luận chính sách"
])

# =========================
# TAB 1 - CÂU 3.4.1
# =========================

with tab1:

    st.markdown('### ⚙️ Chuẩn hoá dữ liệu Min-Max')

    st.write("""
    Mục tiêu:

    • Chuẩn hóa toàn bộ 7 tiêu chí về cùng thang đo [0,1]

    • Đảo dấu chỉ tiêu Risk

    • Xây dựng ma trận chuẩn hóa phục vụ tính Priority Index
    """)

    st.divider()

    # -------------------------
    # CÔNG THỨC
    # -------------------------

    st.markdown('### 📚 Công thức chuẩn hóa')

    st.latex(
        r'''
        \tilde{x}
        =
        \frac{x-min(x)}
        {max(x)-min(x)}
        '''
    )

    st.write(
        "Đối với chỉ tiêu bất lợi (Risk):"
    )

    st.latex(
        r'''
        \tilde{x}
        =
        \frac{max(x)-x}
        {max(x)-min(x)}
        '''
    )

    st.info("""
    Sau chuẩn hóa:

    • Giá trị càng lớn càng tốt

    • Các tiêu chí đều nằm trong khoảng [0,1]

    • Risk được đảo dấu để ngành có rủi ro thấp nhận điểm cao hơn
    """)

    st.divider()

    # -------------------------
    # KPI
    # -------------------------

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số ngành",
        len(df)
    )

    col2.metric(
        "Số tiêu chí",
        7
    )

    col3.metric(
        "Khoảng chuẩn hóa",
        "0 → 1"
    )

    st.divider()

    # -------------------------
    # DỮ LIỆU GỐC
    # -------------------------

    st.markdown('### 📋 Dữ liệu gốc')

    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -------------------------
    # MA TRẬN CHUẨN HÓA
    # -------------------------

    st.markdown('### 📊 Ma trận chuẩn hóa')

    st.dataframe(
        norm_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    # -------------------------
    # KIỂM TRA MIN-MAX
    # -------------------------

    st.markdown('### 🔍 Kiểm tra kết quả')

    check_df = pd.DataFrame({
        "Tiêu chí": positive_cols + ["Risk"],
        "Min": [
            norm_df[col].min()
            for col in positive_cols + ["Risk"]
        ],
        "Max": [
            norm_df[col].max()
            for col in positive_cols + ["Risk"]
        ]
    })

    st.dataframe(
        check_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    if (
        round(check_df["Min"].min(), 4) == 0
        and
        round(check_df["Max"].max(), 4) == 1
    ):

        st.success(
            "Chuẩn hóa thành công. Tất cả tiêu chí đều nằm trong khoảng [0,1]."
        )

    st.divider()

    # -------------------------
    # DOWNLOAD FILE
    # -------------------------

    csv = norm_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải ma trận chuẩn hóa",
        csv,
        "normalized_sectors.csv",
        "text/csv"
    )

    st.divider()

    # -------------------------
    # KẾT LUẬN
    # -------------------------

    st.markdown('### 📌 Kết luận quản trị')

    st.success("""
    Quá trình chuẩn hóa dữ liệu đã hoàn tất.

    • Toàn bộ 7 tiêu chí được đưa về cùng thang đo [0,1].

    • Risk đã được đảo dấu nên ngành có rủi ro tự động hóa thấp sẽ nhận điểm cao hơn.

    • Ma trận chuẩn hóa giúp loại bỏ sự khác biệt về đơn vị đo lường giữa các tiêu chí.

    • Kết quả này là đầu vào cho việc tính toán Priority Index ở câu 3.4.2.
    """)

# =========================
# TAB 2 - CÂU 3.4.2
# =========================

with tab2:

    st.markdown('### 🏆 Xếp hạng Priority Index các ngành')

    # =========================
    # TRỌNG SỐ
    # =========================

    a1 = 0.15
    a2 = 0.15
    a3 = 0.20
    a4 = 0.15
    a5 = 0.10
    a6 = 0.20
    a7 = 0.15

    # =========================
    # TÍNH PRIORITY
    # =========================

    priority_df = norm_df.copy()

    priority_df["Priority"] = (
        a1 * priority_df["Growth"]
        + a2 * priority_df["Productivity"]
        + a3 * priority_df["Spillover"]
        + a4 * priority_df["Export"]
        + a5 * priority_df["Employment"]
        + a6 * priority_df["AIReadiness"]
        + a7 * priority_df["Risk"]
    )

    priority_df = priority_df.sort_values(
        by="Priority",
        ascending=False
    )

    priority_df["Rank"] = range(
        1,
        len(priority_df) + 1
    )

    priority_df = priority_df[
        [
            "Rank",
            "Sector",
            "Priority"
        ]
    ]

    # =========================
    # KPI
    # =========================

    top_sector = priority_df.iloc[0]["Sector"]
    top_score = priority_df.iloc[0]["Priority"]

    avg_score = priority_df["Priority"].mean()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Ngành ưu tiên số 1",
        top_sector
    )

    col2.metric(
        "Điểm Priority cao nhất",
        f"{top_score:.4f}"
    )

    col3.metric(
        "Priority trung bình",
        f"{avg_score:.4f}"
    )

    st.divider()

    # =========================
    # BẢNG XẾP HẠNG
    # =========================

    st.markdown('### 📋 Bảng xếp hạng Priority Index')

    st.dataframe(
        priority_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    st.markdown('### 📊 So sánh Priority Index')

    fig = px.bar(
        priority_df,
        x="Priority",
        y="Sector",
        orientation="h",
        text_auto=".4f",
        title="Xếp hạng ưu tiên chuyển đổi số theo ngành"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=650,
        yaxis={
            "categoryorder": "total ascending"
        }
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD
    # =========================

    csv = priority_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải bảng xếp hạng",
        csv,
        "priority_ranking.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    top3 = priority_df.head(3)

    st.success(f"""
    Theo bộ trọng số mặc định của đề bài,
    ngành được ưu tiên chuyển đổi số cao nhất là:

    • {priority_df.iloc[0]['Sector']}

    với điểm Priority = {priority_df.iloc[0]['Priority']:.4f}

    Top 3 ngành ưu tiên gồm:

    • {top3.iloc[0]['Sector']}

    • {top3.iloc[1]['Sector']}

    • {top3.iloc[2]['Sector']}

    Các ngành này có lợi thế về năng suất,
    khả năng lan tỏa, mức độ sẵn sàng AI
    và tiềm năng tạo giá trị gia tăng cho nền kinh tế.

    Đây là những lĩnh vực nên được ưu tiên
    đầu tư chuyển đổi số và ứng dụng AI
    trong giai đoạn tới.
    """)

# ==================================================
# TAB 3 - ĐỘ NHẠY AI READINESS
# ==================================================

with tab3:

    st.markdown('### 📈 Phân tích độ nhạy trọng số AI Readiness')

    # =====================================
    # TRỌNG SỐ GỐC
    # =====================================

    base_weights = {
        "Growth": 0.15,
        "Productivity": 0.15,
        "Spillover": 0.20,
        "Export": 0.15,
        "Employment": 0.10,
        "AIReadiness": 0.20,
        "Risk": 0.15
    }

    ai_weights = np.arange(
        0.05,
        0.45,
        0.05
    )

    heatmap_data = []

    top1_list = []

    # =====================================
    # CHẠY PHÂN TÍCH ĐỘ NHẠY
    # =====================================

    for ai_w in ai_weights:

        weights = base_weights.copy()

        weights["AIReadiness"] = ai_w

        total_weight = sum(weights.values())

        for k in weights:
            weights[k] = (
                weights[k]
                / total_weight
            )

        temp_df = norm_df.copy()

        temp_df["Priority"] = (
            temp_df["Growth"] * weights["Growth"]
            + temp_df["Productivity"] * weights["Productivity"]
            + temp_df["Spillover"] * weights["Spillover"]
            + temp_df["Export"] * weights["Export"]
            + temp_df["Employment"] * weights["Employment"]
            + temp_df["AIReadiness"] * weights["AIReadiness"]
            + temp_df["Risk"] * weights["Risk"]
        )

        temp_df = temp_df.sort_values(
            "Priority",
            ascending=False
        )

        top3 = temp_df.head(3)

        top1_list.append(
            top3.iloc[0]["Sector"]
        )

        for rank, row in enumerate(
            top3.itertuples(),
            start=1
        ):

            heatmap_data.append({
                "AI Weight": round(ai_w, 2),
                "Rank": f"Top {rank}",
                "Sector": row.Sector
            })

    heatmap_df = pd.DataFrame(
        heatmap_data
    )

    # =====================================
    # KPI
    # =====================================

    unique_top1 = len(
        set(top1_list)
    )

    stable = (
        "Ổn định"
        if unique_top1 == 1
        else "Có thay đổi"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Khoảng AI Weight",
        "0.05 → 0.40"
    )

    col2.metric(
        "Số kịch bản",
        len(ai_weights)
    )

    col3.metric(
        "Top 1",
        stable
    )

    st.divider()

    # =====================================
    # BẢNG KẾT QUẢ
    # =====================================

    st.markdown('### 📋 Top 3 theo từng mức AI Weight')

    st.dataframe(
        heatmap_df,
        use_container_width=True,
        hide_index=True
    )

    # =====================================
    # HEATMAP
    # =====================================

    sector_codes = {
        sector: i + 1
        for i, sector
        in enumerate(
            sorted(
                heatmap_df["Sector"].unique()
            )
        )
    }

    heatmap_plot = heatmap_df.copy()

    heatmap_plot["Sector_Code"] = (
        heatmap_plot["Sector"]
        .map(sector_codes)
    )

    fig = px.density_heatmap(
        heatmap_plot,
        x="AI Weight",
        y="Rank",
        z="Sector_Code",
        text_auto=True,
        title="Heatmap thay đổi Top 3 khi tăng trọng số AI Readiness"
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

    # =====================================
    # DOWNLOAD
    # =====================================

    csv = heatmap_df.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Tải kết quả độ nhạy",
        csv,
        "AI_sensitivity.csv",
        "text/csv"
    )

    # =====================================
    # KẾT LUẬN QUẢN TRỊ
    # =====================================

    st.markdown('### 📌 Kết luận quản trị')

    if unique_top1 == 1:

        st.success(f"""
        Top 1 ngành ưu tiên không thay đổi
        khi trọng số AI Readiness tăng từ
        0.05 đến 0.40.

        Điều này cho thấy kết quả xếp hạng
        có tính ổn định cao và không phụ thuộc
        quá mạnh vào riêng tiêu chí AI Readiness.

        Ngành dẫn đầu vẫn duy trì lợi thế tổng hợp
        trên nhiều tiêu chí khác như năng suất,
        lan tỏa và xuất khẩu.
        """)

    else:

        st.warning(f"""
        Top 1 thay đổi khi trọng số AI Readiness
        được điều chỉnh.

        Điều này cho thấy tiêu chí AI Readiness
        có ảnh hưởng mạnh đến thứ hạng ngành.

        Nhà hoạch định chính sách cần xác định rõ
        mục tiêu ưu tiên trước khi lựa chọn bộ trọng số.
        """)

# ==================================================
# TAB 4 - SO SÁNH KỊCH BẢN TRỌNG SỐ
# ==================================================

with tab4:

    st.markdown('### ⚖️ So sánh hai định hướng chính sách')

    # =====================================
    # BỘ TRỌNG SỐ 1: TĂNG TRƯỞNG
    # =====================================

    growth_weights = {
        "Growth": 0.25,
        "Productivity": 0.25,
        "Spillover": 0.10,
        "Export": 0.20,
        "Employment": 0.05,
        "AIReadiness": 0.10,
        "Risk": 0.05
    }

    # =====================================
    # BỘ TRỌNG SỐ 2: BAO TRÙM
    # =====================================

    inclusive_weights = {
        "Growth": 0.05,
        "Productivity": 0.10,
        "Spillover": 0.25,
        "Export": 0.05,
        "Employment": 0.25,
        "AIReadiness": 0.10,
        "Risk": 0.20
    }

    # =====================================
    # TÍNH PRIORITY - TĂNG TRƯỞNG
    # =====================================

    growth_df = norm_df.copy()

    growth_df["Priority"] = (
        growth_df["Growth"] * growth_weights["Growth"]
        + growth_df["Productivity"] * growth_weights["Productivity"]
        + growth_df["Spillover"] * growth_weights["Spillover"]
        + growth_df["Export"] * growth_weights["Export"]
        + growth_df["Employment"] * growth_weights["Employment"]
        + growth_df["AIReadiness"] * growth_weights["AIReadiness"]
        + growth_df["Risk"] * growth_weights["Risk"]
    )

    growth_df = growth_df.sort_values(
        "Priority",
        ascending=False
    )

    growth_top3 = growth_df.head(3)

    # =====================================
    # TÍNH PRIORITY - BAO TRÙM
    # =====================================

    inclusive_df = norm_df.copy()

    inclusive_df["Priority"] = (
        inclusive_df["Growth"] * inclusive_weights["Growth"]
        + inclusive_df["Productivity"] * inclusive_weights["Productivity"]
        + inclusive_df["Spillover"] * inclusive_weights["Spillover"]
        + inclusive_df["Export"] * inclusive_weights["Export"]
        + inclusive_df["Employment"] * inclusive_weights["Employment"]
        + inclusive_df["AIReadiness"] * inclusive_weights["AIReadiness"]
        + inclusive_df["Risk"] * inclusive_weights["Risk"]
    )

    inclusive_df = inclusive_df.sort_values(
        "Priority",
        ascending=False
    )

    inclusive_top3 = inclusive_df.head(3)

    # =====================================
    # KPI
    # =====================================

    overlap = len(
        set(growth_top3["Sector"])
        &
        set(inclusive_top3["Sector"])
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Top 3 tăng trưởng",
        len(growth_top3)
    )

    col2.metric(
        "Top 3 bao trùm",
        len(inclusive_top3)
    )

    col3.metric(
        "Ngành trùng nhau",
        overlap
    )

    st.divider()

    # =====================================
    # BẢNG KẾT QUẢ
    # =====================================

    st.markdown('### 📋 Top 3 theo từng định hướng')

    compare_table = pd.DataFrame({
        "Xếp hạng": [1, 2, 3],
        "Định hướng tăng trưởng":
            growth_top3["Sector"].values,
        "Định hướng bao trùm":
            inclusive_top3["Sector"].values
    })

    st.dataframe(
        compare_table,
        use_container_width=True,
        hide_index=True
    )

    # =====================================
    # BIỂU ĐỒ
    # =====================================

    chart_df = pd.concat([
        pd.DataFrame({
            "Sector": growth_top3["Sector"],
            "Priority": growth_top3["Priority"],
            "Kịch bản": "Tăng trưởng"
        }),
        pd.DataFrame({
            "Sector": inclusive_top3["Sector"],
            "Priority": inclusive_top3["Priority"],
            "Kịch bản": "Bao trùm"
        })
    ])

    fig = px.bar(
        chart_df,
        x="Sector",
        y="Priority",
        color="Kịch bản",
        barmode="group",
        text_auto=".3f",
        title="So sánh Top 3 ngành ưu tiên"
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

    # =====================================
    # DOWNLOAD
    # =====================================

    csv = compare_table.to_csv(
        index=False
    ).encode("utf-8-sig")

    st.download_button(
        "📥 Tải kết quả so sánh",
        csv,
        "priority_scenarios.csv",
        "text/csv"
    )

    # =====================================
    # KẾT LUẬN QUẢN TRỊ
    # =====================================

    st.markdown('### 📌 Kết luận quản trị')

    if overlap == 3:

        st.success("""
        Hai định hướng chính sách tạo ra
        cùng một Top 3 ngành ưu tiên.

        Điều này cho thấy các ngành dẫn đầu
        có lợi thế toàn diện cả về tăng trưởng,
        năng suất, lan tỏa và khả năng tạo việc làm.

        Việc ưu tiên đầu tư số vào các ngành này
        có thể mang lại hiệu quả cao dưới nhiều
        góc độ phát triển khác nhau.
        """)

    elif overlap >= 1:

        st.info(f"""
        Hai định hướng có {overlap} ngành xuất hiện
        đồng thời trong Top 3.

        Điều này cho thấy mục tiêu tăng trưởng
        và mục tiêu phát triển bao trùm không hoàn toàn
        giống nhau.

        Nhà hoạch định chính sách cần cân nhắc
        giữa hiệu quả kinh tế ngắn hạn và tác động
        xã hội dài hạn khi lựa chọn ngành ưu tiên.
        """)

    else:

        st.warning("""
        Hai định hướng cho ra Top 3 hoàn toàn khác nhau.

        Điều này phản ánh sự đánh đổi rõ rệt giữa:

        • Tối đa hóa tăng trưởng kinh tế

        • Tối đa hóa lợi ích xã hội và bao trùm

        Do đó chiến lược chuyển đổi số quốc gia
        cần xác định rõ mục tiêu ưu tiên trước
        khi phân bổ nguồn lực.
        """)

# ==================================================
# TAB 5 - THẢO LUẬN CHÍNH SÁCH
# ==================================================

with tab5:
    
    st.markdown('### 💬 Thảo luận chính sách')

    # ----------------------------------
    # CÂU A
    # ----------------------------------

    st.markdown("""
    ### a) Theo kết quả của em, ba ngành nào nên được ưu tiên đẩy mạnh chuyển đổi số và AI trước? Kết quả này có phù hợp với Nghị quyết 57-NQ/TW không?

    Theo kết quả Priority Index, ba ngành có mức ưu tiên cao nhất là các ngành đứng đầu trong bảng xếp hạng ở Câu 3.4.2.

    Các ngành này có điểm mạnh về khả năng lan tỏa, năng suất lao động, mức độ sẵn sàng AI và đóng góp cho tăng trưởng kinh tế.

    Kết quả này phù hợp với Nghị quyết 57-NQ/TW khi nghị quyết xác định khoa học công nghệ, đổi mới sáng tạo và chuyển đổi số là động lực phát triển mới của nền kinh tế Việt Nam.
    """)

    # ----------------------------------
    # CÂU B
    # ----------------------------------

    st.markdown("""
    ### b) Tại sao ngành Khai khoáng có năng suất rất cao nhưng vẫn không nằm trong nhóm ưu tiên?

    Mặc dù ngành Khai khoáng có năng suất lao động cao, nhưng đây chỉ là một trong nhiều tiêu chí đánh giá.

    Ngành này có một số hạn chế như:

    - Tốc độ tăng trưởng thấp hoặc âm.
    - Khả năng lan tỏa sang các ngành khác không cao.
    - Quy mô việc làm nhỏ.
    - Mức độ sẵn sàng AI chưa nổi bật.
    - Rủi ro tự động hóa tương đối cao.

    Vì vậy điểm Priority tổng hợp không đủ cao để nằm trong nhóm ưu tiên hàng đầu.
    """)

    # ----------------------------------
    # CÂU C
    # ----------------------------------

    st.markdown("""
    ### c) Bộ trọng số nên do ai quyết định: chuyên gia kỹ thuật, hội đồng chính sách hay quy trình đối thoại công khai?

    Không nên để một nhóm duy nhất quyết định bộ trọng số.

    - Chuyên gia kỹ thuật có lợi thế về dữ liệu và phương pháp định lượng.
    - Hội đồng chính sách có khả năng cân bằng giữa mục tiêu kinh tế, xã hội và chiến lược quốc gia.
    - Đối thoại công khai giúp tăng tính minh bạch và tính chính danh của chính sách.

    Vì vậy cách tiếp cận phù hợp là kết hợp cả ba bên: chuyên gia xây dựng mô hình, hội đồng chính sách thẩm định và các bên liên quan tham gia góp ý trước khi ban hành quyết định cuối cùng.
    """)

    st.divider()

st.caption("""
Nguồn dữ liệu:

• Cục Thống kê Quốc gia Việt Nam (2024)

• Dữ liệu mô phỏng phục vụ học tập và nghiên cứu
""")