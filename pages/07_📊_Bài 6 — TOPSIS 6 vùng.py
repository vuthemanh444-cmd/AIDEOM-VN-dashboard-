import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# =========================
# CẤU HÌNH TRANG
# =========================

st.set_page_config(
    page_title="Bài 6 - TOPSIS AI Regions",
    page_icon="🤖",
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

st.markdown("## 🤖 Bài 6 - Xếp hạng mức độ sẵn sàng AI của các vùng kinh tế Việt Nam")
st.markdown(
    "<span class='badge badge-mod'>TRUNG BÌNH</span>"
    "<span class='badge badge-info'>TOPSIS · Entropy</span>",
    unsafe_allow_html=True
)
st.divider()

st.markdown("""
### Ứng dụng TOPSIS trong lựa chọn vùng ưu tiên triển khai AI

Mục tiêu:

- Đánh giá mức độ sẵn sàng AI của 6 vùng kinh tế xã hội
- Xây dựng mô hình TOPSIS từ đầu bằng NumPy
- So sánh trọng số chuyên gia và trọng số Entropy
- Phân tích độ nhạy của tiêu chí AI Readiness
- Đề xuất vùng ưu tiên phát triển trung tâm AI quốc gia
""")

# =========================
# ĐỌC DỮ LIỆU
# =========================
df = pd.read_csv("data/vietnam_regions_2024_2.csv")

# =========================
# TABS
# =========================

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📋 Dữ liệu",
    "🏆 TOPSIS Chuyên gia",
    "⚖️ TOPSIS Entropy",
    "📈 Phân tích độ nhạy",
    "🔄 So sánh AHP",
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
    "vietnam_regions_2024_2.csv",
    "text/csv"
  )

# =========================
# TAB 2
# =========================

with tab2:

    st.markdown('### 🏆 Xếp hạng TOPSIS với trọng số chuyên gia')

    # =========================
    # DỮ LIỆU
    # =========================

    benefit_cols = [
        "GRDP",
        "FDI",
        "DigitalIndex",
        "AIReadiness",
        "LDDT",
        "R&D/GRDP",
        "Internet"
    ]

    cost_cols = [
        "Gini"
    ]

    criteria = benefit_cols + cost_cols

    X = df[criteria].values.astype(float)

    # =========================
    # TRỌNG SỐ CHUYÊN GIA
    # =========================

    weights = np.array([
        0.10,  # GRDP
        0.10,  # FDI
        0.15,  # Digital Index
        0.20,  # AI Readiness
        0.15,  # Lao động đào tạo
        0.15,  # R&D
        0.05,  # Internet
        0.10   # Gini
    ])

    # =========================
    # BƯỚC 1
    # CHUẨN HÓA VECTOR
    # =========================

    norm_matrix = X / np.sqrt(
        (X ** 2).sum(axis=0)
    )

    # =========================
    # BƯỚC 2
    # MA TRẬN CÓ TRỌNG SỐ
    # =========================

    weighted_matrix = (
        norm_matrix * weights
    )

    # =========================
    # BƯỚC 3
    # IDEAL BEST / WORST
    # =========================

    ideal_best = np.zeros(len(criteria))
    ideal_worst = np.zeros(len(criteria))

    for i, col in enumerate(criteria):

        if col in benefit_cols:

            ideal_best[i] = weighted_matrix[:, i].max()
            ideal_worst[i] = weighted_matrix[:, i].min()

        else:

            ideal_best[i] = weighted_matrix[:, i].min()
            ideal_worst[i] = weighted_matrix[:, i].max()

    # =========================
    # BƯỚC 4
    # KHOẢNG CÁCH
    # =========================

    s_plus = np.sqrt(
        ((weighted_matrix - ideal_best) ** 2).sum(axis=1)
    )

    s_minus = np.sqrt(
        ((weighted_matrix - ideal_worst) ** 2).sum(axis=1)
    )

    # =========================
    # BƯỚC 5
    # TOPSIS SCORE
    # =========================

    ci = s_minus / (
        s_plus + s_minus
    )

    ranking_df = pd.DataFrame({
        "Region": df["Regions"],
        "TOPSIS Score": ci
    })

    ranking_df = ranking_df.sort_values(
        "TOPSIS Score",
        ascending=False
    ).reset_index(drop=True)

    ranking_df["Rank"] = (
        ranking_df.index + 1
    )

    # =========================
    # KPI
    # =========================

    top_region = ranking_df.iloc[0]["Region"]

    top_score = ranking_df.iloc[0]["TOPSIS Score"]

    gap = (
        ranking_df.iloc[0]["TOPSIS Score"]
        -
        ranking_df.iloc[1]["TOPSIS Score"]
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Vùng dẫn đầu",
        top_region
    )

    col2.metric(
        "TOPSIS Score",
        f"{top_score:.4f}"
    )

    col3.metric(
        "Khoảng cách Top1-Top2",
        f"{gap:.4f}"
    )

    st.divider()

    # =========================
    # BẢNG XẾP HẠNG
    # =========================

    st.markdown('### 📋 Bảng xếp hạng TOPSIS')

    st.dataframe(
        ranking_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.bar(
        ranking_df,
        x="Region",
        y="TOPSIS Score",
        color="TOPSIS Score",
        text_auto=".4f",
        title="Xếp hạng TOPSIS các vùng kinh tế"
    )

    fig.update_layout(
        paper_bgcolor='#0f1117',
        plot_bgcolor='#1a1d27',
        font_color='#c0c4d0',
        height=550,
        xaxis_title="Vùng",
        yaxis_title="TOPSIS Score"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    # =========================
    # DOWNLOAD
    # =========================

    csv = ranking_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải bảng xếp hạng",
        csv,
        "topsis_expert_ranking.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.success(
        f"""
        Theo phương pháp TOPSIS với bộ trọng số chuyên gia,
        vùng có mức độ sẵn sàng AI cao nhất là:

        ► {top_region}

        Vùng này có sự kết hợp tốt giữa:

        • Digital Index

        • AI Readiness

        • Chất lượng nguồn nhân lực

        • Năng lực R&D

        Do đó đây là ứng viên phù hợp để ưu tiên
        triển khai trung tâm AI và sandbox dữ liệu
        trong giai đoạn đầu của chiến lược AI quốc gia.
        """
    )

# =========================
# TAB 3
# =========================

with tab3:

    st.markdown('### ⚖️ TOPSIS với trọng số Entropy')

    # =========================
    # DỮ LIỆU
    # =========================

    benefit_cols = [
        "GRDP",
        "FDI",
        "DigitalIndex",
        "AIReadiness",
        "LDDT",
        "R&D/GRDP",
        "Internet"
    ]

    cost_cols = [
        "Gini"
    ]

    criteria = benefit_cols + cost_cols

    X = df[criteria].values.astype(float)

    m, n = X.shape

    # =========================
    # CHUẨN HÓA ENTROPY
    # =========================

    X_entropy = X.copy()

    for j, col in enumerate(criteria):

        if col in benefit_cols:

            X_entropy[:, j] = (
                X[:, j] - X[:, j].min()
            ) / (
                X[:, j].max() - X[:, j].min()
            )

        else:

            X_entropy[:, j] = (
                X[:, j].max() - X[:, j]
            ) / (
                X[:, j].max() - X[:, j].min()
            )

    X_entropy = X_entropy + 1e-12

    # =========================
    # ENTROPY WEIGHTS
    # =========================

    P = X_entropy / X_entropy.sum(axis=0)

    k = 1 / np.log(m)

    entropy = -k * np.sum(
        P * np.log(P),
        axis=0
    )

    diversification = 1 - entropy

    entropy_weights = (
        diversification /
        diversification.sum()
    )

    # =========================
    # TOPSIS
    # =========================

    norm_matrix = X / np.sqrt(
        (X ** 2).sum(axis=0)
    )

    weighted_matrix = (
        norm_matrix *
        entropy_weights
    )

    ideal_best = np.zeros(n)
    ideal_worst = np.zeros(n)

    for i, col in enumerate(criteria):

        if col in benefit_cols:

            ideal_best[i] = weighted_matrix[:, i].max()
            ideal_worst[i] = weighted_matrix[:, i].min()

        else:

            ideal_best[i] = weighted_matrix[:, i].min()
            ideal_worst[i] = weighted_matrix[:, i].max()

    s_plus = np.sqrt(
        ((weighted_matrix - ideal_best) ** 2).sum(axis=1)
    )

    s_minus = np.sqrt(
        ((weighted_matrix - ideal_worst) ** 2).sum(axis=1)
    )

    ci_entropy = (
        s_minus /
        (s_plus + s_minus)
    )

    entropy_rank = pd.DataFrame({
        "Region": df["Regions"],
        "Entropy Score": ci_entropy
    })

    entropy_rank = entropy_rank.sort_values(
        "Entropy Score",
        ascending=False
    ).reset_index(drop=True)

    entropy_rank["Rank Entropy"] = (
        entropy_rank.index + 1
    )

    # =========================
    # SO SÁNH VỚI TAB 2
    # =========================

    compare_df = ranking_df.merge(
        entropy_rank,
        on="Region"
    )

    compare_df = compare_df[
        [
            "Region",
            "Rank",
            "Rank Entropy",
            "TOPSIS Score",
            "Entropy Score"
        ]
    ]

    compare_df["Thay đổi"] = (
        compare_df["Rank"]
        -
        compare_df["Rank Entropy"]
    )

    # =========================
    # KPI
    # =========================

    top_entropy = entropy_rank.iloc[0]["Region"]

    changed_regions = (
        compare_df["Thay đổi"] != 0
    ).sum()

    max_change = (
        compare_df["Thay đổi"]
        .abs()
        .max()
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Top 1 Entropy",
        top_entropy
    )

    col2.metric(
        "Vùng đổi hạng",
        int(changed_regions)
    )

    col3.metric(
        "Mức đổi hạng lớn nhất",
        int(max_change)
    )

    st.divider()

    # =========================
    # TRỌNG SỐ ENTROPY
    # =========================

    st.markdown('### 📋 Trọng số Entropy')

    weight_df = pd.DataFrame({
        "Tiêu chí": criteria,
        "Entropy Weight": entropy_weights
    })

    st.dataframe(
        weight_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BẢNG SO SÁNH
    # =========================

    st.markdown('### 📋 So sánh xếp hạng')

    st.dataframe(
        compare_df.round(4),
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    chart_df = compare_df.melt(
        id_vars="Region",
        value_vars=[
            "TOPSIS Score",
            "Entropy Score"
        ],
        var_name="Phương pháp",
        value_name="Score"
    )

    fig = px.bar(
        chart_df,
        x="Region",
        y="Score",
        color="Phương pháp",
        barmode="group",
        title="So sánh TOPSIS: Chuyên gia vs Entropy"
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
    # DOWNLOAD
    # =========================

    csv = compare_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải kết quả so sánh",
        csv,
        "entropy_vs_expert.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.info(
        f"""
        Khi sử dụng trọng số Entropy,
        vùng đứng đầu là:

        ► {top_entropy}

        Phương pháp Entropy xác định trọng số
        hoàn toàn dựa trên mức độ phân tán dữ liệu,
        không phụ thuộc đánh giá chủ quan của chuyên gia.

        Có {changed_regions} vùng thay đổi thứ hạng
        so với phương pháp trọng số chuyên gia.

        Điều này cho thấy kết quả TOPSIS
        có mức độ nhạy nhất định đối với cách xác định
        trọng số tiêu chí.
        """
    )

# =========================
# TAB 4
# =========================
with tab4:

    st.markdown('### 📈 Phân tích độ nhạy trọng số AI Readiness')

    # =========================
    # DỮ LIỆU
    # =========================

    benefit_cols = [
        "GRDP",
        "FDI",
        "DigitalIndex",
        "AIReadiness",
        "LDDT",
        "R&D/GRDP",
        "Internet"
    ]

    cost_cols = ["Gini"]

    criteria = benefit_cols + cost_cols

    X = df[criteria].values.astype(float)

    # =========================
    # CHUẨN HÓA VECTOR
    # =========================

    norm_matrix = X / np.sqrt(
        (X ** 2).sum(axis=0)
    )

    # =========================
    # BỘ TRỌNG SỐ GỐC
    # =========================

    base_weights = np.array([
        0.10,
        0.10,
        0.15,
        0.20,
        0.15,
        0.15,
        0.05,
        0.10
    ])

    ai_weights = np.arange(
        0.10,
        0.45,
        0.05
    )

    results = []

    # =========================
    # LOOP ĐỘ NHẠY
    # =========================

    for w_ai in ai_weights:

        new_weights = base_weights.copy()

        # AIReadiness nằm vị trí thứ 4
        new_weights[3] = w_ai

        # chuẩn hóa tổng = 1
        new_weights = (
            new_weights /
            new_weights.sum()
        )

        weighted_matrix = (
            norm_matrix *
            new_weights
        )

        ideal_best = np.zeros(len(criteria))
        ideal_worst = np.zeros(len(criteria))

        for i, col in enumerate(criteria):

            if col in benefit_cols:

                ideal_best[i] = weighted_matrix[:, i].max()
                ideal_worst[i] = weighted_matrix[:, i].min()

            else:

                ideal_best[i] = weighted_matrix[:, i].min()
                ideal_worst[i] = weighted_matrix[:, i].max()

        s_plus = np.sqrt(
            ((weighted_matrix - ideal_best) ** 2)
            .sum(axis=1)
        )

        s_minus = np.sqrt(
            ((weighted_matrix - ideal_worst) ** 2)
            .sum(axis=1)
        )

        score = (
            s_minus /
            (s_plus + s_minus)
        )

        rank_df = pd.DataFrame({
            "Region": df["Regions"],
            "Score": score
        })

        rank_df = rank_df.sort_values(
            "Score",
            ascending=False
        )

        top3 = (
            rank_df["Region"]
            .head(3)
            .tolist()
        )

        results.append({
            "w_AI": round(w_ai, 2),
            "Top 1": top3[0],
            "Top 2": top3[1],
            "Top 3": top3[2]
        })

    sensitivity_df = pd.DataFrame(results)

    # =========================
    # KPI
    # =========================

    unique_top3 = (
        sensitivity_df[
            ["Top 1", "Top 2", "Top 3"]
        ]
        .drop_duplicates()
        .shape[0]
    )

    stable = (
        "Có"
        if unique_top3 == 1
        else "Không"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số kịch bản",
        len(ai_weights)
    )

    col2.metric(
        "Top-3 ổn định?",
        stable
    )

    col3.metric(
        "Số cấu trúc Top-3",
        unique_top3
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
    # HEATMAP
    # =========================

    region_list = df["Regions"].tolist()

    heatmap_rows = []

    for _, row in sensitivity_df.iterrows():

        w = row["w_AI"]

        for rank_pos in [1, 2, 3]:

            heatmap_rows.append({
                "w_AI": w,
                "Rank": rank_pos,
                "Region": row[f"Top {rank_pos}"]
            })

    heatmap_df = pd.DataFrame(
        heatmap_rows
    )

    region_map = {
        region: i + 1
        for i, region
        in enumerate(region_list)
    }

    heatmap_df["Code"] = (
        heatmap_df["Region"]
        .map(region_map)
    )

    fig = px.imshow(
        heatmap_df.pivot(
            index="Rank",
            columns="w_AI",
            values="Code"
        ),
        aspect="auto",
        title="Độ ổn định Top-3 khi thay đổi w_AI"
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

    csv = sensitivity_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải kết quả độ nhạy",
        csv,
        "ai_weight_sensitivity.csv",
        "text/csv"
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    if stable == "Có":

        st.success(
            """
            Top-3 vùng hoàn toàn ổn định
            khi thay đổi trọng số AI Readiness.

            Điều này cho thấy kết quả TOPSIS
            có độ vững cao và không phụ thuộc
            quá nhiều vào đánh giá chủ quan
            về tiêu chí AI.
            """
        )

    else:

        st.warning(
            """
            Top-3 thay đổi khi điều chỉnh
            trọng số AI Readiness.

            Điều này cho thấy tiêu chí AI
            có ảnh hưởng đáng kể tới kết quả
            xếp hạng và cần được cân nhắc kỹ
            khi xây dựng chính sách.
            """
        )

# =========================
# TAB 5
# =========================
with tab5:

    st.markdown('### ⚖️ So sánh AHP và TOPSIS')

    # =========================
    # DỮ LIỆU
    # =========================

    benefit_cols = [
        "GRDP",
        "FDI",
        "DigitalIndex",
        "AIReadiness",
        "LDDT",
        "R&D/GRDP",
        "Internet"
    ]

    cost_cols = ["Gini"]

    criteria = benefit_cols + cost_cols

    X = df[criteria].values.astype(float)

    # =========================
    # CHUẨN HÓA VECTOR
    # =========================

    norm_matrix = X / np.sqrt(
        (X ** 2).sum(axis=0)
    )

    # =========================
    # TRỌNG SỐ AHP GIẢ ĐỊNH
    # =========================

    ahp_weights = np.array([
        0.08,   # GRDP
        0.08,   # FDI
        0.15,   # DigitalIndex
        0.25,   # AIReadiness
        0.12,   # LDDT
        0.15,   # R&D
        0.07,   # Internet
        0.10    # Gini
    ])

    ahp_weights = (
        ahp_weights /
        ahp_weights.sum()
    )

    # =========================
    # TOPSIS + AHP WEIGHT
    # =========================

    weighted_matrix = (
        norm_matrix *
        ahp_weights
    )

    ideal_best = np.zeros(len(criteria))
    ideal_worst = np.zeros(len(criteria))

    for i, col in enumerate(criteria):

        if col in benefit_cols:

            ideal_best[i] = weighted_matrix[:, i].max()
            ideal_worst[i] = weighted_matrix[:, i].min()

        else:

            ideal_best[i] = weighted_matrix[:, i].min()
            ideal_worst[i] = weighted_matrix[:, i].max()

    s_plus = np.sqrt(
        ((weighted_matrix - ideal_best) ** 2)
        .sum(axis=1)
    )

    s_minus = np.sqrt(
        ((weighted_matrix - ideal_worst) ** 2)
        .sum(axis=1)
    )

    ahp_score = (
        s_minus /
        (s_plus + s_minus)
    )

    ahp_df = pd.DataFrame({
        "Region": df["Regions"],
        "AHP Score": ahp_score
    })

    ahp_df["Rank AHP"] = (
        ahp_df["AHP Score"]
        .rank(
            ascending=False,
            method="dense"
        )
        .astype(int)
    )

    ahp_df = ahp_df.sort_values(
        "Rank AHP"
    )

    # =========================
    # KPI
    # =========================

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Số vùng",
        len(df)
    )

    col2.metric(
        "Top 1",
        ahp_df.iloc[0]["Region"]
    )

    col3.metric(
        "Điểm cao nhất",
        f"{ahp_df.iloc[0]['AHP Score']:.3f}"
    )

    st.divider()

    # =========================
    # BẢNG KẾT QUẢ
    # =========================

    st.markdown('### 📋 Xếp hạng theo AHP')

    st.dataframe(
        ahp_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # BIỂU ĐỒ
    # =========================

    fig = px.bar(
        ahp_df,
        x="Region",
        y="AHP Score",
        text_auto=".3f",
        title="Xếp hạng vùng theo AHP"
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

    csv = ahp_df.to_csv(
        index=False
    )

    st.download_button(
        "📥 Tải kết quả AHP",
        csv,
        "ahp_ranking.csv",
        "text/csv"
    )

    # =========================
    # SO SÁNH VỚI TOPSIS
    # =========================

    compare_df = pd.DataFrame({
        "Region": ranking_df["Region"],
        "Rank TOPSIS": ranking_df["Rank"],
        "Rank AHP": ahp_df.set_index(
            "Region"
        ).loc[
            ranking_df["Region"],
            "Rank AHP"
        ].values
    })

    st.markdown('### 🔍 So sánh thứ hạng')

    st.dataframe(
        compare_df,
        use_container_width=True,
        hide_index=True
    )

    # =========================
    # KẾT LUẬN
    # =========================

    st.markdown('### 📌 Kết luận quản trị')

    st.info(
        """
        Phương pháp TOPSIS đánh giá
        mức độ gần với phương án lý tưởng,
        trong khi AHP phản ánh ưu tiên
        của chuyên gia thông qua trọng số.

        Nếu thứ hạng giữa hai phương pháp
        tương đối giống nhau thì kết quả
        có tính ổn định cao.

        Nếu có sự khác biệt lớn,
        điều đó cho thấy kết quả phụ thuộc
        đáng kể vào cách xác định trọng số
        và cần xem xét thêm trong quá trình
        hoạch định chính sách.
        """
    )

# =========================
# TAB 6
# =========================

with tab6:

    st.markdown('### 💬 Thảo luận chính sách')

    st.markdown("""
### a) Vùng nào dẫn đầu theo TOPSIS với trọng số chuyên gia? Đây có phải vùng nên triển khai trung tâm AI quốc gia đầu tiên không?

Theo kết quả TOPSIS với bộ trọng số chuyên gia, **Đông Nam Bộ** là vùng có điểm số cao nhất và đứng đầu bảng xếp hạng.

Kết quả này là hợp lý vì vùng Đông Nam Bộ có:

- GRDP bình quân đầu người cao.
- Chỉ số Digital Index và AI Readiness thuộc nhóm dẫn đầu.
- Hệ sinh thái doanh nghiệp công nghệ phát triển.
- Khả năng thu hút đầu tư và nhân lực chất lượng cao.

Do đó, Đông Nam Bộ là ứng viên phù hợp để triển khai trung tâm AI quốc gia đầu tiên. Tuy nhiên, việc lựa chọn cuối cùng không nên chỉ dựa trên tiêu chí kinh tế mà còn cần xem xét yếu tố chiến lược quốc gia và cân bằng phát triển vùng.

---

### b) Khi dùng trọng số Entropy, vùng nào có sự thay đổi xếp hạng lớn nhất? Vì sao?

Việc sử dụng trọng số Entropy có thể làm thay đổi điểm TOPSIS của các vùng do trọng số được xác định khách quan từ dữ liệu thay vì từ đánh giá chuyên gia.

Vùng có mức thay đổi thứ hạng lớn nhất thường là những vùng có kết quả nổi bật ở một số tiêu chí nhưng không đồng đều trên toàn bộ hệ tiêu chí. Khi Entropy tăng trọng số cho các tiêu chí có độ phân tán lớn, lợi thế hoặc bất lợi của các vùng này sẽ được khuếch đại.

Nếu kết quả thực nghiệm cho thấy thứ hạng gần như không thay đổi, điều đó cho thấy mô hình có tính ổn định cao và không quá nhạy cảm với phương pháp xác định trọng số.

---

### c) TOPSIS giả định độc lập tuyến tính giữa các tiêu chí. Trong thực tế, AI Readiness và Internet penetration có thể tương quan rất cao. Điều này ảnh hưởng đến kết quả như thế nào? Đề xuất cách xử lý.

Nếu hai tiêu chí có tương quan rất cao, mô hình TOPSIS có thể vô tình "đếm hai lần" cùng một lợi thế.

Ví dụ:

- Vùng có hạ tầng Internet tốt thường cũng có AI Readiness cao.
- Khi cả hai tiêu chí đều được đưa vào với trọng số đáng kể, lợi thế của vùng đó có thể bị khuếch đại quá mức.

Điều này làm giảm tính khách quan của kết quả xếp hạng.

Một số hướng xử lý:

- Kiểm tra hệ số tương quan giữa các tiêu chí trước khi áp dụng TOPSIS.
- Loại bỏ hoặc gộp các tiêu chí có tương quan quá cao.
- Sử dụng PCA (Principal Component Analysis) để giảm chiều dữ liệu.
- Điều chỉnh trọng số nhằm hạn chế hiện tượng trùng lặp thông tin.

---

### d) Theo Quyết định 127/QĐ-TTg, Việt Nam đặt mục tiêu xây dựng 3 trung tâm AI lớn. Em sẽ chọn 3 vùng nào dựa trên kết quả TOPSIS? Có cần điều chỉnh thêm tiêu chí địa - chính trị không?

Nếu căn cứ hoàn toàn vào kết quả TOPSIS, ba vùng nên được ưu tiên gồm:

1. Đông Nam Bộ.
2. Đồng bằng sông Hồng.
3. Bắc Trung Bộ và Duyên hải Trung Bộ.

Ba vùng này có mức độ sẵn sàng AI cao nhất và sở hữu nhiều lợi thế về hạ tầng số, nguồn nhân lực, kết nối Internet và năng lực đổi mới sáng tạo.

Tuy nhiên, trong hoạch định chính sách quốc gia cần bổ sung thêm các yếu tố ngoài mô hình định lượng như:

- An ninh quốc gia.
- Khả năng kết nối liên vùng.
- Cân bằng phát triển giữa các khu vực.
- Phân bố dân cư và nguồn nhân lực dài hạn.
- Khả năng dự phòng rủi ro thiên tai hoặc gián đoạn hạ tầng.

Vì vậy, kết quả TOPSIS nên được xem là công cụ hỗ trợ ra quyết định, còn quyết định cuối cùng cần kết hợp với các tiêu chí địa - chính trị và chiến lược phát triển quốc gia.
""")

st.divider()

st.caption("""
Nguồn dữ liệu:
• Quyết định 127/QĐ-TTg về Chiến lược quốc gia nghiên cứu, phát triển và ứng dụng AI đến năm 2030
""")