"""
Bài 11 — Học tăng cường (Q-learning) cho chính sách kinh tế thích nghi
AIDEOM-VN Dashboard | Streamlit
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import time

# ─────────────────────────────────────────────
# CẤU HÌNH TRANG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bài 11 — Q-learning | AIDEOM-VN",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CSS DARK THEME
# ─────────────────────────────────────────────
st.markdown("""
<style>
/* Nền tổng thể */
[data-testid="stAppViewContainer"] {
    background-color: #0f1117;
    color: #e0e0e0;
}
[data-testid="stSidebar"] {
    background-color: #1a1d27;
    border-right: 1px solid #2d2f3e;
}
[data-testid="stSidebar"] * {
    color: #c0c4d0 !important;
}

/* Card metric */
.metric-card {
    background: #1e2130;
    border: 1px solid #2d3250;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.metric-card .label {
    font-size: 0.78rem;
    color: #8b8fa8;
    margin-bottom: 4px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}
.metric-card .value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #a78bfa;
}
.metric-card .delta {
    font-size: 0.8rem;
    color: #6ee7b7;
    margin-top: 2px;
}

/* Badge cấp độ */
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.04em;
    margin-right: 6px;
}
.badge-hard   { background:#7c3aed22; color:#a78bfa; border:1px solid #7c3aed55; }
.badge-rl     { background:#0ea5e922; color:#38bdf8; border:1px solid #0ea5e955; }
.badge-info   { background:#10b98122; color:#6ee7b7; border:1px solid #10b98155; }

/* Thông báo kết quả */
.result-box {
    background: #14532d33;
    border: 1px solid #16a34a66;
    border-radius: 8px;
    padding: 10px 16px;
    color: #86efac;
    font-size: 0.9rem;
}
.warn-box {
    background: #78350f33;
    border: 1px solid #d9770666;
    border-radius: 8px;
    padding: 10px 16px;
    color: #fcd34d;
    font-size: 0.9rem;
}

/* Bảng */
.stDataFrame { border-radius: 8px; overflow: hidden; }

/* Nút */
.stButton > button {
    background: linear-gradient(135deg,#7c3aed,#6d28d9);
    color: white;
    border: none;
    border-radius: 8px;
    padding: 8px 22px;
    font-weight: 600;
    font-size: 0.95rem;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }

/* Slider label */
[data-testid="stSlider"] label { color: #a0a4b8 !important; font-size:0.85rem; }

h2, h3 { color: #e2e8f0 !important; }
hr { border-color: #2d2f3e; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# DỮ LIỆU HÀNH ĐỘNG (từ file Action_at.csv)
# ─────────────────────────────────────────────
ACTIONS = {
    0: {"name": "a0 · Truyền thống", "alloc": np.array([0.70, 0.10, 0.10, 0.10])},
    1: {"name": "a1 · Cân bằng",     "alloc": np.array([0.40, 0.25, 0.15, 0.20])},
    2: {"name": "a2 · Số hóa nhanh", "alloc": np.array([0.25, 0.45, 0.15, 0.15])},
    3: {"name": "a3 · AI dẫn dắt",   "alloc": np.array([0.20, 0.20, 0.45, 0.15])},
    4: {"name": "a4 · Bao trùm",     "alloc": np.array([0.30, 0.20, 0.10, 0.40])},
}
N_ACTIONS = 5

# Trọng số reward
W = np.array([0.40, 0.25, 0.20, 0.15])  # GDP, unemployment, cyber, emission

# Tham số Cobb-Douglas (từ Bài 1)
ALPHA = 0.33; BETA_L = 0.42; GAMMA_D = 0.10; DELTA_AI = 0.08; THETA_H = 0.07
BUDGET_ANNUAL = 1000   # nghìn tỷ VND / năm

# Trạng thái ban đầu VN 2026 (thực tế)
INIT_STATE_VN2026 = np.array([1, 1, 0, 1])  # GDP=medium, D=medium, AI=low, U=medium


# ─────────────────────────────────────────────
# MÔI TRƯỜNG MDP
# ─────────────────────────────────────────────
class VietnamEconomyEnv:
    """
    MDP đơn giản hóa nền kinh tế Việt Nam.
    Trạng thái: (GDP_growth, Digital_index, AI_capacity, Unemploy_risk)
    Mỗi chiều: 0=low, 1=medium, 2=high  → 3^4 = 81 trạng thái
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)
        self.T = 10  # 10 năm / episode
        self.reset()

    def reset(self):
        self.state = INIT_STATE_VN2026.copy()
        self.t = 0
        # Giá trị liên tục
        self.K  = 27_500.0   # nghìn tỷ VND
        self.D  = 20.3        # % GDP
        self.AI = 86.0        # nghìn DN số
        self.H  = 30.0        # % lao động qua ĐT
        self.A  = 1.0         # TFP (chuẩn hóa)
        self.L  = 53.9        # triệu lao động (cố định nhẹ)
        return self.state.copy()

    def _compute_Y(self):
        K = max(self.K, 1); D = max(self.D, 0.1)
        AI = max(self.AI, 0.1); H = max(self.H, 0.1)
        return self.A * (K ** ALPHA) * (self.L ** BETA_L) * \
               (D ** GAMMA_D) * (AI ** DELTA_AI) * (H ** THETA_H)

    def step(self, action: int):
        alloc = ACTIONS[action]["alloc"]
        B = BUDGET_ANNUAL

        # Cập nhật vốn
        dK  = alloc[0] * B
        dD  = alloc[1] * B / 500.0
        dAI = alloc[2] * B / 20.0
        dH  = alloc[3] * B / 200.0

        Y_prev = self._compute_Y()

        self.K  = (1 - 0.05)  * self.K  + dK
        self.D  = (1 - 0.12)  * self.D  + dD
        self.AI = (1 - 0.15)  * self.AI + dAI
        self.H  = self.H + 0.8 * dH - 0.02 * self.H
        # TFP nội sinh
        self.A  = self.A * (1 + 0.003 * self.D + 0.002 * self.AI + 0.004 * self.H) / 100

        Y_new = self._compute_Y()
        gdp_growth = (Y_new - Y_prev) / (Y_prev + 1e-9)

        # ── Reward ──────────────────────────────
        delta_gdp    =  min(gdp_growth * 10, 3.0)
        delta_unemp  = -0.05 * alloc[2]          # AI tăng → rủi ro việc làm
        cyber_risk   = -0.08 * alloc[2] + 0.04 * alloc[3]
        emission     = -0.06 * (alloc[0] + alloc[2])
        reward = (W[0] * delta_gdp +
                  W[1] * delta_unemp +
                  W[2] * cyber_risk  +
                  W[3] * emission)

        # ── Cập nhật trạng thái rời rạc ─────────
        def disc(val, low, high):
            if val < low: return 0
            if val < high: return 1
            return 2

        new_gdp = disc(gdp_growth * 100, 5, 8)
        new_d   = disc(self.D, 15, 22)
        new_ai  = disc(self.AI, 70, 90)
        # unemployment risk: AI cao → tăng rủi ro
        u_raw   = 0.4 - alloc[2] * 0.3 + alloc[3] * 0.1
        new_u   = disc(u_raw, 0.2, 0.35)

        self.state = np.array([new_gdp, new_d, new_ai, new_u])
        self.t += 1
        done = self.t >= self.T
        return self.state.copy(), reward, done


# ─────────────────────────────────────────────
# Q-LEARNING
# ─────────────────────────────────────────────
def train_qlearning(n_episodes: int, alpha: float, gamma: float,
                    seed: int, progress_bar) -> dict:
    env = VietnamEconomyEnv(seed=seed)
    rng = np.random.default_rng(seed)
    Q = np.zeros((3, 3, 3, 3, N_ACTIONS))

    ep_rewards = []
    eps_start, eps_end = 1.0, 0.05

    for ep in range(n_episodes):
        s = env.reset()
        total_r = 0.0
        eps = max(eps_end, eps_start - (eps_start - eps_end) * ep / max(n_episodes * 0.8, 1))

        while True:
            if rng.random() < eps:
                a = rng.integers(0, N_ACTIONS)
            else:
                a = int(np.argmax(Q[tuple(s)]))

            s2, r, done = env.step(a)
            best_next = np.max(Q[tuple(s2)])
            Q[tuple(s) + (a,)] += alpha * (r + gamma * best_next - Q[tuple(s) + (a,)])
            s = s2
            total_r += r
            if done:
                break

        ep_rewards.append(total_r)

        # Cập nhật progress mỗi 200 episode
        if (ep + 1) % max(1, n_episodes // 50) == 0:
            progress_bar.progress((ep + 1) / n_episodes)

    return {"Q": Q, "rewards": ep_rewards}


# ─────────────────────────────────────────────
# ĐÁNH GIÁ RULE-BASED
# ─────────────────────────────────────────────
def evaluate_policy(policy_fn, n_eval: int = 200, seed: int = 42) -> list:
    rewards = []
    for i in range(n_eval):
        env = VietnamEconomyEnv(seed=seed + i)
        s = env.reset()
        total = 0.0
        while True:
            a = policy_fn(s)
            s, r, done = env.step(a)
            total += r
            if done:
                break
        rewards.append(total)
    return rewards


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🇻🇳 AIDEOM-VN")
    st.caption("Mô hình ra quyết định phát triển kinh tế VN trong kỉ nguyên AI")
    
    st.divider()
    st.caption("📂 Dữ liệu: NSO, MoST, MIC, MPI, WB, GII 2025")
    st.caption("⚙️ Tools: Python, Streamlit, PuLP,..")
    st.caption("📘 Dựa trên giáo trình AIDEOM-VN 2026")


# ─────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────
st.markdown("## 🤖 Bài 11 — Q-learning cho chính sách kinh tế thích nghi")

st.markdown("""
<span class='badge badge-hard'>CẤP ĐỘ KHÓ</span>
<span class='badge badge-rl'>RL tabular</span>
<span class='badge badge-info'>81 trạng thái × 5 hành động</span>
""", unsafe_allow_html=True)

st.markdown("""
**MDP:** trạng thái = (GDP_growth, D, AI, Unemploy_risk) × 3 mức = 81 trạng thái.

**5 hành động:** a0 truyền thống · a1 cân bằng · a2 số hóa nhanh · a3 AI dẫn dắt · a4 bao trùm.

**Reward:** $R = w_1\\Delta GDP - w_2\\Delta U - w_3 CyberRisk - w_4 Emission,\\; w = (0.40, 0.25, 0.20, 0.15)$.
""")

st.divider()

# ─── Tab layout ───────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "⚙️ Huấn luyện",
    "📊 Kết quả chính sách",
    "🔄 So sánh rule-based",
    "📋 Chi tiết Q-table",
    "💬 Thảo luận chính sách",
])


# ════════════════════════════════════════════
# TAB 1 — HUẤN LUYỆN
# ════════════════════════════════════════════
with tab1:
    st.markdown("### 🎛️ Tham số huấn luyện")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        n_episodes = st.slider("Số episode", 500, 20_000, 3_000, step=500)
    with col2:
        alpha = st.slider("α (learning rate)", 0.01, 0.50, 0.10, step=0.01)
    with col3:
        gamma = st.slider("γ (discount)", 0.80, 0.99, 0.95, step=0.01)
    with col4:
        seed = st.number_input("Seed", min_value=0, max_value=9999, value=42, step=1)

    st.markdown("**Epsilon-greedy:** giảm tuyến tính từ 1.0 → 0.05 trong 80% đầu số episode.")

    btn_train = st.button("🚀 Train Q-learning")

    if btn_train:
        prog = st.progress(0)
        status = st.empty()
        status.info("⏳ Đang huấn luyện…")

        t0 = time.time()
        result = train_qlearning(n_episodes, alpha, gamma, int(seed), prog)
        elapsed = time.time() - t0

        st.session_state["q_result"] = result
        st.session_state["train_params"] = {
            "n_episodes": n_episodes, "alpha": alpha,
            "gamma": gamma, "seed": int(seed)
        }

        rewards = result["rewards"]
        mean_last = np.mean(rewards[-100:])
        prog.progress(1.0)
        status.markdown(
            f"<div class='result-box'>✅ Done. Mean reward 100 ep cuối: "
            f"<b>{mean_last:.4f}</b> &nbsp;|&nbsp; ⏱ {elapsed:.1f}s</div>",
            unsafe_allow_html=True
        )

    # Learning curve
    if "q_result" in st.session_state:
        rewards = st.session_state["q_result"]["rewards"]
        st.markdown("### 📈 Learning curve")

        window = max(1, len(rewards) // 50)
        smoothed = pd.Series(rewards).rolling(window, min_periods=1).mean().tolist()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=rewards, mode="lines",
            line=dict(color="#4b5563", width=0.8),
            name="Per-episode"
        ))
        fig.add_trace(go.Scatter(
            y=smoothed, mode="lines",
            line=dict(color="#f87171", width=2.0),
            name="Smoothed"
        ))
        fig.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0",
            xaxis=dict(title="Episode", gridcolor="#2d2f3e"),
            yaxis=dict(title="Cumulative Reward", gridcolor="#2d2f3e"),
            legend=dict(bgcolor="#1e2130", bordercolor="#2d3250"),
            height=360, margin=dict(l=50, r=20, t=20, b=50)
        )
        st.plotly_chart(fig, use_container_width=True)

    else:
        st.markdown(
            "<div class='warn-box'>⚠️ Chưa có kết quả. Nhấn <b>Train Q-learning</b> để bắt đầu.</div>",
            unsafe_allow_html=True
        )


# ════════════════════════════════════════════
# TAB 2 — KẾT QUẢ CHÍNH SÁCH
# ════════════════════════════════════════════
with tab2:
    if "q_result" not in st.session_state:
        st.markdown(
            "<div class='warn-box'>⚠️ Hãy Train Q-learning ở tab 1 trước.</div>",
            unsafe_allow_html=True
        )
    else:
        Q = st.session_state["q_result"]["Q"]

        st.markdown("### 🎯 Chính sách π*(s) — trạng thái thực tế VN 2026")

        # Trạng thái khởi đầu VN 2026
        s0 = tuple(INIT_STATE_VN2026)
        best_a = int(np.argmax(Q[s0]))
        q_vals = Q[s0]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Hành động tối ưu π*(s₀)</div>
                <div class='value'>{ACTIONS[best_a]['name']}</div>
                <div class='delta'>VN 2026: GDP=mid, D=mid, AI=low, U=mid</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Q-value tối ưu</div>
                <div class='value'>{q_vals[best_a]:.4f}</div>
                <div class='delta'>Tại trạng thái (1,1,0,1)</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            params = st.session_state["train_params"]
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Episodes đã train</div>
                <div class='value'>{params['n_episodes']:,}</div>
                <div class='delta'>α={params['alpha']} · γ={params['gamma']}</div>
            </div>""", unsafe_allow_html=True)

        # Q-values tại s0 dạng bar chart
        st.markdown("#### Q-values tại trạng thái VN 2026")
        action_names = [ACTIONS[i]["name"] for i in range(N_ACTIONS)]
        colors = ["#a78bfa" if i == best_a else "#4b5680" for i in range(N_ACTIONS)]

        fig2 = go.Figure(go.Bar(
            x=action_names, y=q_vals,
            marker_color=colors,
            text=[f"{v:.4f}" for v in q_vals],
            textposition="outside"
        ))
        fig2.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0",
            xaxis=dict(gridcolor="#2d2f3e"),
            yaxis=dict(title="Q-value", gridcolor="#2d2f3e"),
            height=320, margin=dict(l=50, r=20, t=30, b=80),
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

        # Bảng 5 trạng thái đặc trưng
        st.markdown("#### 🗂️ Chính sách π*(s) tại 5 trạng thái đặc trưng")
        scenarios = {
            "VN 2026 (thực tế)":         (1, 1, 0, 1),
            "Khủng hoảng (GDP thấp)":     (0, 0, 0, 2),
            "Bùng nổ số (D cao, AI cao)": (2, 2, 2, 0),
            "Tăng trưởng cao":            (2, 1, 1, 0),
            "AI thấp, nhân lực yếu":      (0, 0, 0, 1),
        }
        rows = []
        for desc, s in scenarios.items():
            a_star = int(np.argmax(Q[s]))
            alloc  = ACTIONS[a_star]["alloc"]
            rows.append({
                "Trạng thái": desc,
                "s = (G,D,AI,U)": str(s),
                "Hành động π*(s)": ACTIONS[a_star]["name"],
                "K (%)": f"{alloc[0]*100:.0f}%",
                "D (%)": f"{alloc[1]*100:.0f}%",
                "AI (%)": f"{alloc[2]*100:.0f}%",
                "H (%)": f"{alloc[3]*100:.0f}%",
            })
        df_policy = pd.DataFrame(rows)
        st.dataframe(df_policy, use_container_width=True, hide_index=True)

        # Phân phối hành động trên toàn bộ 81 trạng thái
        st.markdown("#### 📊 Phân phối hành động trên 81 trạng thái")
        all_actions = []
        for g in range(3):
            for d in range(3):
                for ai in range(3):
                    for u in range(3):
                        all_actions.append(int(np.argmax(Q[g, d, ai, u])))

        counts = [all_actions.count(i) for i in range(N_ACTIONS)]
        fig3 = go.Figure(go.Pie(
            labels=action_names,
            values=counts,
            hole=0.45,
            marker_colors=["#6366f1", "#22d3ee", "#f59e0b", "#a78bfa", "#34d399"]
        ))
        fig3.update_layout(
            paper_bgcolor="#0f1117",
            font_color="#c0c4d0",
            height=320,
            legend=dict(bgcolor="#1e2130"),
            margin=dict(l=20, r=20, t=20, b=20)
        )
        st.plotly_chart(fig3, use_container_width=True)


# ════════════════════════════════════════════
# TAB 3 — SO SÁNH RULE-BASED
# ════════════════════════════════════════════
with tab3:
    if "q_result" not in st.session_state:
        st.markdown(
            "<div class='warn-box'>⚠️ Hãy Train Q-learning ở tab 1 trước.</div>",
            unsafe_allow_html=True
        )
    else:
        Q = st.session_state["q_result"]["Q"]
        params = st.session_state["train_params"]

        st.markdown("### 🔄 So sánh Q-learning với các chính sách rule-based")
        st.caption("Đánh giá trên 200 episode độc lập, trạng thái khởi đầu VN 2026.")

        with st.spinner("Đang đánh giá 4 chính sách…"):
            # Q-learning policy
            def pi_qlearning(s):
                return int(np.argmax(Q[tuple(s)]))

            # Rule-based
            def pi_a1(s): return 1
            def pi_a3(s): return 3
            def pi_random(s): return np.random.randint(0, N_ACTIONS)

            seed_eval = params["seed"]
            r_q  = evaluate_policy(pi_qlearning, seed=seed_eval)
            r_a1 = evaluate_policy(pi_a1,        seed=seed_eval)
            r_a3 = evaluate_policy(pi_a3,        seed=seed_eval)
            r_rnd= evaluate_policy(pi_random,    seed=seed_eval)

        policies = {
            "Q-learning π*":  r_q,
            "Luôn a1 (cân bằng)": r_a1,
            "Luôn a3 (AI)":   r_a3,
            "Random":         r_rnd,
        }

        # Metric cards
        cols = st.columns(4)
        colors_m = ["#a78bfa", "#38bdf8", "#f59e0b", "#94a3b8"]
        for i, (name, rw) in enumerate(policies.items()):
            with cols[i]:
                mean_r = np.mean(rw)
                std_r  = np.std(rw)
                st.markdown(f"""<div class='metric-card'>
                    <div class='label'>{name}</div>
                    <div class='value' style='color:{colors_m[i]}'>{mean_r:.4f}</div>
                    <div class='delta'>± {std_r:.4f}</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Box plot
        fig4 = go.Figure()
        for (name, rw), col in zip(policies.items(), colors_m):
            fig4.add_trace(go.Box(
                y=rw, name=name,
                marker_color=col,
                boxmean="sd",
                line_color=col
            ))
        fig4.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0",
            yaxis=dict(title="Cumulative Reward / episode", gridcolor="#2d2f3e"),
            xaxis=dict(gridcolor="#2d2f3e"),
            height=380, margin=dict(l=60, r=20, t=30, b=60),
            showlegend=False
        )
        st.plotly_chart(fig4, use_container_width=True)

        # Bảng tổng hợp
        summary = []
        for name, rw in policies.items():
            summary.append({
                "Chính sách": name,
                "Mean reward": f"{np.mean(rw):.5f}",
                "Std":         f"{np.std(rw):.5f}",
                "Min":         f"{np.min(rw):.5f}",
                "Max":         f"{np.max(rw):.5f}",
                "% tốt hơn Random": f"{(np.mean(rw) - np.mean(r_rnd)) / (abs(np.mean(r_rnd)) + 1e-9) * 100:.1f}%",
            })
        st.dataframe(pd.DataFrame(summary), use_container_width=True, hide_index=True)


# ════════════════════════════════════════════
# TAB 4 — CHI TIẾT Q-TABLE
# ════════════════════════════════════════════
with tab4:
    if "q_result" not in st.session_state:
        st.markdown(
            "<div class='warn-box'>⚠️ Hãy Train Q-learning ở tab 1 trước.</div>",
            unsafe_allow_html=True
        )
    else:
        Q = st.session_state["q_result"]["Q"]

        st.markdown("### 📋 Chi tiết Q-table (81 trạng thái)")
        st.caption("Chọn slice theo giá trị GDP_growth và Digital_index để xem Q-values.")

        col1, col2 = st.columns(2)
        with col1:
            g_filter = st.selectbox("GDP_growth", [0, 1, 2],
                                    format_func=lambda x: ["Low","Medium","High"][x], index=1)
        with col2:
            d_filter = st.selectbox("Digital_index", [0, 1, 2],
                                    format_func=lambda x: ["Low","Medium","High"][x], index=1)

        rows = []
        lvl = ["Low", "Medium", "High"]
        action_names = [ACTIONS[i]["name"] for i in range(N_ACTIONS)]
        for ai in range(3):
            for u in range(3):
                q_row = Q[g_filter, d_filter, ai, u]
                best  = int(np.argmax(q_row))
                row = {
                    "AI_capacity": lvl[ai],
                    "Unemploy_risk": lvl[u],
                    "π*(s)": ACTIONS[best]["name"],
                }
                for i, an in enumerate(action_names):
                    row[an] = f"{q_row[i]:.5f}"
                rows.append(row)

        df_q = pd.DataFrame(rows)
        st.dataframe(df_q, use_container_width=True, hide_index=True)

        # Heatmap Q-value của hành động tối ưu
        st.markdown("#### 🌡️ Heatmap Q*(s) — Giá trị Q tối ưu")
        z_matrix = np.zeros((3, 3))
        for ai in range(3):
            for u in range(3):
                z_matrix[ai, u] = np.max(Q[g_filter, d_filter, ai, u])

        fig5 = go.Figure(go.Heatmap(
            z=z_matrix,
            x=["U=Low", "U=Medium", "U=High"],
            y=["AI=Low", "AI=Medium", "AI=High"],
            colorscale="Viridis",
            text=[[f"{z_matrix[ai,u]:.4f}" for u in range(3)] for ai in range(3)],
            texttemplate="%{text}",
        ))
        fig5.update_layout(
            paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
            font_color="#c0c4d0",
            xaxis_title="Unemployment risk",
            yaxis_title="AI capacity",
            height=320, margin=dict(l=80, r=20, t=30, b=60)
        )
        st.plotly_chart(fig5, use_container_width=True)

        st.markdown("#### 📌 Ghi chú giải thích")
        st.markdown("""
| Ký hiệu | Ý nghĩa |
|---------|---------|
| **G** | GDP_growth: 0=Low (<5%), 1=Medium (5–8%), 2=High (>8%) |
| **D** | Digital_index: 0=Low (<15%), 1=Medium (15–22%), 2=High (>22%) |
| **AI** | AI_capacity: 0=Low (<70k DN), 1=Medium (70–90k), 2=High (>90k) |
| **U** | Unemploy_risk: 0=Low, 1=Medium, 2=High |
| **π*(s)** | Hành động tối ưu theo Q-learning tại trạng thái s |
""")

        st.markdown("""
> **Lưu ý học thuật:** Mô hình này mang tính minh họa kỹ thuật. 
> Theo bài báo nguồn (Mục 11), **AI hỗ trợ ra quyết định, không thay thế trách nhiệm chính trị – xã hội**.
""")


# ════════════════════════════════════════════
# TAB 5 — THẢO LUẬN CHÍNH SÁCH (Mục 11.4)
# ════════════════════════════════════════════
with tab5:
    st.markdown("### 💬 Câu hỏi thảo luận chính sách")
    st.caption("Phân tích kết quả Q-learning dưới góc nhìn chính sách công Việt Nam")

    # ── Kiểm tra đã train chưa ──────────────────
    if "q_result" not in st.session_state:
        st.markdown(
            "<div class='warn-box'>⚠️ Hãy Train Q-learning ở tab 1 trước để xem phân tích động.</div>",
            unsafe_allow_html=True
        )
        Q_available = False
    else:
        Q = st.session_state["q_result"]["Q"]
        Q_available = True

    st.divider()

    # ════════════════════════════════════════════
    # CÂU A
    # ════════════════════════════════════════════
    st.markdown("#### 🅐 Câu a — Trạng thái khủng hoảng: GDP thấp · D thấp · U cao")

    st.markdown("""
    > *"Khi nền kinh tế ở trạng thái GDP growth thấp, D thấp, U cao —  
    > chính sách π*(s) chọn hành động gì? Có khớp với **'quick win'** hay không?"*
    """)

    # Trạng thái: G=0(low), D=0(low), AI bất kỳ, U=2(high)
    state_a = (0, 0, 0, 2)   # worst case

    if Q_available:
        a_star_a = int(np.argmax(Q[state_a]))
        alloc_a  = ACTIONS[a_star_a]["alloc"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Trạng thái s = (0, 0, 0, 2)</div>
                <div class='value' style='font-size:1.1rem'>{ACTIONS[a_star_a]["name"]}</div>
                <div class='delta'>GDP=Low · D=Low · AI=Low · U=High</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            # So sánh phân bổ hành động được chọn vs a3 AI dẫn dắt
            fig_a = go.Figure()
            labels = ["K (Vốn)", "D (Số hóa)", "AI", "H (Nhân lực)"]
            fig_a.add_trace(go.Bar(
                name=ACTIONS[a_star_a]["name"],
                x=labels, y=alloc_a * 100,
                marker_color="#a78bfa"
            ))
            fig_a.add_trace(go.Bar(
                name="a3 · AI dẫn dắt (so sánh)",
                x=labels, y=ACTIONS[3]["alloc"] * 100,
                marker_color="#374151"
            ))
            fig_a.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font_color="#c0c4d0", barmode="group",
                yaxis=dict(title="%", gridcolor="#2d2f3e"),
                height=240, margin=dict(l=40, r=10, t=20, b=40),
                legend=dict(bgcolor="#1e2130", font_size=11)
            )
            st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("""
    **📌 Phân tích:**

    Trong trạng thái khủng hoảng *(GDP thấp, số hóa thấp, thất nghiệp cao)*, chiến lược "quick win"  
    đúng nghĩa cần **tạo việc làm nhanh + nâng năng lực số cơ bản**, không phải đẩy AI ngay.

    - Nếu mô hình chọn **a4 · Bao trùm** (H=40%): ✅ *Khớp hoàn toàn* — ưu tiên đào tạo nhân lực  
      giúp giảm thất nghiệp nhanh, tạo nền tảng hấp thụ công nghệ sau.
    - Nếu mô hình chọn **a1 · Cân bằng**: ✅ *Khớp một phần* — phân bổ đa dạng giảm rủi ro.
    - Nếu mô hình chọn **a3 · AI dẫn dắt**: ⚠️ *Không khớp* — đầu tư AI khi nhân lực yếu,  
      D thấp sẽ không phát huy hiệu quả và dễ làm trầm trọng thêm thất nghiệp.

    > **Kết luận:** Hành động "quick win" phù hợp nhất là **a4 (Bao trùm)** hoặc **a1 (Cân bằng)**.  
    > Nếu π* chọn đúng → mô hình Q-learning đã học được ưu tiên chính sách hợp lý.  
    > Điều này khớp với Nghị quyết 57-NQ/TW nhấn mạnh "không để ai bị bỏ lại phía sau" trong  
    > chuyển đổi số.
    """)

    st.divider()

    # ════════════════════════════════════════════
    # CÂU B
    # ════════════════════════════════════════════
    st.markdown("#### 🅑 Câu b — Trạng thái bùng nổ: GDP cao · AI cao · U thấp")

    st.markdown("""
    > *"Khi GDP growth cao, AI cao, U thấp —  
    > chính sách chọn gì? Phù hợp với **'consolidation'** không?"*
    """)

    state_b = (2, 2, 2, 0)   # G=high, D=high, AI=high, U=low

    if Q_available:
        a_star_b = int(np.argmax(Q[state_b]))
        alloc_b  = ACTIONS[a_star_b]["alloc"]

        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown(f"""<div class='metric-card'>
                <div class='label'>Trạng thái s = (2, 2, 2, 0)</div>
                <div class='value' style='font-size:1.1rem'>{ACTIONS[a_star_b]["name"]}</div>
                <div class='delta'>GDP=High · D=High · AI=High · U=Low</div>
            </div>""", unsafe_allow_html=True)

        with col2:
            fig_b = go.Figure()
            fig_b.add_trace(go.Bar(
                name=ACTIONS[a_star_b]["name"],
                x=labels, y=alloc_b * 100,
                marker_color="#34d399"
            ))
            fig_b.add_trace(go.Bar(
                name="a4 · Bao trùm (so sánh)",
                x=labels, y=ACTIONS[4]["alloc"] * 100,
                marker_color="#374151"
            ))
            fig_b.update_layout(
                paper_bgcolor="#0f1117", plot_bgcolor="#1a1d27",
                font_color="#c0c4d0", barmode="group",
                yaxis=dict(title="%", gridcolor="#2d2f3e"),
                height=240, margin=dict(l=40, r=10, t=20, b=40),
                legend=dict(bgcolor="#1e2130", font_size=11)
            )
            st.plotly_chart(fig_b, use_container_width=True)

    st.markdown("""
    **📌 Phân tích:**

    Trạng thái bùng nổ *(GDP cao, AI cao, thất nghiệp thấp)* là thời điểm lý tưởng để  
    **"consolidation"** — củng cố thành quả, đẩy mạnh AI và R&D, duy trì đà tăng trưởng.

    - Nếu mô hình chọn **a3 · AI dẫn dắt** (AI=45%): ✅ *Khớp hoàn toàn* — tận dụng nền tảng  
      nhân lực sẵn có, đẩy mạnh AI khi hạ tầng và con người đã sẵn sàng.
    - Nếu mô hình chọn **a2 · Số hóa nhanh**: ✅ *Khớp một phần* — tiếp tục đẩy D khi đang  
      tăng trưởng tốt.
    - Nếu mô hình chọn **a4 · Bao trùm**: ⚠️ *Không tối ưu* — lãng phí cơ hội khi điều kiện  
      thuận lợi cho đầu tư AI chiều sâu.

    > **Kết luận:** Consolidation đúng nghĩa = **a3 (AI dẫn dắt)**. Điều này phù hợp với  
    > Quyết định 127/QĐ-TTg về chiến lược quốc gia AI đến 2030 — đặt mục tiêu Việt Nam  
    > trở thành **trung tâm AI của ASEAN** khi đã có nền tảng số vững chắc.
    """)

    st.divider()

    # ════════════════════════════════════════════
    # CÂU C
    # ════════════════════════════════════════════
    st.markdown("#### 🅒 Câu c — Tích hợp π* vào hoạch định chính sách không vi phạm nguyên tắc")

    st.markdown("""
    > *"Mục 11 của bài báo nguồn nhấn mạnh 'AI không thay thế quyết định chính trị – xã hội'.  
    > Làm thế nào để tích hợp π* vào quy trình hoạch định chính sách Việt Nam?"*
    """)

    # Sơ đồ quy trình tích hợp
    st.markdown("##### 🔄 Quy trình tích hợp π* được đề xuất")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card' style='border-color:#3b82f655'>
            <div class='label' style='color:#60a5fa'>BƯỚC 1 — Đầu vào</div>
            <div style='font-size:0.88rem;color:#c0c4d0;margin-top:8px;text-align:left'>
            📊 Thu thập dữ liệu KT-XH<br>
            🗺️ Xác định trạng thái s hiện tại<br>
            ⚙️ Chạy mô hình Q-learning<br>
            📋 Xuất khuyến nghị π*(s)
            </div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card' style='border-color:#f59e0b55'>
            <div class='label' style='color:#fcd34d'>BƯỚC 2 — Thẩm định</div>
            <div style='font-size:0.88rem;color:#c0c4d0;margin-top:8px;text-align:left'>
            👥 Hội đồng chuyên gia phản biện<br>
            🏛️ Tham vấn Bộ KH&ĐT, Bộ KH&CN<br>
            📣 Đối thoại công khai (nếu lớn)<br>
            ⚖️ Đánh giá tác động xã hội
            </div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card' style='border-color:#10b98155'>
            <div class='label' style='color:#6ee7b7'>BƯỚC 3 — Quyết định</div>
            <div style='font-size:0.88rem;color:#c0c4d0;margin-top:8px;text-align:left'>
            ✍️ Lãnh đạo chính trị quyết định<br>
            📝 Ban hành Nghị quyết / QĐ<br>
            🔍 Giám sát thực thi<br>
            🔄 Cập nhật mô hình theo phản hồi
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("""
    **📌 Nguyên tắc tích hợp không vi phạm trách nhiệm chính trị:**

    | Nguyên tắc | Cách thực hiện |
    |-----------|---------------|
    | **π* là gợi ý, không phải lệnh** | Dashboard hiển thị nhiều phương án thay thế, không chỉ một hành động duy nhất |
    | **Minh bạch hóa mô hình** | Công khai hàm reward, tham số, giả định → chuyên gia và công chúng có thể phản biện |
    | **Con người kiểm soát vòng lặp** | Mọi quyết định ngân sách cuối cùng phải qua Quốc hội / Thủ tướng phê duyệt |
    | **Cập nhật dân chủ** | Trọng số reward *w* (GDP vs bao trùm vs môi trường) cần được xã hội đồng thuận, không do kỹ thuật viên tự quyết |
    | **Phân tích độ nhạy bắt buộc** | Luôn trình bày "nếu đổi trọng số w → chính sách thay đổi như thế nào?" |

    > **Kết luận:** Mô hình Q-learning đóng vai trò **"tham mưu định lượng"** —  
    > cung cấp bằng chứng số để hỗ trợ tranh luận chính sách, tương tự như các báo cáo  
    > của CIEM, VEPR hay World Bank. Quyết định cuối cùng vẫn thuộc về **con người**  
    > và thể chế dân chủ, phù hợp với tinh thần Nghị quyết 57-NQ/TW về  
    > *"phát triển KH-CN phải gắn với lợi ích của nhân dân và sự lãnh đạo của Đảng"*.
    """)

    st.divider()
    st.markdown("""
    <div style='background:#1e2130;border:1px solid #2d3250;border-radius:10px;
    padding:14px 20px;font-size:0.85rem;color:#8b8fa8'>
    📚 <b>Tài liệu tham chiếu:</b> Nghị quyết 57-NQ/TW (2024) · QĐ 127/QĐ-TTg (2021) ·
    QĐ 749/QĐ-TTg (2020) · Sutton & Barto (2018) <i>Reinforcement Learning: An Introduction</i>
    </div>
    """, unsafe_allow_html=True)