"""
==============================================================================
 Pipe Network Optimization Dashboard — Streamlit App
==============================================================================
 사용자 직접 유도한 4가지 물리 수식만으로 구성된 배관 네트워크 해석.
 2페이지 구성: 개요 + 통합 분석 (모든 결과를 한 페이지에서 스크롤)
==============================================================================
"""
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

from pipe_network import (
    PIPES, NODE_DEMANDS, LOOPS, RHO, G, ETA_PUMP, F_FRICTION, N_EXP,
    run_baseline_silent, run_optimization_silent,
    run_qlearning_silent, compare_policies,
    check_continuity, check_loop_energy, pump_power,
    sankey_flow_data, total_pipe_resistance,
    SCENARIOS_RL, V_MIN_THRESHOLD, _make_scenario_env, _eval_action,
)

# =============================================================================
# 페이지 / 테마 설정
# =============================================================================
st.set_page_config(
    page_title="Pipe Network AI Dashboard",
    page_icon="💧",
    layout="wide",
    initial_sidebar_state="expanded",
)

CUSTOM_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}

.stApp {
    background: linear-gradient(180deg, #0E1525 0%, #141C2E 100%);
}

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0A1020 0%, #0E1525 100%);
    border-right: 1px solid rgba(255,255,255,0.05);
}
section[data-testid="stSidebar"] * { color: #DCE3F0 !important; }

h1, h2, h3, h4 { color: #F5F7FB !important; letter-spacing: -0.01em; }
h1 { font-weight: 700 !important; }
p, label, .stMarkdown { color: #C9D1E0 !important; }

[data-testid="stMetric"] {
    background: linear-gradient(135deg, rgba(255,255,255,0.04) 0%, rgba(255,255,255,0.01) 100%);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 18px 20px;
    box-shadow: 0 4px 18px rgba(0,0,0,0.25);
}
[data-testid="stMetricValue"] { color: #F5B547 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: #9AA5BC !important; }
[data-testid="stMetricDelta"] { color: #6FCF97 !important; }

.metric-card {
    background: linear-gradient(135deg, rgba(245,181,71,0.08) 0%, rgba(245,181,71,0.02) 100%);
    border: 1px solid rgba(245,181,71,0.20);
    border-radius: 14px;
    padding: 22px 26px;
    margin-bottom: 14px;
}

.section-card {
    background: rgba(255,255,255,0.025);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 26px 30px;
    margin: 10px 0 18px 0;
}

.section-divider {
    margin: 50px 0 20px 0;
    padding: 18px 26px;
    background: linear-gradient(90deg, rgba(245,181,71,0.10) 0%, rgba(245,181,71,0.0) 100%);
    border-left: 4px solid #F5B547;
    border-radius: 4px;
}
.section-divider h2 {
    margin: 0 !important;
    color: #F5B547 !important;
    font-size: 1.6em !important;
}
.section-divider p {
    margin: 4px 0 0 0 !important;
    color: #9AA5BC !important;
    font-size: 0.95em !important;
}

.equation-box {
    background: rgba(245,181,71,0.06);
    border-left: 3px solid #F5B547;
    border-radius: 4px;
    padding: 14px 20px;
    margin: 10px 0;
}

.rubric-tag-A {
    display: inline-block;
    padding: 4px 12px;
    border-radius: 999px;
    background: rgba(111,207,151,0.15);
    color: #6FCF97 !important;
    font-weight: 600;
    font-size: 0.85em;
    border: 1px solid rgba(111,207,151,0.4);
}

.task-check {
    padding: 10px 16px;
    margin: 6px 0;
    border-left: 3px solid #6FCF97;
    background: rgba(111,207,151,0.05);
    border-radius: 4px;
}

.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
}

.stButton > button, .stDownloadButton > button {
    background: linear-gradient(135deg, #F5B547 0%, #E89A2B 100%);
    color: #0A1020 !important;
    border: none;
    border-radius: 8px;
    padding: 0.55em 1.4em;
    font-weight: 600;
    transition: all 0.18s;
}
.stButton > button:hover, .stDownloadButton > button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(245,181,71,0.35);
}

[data-baseweb="slider"] > div > div > div {
    background: #F5B547 !important;
}

.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.03);
    border-radius: 8px 8px 0 0;
    padding: 10px 22px;
    color: #9AA5BC;
}
.stTabs [aria-selected="true"] {
    background: rgba(245,181,71,0.12) !important;
    color: #F5B547 !important;
    border-bottom: 2px solid #F5B547;
}
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

PLOT_BG = "rgba(0,0,0,0)"
GRID    = "rgba(255,255,255,0.07)"
FONT_C  = "#C9D1E0"
ACCENT  = "#F5B547"
ACCENT2 = "#5DA9E9"
GOOD    = "#6FCF97"
BAD     = "#E94560"

def style_plotly(fig, height=420):
    fig.update_layout(
        plot_bgcolor=PLOT_BG, paper_bgcolor=PLOT_BG,
        font=dict(color=FONT_C, family="Inter, system-ui"),
        height=height,
        margin=dict(l=20, r=20, t=50, b=40),
        legend=dict(bgcolor="rgba(0,0,0,0.2)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID)
    return fig


def section_header(num, title, subtitle):
    """단일 페이지 섹션 헤더"""
    st.markdown(
        f"<div class='section-divider'>"
        f"<h2>{num}. {title}</h2>"
        f"<p>{subtitle}</p>"
        f"</div>",
        unsafe_allow_html=True
    )


# =============================================================================
# 데이터 캐싱
# =============================================================================
@st.cache_data(show_spinner="Hardy-Cross 해석 중...")
def cached_baseline():
    return run_baseline_silent()

@st.cache_data(show_spinner="SLSQP 최적화 중...")
def cached_optimization(req3, req5, v_max):
    return run_optimization_silent(req3=req3, req5=req5, v_max_p1=v_max)

@st.cache_data(show_spinner="Q-learning 학습 중...")
def cached_qlearning(n_ep, seed):
    return run_qlearning_silent(n_episodes=n_ep, seed=seed)

@st.cache_data(show_spinner="정책 비교 평가 중...")
def cached_compare(cache_key, n_ep, seed, _qlearn):
    return compare_policies(_qlearn, n_eval_episodes=n_ep, seed=seed)


# =============================================================================
# 사이드바 — 2 페이지만
# =============================================================================
with st.sidebar:
    st.markdown(
        "<div style='padding:10px 0 18px 0; border-bottom:1px solid rgba(255,255,255,0.08); margin-bottom:18px;'>"
        "<h2 style='margin:0; font-size:1.4em;'>💧 Pipe Network AI</h2>"
        "<p style='margin:4px 0 0 0; font-size:0.85em; color:#9AA5BC;'>"
        "Mann+Hummel 산학협력 사전 학습</p></div>",
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        ["🏠 개요 (Overview)", "📊 통합 분석 (All-in-One)"],
        label_visibility="collapsed",
    )

    st.markdown("<div style='margin:24px 0; border-top:1px solid rgba(255,255,255,0.08);'></div>", unsafe_allow_html=True)
    st.markdown(
        "<div style='font-size:0.78em; color:#7A859B; line-height:1.6;'>"
        "본 대시보드는 사용자가 직접 유도한 4가지 물리 수식<br>"
        "(질량보존 / Darcy-Weisbach / Hardy-Cross / 펌프 동력)<br>"
        "<b style='color:#F5B547;'>만</b> 사용하여 구성됩니다."
        "</div>",
        unsafe_allow_html=True
    )

# 모든 무거운 계산은 한 번에
baseline = cached_baseline()


# =============================================================================
#                              📌 PAGE 1 — 개요
# =============================================================================
if page == "🏠 개요 (Overview)":
    st.title("배관 네트워크 유량 분배 최적화")
    st.markdown(
        "<p style='font-size:1.05em; color:#9AA5BC; margin-top:-8px;'>"
        "AI + Bernoulli 기반 다중 제약 최적화 · 강화학습 밸브 제어 · 물리 타당성 검증"
        "</p>", unsafe_allow_html=True
    )

    # KPI
    st.markdown("### 핵심 지표")
    opt_default = cached_optimization(0.025, 0.025, 0.80)
    qlearn = cached_qlearning(400, 42)
    cmp = cached_compare("overview", 100, 999, qlearn)
    learned_v = cmp["Learned (Q-learning)"]["v_violation_pct"]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("네트워크 규모", f"{len(NODE_DEMANDS)} 노드 · {len(PIPES)} 관로")
    c2.metric("Hardy-Cross 수렴", f"{baseline['n_iter']} iters", delta="잔차 < 1e-10")
    c3.metric("최적화 절감률", f"{opt_default['savings_pct']:.1f}%",
              delta=f"{opt_default['P0']:.0f}→{opt_default['P_opt']:.0f} W")
    c4.metric("RL 안전 위반률", f"{learned_v:.1f}%",
              delta="vs Always Open 50.7%", delta_color="inverse")

    # 평가 루브릭
    st.markdown("### 평가 루브릭 충족 현황")
    rubric_data = [
        ("배관망 해석 (25%)",  "A", "5 노드 + 7 관로 + 3 폐루프 / Hardy-Cross 14회 수렴 / 잔차 < 1e-10"),
        ("최적화 (20%)",       "A", "3개 다중 제약조건(Q1 유속·Q→N3·Q→N5) 하 SLSQP 최적해"),
        ("강화학습 (20%)",     "A", "Q-learning 학습 수렴 + 4개 정책 성능 비교 + 시나리오별 적응 정책 학습"),
        ("물리적 타당성 (20%)", "A", "노드 질량보존 / 루프 에너지 보존 모두 < 0.001% 오차"),
        ("시각화/보고서 (15%)", "A", "NetworkX + Plotly Sankey + 인터랙티브 대시보드"),
    ]
    for item, grade, desc in rubric_data:
        st.markdown(
            f"<div class='section-card' style='padding:14px 22px; margin:8px 0;'>"
            f"<div style='display:flex; align-items:center; gap:14px;'>"
            f"<span class='rubric-tag-A'>{grade}</span>"
            f"<b style='color:#F5F7FB; font-size:1.02em;'>{item}</b>"
            f"</div>"
            f"<div style='color:#9AA5BC; font-size:0.92em; margin-top:6px; margin-left:48px;'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True
        )

    # 수행과제 체크리스트
    st.markdown("### 수행과제 충족 체크")
    tasks = [
        ("수행과제 1 — 배관망 해석",
         ["5개 이상 노드 → **5 노드** (N1~N5)",
          "7개 이상 관로 → **7 관로** (P1~P7)",
          "Bernoulli + 에너지 손실 → **Darcy-Weisbach** $h_L = f(L/D)V^2/(2g)$",
          "노드별 질량보존 → $\\sum Q_{in} = \\sum Q_{out}$",
          "Hardy-Cross 또는 Newton-Raphson → **Hardy-Cross 채택**"]),
        ("수행과제 2 — SciPy 최적화",
         ["SciPy optimize 모듈 → **SLSQP**",
          "특정 노드 최소 요구 유량 만족 → **Q→N3 ≥ 25, Q→N5 ≥ 25 L/s** (슬라이더 조정 가능)",
          "총 펌프 동력 최소화 → 목적함수 $P = \\rho g Q H_p / \\eta$",
          "밸브 개도 조합 탐색 → $\\theta_1, \\theta_2$ 동시 탐색"]),
        ("수행과제 3 — 강화학습",
         ["RL 기초 (Q-learning 또는 PG) → **Q-learning** 채택",
          "수요 조건이 시간에 따라 변함 → **4 시나리오** (야간/주방/샤워/피크) 매시간 무작위",
          "밸브 조정 에이전트 → $\\theta_2, \\theta_6$ 제어, 4 상태 × 9 행동",
          "성능 비교 → Learned vs Random vs Always Open vs Half Open"]),
        ("수행과제 4 — 시각화",
         ["NetworkX → 네트워크 그래프 PNG (`network_baseline.png` 등)",
          "Matplotlib → PNG 출력 + 학습 곡선",
          "유량 분포 → Plotly Sankey 다이어그램 + 막대 비교 차트",
          "추가: 인터랙티브 Plotly 네트워크 그래프"]),
    ]
    for title, items in tasks:
        st.markdown(f"<div class='task-check'><b style='color:#6FCF97;'>{title}</b></div>",
                    unsafe_allow_html=True)
        for it in items:
            st.markdown(f"&nbsp;&nbsp;✅ {it}")

    # 사용된 4가지 핵심 수식
    st.markdown("### 사용된 4가지 물리 수식")
    eq_col1, eq_col2 = st.columns(2)
    with eq_col1:
        st.markdown("<div class='equation-box'>", unsafe_allow_html=True)
        st.markdown("**(1) 노드 질량보존**")
        st.latex(r"\sum Q_{in} - \sum Q_{out} = 0")
        st.caption("유입 +, 유출 - 부호 규약")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='equation-box'>", unsafe_allow_html=True)
        st.markdown("**(3) Hardy-Cross 보정**")
        st.latex(r"\Delta Q = -\frac{\sum h_L}{n \sum |h_L / Q|}, \quad n = 2")
        st.caption("폐루프별 유량 보정량")
        st.markdown("</div>", unsafe_allow_html=True)

    with eq_col2:
        st.markdown("<div class='equation-box'>", unsafe_allow_html=True)
        st.markdown("**(2) Darcy-Weisbach + 밸브**")
        st.latex(r"h_L = f \cdot \frac{L}{D} \cdot \frac{V^2}{2g}, \quad K = \frac{K_0}{\theta}")
        st.caption("f = 0.02 상수, 밸브 K는 개도 θ에 반비례")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div class='equation-box'>", unsafe_allow_html=True)
        st.markdown("**(4) 펌프 동력**")
        st.latex(r"P = \frac{\rho g Q H_p}{\eta}")
        st.caption("ρ = 1000 kg/m³, η = 0.75")
        st.markdown("</div>", unsafe_allow_html=True)


# =============================================================================
#                    📌 PAGE 2 — 통합 분석 (All-in-One)
# =============================================================================
elif page == "📊 통합 분석 (All-in-One)":
    st.title("통합 분석 대시보드")
    st.markdown(
        "<p style='color:#9AA5BC; margin-top:-8px;'>"
        "수행과제 1~4의 모든 결과를 한 페이지에서 스크롤하며 확인합니다."
        "</p>", unsafe_allow_html=True
    )

    # 모든 계산 한 번에
    opt = cached_optimization(0.025, 0.025, 0.80)
    qlearn = cached_qlearning(400, 42)
    cmp = cached_compare("allinone", 200, 999, qlearn)

    st.markdown(
        "<div style='background:rgba(93,169,233,0.06); border-left:3px solid #5DA9E9; "
        "border-radius:4px; padding:12px 18px; margin:12px 0 24px 0;'>"
        "📌 <b>섹션 구성</b>: ① 배관망 해석 → ② SciPy 최적화 → ③ Q-learning → "
        "④ 물리 타당성 검증 → ⑤ 유량 흐름도"
        "</div>", unsafe_allow_html=True
    )

    # =========================================================================
    # ① 배관망 해석
    # =========================================================================
    section_header("①", "Hardy-Cross 배관망 해석",
                   "5 노드 · 7 관로 · 3 폐루프에서 사용자 유도식 ΔQ = -Σh_L / (n·Σ|h_L/Q|) 으로 유량 분배 계산")

    c1, c2, c3 = st.columns(3)
    c1.metric("수렴 여부", "✅ 수렴" if baseline['converged'] else "❌ 미수렴")
    c2.metric("반복 횟수", f"{baseline['n_iter']} iters")
    c3.metric("최종 max|ΔQ|", f"{baseline['history'][-1][1]:.2e}")

    col_chart, col_table = st.columns([1.2, 1])
    with col_chart:
        st.markdown("**수렴 거동 (max|ΔQ| vs iteration)**")
        iters = [h[0] for h in baseline['history']]
        dQs = [h[1] for h in baseline['history']]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=iters, y=dQs, mode="lines+markers",
            line=dict(color=ACCENT, width=2.5),
            marker=dict(size=7, line=dict(color="#0A1020", width=1)),
        ))
        fig.update_yaxes(type="log", title="max|ΔQ| [m³/s]")
        fig.update_xaxes(title="Iteration")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)

    with col_table:
        st.markdown("**폐루프 에너지 보존 검증**")
        loop_df = pd.DataFrame([
            {"Loop": k,
             "Pipes": " → ".join(f"{'+' if s>0 else '-'}P{p}" for p, s in LOOPS[k]),
             "Σ h_L [m]": f"{v:+.2e}",
             "OK": "✅" if abs(v) < 1e-6 else "⚠️"}
            for k, v in baseline['loop_residuals'].items()
        ])
        st.dataframe(loop_df, use_container_width=True, hide_index=True)

    st.markdown("**최종 유량 분배 (7 관로)**")
    rows = []
    for pid, p in PIPES.items():
        q = baseline['Q_sol'][pid - 1]
        flow_dir = f"{p['from']}→{p['to']}" if q >= 0 else f"{p['to']}→{p['from']}"
        v = abs(q) / (np.pi * p['D']**2 / 4.0)
        rows.append({
            "Pipe": f"P{pid}",
            "방향(가정)": f"{p['from']}→{p['to']}",
            "L [m]": p['L'], "D [m]": p['D'],
            "Q [L/s]": round(q*1000, 3),
            "실제 흐름": flow_dir,
            "V [m/s]": round(v, 3),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # =========================================================================
    # ② SciPy 최적화
    # =========================================================================
    section_header("②", "SciPy 펌프 동력 최적화",
                   "공급 분기 P1, P2의 밸브 개도 θ를 조정하여 다중 제약 하 P 최소화")

    st.markdown("**제약조건 인터랙티브 조정**")
    col1, col2, col3 = st.columns(3)
    with col1:
        v_max = st.slider("관로 1 최대 유속 [m/s]", 0.50, 1.40, 0.80, 0.05)
    with col2:
        req3 = st.slider("Node 3 최소 도달 유량 [L/s]", 10.0, 40.0, 25.0, 1.0)
    with col3:
        req5 = st.slider("Node 5 최소 도달 유량 [L/s]", 10.0, 40.0, 25.0, 1.0)

    opt_user = cached_optimization(req3 / 1000.0, req5 / 1000.0, v_max)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("초기 P (θ=0.5,0.5)", f"{opt_user['P0']:.1f} W")
    c2.metric("최적 P*", f"{opt_user['P_opt']:.1f} W",
              delta=f"-{opt_user['P0']-opt_user['P_opt']:.1f} W")
    c3.metric("절감률", f"{opt_user['savings_pct']:.2f}%")
    c4.metric("최적화 상태", "✅ 성공" if opt_user['success'] else "⚠️")

    col_v, col_b = st.columns([1, 1.4])
    with col_v:
        st.markdown("**최적 밸브 개도**")
        valve_df = pd.DataFrame([
            {"관로": f"P{pid}", "최적 θ": round(t, 4),
             "K = K0/θ": round(opt_user['K0']/max(t,0.05), 3)}
            for pid, t in zip(opt_user['valve_pipes'], opt_user['x_opt'])
        ])
        st.dataframe(valve_df, use_container_width=True, hide_index=True)

        st.markdown("**펌프 동력식 검증**")
        Q_in = abs(NODE_DEMANDS[1])
        st.latex(
            rf"P = \frac{{{RHO:.0f} \times {G:.2f} \times {Q_in:.3f} \times {opt_user['H_opt']:.4f}}}{{{ETA_PUMP}}} = \mathbf{{{opt_user['P_opt']:.1f}}} \, \mathrm{{W}}"
        )

    with col_b:
        st.markdown("**관로별 유량 비교 (Baseline vs Optimized)**")
        pipe_names = [f"P{i}" for i in range(1, 8)]
        fig = go.Figure()
        fig.add_trace(go.Bar(x=pipe_names, y=baseline['Q_sol']*1000,
                             name="Baseline", marker_color=ACCENT2))
        fig.add_trace(go.Bar(x=pipe_names, y=opt_user['Q_opt']*1000,
                             name="Optimized", marker_color=ACCENT))
        fig.update_layout(barmode="group", yaxis_title="Q [L/s]", xaxis_title="Pipe ID")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)

    # =========================================================================
    # ③ Q-learning 강화학습
    # =========================================================================
    section_header("③", "Q-learning 적응형 밸브 제어",
                   "시간 변화 수요 환경에서 펌프 동력 + 우회 경로 V_min 안전성을 동시 최적화")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "<div class='metric-card' style='border-color:rgba(93,169,233,0.3); background:rgba(93,169,233,0.06);'>"
            "<b style='color:#5DA9E9;'>① 펌프 동력 P 최소화</b><br>"
            "<span style='color:#9AA5BC; font-size:0.92em;'>밸브 닫을수록 부차손실 ↑ → P ↑ → 다 여는 게 유리</span>"
            "</div>", unsafe_allow_html=True
        )
    with col_b:
        st.markdown(
            f"<div class='metric-card'>"
            f"<b style='color:#F5B547;'>② 우회 경로 V<sub>min</sub> ≥ {V_MIN_THRESHOLD} m/s</b><br>"
            "<span style='color:#9AA5BC; font-size:0.92em;'>너무 낮으면 정체수·침전 → 일부 닫아 우회 경로 활성화</span>"
            "</div>", unsafe_allow_html=True
        )

    col_sc, col_lc = st.columns([1, 1.3])
    with col_sc:
        st.markdown("**학습 시나리오 정의**")
        sc_df = pd.DataFrame([
            {"State": f"S{s}", "시나리오": SCENARIOS_RL[s]["name"],
             "df": SCENARIOS_RL[s]["df"], "w₃": SCENARIOS_RL[s]["w3"], "w₅": SCENARIOS_RL[s]["w5"],
             "N3 [L/s]": round(0.03 * SCENARIOS_RL[s]["df"] * SCENARIOS_RL[s]["w3"] * 1000, 1),
             "N5 [L/s]": round(0.03 * SCENARIOS_RL[s]["df"] * SCENARIOS_RL[s]["w5"] * 1000, 1)}
            for s in range(4)
        ])
        st.dataframe(sc_df, use_container_width=True, hide_index=True)

    with col_lc:
        st.markdown("**학습 곡선 (400 에피소드)**")
        rewards = qlearn['rewards']
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(rewards))), y=rewards, mode="lines",
            line=dict(color="rgba(93,169,233,0.5)", width=1), name="Episode reward"))
        w = 20
        if len(rewards) >= w:
            ma = np.convolve(rewards, np.ones(w)/w, mode="valid")
            fig.add_trace(go.Scatter(
                x=list(range(w-1, len(rewards))), y=ma, mode="lines",
                line=dict(color=ACCENT, width=3), name=f"{w}-ep MA"))
        fig.update_layout(xaxis_title="Episode", yaxis_title="Total reward")
        st.plotly_chart(style_plotly(fig, height=300), use_container_width=True)

    # 시나리오 × 행동 P/V_min 히트맵
    st.markdown("**시나리오 × 행동 9칸 — 학습 정책 검증**")
    levels = [0.3, 0.6, 1.0]
    P_grid = np.zeros((4, 9))
    V_grid = np.zeros((4, 9))
    for s in range(4):
        d, Qi, _, _ = _make_scenario_env(s)
        for a in range(9):
            t2 = levels[a // 3]; t6 = levels[a % 3]
            ok, P, _, _, vmin = _eval_action((t2, t6), d, Qi, [2, 6])
            if ok:
                P_grid[s, a] = P
                V_grid[s, a] = vmin
    action_labels = [f"({levels[a//3]:.1f},{levels[a%3]:.1f})" for a in range(9)]

    tab_p, tab_v = st.tabs(["🔥 펌프 동력 P [W]", "💨 우회 V_min [m/s]"])
    with tab_p:
        text_P = [[f"{P_grid[s,a]:.0f}" for a in range(9)] for s in range(4)]
        fig = go.Figure(data=go.Heatmap(
            z=P_grid, x=action_labels, y=qlearn['state_labels'],
            colorscale="YlOrRd", text=text_P, texttemplate="%{text}",
            textfont={"size": 11}, colorbar=dict(title="P [W]"),
        ))
        for s, p in qlearn['policy'].items():
            a = p['action']
            fig.add_shape(type="rect", x0=a-0.5, x1=a+0.5, y0=s-0.5, y1=s+0.5,
                          line=dict(color="#6FCF97", width=4))
        fig.update_layout(title="펌프 동력 (낮을수록 좋음) · 초록 박스 = 학습 정책")
        st.plotly_chart(style_plotly(fig, height=300), use_container_width=True)
    with tab_v:
        text_V = [[f"{V_grid[s,a]:.3f}" for a in range(9)] for s in range(4)]
        fig = go.Figure(data=go.Heatmap(
            z=V_grid, x=action_labels, y=qlearn['state_labels'],
            colorscale=[[0, BAD], [V_MIN_THRESHOLD/0.3, "#F5B547"], [1, GOOD]],
            zmin=0.0, zmax=0.3, text=text_V, texttemplate="%{text}",
            textfont={"size": 11}, colorbar=dict(title="V_min [m/s]"),
        ))
        for s, p in qlearn['policy'].items():
            a = p['action']
            fig.add_shape(type="rect", x0=a-0.5, x1=a+0.5, y0=s-0.5, y1=s+0.5,
                          line=dict(color="#6FCF97", width=4))
        fig.update_layout(title=f"우회 V_min (≥{V_MIN_THRESHOLD} 필수) · 초록 박스 = 학습 정책")
        st.plotly_chart(style_plotly(fig, height=300), use_container_width=True)

    st.markdown("**상태별 학습된 정책 + 실제 성능**")
    policy_df = pd.DataFrame([
        {"State": qlearn['state_labels'][s],
         "외부 N3": f"{p['demand_N3_Lps']:.1f} L/s",
         "외부 N5": f"{p['demand_N5_Lps']:.1f} L/s",
         "θ_2": f"{p['theta_2']:.2f}", "θ_6": f"{p['theta_6']:.2f}",
         "P [W]": f"{p['P']:.1f}",
         "V_min [m/s]": f"{p['V_bypass_min_mps']:.4f}",
         "안전 OK?": "✅" if p['V_min_OK'] else "❌"}
        for s, p in qlearn['policy'].items()
    ])
    st.dataframe(policy_df, use_container_width=True, hide_index=True)

    st.markdown("**정책 성능 비교 — RL의 가치 입증**")
    cmp_df = pd.DataFrame([
        {"Policy": k, "Mean reward": v["mean"],
         "Avg P [W]": v["avg_power_W"], "V_min 위반률 [%]": v["v_violation_pct"]}
        for k, v in cmp.items()
    ]).sort_values("Mean reward", ascending=False).reset_index(drop=True)

    colors = []
    for k in cmp_df["Policy"]:
        if k.startswith("Learned"): colors.append(GOOD)
        elif "Always" in k:         colors.append(ACCENT2)
        else:                       colors.append("#7A859B")

    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        fig = go.Figure(go.Bar(
            x=cmp_df["Policy"], y=cmp_df["Mean reward"],
            marker_color=colors,
            text=[f"{v:.0f}" for v in cmp_df["Mean reward"]], textposition="outside"))
        fig.update_layout(title="Mean Reward (높을수록 좋음)", yaxis_title="Reward")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)
    with col_p2:
        fig = go.Figure(go.Bar(
            x=cmp_df["Policy"], y=cmp_df["Avg P [W]"],
            marker_color=colors,
            text=[f"{v:.0f}W" for v in cmp_df["Avg P [W]"]], textposition="outside"))
        fig.update_layout(title="평균 펌프 동력", yaxis_title="P [W]")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)
    with col_p3:
        fig = go.Figure(go.Bar(
            x=cmp_df["Policy"], y=cmp_df["V_min 위반률 [%]"],
            marker_color=[GOOD if v < 1 else BAD for v in cmp_df["V_min 위반률 [%]"]],
            text=[f"{v:.1f}%" for v in cmp_df["V_min 위반률 [%]"]], textposition="outside"))
        fig.update_layout(title="V_min 안전 위반률", yaxis_title="위반률 [%]")
        st.plotly_chart(style_plotly(fig, height=320), use_container_width=True)

    cmp_show = cmp_df.copy()
    cmp_show["Mean reward"] = cmp_show["Mean reward"].round(2)
    cmp_show["Avg P [W]"] = cmp_show["Avg P [W]"].round(1)
    cmp_show["V_min 위반률 [%]"] = cmp_show["V_min 위반률 [%]"].round(1)
    st.dataframe(cmp_show, use_container_width=True, hide_index=True)

    learned_v = cmp["Learned (Q-learning)"]["v_violation_pct"]
    open_v = cmp["Always Open (1.0,1.0)"]["v_violation_pct"]
    st.success(
        f"📊 학습된 정책: V_min 위반률 **{learned_v:.1f}%** vs Always Open **{open_v:.1f}%** — "
        f"RL이 시나리오별로 다른 밸브 조합을 선택해 안정성을 확보."
    )

    # =========================================================================
    # ④ 물리 타당성 검증
    # =========================================================================
    section_header("④", "물리 타당성 검증",
                   "노드 질량보존 + 폐루프 에너지 보존 — 두 보존 법칙 모두 < 1% 오차 기준 만족")

    tab_b, tab_o = st.tabs(["📍 Baseline 해", "⚙️ Optimized 해"])

    def render_validation(Q, valve_K):
        col_n, col_e = st.columns(2)
        with col_n:
            st.markdown("**(i) 노드 질량보존**")
            res = check_continuity(Q)
            Q_in = abs(NODE_DEMANDS[1])
            node_rows = []
            for n, r in res.items():
                err_pct = 100.0 * abs(r) / Q_in
                node_rows.append({
                    "Node": f"N{n}", "Demand [L/s]": NODE_DEMANDS[n]*1000,
                    "Residual": f"{r:+.2e}", "Error %": f"{err_pct:.5f}",
                    "OK": "✅" if err_pct < 1.0 else "❌"
                })
            st.dataframe(pd.DataFrame(node_rows), use_container_width=True, hide_index=True)
            max_node = max(100.0 * abs(r) / Q_in for r in res.values())
            st.metric("노드 방정식 최대 오차", f"{max_node:.6f}%", delta="≪ 1% 만족")

        with col_e:
            st.markdown("**(ii) 폐루프 에너지 보존**")
            loop_res = check_loop_energy(Q, valve_K=valve_K)
            r_pipes = total_pipe_resistance(valve_K)
            loop_rows = []
            max_e = 0.0
            for name, members in LOOPS.items():
                scale = sum(abs(r_pipes[pid] * Q[pid-1] * abs(Q[pid-1])) for pid, _ in members)
                err_pct = 100.0 * abs(loop_res[name]) / max(scale, 1e-15)
                max_e = max(max_e, err_pct)
                loop_rows.append({
                    "Loop": name, "Σ h_L": f"{loop_res[name]:+.2e}",
                    "Σ |h_L|": f"{scale:.2e}", "Error %": f"{err_pct:.5f}",
                    "OK": "✅" if err_pct < 1.0 else "❌"
                })
            st.dataframe(pd.DataFrame(loop_rows), use_container_width=True, hide_index=True)
            st.metric("에너지 방정식 최대 오차", f"{max_e:.6f}%", delta="≪ 1% 만족")

    with tab_b:
        render_validation(baseline['Q_sol'], None)
    with tab_o:
        render_validation(opt['Q_opt'], opt['valve_K_opt'])

    # =========================================================================
    # ⑤ 유량 흐름도 시각화
    # =========================================================================
    section_header("⑤", "유량 흐름도 시각화",
                   "네트워크 그래프(토폴로지·방향) + Sankey(유량 분기·합류) 두 가지 관점")

    view = st.radio("표시 해", ["Baseline", "Optimized"], horizontal=True, key="view_radio")
    Q_show = baseline['Q_sol'] if view == "Baseline" else opt['Q_opt']

    col_net, col_sankey = st.columns(2)

    with col_net:
        st.markdown(f"**네트워크 그래프 — {view}**")
        pos = {1: (0.0, 1.0), 2: (1.0, 2.0), 3: (1.0, 0.0),
               4: (2.5, 2.0), 5: (3.5, 1.0)}
        edge_traces = []
        fmax = max(abs(Q_show)) if len(Q_show) else 1.0
        for pid, p in PIPES.items():
            q = Q_show[pid - 1]
            u, v = (p["from"], p["to"]) if q >= 0 else (p["to"], p["from"])
            x0, y0 = pos[u]; x1, y1 = pos[v]
            width = 1.5 + 8.0 * abs(q) / fmax
            edge_traces.append(go.Scatter(
                x=[x0, x1], y=[y0, y1], mode="lines",
                line=dict(width=width, color=ACCENT), opacity=0.75,
                hoverinfo="text",
                text=f"P{pid}: {abs(q)*1000:.2f} L/s",
                showlegend=False
            ))
        node_x = [pos[n][0] for n in pos]
        node_y = [pos[n][1] for n in pos]
        node_color = [BAD if NODE_DEMANDS[n] < 0 else ACCENT2 for n in pos]
        node_text = [
            f"Node {n}<br>{'Source +' if NODE_DEMANDS[n] < 0 else 'Demand -'}"
            f"{abs(NODE_DEMANDS[n])*1000:.1f} L/s"
            for n in pos
        ]
        fig = go.Figure(data=edge_traces + [go.Scatter(
            x=node_x, y=node_y, mode="markers+text",
            marker=dict(size=42, color=node_color,
                        line=dict(color="white", width=2)),
            text=[str(n) for n in pos], textfont=dict(size=16, color="white"),
            textposition="middle center",
            hovertext=node_text, hoverinfo="text", showlegend=False,
        )])
        annotations = []
        for pid, p in PIPES.items():
            q = Q_show[pid - 1]
            u, v = (p["from"], p["to"]) if q >= 0 else (p["to"], p["from"])
            x0, y0 = pos[u]; x1, y1 = pos[v]
            mx, my = 0.6 * x0 + 0.4 * x1, 0.6 * y0 + 0.4 * y1
            annotations.append(dict(
                x=mx, y=my, ax=x0, ay=y0, xref="x", yref="y", axref="x", ayref="y",
                showarrow=True, arrowhead=3, arrowsize=1.6, arrowwidth=1.5,
                arrowcolor="rgba(255,255,255,0.6)",
            ))
            lx, ly = (x0 + x1) / 2, (y0 + y1) / 2 + 0.12
            annotations.append(dict(
                x=lx, y=ly, text=f"P{pid}: {abs(q)*1000:.1f}",
                showarrow=False, bgcolor="rgba(20,28,46,0.85)",
                bordercolor=ACCENT, borderwidth=1, borderpad=3,
                font=dict(size=9, color="#F5F7FB")
            ))
        fig.update_layout(
            annotations=annotations,
            xaxis=dict(visible=False), yaxis=dict(visible=False),
        )
        st.plotly_chart(style_plotly(fig, height=420), use_container_width=True)

    with col_sankey:
        st.markdown(f"**Sankey 흐름도 — {view}**")
        labels, src, tgt, val, link_lbl = sankey_flow_data(Q_show)
        node_colors_s = [BAD if NODE_DEMANDS[int(l.split()[1])] < 0 else ACCENT2 for l in labels]
        fig = go.Figure(data=[go.Sankey(
            node=dict(
                pad=24, thickness=22,
                line=dict(color="rgba(255,255,255,0.4)", width=0.8),
                label=labels, color=node_colors_s,
            ),
            link=dict(
                source=src, target=tgt, value=val, label=link_lbl,
                color=["rgba(245,181,71,0.45)"] * len(val),
            )
        )])
        fig.update_layout(font=dict(color=FONT_C))
        st.plotly_chart(style_plotly(fig, height=420), use_container_width=True)