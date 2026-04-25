"""
RAIE–SCIS CCUS Dashboard — Streamlit Interface
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import streamlit as st
import pandas as pd
from io import BytesIO

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAIE–SCIS · CCUS Resilience Simulator",
    page_icon="⚗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# CUSTOM CSS  (industrial / dark-blue aesthetic)
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'IBM Plex Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1117 0%, #0f1f35 60%, #0d1117 100%);
    color: #c9d1d9;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0d1b2a !important;
    border-right: 1px solid #1f4068;
}
[data-testid="stSidebar"] * { color: #a8c0d6 !important; }

/* Metric cards */
[data-testid="metric-container"] {
    background: #0f2744;
    border: 1px solid #1f4068;
    border-radius: 6px;
    padding: 0.8rem 1rem;
}
[data-testid="stMetricValue"] {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 1.6rem !important;
    color: #58a6ff !important;
}
[data-testid="stMetricLabel"] { color: #8b9cb6 !important; }
[data-testid="stMetricDelta"] > div { font-family: 'IBM Plex Mono', monospace; }

/* Headers */
h1 { 
    font-family: 'IBM Plex Mono', monospace !important;
    color: #58a6ff !important;
    letter-spacing: -1px;
    font-size: 1.9rem !important;
}
h2, h3 {
    color: #79c0ff !important;
    font-weight: 600;
    letter-spacing: -0.3px;
}

/* Dividers */
hr { border-color: #1f4068 !important; }

/* Tabs */
[data-baseweb="tab-list"] { background: #0d1b2a !important; border-bottom: 1px solid #1f4068; }
[data-baseweb="tab"] { color: #8b9cb6 !important; font-family: 'IBM Plex Mono', monospace; }
[aria-selected="true"] { color: #58a6ff !important; border-bottom: 2px solid #58a6ff !important; }

/* Sliders */
[data-baseweb="slider"] { color: #58a6ff; }
.stSlider > div > div > div > div { background: #58a6ff !important; }

/* Selectbox */
[data-baseweb="select"] > div { background: #0f2744 !important; border-color: #1f4068 !important; }

/* Buttons */
.stButton > button {
    background: #1f4068 !important;
    color: #58a6ff !important;
    border: 1px solid #58a6ff !important;
    border-radius: 4px;
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    transition: all 0.2s;
}
.stButton > button:hover {
    background: #58a6ff !important;
    color: #0d1117 !important;
}

/* Info boxes */
.stInfo { background: #0f2744 !important; border-left-color: #58a6ff !important; }
.stSuccess { background: #0f2e1f !important; border-left-color: #3fb950 !important; }
.stWarning { background: #2d1f0f !important; border-left-color: #d29922 !important; }
.stError { background: #2d0f0f !important; border-left-color: #f85149 !important; }

/* Expander */
[data-testid="stExpander"] {
    background: #0f2744 !important;
    border: 1px solid #1f4068 !important;
    border-radius: 6px;
}

/* Table */
.dataframe { color: #c9d1d9 !important; background: #0f2744 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# MODEL CODE (embedded)
# ─────────────────────────────────────────────

STATE_INDEX = {
    "xB": 0, "R": 1, "D": 2, "Vs": 3, "A": 4,
    "M": 5, "P": 6, "Vbar": 7, "S": 8, "C": 9,
}

def eq3_xB_dot(xB, R, D, Vs, lambda_1, eta_1, eta_2, eta_3):
    return -lambda_1 * xB + eta_1 * R + eta_2 * D + eta_3 * Vs

def eq4_R_dot(R, D, rho_R, R_star, delta_R, psi_R):
    return rho_R * R * (1 - R / R_star) - delta_R * R + psi_R * D

def eq5_D_dot(D, Vs, rho_D, D_star, delta_D, psi_D):
    return rho_D * D * (1 - D / D_star) - delta_D * D + psi_D * Vs

def eq6_Vs_dot(Vs, corr, rho_V, Vs_star, delta_V, psi_V):
    return rho_V * Vs * (1 - Vs / Vs_star) - delta_V * Vs + psi_V * corr

def compute_N(xB, R, D, Vs, wB, wR, wD, wV):
    return np.tanh(wB * xB + wR * R + wD * D + wV * Vs)

def eq7_zI(e_y, e_u, xB, theta_I, lambda_1):
    return theta_I * (abs(e_y) + abs(e_u)) / (1 + lambda_1 * xB)

def eq8_uI(e_y, e_u, zI, alpha_I, beta_I, gamma_I, lambda_I):
    return alpha_I * (abs(e_y) + abs(e_u)) + beta_I * 0.0 + gamma_I * np.tanh(lambda_I * zI)

def eq_fA(e_y, R, kappa_A, eta_A, sigma_A):
    return kappa_A * (np.tanh(eta_A * abs(e_y)) + sigma_A * R)

def eq9_A_dot(A, fA, lambda_2):
    return -lambda_2 * A + fA

def eq11_M_dot(M, A, lambda_3, phi_M):
    return -lambda_3 * M + phi_M * A

def eq12_Deff(e_y, M, N, lambda_m):
    return max(0, (1 - lambda_m * M) * abs(e_y) * (1 - N))

def compute_alpha_t(M, alpha_0, alpha_M):
    return float(np.clip(alpha_0 + alpha_M * M, alpha_0, alpha_0 + alpha_M))

def eq13_P_dot(P, alpha_t, D_eff, beta, P_star):
    return alpha_t * (P_star - P) - beta * D_eff * P

def eq18_Vbar_dot(Vbar, P, Delta):
    return (P - Vbar) / Delta

def eq19_S_dot(S, y_r, y_p, Delta):
    return (y_r / (y_p + 1e-9) - S) / Delta

def eq20_C_dot(C, A, U, R, P, c_A, c_U, c_R, c_P, Delta):
    return (c_A * A + c_U * U + c_R * R + c_P * (1 - P) - C) / Delta

def saturate_phi(phi_max, kappa):
    return phi_max * np.tanh(kappa)

def eq_vy(e_y, e_u, kappa_v):
    return kappa_v * (abs(e_y) + abs(e_u))

def raie_scis_rhs(t, X, params, shock_fn=None):
    xB, R, D, Vs, A, M, P, Vbar, S, C = X
    sp, ip, ap, pp, rp, wt = (
        params["struct"], params["innate"], params["adapt"],
        params["perf"], params["rim"], params["weights"]
    )
    y_p, y_r = pp["P_star"], P
    e_y = abs(y_p - y_r)
    e_u = 0.0
    corr = float(np.clip(1 - abs(y_p - y_r) / (abs(y_p) + 1e-9), 0, 1))

    xB_dot = eq3_xB_dot(xB, R, D, Vs, sp["lambda_1"], sp["eta_1"], sp["eta_2"], sp["eta_3"])
    R_dot  = eq4_R_dot(R, D, sp["rho_R"], sp["R_star"], sp["delta_R"], sp["psi_R"])
    D_dot  = eq5_D_dot(D, Vs, sp["rho_D"], sp["D_star"], sp["delta_D"], sp["psi_D"])
    Vs_dot = eq6_Vs_dot(Vs, corr, sp["rho_V"], sp["Vs_star"], sp["delta_V"], sp["psi_V"])

    N  = compute_N(xB, R, D, Vs, wt["wB"], wt["wR"], wt["wD"], wt["wV"])
    zI = eq7_zI(e_y, e_u, xB, ip["theta_I"], sp["lambda_1"])
    uI = eq8_uI(e_y, e_u, zI, ip["alpha_I"], ip["beta_I"], ip["gamma_I"], ip["lambda_I"])
    U_eff = abs(uI)

    fA    = eq_fA(e_y, R, ap["kappa_A"], ap["eta_A"], ap["sigma_A"])
    A_dot = eq9_A_dot(A, fA, ap["lambda_2"])
    vy    = eq_vy(e_y, e_u, rp["kappa_v"])
    phi_M_eff = saturate_phi(ap["phi_M_max"], vy)
    M_dot = eq11_M_dot(M, A, ap["lambda_3"], phi_M_eff)

    D_eff   = eq12_Deff(e_y, M, N, ap["lambda_m"])
    alpha_t = compute_alpha_t(M, pp["alpha_0"], pp["alpha_M"])
    P_dot   = eq13_P_dot(P, alpha_t, D_eff, pp["beta_P"], pp["P_star"])

    Vbar_dot = eq18_Vbar_dot(Vbar, P, rp["Delta"])
    S_dot    = eq19_S_dot(S, y_r, y_p, rp["Delta"])
    C_dot    = eq20_C_dot(C, A, U_eff, R, P, rp["c_A"], rp["c_U"], rp["c_R"], rp["c_P"], rp["Delta"])

    dXdt = np.array([xB_dot, R_dot, D_dot, Vs_dot, A_dot, M_dot, P_dot, Vbar_dot, S_dot, C_dot])
    if shock_fn is not None:
        dXdt[STATE_INDEX["P"]] += shock_fn(t)
    return dXdt


def triangular_pulse(t, t0, duration):
    t1, t2 = t0 + 0.5 * duration, t0 + duration
    if t < t0 or t > t2:
        return 0.0
    if t <= t1:
        return (t - t0) / (t1 - t0 + 1e-9)
    return max(0.0, 1 - (t - t1) / (t2 - t1 + 1e-9))


def simulate(params, scenario_meta, with_shock=True, beta_shock=1.0, noise=0.005, T_end=365, dt=0.1, seed=42):
    times = np.arange(0, T_end + dt, dt)
    X0 = np.array([0.55, 0.80, 0.70, 0.70, 0.10, 0.10, 0.99, 0.99, 0.99, 0.40])
    X = np.zeros((len(times), 10))
    X[0] = X0
    rng = np.random.default_rng(seed)

    def shock_fn(t):
        if not with_shock:
            return 0.0
        xi = scenario_meta["severity"] * triangular_pulse(t, scenario_meta["t0"], scenario_meta["duration"])
        return -beta_shock * xi * max(X[k, STATE_INDEX["P"]], 0)

    for k in range(len(times) - 1):
        dXdt = raie_scis_rhs(times[k], X[k], params, shock_fn=shock_fn)
        X[k + 1] = X[k] + dt * dXdt
        if noise > 0:
            X[k + 1, STATE_INDEX["P"]] += noise * rng.standard_normal()
        X[k + 1, STATE_INDEX["P"]] = np.clip(X[k + 1, STATE_INDEX["P"]], 0, 1.05)

    return times, X


# ─────────────────────────────────────────────
# DEFAULT SCENARIOS
# ─────────────────────────────────────────────
DEFAULT_SCENARIOS = {
    "S1 – Compression-Liquefaction Failure": {"severity": 0.95, "duration": 20.0, "t0": 40.0},
    "S2 – Recurrent Logistics Restrictions": {"severity": 0.45, "duration": 90.0, "t0": 80.0},
    "S3 – Supply Fluctuations at Capture":   {"severity": 0.65, "duration": 65.0, "t0": 120.0},
    "S4 – Impurity Excursions":              {"severity": 0.50, "duration": 75.0, "t0": 150.0},
    "S5 – Storage Throttling":               {"severity": 0.70, "duration": 95.0, "t0": 180.0},
    "S6 – Power Curtailments":               {"severity": 0.35, "duration": 80.0, "t0": 210.0},
}

SCEN_COLORS = {
    "S1 – Compression-Liquefaction Failure": "#f85149",
    "S2 – Recurrent Logistics Restrictions": "#58a6ff",
    "S3 – Supply Fluctuations at Capture":   "#3fb950",
    "S4 – Impurity Excursions":              "#e3b341",
    "S5 – Storage Throttling":               "#bc8cff",
    "S6 – Power Curtailments":               "#ff7b72",
}

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚗️ RAIE–SCIS CCUS")
    st.markdown("**Resilience Simulator**")
    st.markdown("---")

    st.markdown("### 📋 Scenario")
    selected_scen = st.selectbox("Select disruption scenario", list(DEFAULT_SCENARIOS.keys()))
    meta = DEFAULT_SCENARIOS[selected_scen].copy()

    with st.expander("🎚️ Modify scenario", expanded=False):
        meta["severity"] = st.slider("Severity |ξ|", 0.0, 1.0, meta["severity"], 0.05)
        meta["t0"]       = st.slider("Start day (t₀)", 0, 300, int(meta["t0"]), 5)
        meta["duration"] = st.slider("Duration (days)", 5, 180, int(meta["duration"]), 5)

    compare_baseline = st.checkbox("Show baseline (no shock)", value=True)
    compare_all      = st.checkbox("Overlay all scenarios", value=False)

    st.markdown("---")
    st.markdown("### ⚙️ Simulation")
    T_end     = st.slider("Horizon (days)", 180, 730, 365, 30)
    beta_shock = st.slider("Shock amplifier β", 0.1, 3.0, 1.0, 0.1)
    noise_lvl  = st.slider("Process noise σ", 0.0, 0.02, 0.005, 0.001)

    st.markdown("---")
    st.markdown("### 🔬 Model Parameters")
    with st.expander("Structural layer"):
        lambda_1 = st.slider("λ₁ (absorption decay)", 0.005, 0.10, 0.02, 0.005)
        rho_R    = st.slider("ρ_R (redundancy growth)", 0.01, 0.10, 0.03, 0.005)
        rho_D    = st.slider("ρ_D (diversif. growth)",  0.01, 0.10, 0.03, 0.005)
        psi_V    = st.slider("ψ_V (sensing coupling)",  0.1, 1.0, 0.5, 0.05)

    with st.expander("Adaptive layer"):
        kappa_A  = st.slider("κ_A", 0.1, 2.0, 0.7, 0.05)
        lambda_2 = st.slider("λ₂ (adapt. decay)",  0.01, 0.30, 0.10, 0.01)
        lambda_3 = st.slider("λ₃ (memory decay)",  0.005, 0.10, 0.03, 0.005)
        lambda_m = st.slider("λ_M (memory mitig.)", 0.1, 1.5, 0.70, 0.05)

    with st.expander("Performance layer"):
        alpha_0  = st.slider("α₀ (base recovery)",   0.01, 0.15, 0.051, 0.005)
        alpha_M  = st.slider("α_M (memory recovery)", 0.05, 0.30, 0.128, 0.005)
        beta_P   = st.slider("β_P (degradation gain)", 0.1, 1.0, 0.35, 0.05)

    run_btn = st.button("▶  Run Simulation", use_container_width=True)

# ─────────────────────────────────────────────
# BUILD PARAMS FROM SIDEBAR
# ─────────────────────────────────────────────
params = {
    "struct": {
        "lambda_1": lambda_1, "eta_1": 0.05, "eta_2": 0.03, "eta_3": 0.04,
        "rho_R": rho_R, "R_star": 1.0, "delta_R": 0.01, "psi_R": 0.02,
        "rho_D": rho_D, "D_star": 1.0, "delta_D": 0.01, "psi_D": 0.02,
        "rho_V": 0.04,  "Vs_star": 1.0,"delta_V": 0.01, "psi_V": psi_V,
    },
    "innate": {"theta_I": 1.0, "alpha_I": 0.3, "beta_I": 0.4, "gamma_I": 0.5, "lambda_I": 1.0},
    "adapt":  {"kappa_A": kappa_A, "eta_A": 3.0, "sigma_A": 0.3,
               "lambda_2": lambda_2, "lambda_3": lambda_3,
               "phi_M_max": 0.22, "lambda_m": lambda_m},
    "perf":   {"alpha_0": alpha_0, "alpha_M": alpha_M, "beta_P": beta_P, "P_star": 1.0},
    "rim":    {"Delta": 30.0, "kappa_v": 0.12, "c_A": 0.30, "c_U": 0.25, "c_R": 0.25, "c_P": 0.20},
    "weights":{"wB": 0.2, "wR": 0.3, "wD": 0.3, "wV": 0.3},
}

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown("# RAIE–SCIS · CCUS Resilience Simulator")
st.markdown(
    "**Resilience-Aware Immune Engineering – Systemic Criticality & Immunity Sensing** "
    "applied to Carbon Capture, Utilization and Storage networks."
)
st.divider()

# ─────────────────────────────────────────────
# RUN SIMULATION (always run, button just triggers re-run)
# ─────────────────────────────────────────────
with st.spinner("Integrating RAIE–SCIS dynamics..."):
    t_sim, X_sim = simulate(
        params, meta,
        with_shock=True, beta_shock=beta_shock,
        noise=noise_lvl, T_end=T_end
    )
    t_base, X_base = simulate(
        params, meta,
        with_shock=False, beta_shock=0, noise=noise_lvl, T_end=T_end
    )

P  = X_sim[:, STATE_INDEX["P"]]
Vb = X_sim[:, STATE_INDEX["Vbar"]]
S  = X_sim[:, STATE_INDEX["S"]]
C  = X_sim[:, STATE_INDEX["C"]]
A  = X_sim[:, STATE_INDEX["A"]]
M  = X_sim[:, STATE_INDEX["M"]]
R  = X_sim[:, STATE_INDEX["R"]]
D  = X_sim[:, STATE_INDEX["D"]]
Vs = X_sim[:, STATE_INDEX["Vs"]]
xB = X_sim[:, STATE_INDEX["xB"]]

P_base = X_base[:, STATE_INDEX["P"]]

# ─────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────
col1, col2, col3, col4, col5 = st.columns(5)

P_min     = P.min()
t_min     = t_sim[P.argmin()]
recovery  = P[-1]
resilience_loss = float(np.trapz(P_base - np.clip(P, 0, P_base), t_sim))
col1.metric("P_min (trough)",   f"{P_min:.3f}", f"at day {t_min:.0f}")
col2.metric("P_final",          f"{recovery:.3f}", f"Δ {recovery - P_min:.3f} recovery")
col3.metric("Viability V̄",     f"{Vb[-1]:.3f}")
col4.metric("Adaptive Cost C",  f"{C[-1]:.3f}")
col5.metric("Resilience Loss",  f"{resilience_loss:.2f}", delta_color="inverse")

st.divider()

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance & RIM", "🔬 System States", "🆚 Scenario Comparison", "📊 Data Table"])

PLOT_STYLE = {
    "figure.facecolor": "#0d1117",
    "axes.facecolor":   "#0f2744",
    "axes.edgecolor":   "#1f4068",
    "axes.labelcolor":  "#8b9cb6",
    "xtick.color":      "#8b9cb6",
    "ytick.color":      "#8b9cb6",
    "grid.color":       "#1f4068",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "text.color":       "#c9d1d9",
    "legend.facecolor": "#0f2744",
    "legend.edgecolor": "#1f4068",
}

# ── TAB 1: Performance & RIM ──────────────────
with tab1:
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 2, figsize=(13, 7))
        fig.patch.set_facecolor("#0d1117")

        # 1a. Performance P(t)
        ax = axes[0, 0]
        if compare_baseline:
            ax.plot(t_base, P_base, "--", color="#444d6e", lw=1.5, label="Baseline (no shock)")
        ax.plot(t_sim, P, color="#58a6ff", lw=2, label=f"P(t) – {selected_scen[:2]}")
        ax.axvspan(meta["t0"], meta["t0"] + meta["duration"], alpha=0.12, color="#f85149", label="Disruption window")
        ax.axhline(P_min, color="#f85149", lw=0.8, ls=":")
        ax.set_title("Performance P(t)", color="#79c0ff", fontweight="bold")
        ax.set_xlabel("Day"); ax.set_ylabel("P(t)")
        ax.legend(fontsize=8); ax.grid(True)

        # 1b. Viability index
        ax = axes[0, 1]
        ax.plot(t_sim, Vb, color="#3fb950", lw=2, label="Viability V̄")
        ax.axhline(0.9, color="#e3b341", lw=0.8, ls="--", label="Threshold 0.9")
        ax.set_title("Viability Index V̄(t)", color="#79c0ff", fontweight="bold")
        ax.set_xlabel("Day"); ax.legend(fontsize=8); ax.grid(True)

        # 1c. Survival metric
        ax = axes[1, 0]
        ax.plot(t_sim, S, color="#bc8cff", lw=2, label="Survival S(t)")
        ax.set_title("Functional Survival S(t)", color="#79c0ff", fontweight="bold")
        ax.set_xlabel("Day"); ax.legend(fontsize=8); ax.grid(True)

        # 1d. Adaptive cost
        ax = axes[1, 1]
        ax.plot(t_sim, C, color="#e3b341", lw=2, label="Cost C(t)")
        ax.set_title("Adaptive Cost C(t)", color="#79c0ff", fontweight="bold")
        ax.set_xlabel("Day"); ax.legend(fontsize=8); ax.grid(True)

        fig.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close(fig)

    # Download button
    buf = BytesIO()
    fig2, ax2 = plt.subplots(figsize=(10, 4))
    ax2.plot(t_sim, P, color="#58a6ff", lw=2)
    ax2.set_title("Performance P(t)")
    fig2.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig2)
    st.download_button("⬇ Download P(t) plot", buf.getvalue(), "performance.png", "image/png")

# ── TAB 2: System States ──────────────────────
with tab2:
    with plt.style.context(PLOT_STYLE):
        fig, axes = plt.subplots(2, 3, figsize=(14, 7))
        fig.patch.set_facecolor("#0d1117")

        series = [
            (xB, "#58a6ff", "Structural Absorption x_B"),
            (R,  "#3fb950", "Redundancy R(t)"),
            (D,  "#e3b341", "Diversification D(t)"),
            (Vs, "#bc8cff", "Visibility/Sensing Vs(t)"),
            (A,  "#ff7b72", "Adaptive Activation A(t)"),
            (M,  "#f0883e", "Memory M(t)"),
        ]
        for ax, (y, c, lbl) in zip(axes.flat, series):
            ax.plot(t_sim, y, color=c, lw=1.8)
            ax.axvspan(meta["t0"], meta["t0"] + meta["duration"], alpha=0.10, color="#f85149")
            ax.set_title(lbl, color="#79c0ff", fontsize=9, fontweight="bold")
            ax.set_xlabel("Day", fontsize=8)
            ax.grid(True)

        fig.tight_layout(pad=2.0)
        st.pyplot(fig)
        plt.close(fig)

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("#### End-state summary")
        state_df = pd.DataFrame({
            "State": ["x_B", "R", "D", "Vs", "A", "M"],
            "t=0":   [X_sim[0, i] for i in [0,1,2,3,4,5]],
            "t=final": [X_sim[-1, i] for i in [0,1,2,3,4,5]],
            "Min":   [X_sim[:, i].min() for i in [0,1,2,3,4,5]],
            "Max":   [X_sim[:, i].max() for i in [0,1,2,3,4,5]],
        })
        st.dataframe(
        state_df.style.format({
            "t=0": "{:.4f}",
            "t=final": "{:.4f}",
            "Min": "{:.4f}",
            "Max": "{:.4f}",
        }),
        use_container_width=True
    )

    with col_b:
        # Natural barrier N over time
        N_arr = np.array([
            compute_N(X_sim[k,0], X_sim[k,1], X_sim[k,2], X_sim[k,3],
                      params["weights"]["wB"], params["weights"]["wR"],
                      params["weights"]["wD"], params["weights"]["wV"])
            for k in range(len(t_sim))
        ])
        with plt.style.context(PLOT_STYLE):
            fig3, ax3 = plt.subplots(figsize=(6, 3))
            fig3.patch.set_facecolor("#0d1117")
            ax3.plot(t_sim, N_arr, color="#58a6ff", lw=1.8, label="N(t) – Natural barriers")
            ax3.axvspan(meta["t0"], meta["t0"] + meta["duration"], alpha=0.12, color="#f85149")
            ax3.set_title("Natural Barriers N(t)", color="#79c0ff", fontweight="bold")
            ax3.set_xlabel("Day"); ax3.grid(True); ax3.legend(fontsize=8)
            st.pyplot(fig3)
            plt.close(fig3)

# ── TAB 3: Scenario Comparison ────────────────
with tab3:
    with st.spinner("Simulating all scenarios..."):
        all_results = {}
        for sname, smeta in DEFAULT_SCENARIOS.items():
            _t, _X = simulate(params, smeta, with_shock=True,
                              beta_shock=beta_shock, noise=noise_lvl, T_end=T_end)
            all_results[sname] = (_t, _X[:, STATE_INDEX["P"]])

    with plt.style.context(PLOT_STYLE):
        fig4, ax4 = plt.subplots(figsize=(13, 5))
        fig4.patch.set_facecolor("#0d1117")

        if compare_baseline:
            ax4.plot(t_base, P_base, "--", color="#444d6e", lw=1.5, label="Baseline", zorder=1)

        for sname, (tt, pp_s) in all_results.items():
            lw = 2.5 if sname == selected_scen else 1.2
            alpha = 1.0 if sname == selected_scen else 0.65
            ax4.plot(tt, pp_s, color=SCEN_COLORS[sname], lw=lw, alpha=alpha, label=sname[:2])

        ax4.axvspan(meta["t0"], meta["t0"] + meta["duration"], alpha=0.10, color="#f85149",
                    label=f"Selected disruption window")
        ax4.set_title("All Scenarios – Performance P(t)", color="#79c0ff", fontweight="bold")
        ax4.set_xlabel("Day"); ax4.set_ylabel("P(t)")
        ax4.legend(fontsize=9, ncol=4); ax4.grid(True)
        fig4.tight_layout()
        st.pyplot(fig4)
        plt.close(fig4)

    # Summary table
    st.markdown("#### Comparative metrics")
    rows = []
    for sname, (tt, pp_s) in all_results.items():
        _, _X2 = simulate(params, DEFAULT_SCENARIOS[sname], with_shock=False,
                          beta_shock=0, noise=noise_lvl, T_end=T_end)
        pb_s = _X2[:, STATE_INDEX["P"]]
        rl   = float(np.trapz(pb_s - np.clip(pp_s, 0, pb_s), tt))
        rows.append({
            "Scenario": sname,
            "Severity": DEFAULT_SCENARIOS[sname]["severity"],
            "P_min":    round(pp_s.min(), 4),
            "Day of trough": int(tt[pp_s.argmin()]),
            "P_final":  round(pp_s[-1], 4),
            "Resilience Loss": round(rl, 2),
        })
    df_cmp = pd.DataFrame(rows).sort_values("Resilience Loss", ascending=False)
    st.dataframe(df_cmp, use_container_width=True, hide_index=True)

# ── TAB 4: Data Table ─────────────────────────
with tab4:
    st.markdown("#### Full simulation output (every 10th step)")
    df_out = pd.DataFrame({
        "Day":       t_sim[::10],
        "P(t)":      X_sim[::10, STATE_INDEX["P"]],
        "V_bar(t)":  X_sim[::10, STATE_INDEX["Vbar"]],
        "S(t)":      X_sim[::10, STATE_INDEX["S"]],
        "C(t)":      X_sim[::10, STATE_INDEX["C"]],
        "R(t)":      X_sim[::10, STATE_INDEX["R"]],
        "D(t)":      X_sim[::10, STATE_INDEX["D"]],
        "Vs(t)":     X_sim[::10, STATE_INDEX["Vs"]],
        "A(t)":      X_sim[::10, STATE_INDEX["A"]],
        "M(t)":      X_sim[::10, STATE_INDEX["M"]],
        "xB(t)":     X_sim[::10, STATE_INDEX["xB"]],
    }).round(5)

    st.dataframe(df_out, use_container_width=True, height=400)

    csv = df_out.to_csv(index=False).encode()
    st.download_button("⬇ Download CSV", csv, "raie_scis_output.csv", "text/csv")

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.divider()
st.markdown(
    "<div style='text-align:center;color:#444d6e;font-size:0.78rem;font-family:IBM Plex Mono,monospace'>"
    "RAIE–SCIS · Resilience-Aware Immune Engineering for CCUS Networks · "
    "Euler integration Δt=0.1 d · Model as per Tables 3–5"
    "</div>",
    unsafe_allow_html=True,
)
