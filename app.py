import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom, poisson, norm, probplot
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FIFA 22 · Player Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── THEME & CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:ital,wght@0,300;0,400;0,600;0,700;0,800;1,700&family=DM+Sans:wght@300;400;500&family=Space+Mono:wght@400;700&display=swap');

:root {
  --bg:     #05080f;
  --card:   #0b1120;
  --card2:  #0f1a2e;
  --cyan:   #00d4ff;
  --pink:   #f72585;
  --purple: #7c3aed;
  --gold:   #fbbf24;
  --green:  #10b981;
  --red:    #ef4444;
  --text:   #e2e8f0;
  --muted:  #64748b;
  --border: rgba(0,212,255,0.12);
  --glow:   0 0 20px rgba(0,212,255,0.15);
}

html, body, .stApp { background-color: var(--bg) !important; color: var(--text); font-family: 'DM Sans', sans-serif; }
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem !important; padding-bottom: 2rem !important; max-width: 1400px; }

/* Sidebar */
[data-testid="stSidebar"] {
  background: linear-gradient(180deg, #040710 0%, #08101e 50%, #040d1a 100%) !important;
  border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] * { color: var(--text) !important; }
[data-testid="stSidebar"] .stRadio label { font-family: 'Exo 2', sans-serif; font-size: 0.9rem; cursor: pointer; }

/* Metrics */
[data-testid="metric-container"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 14px !important;
  padding: 1rem 1.2rem !important;
  box-shadow: var(--glow);
}
[data-testid="stMetricValue"] { color: var(--cyan) !important; font-family: 'Space Mono', monospace !important; font-size: 1.8rem !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"] { color: var(--muted) !important; font-size: 0.75rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="stMetricDelta"] > div { font-family: 'Space Mono', monospace !important; font-size: 0.8rem !important; }

/* Headings */
h1, h2, h3 { font-family: 'Exo 2', sans-serif !important; }
h1 { color: var(--cyan) !important; font-weight: 800 !important; letter-spacing: -0.03em; }
h2 { color: var(--text) !important; font-weight: 700 !important; letter-spacing: -0.02em; }
h3 { color: var(--text) !important; font-weight: 600 !important; }

hr { border-color: var(--border) !important; margin: 1.5rem 0 !important; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  border-radius: 10px !important;
  padding: 4px !important;
  gap: 2px !important;
}
.stTabs [data-baseweb="tab"] {
  color: var(--muted) !important;
  font-family: 'Exo 2', sans-serif !important;
  font-size: 0.85rem !important;
  border-radius: 6px !important;
  padding: 6px 16px !important;
}
.stTabs [aria-selected="true"] {
  background: linear-gradient(135deg, var(--cyan), #0090b3) !important;
  color: #000 !important;
  font-weight: 600 !important;
}

/* Inputs */
.stSelectbox > div > div, .stMultiSelect > div > div {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}
.stSlider [data-testid="stThumb"] { background: var(--cyan) !important; }
.stSlider [data-testid="stTrackFill"] { background: var(--cyan) !important; }
input[type="number"], .stNumberInput input {
  background: var(--card) !important;
  border: 1px solid var(--border) !important;
  color: var(--text) !important;
  border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
  background: linear-gradient(135deg, var(--cyan), #0090b3) !important;
  color: #000 !important;
  font-family: 'Exo 2', sans-serif !important;
  font-weight: 700 !important;
  border: none !important;
  border-radius: 10px !important;
  padding: 0.6rem 2rem !important;
  font-size: 0.95rem !important;
  letter-spacing: 0.05em;
  transition: all 0.2s;
}
.stButton > button:hover { transform: translateY(-1px); box-shadow: 0 4px 20px rgba(0,212,255,0.3) !important; }

/* Info/Success boxes */
.stAlert { background: var(--card) !important; border-radius: 10px !important; border-left-width: 3px !important; }

/* DataFrame */
.stDataFrame { background: var(--card) !important; border-radius: 10px !important; }

/* Custom classes */
.hero { padding: 2rem 0 1rem; }
.hero-title {
  font-family: 'Exo 2', sans-serif;
  font-size: 3.2rem;
  font-weight: 800;
  line-height: 1.1;
  background: linear-gradient(135deg, var(--cyan) 0%, #7c3aed 60%, var(--pink) 100%);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  background-clip: text;
}
.hero-sub {
  font-family: 'Space Mono', monospace;
  font-size: 0.8rem;
  color: var(--muted);
  letter-spacing: 0.15em;
  text-transform: uppercase;
  margin-top: 0.4rem;
}
.card {
  background: var(--card);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 1.5rem;
  margin-bottom: 1rem;
}
.card-accent { border-left: 3px solid var(--cyan); }
.card-pink   { border-left: 3px solid var(--pink); }
.card-gold   { border-left: 3px solid var(--gold); }
.card-green  { border-left: 3px solid var(--green); }

.badge {
  display: inline-block;
  padding: 3px 12px;
  border-radius: 20px;
  font-family: 'Space Mono', monospace;
  font-size: 0.65rem;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  margin-bottom: 0.6rem;
}
.badge-cyan   { background: rgba(0,212,255,0.1);  border: 1px solid var(--cyan);   color: var(--cyan);   }
.badge-pink   { background: rgba(247,37,133,0.1); border: 1px solid var(--pink);   color: var(--pink);   }
.badge-gold   { background: rgba(251,191,36,0.1); border: 1px solid var(--gold);   color: var(--gold);   }
.badge-green  { background: rgba(16,185,129,0.1); border: 1px solid var(--green);  color: var(--green);  }
.badge-purple { background: rgba(124,58,237,0.1); border: 1px solid var(--purple); color: var(--purple); }

.stat-row { display: flex; align-items: center; gap: 1rem; margin: 0.6rem 0; }
.stat-label { color: var(--muted); font-size: 0.85rem; min-width: 120px; }
.stat-val   { font-family: 'Space Mono', monospace; color: var(--cyan); font-size: 0.95rem; font-weight: 700; }

.formula-box {
  background: #0a1020;
  border: 1px solid rgba(124,58,237,0.3);
  border-radius: 10px;
  padding: 1rem 1.5rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.85rem;
  color: #a78bfa;
  margin: 0.8rem 0;
}
.result-good { color: var(--green); font-family: 'Space Mono', monospace; font-weight: 700; }
.result-bad  { color: var(--red);   font-family: 'Space Mono', monospace; font-weight: 700; }

.player-card {
  background: linear-gradient(135deg, #0b1120 60%, #0f1a2e);
  border: 1px solid var(--border);
  border-radius: 16px;
  padding: 1.2rem 1.5rem;
  text-align: center;
}
.player-rating {
  font-family: 'Exo 2', sans-serif;
  font-size: 4rem;
  font-weight: 800;
  line-height: 1;
}
</style>
""", unsafe_allow_html=True)

# ─── PLOTLY DARK TEMPLATE ────────────────────────────────────────────────────────
PLOT_THEME = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(11,17,32,0.5)",
    font=dict(family="DM Sans", color="#e2e8f0"),
    title_font=dict(family="Exo 2", size=15, color="#e2e8f0"),
    margin=dict(l=10, r=10, t=45, b=10),
    colorway=["#00d4ff","#f72585","#7c3aed","#fbbf24","#10b981","#fb923c","#818cf8","#34d399"],
)

def apply_theme(fig, title=None):
    fig.update_layout(**PLOT_THEME)
    if title:
        fig.update_layout(title_text=title)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

# ─── DATA LOADING ────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("fifa_cleaned.csv", low_memory=False)
    # Basic cleaning
    df["value_eur"] = df["value_eur"].fillna(df["value_eur"].median())
    df["wage_eur"]  = df["wage_eur"].fillna(df["wage_eur"].median())
    # Outfield
    df_out = df[~df["player_positions"].str.contains("GK", na=False)].copy()
    # Positions
    df_out["primary_position"] = df_out["player_positions"].str.split(",").str[0].str.strip()
    def pg(p):
        if p in ["ST","CF","LW","RW"]: return "Attacker"
        elif p in ["CM","CAM","CDM","LM","RM","LWB","RWB"]: return "Midfielder"
        elif p in ["CB","LB","RB"]: return "Defender"
        else: return "Other"
    df_out["position_group"] = df_out["primary_position"].apply(pg)
    return df, df_out

@st.cache_data
def run_regression(df_out):
    feats = [
        "age","value_eur","wage_eur","movement_reactions","mentality_composure",
        "attacking_short_passing","skill_ball_control","mentality_vision","mentality_positioning",
        "attacking_finishing","skill_dribbling","skill_long_passing","power_stamina",
        "attacking_volleys","skill_curve","power_shot_power","power_long_shots",
        "mentality_aggression","attacking_crossing","skill_fk_accuracy"
    ]
    clean = df_out[feats + ["overall"]].dropna()
    X = clean[feats].values
    y = clean["overall"].values

    sc = StandardScaler()
    Xs = sc.fit_transform(X)
    Xb = np.c_[np.ones(len(Xs)), Xs]

    # Normal Equation
    W_ne = np.linalg.inv(Xb.T @ Xb) @ Xb.T @ y
    y_ne = Xb @ W_ne

    # Gradient Descent
    def gd(X, y, alpha=0.01, iters=1000):
        n = len(y)
        W = np.zeros(X.shape[1])
        hist = []
        for _ in range(iters):
            e = X @ W - y
            W -= (alpha/n) * (X.T @ e)
            hist.append((1/(2*n)) * np.sum(e**2))
        return W, hist

    W_gd, cost_hist = gd(Xb, y)
    y_gd = Xb @ W_gd

    metrics_ne = dict(
        r2=r2_score(y, y_ne),
        mae=mean_absolute_error(y, y_ne),
        rmse=np.sqrt(mean_squared_error(y, y_ne))
    )
    metrics_gd = dict(
        r2=r2_score(y, y_gd),
        mae=mean_absolute_error(y, y_gd),
        rmse=np.sqrt(mean_squared_error(y, y_gd))
    )
    coefs = dict(zip(feats, W_ne[1:]))
    return W_ne, W_gd, y, y_ne, y_gd, cost_hist, metrics_ne, metrics_gd, sc, feats, coefs

df, df_out = load_data()
W_ne, W_gd, y_all, y_ne, y_gd, cost_hist, m_ne, m_gd, scaler, FEATURES, coefs = run_regression(df_out)

# ─── SIDEBAR ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1.2rem 0 0.8rem">
      <div style="font-size:2.8rem;line-height:1">⚽</div>
      <div style="font-family:'Exo 2',sans-serif;font-size:1.3rem;font-weight:800;color:#00d4ff;margin-top:0.3rem">FIFA 22</div>
      <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:#64748b;letter-spacing:0.15em">PLAYER ANALYTICS</div>
    </div>
    """, unsafe_allow_html=True)
    st.divider()
    page = st.radio("Navigate", [
        "🏠  Overview",
        "🧹  Data Cleaning",
        "📊  EDA",
        "🔗  Correlation",
        "📐  Distributions",
        "📈  Regression",
        "🎮  Predict a Player"
    ], label_visibility="collapsed")
    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem;color:#64748b;text-align:center;font-family:'Space Mono',monospace;line-height:1.8">
      19,239 Players · 110 Features<br>
      FIFA 22 (SoFIFA)<br>
      <span style="color:#00d4ff">Prob &amp; Stats Project</span>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero"><div class="hero-title">FIFA 22<br>Player Analytics</div><div class="hero-sub">Probability & Statistics · EDA · Linear Regression</div></div>', unsafe_allow_html=True)
    st.markdown("---")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Players",      f"{len(df):,}")
    c2.metric("Nationalities",      f"{df['nationality_name'].nunique()}")
    c3.metric("Clubs",              f"{df['club_name'].nunique()}")
    c4.metric("Avg Overall Rating", f"{df['overall'].mean():.1f}")
    c5.metric("Features",           "110")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.3, 1])

    with col1:
        st.markdown("### 🌍 Top 15 Nationalities")
        top_nat = df["nationality_name"].value_counts().head(15).reset_index()
        top_nat.columns = ["Nation", "Players"]
        fig = px.bar(top_nat, x="Players", y="Nation", orientation="h",
                     color="Players", color_continuous_scale=["#1a2744","#00d4ff"])
        apply_theme(fig)
        fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=380)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚽ Players by Position")
        pos_counts = df_out["position_group"].value_counts().reset_index()
        pos_counts.columns = ["Group", "Count"]
        fig2 = px.pie(pos_counts, names="Group", values="Count",
                      color_discrete_sequence=["#00d4ff","#7c3aed","#f72585","#fbbf24"],
                      hole=0.55)
        apply_theme(fig2)
        fig2.update_traces(textfont_color="white", textfont_family="Space Mono")
        fig2.update_layout(height=380, legend=dict(orientation="h", y=-0.05))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏆 Top 20 Rated Players")
    top20 = df.nlargest(20, "overall")[["short_name","nationality_name","club_name","player_positions","overall","potential","age","value_eur","wage_eur"]].copy()
    top20["value_eur"] = top20["value_eur"].apply(lambda x: f"€{x/1e6:.1f}M")
    top20["wage_eur"]  = top20["wage_eur"].apply(lambda x: f"€{x/1000:.0f}K/w")
    top20.columns = ["Name","Nation","Club","Position","Overall","Potential","Age","Value","Wage"]
    top20 = top20.reset_index(drop=True)
    top20.index = top20.index + 1
    st.dataframe(top20, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🧹 DATA CLEANING
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🧹  Data Cleaning":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">What it is · What was done · Is this dataset clean?</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📖 What is Data Cleaning?", "🔍 Missing Values", "📦 Outliers", "✅ Summary"])

    with tab1:
        st.markdown("""
        <div class="card card-accent">
          <span class="badge badge-cyan">Definition</span>
          <h3 style="margin:0.3rem 0 0.8rem">What is Data Cleaning?</h3>
          <p style="color:#94a3b8;line-height:1.7">
            Data cleaning is the process of <strong style="color:#00d4ff">detecting and fixing problems</strong> in a dataset
            before analysis. A dirty dataset produces wrong conclusions — "garbage in, garbage out."
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("""
            <div class="card">
              <span class="badge badge-pink">Problem Types</span>
              <div class="stat-row"><span class="stat-label">❌ Missing values</span><span style="color:#94a3b8;font-size:0.85rem">NaN, null, empty cells</span></div>
              <div class="stat-row"><span class="stat-label">❌ Outliers</span><span style="color:#94a3b8;font-size:0.85rem">Extreme unusual values</span></div>
              <div class="stat-row"><span class="stat-label">❌ Wrong types</span><span style="color:#94a3b8;font-size:0.85rem">Text where number expected</span></div>
              <div class="stat-row"><span class="stat-label">❌ Duplicates</span><span style="color:#94a3b8;font-size:0.85rem">Same row appears twice</span></div>
              <div class="stat-row"><span class="stat-label">❌ Inconsistent</span><span style="color:#94a3b8;font-size:0.85rem">"England" vs "england"</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            st.markdown("""
            <div class="card">
              <span class="badge badge-green">Fix Methods</span>
              <div class="stat-row"><span class="stat-label">✅ Fill missing</span><span style="color:#94a3b8;font-size:0.85rem">mean / median / mode</span></div>
              <div class="stat-row"><span class="stat-label">✅ Drop missing</span><span style="color:#94a3b8;font-size:0.85rem">if too many NaN in row/col</span></div>
              <div class="stat-row"><span class="stat-label">✅ IQR clipping</span><span style="color:#94a3b8;font-size:0.85rem">cap extreme outliers</span></div>
              <div class="stat-row"><span class="stat-label">✅ Encode</span><span style="color:#94a3b8;font-size:0.85rem">text → numbers (0/1)</span></div>
              <div class="stat-row"><span class="stat-label">✅ Filter rows</span><span style="color:#94a3b8;font-size:0.85rem">remove irrelevant subset</span></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-gold" style="margin-top:1rem">
          <span class="badge badge-gold">Real Example from YOUR notebook</span>
          <div class="formula-box" style="color:#fbbf24">
# Problem: value_eur has missing values<br>
df['value_eur'].isnull().sum()  → 243 missing<br><br>
# Fix: fill with median (not mean — median is robust to outliers)<br>
df['value_eur'] = df['value_eur'].fillna(df['value_eur'].median())<br><br>
# Result: 0 missing values ✅
          </div>
        </div>
        """, unsafe_allow_html=True)

    with tab2:
        st.markdown("### 🔍 Missing Values in the Dataset")
        missing = df.isnull().sum()
        missing_pct = (missing / len(df) * 100).round(1)
        miss_df = pd.DataFrame({"Column": missing.index, "Missing": missing.values, "Pct (%)": missing_pct.values})
        miss_df = miss_df[miss_df["Missing"] > 0].sort_values("Missing", ascending=False)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig = px.bar(miss_df.head(20), x="Pct (%)", y="Column", orientation="h",
                         color="Pct (%)", color_continuous_scale=["#1a3a4a","#f72585"],
                         title="Top 20 Columns with Missing Values (%)")
            apply_theme(fig)
            fig.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False, height=420)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("""
            <div class="card card-accent" style="margin-top:1rem">
              <span class="badge badge-cyan">Explanation</span>
              <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
                <strong style="color:#00d4ff">nation_* columns (96% missing):</strong><br>
                Most players don't play for national teams. This is expected — not a problem.<br><br>
                <strong style="color:#00d4ff">club_loaned_from (94% missing):</strong><br>
                Most players are not on loan. Normal.<br><br>
                <strong style="color:#00d4ff">pace / physic (11% missing):</strong><br>
                These are GK players — they don't have the same stats as outfield players. We <strong>remove GKs</strong> before regression.<br><br>
                <strong style="color:#fbbf24">value_eur / wage_eur:</strong><br>
                We fill these with the <strong>median</strong> value.
              </p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("#### Key Columns — Missing Values After Cleaning")
        key_cols = ["overall","age","value_eur","wage_eur","movement_reactions","skill_ball_control","mentality_composure"]
        after = df[key_cols].isnull().sum()
        adf = pd.DataFrame({"Column": after.index, "Missing After Clean": after.values})
        adf["Status"] = adf["Missing After Clean"].apply(lambda x: "✅ Clean" if x==0 else f"⚠️ {x} left")
        st.dataframe(adf, use_container_width=True)

    with tab3:
        st.markdown("### 📦 Outlier Detection & Removal (IQR Method)")
        st.markdown("""
        <div class="card card-accent">
          <span class="badge badge-cyan">IQR Method</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7">
            The <strong style="color:#00d4ff">Interquartile Range (IQR)</strong> method finds outliers by:
          </p>
          <div class="formula-box">
Q1 = 25th percentile, &nbsp; Q3 = 75th percentile<br>
IQR = Q3 − Q1<br><br>
Lower Fence = Q1 − 1.5 × IQR<br>
Upper Fence = Q3 + 1.5 × IQR<br><br>
Any value &lt; Lower Fence or &gt; Upper Fence → <span style="color:#f72585">OUTLIER</span><br>
Fix: clip(lower=Lower Fence, upper=Upper Fence)
          </div>
        </div>
        """, unsafe_allow_html=True)

        demo_feat = st.selectbox("Select feature to visualize outliers:", ["movement_reactions","skill_ball_control","value_eur","age","attacking_finishing"])
        col1, col2 = st.columns(2)
        data_before = df_out[demo_feat].dropna()
        Q1, Q3 = data_before.quantile(0.25), data_before.quantile(0.75)
        IQR = Q3 - Q1
        lf, uf = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        data_after = data_before.clip(lower=lf, upper=uf)
        n_out = ((data_before < lf) | (data_before > uf)).sum()
        with col1:
            fig = go.Figure()
            fig.add_trace(go.Box(y=data_before, name="Before Cleaning", marker_color="#f72585", boxmean=True))
            apply_theme(fig, "Before IQR Clipping")
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure()
            fig2.add_trace(go.Box(y=data_after, name="After Cleaning", marker_color="#10b981", boxmean=True))
            apply_theme(fig2, "After IQR Clipping")
            fig2.update_layout(height=350)
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="card card-green">
          <div class="stat-row"><span class="stat-label">Outliers found</span><span class="stat-val">{n_out:,}</span></div>
          <div class="stat-row"><span class="stat-label">Lower fence</span><span class="stat-val">{lf:.2f}</span></div>
          <div class="stat-row"><span class="stat-label">Upper fence</span><span class="stat-val">{uf:.2f}</span></div>
          <div class="stat-row"><span class="stat-label">Action taken</span><span style="color:#10b981;font-family:'Space Mono',monospace">Clipped (not deleted)</span></div>
        </div>
        """, unsafe_allow_html=True)

    with tab4:
        st.markdown("### ✅ Is Our Dataset Already Clean?")
        st.markdown("""
        <div class="card card-green">
          <span class="badge badge-green">Answer: Mostly YES — with minor preprocessing needed</span>
          <p style="color:#94a3b8;line-height:1.7;margin-top:0.8rem">
            The file is named <strong style="color:#00d4ff">"fifa_cleaned.csv"</strong> — the raw FIFA 22 data
            from SoFIFA was already processed. The major cleaning steps our notebook adds:
          </p>
        </div>
        """, unsafe_allow_html=True)

        steps = [
            ("✅ Done before","No duplicates","The 19,239 rows are all unique players","badge-green"),
            ("✅ Done before","Consistent types","Numbers are numbers, strings are strings","badge-green"),
            ("✅ Done before","Column names standardized","snake_case naming throughout","badge-green"),
            ("⚙️ Done in notebook","Fill value_eur / wage_eur","fillna(median) — 243 and 51 values","badge-cyan"),
            ("⚙️ Done in notebook","Remove GK players","Different skill set → analyzed separately","badge-cyan"),
            ("⚙️ Done in notebook","Clip outliers with IQR","Caps extreme values in all skill features","badge-cyan"),
            ("⚙️ Done in notebook","Encode preferred_foot","Right→1, Left→0 (label encoding)","badge-cyan"),
            ("⚙️ Done in notebook","Create position_group","ST/CF→Attacker, CM/CDM→Midfielder, etc.","badge-cyan"),
            ("⚠️ Acceptable missing","Nation columns (96%)","Only top players represent national teams","badge-gold"),
        ]
        for status, title, desc, badge in steps:
            st.markdown(f"""
            <div class="card" style="padding:0.8rem 1.2rem;margin-bottom:0.5rem">
              <span class="badge {badge}" style="margin-bottom:0">{status}</span>
              <strong style="margin-left:0.5rem">{title}</strong>
              <span style="color:#64748b;font-size:0.82rem;margin-left:0.5rem">— {desc}</span>
            </div>
            """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📊 EDA
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Understanding the data through statistics and visualization</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📉 Rating Distribution", "🏃 Age & Physical", "💰 Value & Wage", "📌 Positions"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.histogram(df, x="overall", nbins=35, title="Overall Rating Distribution",
                               color_discrete_sequence=["#00d4ff"])
            fig.update_traces(marker_line_color="#0b1120", marker_line_width=1)
            apply_theme(fig)
            mu_, sg_ = df["overall"].mean(), df["overall"].std()
            fig.add_vline(x=mu_, line_color="#f72585", line_dash="dash", annotation_text=f"Mean={mu_:.1f}", annotation_font_color="#f72585")
            fig.add_vline(x=mu_+sg_, line_color="#fbbf24", line_dash="dot", annotation_text=f"+1σ={mu_+sg_:.0f}", annotation_font_color="#fbbf24")
            fig.add_vline(x=mu_-sg_, line_color="#fbbf24", line_dash="dot", annotation_text=f"-1σ={mu_-sg_:.0f}", annotation_font_color="#fbbf24")
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(df_out, x="position_group", y="overall", color="position_group",
                          title="Overall Rating by Position Group",
                          color_discrete_sequence=["#00d4ff","#f72585","#7c3aed","#fbbf24"])
            apply_theme(fig2)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean", f"{df['overall'].mean():.2f}")
        c2.metric("Median", f"{df['overall'].median():.0f}")
        c3.metric("Std Dev", f"{df['overall'].std():.2f}")
        c4.metric("Range", f"{df['overall'].min():.0f} – {df['overall'].max():.0f}")

        st.markdown("#### Rating Distribution by League (Top 10 Leagues)")
        top_leagues = df["league_name"].value_counts().head(10).index
        fig3 = px.violin(df[df["league_name"].isin(top_leagues)], x="league_name", y="overall",
                         color="league_name", box=True, title="Rating Distribution by League")
        apply_theme(fig3)
        fig3.update_layout(showlegend=False, xaxis_tickangle=-25, height=400)
        st.plotly_chart(fig3, use_container_width=True)

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df_out.sample(3000, random_state=42), x="age", y="overall",
                             color="position_group", opacity=0.6,
                             title="Age vs Overall Rating",
                             color_discrete_sequence=["#00d4ff","#f72585","#7c3aed","#fbbf24"])
            fig.update_traces(marker_size=4)
            apply_theme(fig)
            corr_age = df_out["age"].corr(df_out["overall"])
            fig.add_annotation(text=f"Correlation: {corr_age:.3f}", xref="paper", yref="paper",
                               x=0.02, y=0.98, showarrow=False, font=dict(color="#fbbf24", size=13))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(df, x="age", nbins=30, title="Age Distribution of All Players",
                                color_discrete_sequence=["#7c3aed"])
            fig2.update_traces(marker_line_color="#0b1120", marker_line_width=1)
            apply_theme(fig2)
            fig2.add_vline(x=df["age"].mean(), line_color="#f72585", line_dash="dash",
                           annotation_text=f"Mean={df['age'].mean():.1f}y")
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.scatter(df_out.sample(2000, random_state=1), x="height_cm", y="weight_kg",
                              color="position_group", opacity=0.6,
                              title="Height vs Weight by Position",
                              color_discrete_sequence=["#00d4ff","#f72585","#7c3aed","#fbbf24"])
            fig3.update_traces(marker_size=4)
            apply_theme(fig3)
            st.plotly_chart(fig3, use_container_width=True)
        with col2:
            foot = df_out["preferred_foot"].value_counts().reset_index()
            foot.columns = ["Foot","Count"]
            fig4 = px.pie(foot, names="Foot", values="Count", title="Preferred Foot Distribution",
                          color_discrete_sequence=["#00d4ff","#f72585"], hole=0.5)
            apply_theme(fig4)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            fig = px.scatter(df_out[df_out["value_eur"]>0].sample(3000, random_state=5),
                             x="overall", y="value_eur",
                             color="position_group", opacity=0.6,
                             title="Overall Rating vs Market Value",
                             color_discrete_sequence=["#00d4ff","#f72585","#7c3aed"])
            fig.update_traces(marker_size=4)
            apply_theme(fig)
            fig.update_layout(yaxis_title="Market Value (€)")
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = px.scatter(df_out[df_out["wage_eur"]>0].sample(3000, random_state=3),
                              x="overall", y="wage_eur",
                              color="position_group", opacity=0.6,
                              title="Overall Rating vs Weekly Wage",
                              color_discrete_sequence=["#00d4ff","#f72585","#7c3aed"])
            fig2.update_traces(marker_size=4)
            apply_theme(fig2)
            fig2.update_layout(yaxis_title="Weekly Wage (€)")
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Top 10 Highest Value Players")
        top_val = df_out.nlargest(10,"value_eur")[["short_name","club_name","position_group","overall","value_eur","wage_eur"]]
        top_val["value_eur"] = top_val["value_eur"].apply(lambda x: f"€{x/1e6:.1f}M")
        top_val["wage_eur"]  = top_val["wage_eur"].apply(lambda x: f"€{x/1000:.0f}K/w")
        top_val.columns = ["Player","Club","Position","Overall","Value","Wage"]
        st.dataframe(top_val.reset_index(drop=True), use_container_width=True)

    with tab4:
        col1, col2 = st.columns(2)
        with col1:
            pc = df_out["primary_position"].value_counts().head(15).reset_index()
            pc.columns = ["Position","Count"]
            fig = px.bar(pc, x="Position", y="Count", color="Count",
                         title="Player Count by Position",
                         color_continuous_scale=["#1a2744","#00d4ff"])
            apply_theme(fig)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            pos_rating = df_out.groupby("position_group")["overall"].agg(["mean","median","std"]).reset_index()
            pos_rating.columns = ["Group","Mean","Median","Std"]
            fig2 = px.bar(pos_rating, x="Group", y="Mean", error_y="Std",
                          color="Group", title="Avg Rating by Position Group",
                          color_discrete_sequence=["#00d4ff","#f72585","#7c3aed","#fbbf24"])
            apply_theme(fig2)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        star_feats = ["skill_moves","weak_foot","international_reputation"]
        col1, col2, col3 = st.columns(3)
        for col, feat in zip([col1,col2,col3], star_feats):
            vc = df_out[feat].value_counts().sort_index().reset_index()
            vc.columns = [feat, "Count"]
            fig = px.bar(vc, x=feat, y="Count", title=feat.replace("_"," ").title(),
                         color="Count", color_continuous_scale=["#1a2744","#f72585"])
            apply_theme(fig)
            fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=40,b=20))
            col.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🔗 CORRELATION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🔗  Correlation":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Which features predict Overall Rating best?</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔥 Heatmap", "📊 Feature Correlations"])

    with tab1:
        heat_cols = FEATURES[:16] + ["overall"]
        corr_mat = df_out[heat_cols].dropna().corr()
        fig = px.imshow(corr_mat, color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
                        title="Correlation Heatmap (Top Features + Overall)", aspect="auto",
                        text_auto=".2f")
        apply_theme(fig)
        fig.update_layout(height=620, coloraxis_colorbar=dict(title="r"))
        fig.update_traces(textfont_size=9)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        corr_vals = df_out[FEATURES + ["overall"]].dropna().corr()["overall"].drop("overall").sort_values()
        colors = ["#f72585" if v < 0 else "#00d4ff" for v in corr_vals.values]
        fig = go.Figure(go.Bar(
            x=corr_vals.values, y=corr_vals.index,
            orientation="h",
            marker=dict(color=colors, line=dict(color="#0b1120", width=1))
        ))
        apply_theme(fig, "Correlation of Each Feature with Overall Rating")
        fig.update_layout(height=520, yaxis=dict(autorange="reversed"))
        fig.add_vline(x=0, line_color="#64748b", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Top 5 Most Correlated Features")
        top5 = corr_vals.abs().sort_values(ascending=False).head(5)
        c1,c2,c3,c4,c5 = st.columns(5)
        for col, (feat, val) in zip([c1,c2,c3,c4,c5], top5.items()):
            col.metric(feat.replace("_"," ").title(), f"{val:.3f}")

        st.markdown("---")
        st.markdown("#### Pairplot — Top 4 Features vs Overall")
        top4_feats = corr_vals.abs().sort_values(ascending=False).head(4).index.tolist()
        samp = df_out[top4_feats + ["overall","position_group"]].dropna().sample(1500, random_state=7)
        fig2 = px.scatter_matrix(samp, dimensions=top4_feats + ["overall"],
                                 color="position_group",
                                 color_discrete_sequence=["#00d4ff","#f72585","#7c3aed","#fbbf24"],
                                 opacity=0.5)
        apply_theme(fig2)
        fig2.update_traces(marker_size=2)
        fig2.update_layout(height=600)
        st.plotly_chart(fig2, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📐 PROBABILITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📐  Distributions":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Probability Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Binomial · Poisson · Normal — Applied to FIFA Player Ratings</div>', unsafe_allow_html=True)
    st.markdown("---")

    ratings = df_out["overall"].dropna().values
    mu_r, sg_r = np.mean(ratings), np.std(ratings)
    p_elite = np.mean(ratings >= 80)
    p_wc    = np.mean(ratings >= 88)

    tab1, tab2, tab3 = st.tabs(["🎲 Binomial Distribution", "⚡ Poisson Distribution", "🔔 Normal Distribution"])

    with tab1:
        st.markdown("""
        <div class="card card-accent">
          <span class="badge badge-cyan">Binomial</span>
          <strong>Question:</strong> If we randomly pick <em>n</em> players, what is the probability of getting exactly <em>k</em>
          elite players (overall ≥ 80)?<br>
          <span style="color:#64748b;font-size:0.85rem">Each pick = Bernoulli trial: success = elite player (p = proportion of players ≥ 80)</span>
        </div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns([1,1])
        with c1:
            n_trials = st.slider("Number of players picked (n)", 5, 50, 20)
        with c2:
            st.markdown(f"""
            <div class="card">
              <div class="stat-row"><span class="stat-label">P(elite, ≥80)</span><span class="stat-val">{p_elite:.4f} ({p_elite*100:.1f}%)</span></div>
              <div class="stat-row"><span class="stat-label">Expected elite</span><span class="stat-val">{n_trials*p_elite:.2f}</span></div>
              <div class="stat-row"><span class="stat-label">Std deviation</span><span class="stat-val">{np.sqrt(n_trials*p_elite*(1-p_elite)):.2f}</span></div>
            </div>
            """, unsafe_allow_html=True)

        bd = binom(n=n_trials, p=p_elite)
        k_vals = np.arange(0, n_trials+1)
        pmf_vals = bd.pmf(k_vals)

        fig = make_subplots(rows=1, cols=2, subplot_titles=["PMF — P(X = k)", "CDF — P(X ≤ k)"])
        fig.add_trace(go.Bar(x=k_vals, y=pmf_vals, marker_color="#00d4ff",
                             marker_line=dict(color="#0b1120",width=1), name="PMF"), row=1, col=1)
        fig.add_vline(x=bd.mean(), line_color="#f72585", line_dash="dash",
                      annotation_text=f"Mean={bd.mean():.1f}", annotation_font_color="#f72585", row=1, col=1)
        fig.add_trace(go.Scatter(x=k_vals, y=bd.cdf(k_vals), mode="lines+markers",
                                 line=dict(color="#7c3aed", width=2.5),
                                 marker=dict(color="#7c3aed", size=6), name="CDF"), row=1, col=2)
        apply_theme(fig)
        fig.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig, use_container_width=True)

        k_q = st.slider("Find P(exactly k elite players):", 0, n_trials, 3)
        cc1,cc2,cc3 = st.columns(3)
        cc1.metric(f"P(X = {k_q})",    f"{bd.pmf(k_q):.4f}")
        cc2.metric(f"P(X ≤ {k_q})",    f"{bd.cdf(k_q):.4f}")
        cc3.metric(f"P(X ≥ {k_q})",    f"{bd.sf(k_q-1):.4f}")

    with tab2:
        st.markdown("""
        <div class="card card-pink">
          <span class="badge badge-pink">Poisson</span>
          <strong>Question:</strong> In a random batch of <em>n</em> players, how many world-class players (≥88) do we expect?<br>
          <span style="color:#64748b;font-size:0.85rem">Poisson models rare events — λ = average count per batch</span>
        </div>
        """, unsafe_allow_html=True)

        batch = st.slider("Batch size (n players):", 20, 200, 50)
        lam = batch * p_wc
        pd_dist = poisson(mu=lam)
        k_p = np.arange(0, max(15, int(lam*3)+5))
        pmf_p = pd_dist.pmf(k_p)
        sim = np.random.poisson(lam, size=5000)

        c1,c2 = st.columns([1,1])
        with c1:
            st.markdown(f"""
            <div class="card">
              <div class="stat-row"><span class="stat-label">P(world-class ≥88)</span><span class="stat-val">{p_wc:.4f}</span></div>
              <div class="stat-row"><span class="stat-label">λ (expected)</span><span class="stat-val">{lam:.3f}</span></div>
              <div class="stat-row"><span class="stat-label">P(0 world-class)</span><span class="stat-val">{pd_dist.pmf(0):.4f}</span></div>
              <div class="stat-row"><span class="stat-label">P(≥3 world-class)</span><span class="stat-val">{pd_dist.sf(2):.4f}</span></div>
            </div>
            """, unsafe_allow_html=True)
        with c2:
            fig = make_subplots(rows=1, cols=2, subplot_titles=["PMF", "Simulation vs Theory"])
            fig.add_trace(go.Bar(x=k_p, y=pmf_p, marker_color="#f72585",
                                 marker_line=dict(color="#0b1120",width=1), name="PMF"), row=1, col=1)
            fig.add_vline(x=lam, line_color="#fbbf24", line_dash="dash",
                          annotation_text=f"λ={lam:.2f}", annotation_font_color="#fbbf24", row=1, col=1)
            bins_p = np.arange(0, k_p.max()+1) - 0.5
            sim_hist, _ = np.histogram(sim, bins=bins_p, density=True)
            sim_x = (bins_p[:-1] + bins_p[1:]) / 2
            fig.add_trace(go.Bar(x=sim_x, y=sim_hist, marker_color="#7c3aed", opacity=0.6, name="Simulated"), row=1, col=2)
            fig.add_trace(go.Scatter(x=k_p, y=pmf_p, mode="lines+markers",
                                     line=dict(color="#f72585", width=2), name="Theoretical"), row=1, col=2)
            apply_theme(fig)
            fig.update_layout(showlegend=False, height=300)
            st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.markdown("""
        <div class="card card-gold">
          <span class="badge badge-gold">Normal (Gaussian)</span>
          <strong>Question:</strong> Does overall_rating follow a Normal distribution?<br>
          <span style="color:#64748b;font-size:0.85rem">Central Limit Theorem: large samples of real measurements cluster around a mean</span>
        </div>
        """, unsafe_allow_html=True)

        mu_f, sig_f = norm.fit(ratings)
        x_r = np.linspace(ratings.min()-5, ratings.max()+5, 300)
        pdf_f = norm.pdf(x_r, mu_f, sig_f)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("μ (Mean)",     f"{mu_f:.2f}")
        c2.metric("σ (Std Dev)",  f"{sig_f:.2f}")
        c3.metric("P(rating>80)", f"{norm.sf(80,mu_f,sig_f)*100:.1f}%")
        c4.metric("95th pct",     f"{norm.ppf(0.95,mu_f,sig_f):.1f}")

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=["Histogram + Normal Fit", "CDF", "Q-Q Plot (Normality Check)"])
        # Histogram
        fig.add_trace(go.Histogram(x=ratings, nbinsx=35, histnorm="probability density",
                                   marker_color="#7c3aed", opacity=0.75,
                                   marker_line=dict(color="#0b1120",width=0.5), name="Data"), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_r, y=pdf_f, mode="lines",
                                 line=dict(color="#f72585",width=2.5), name=f"Normal(μ={mu_f:.1f})"), row=1, col=1)
        fig.add_vline(x=mu_f, line_color="#fbbf24", line_dash="dash", row=1, col=1)
        # CDF
        fig.add_trace(go.Scatter(x=x_r, y=norm.cdf(x_r, mu_f, sig_f), mode="lines",
                                 line=dict(color="#7c3aed",width=2.5), name="CDF"), row=1, col=2)
        fig.add_hline(y=0.95, line_color="#f72585", line_dash="dot", row=1, col=2)
        fig.add_hline(y=0.50, line_color="#fbbf24", line_dash="dot", row=1, col=2)
        # Q-Q
        (osm, osr), (slope, intercept, r_qq) = probplot(ratings, dist="norm")
        fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers",
                                 marker=dict(color="#00d4ff",size=2,opacity=0.5), name="Q-Q"), row=1, col=3)
        fit_y = slope*np.array(osm)+intercept
        fig.add_trace(go.Scatter(x=osm, y=fit_y, mode="lines",
                                 line=dict(color="#f72585",width=2), name=f"R²={r_qq**2:.3f}"), row=1, col=3)
        apply_theme(fig)
        fig.update_layout(showlegend=False, height=400)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card card-green">
          <span class="badge badge-green">Interpretation</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin:0.5rem 0 0">
            <strong style="color:#10b981">Q-Q R² = {r_qq**2:.4f}</strong> — very close to 1, meaning the overall rating
            distribution closely follows a Normal distribution. The Q-Q plot points align tightly along the diagonal line.
            This confirms we can apply Normal distribution probabilities to FIFA player ratings.
          </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 📈 REGRESSION
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Normal Equation · Gradient Descent · Model Evaluation</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3, tab4 = st.tabs(["📐 Normal Equation", "⚙️ Gradient Descent", "📊 Comparison", "🔢 Coefficients"])

    with tab1:
        st.markdown("""
        <div class="card card-accent">
          <span class="badge badge-cyan">Normal Equation</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin:0.5rem 0 0">
            Solves for the optimal weights <strong style="color:#00d4ff">W</strong> analytically (in one step, no iterations).
            Uses the closed-form solution from matrix algebra.
          </p>
          <div class="formula-box">
W = (Xᵀ · X)⁻¹ · Xᵀ · y<br><br>
where:<br>
  X = feature matrix (with bias column of 1s)<br>
  y = target vector (overall rating)<br>
  W = weight vector (what we want to find)
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score",  f"{m_ne['r2']:.4f}", "88.8% variance explained")
        c2.metric("MAE",       f"{m_ne['mae']:.4f}", "±1.76 rating points")
        c3.metric("RMSE",      f"{m_ne['rmse']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_all, y=y_ne, mode="markers",
                                 marker=dict(color="#00d4ff", size=3, opacity=0.3), name="Predictions"))
        fig.add_trace(go.Scatter(x=[y_all.min(),y_all.max()], y=[y_all.min(),y_all.max()],
                                 mode="lines", line=dict(color="#f72585",dash="dash",width=2), name="Perfect Prediction"))
        apply_theme(fig, "Actual vs Predicted — Normal Equation")
        fig.update_layout(xaxis_title="Actual Rating", yaxis_title="Predicted Rating", height=420)
        st.plotly_chart(fig, use_container_width=True)

        # Residuals
        residuals = y_all - y_ne
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_ne, y=residuals, mode="markers",
                                  marker=dict(color="#7c3aed", size=3, opacity=0.3), name="Residuals"))
        fig2.add_hline(y=0, line_color="#f72585", line_dash="dash")
        apply_theme(fig2, "Residual Plot — Errors Should Be Random Around 0")
        fig2.update_layout(xaxis_title="Predicted Rating", yaxis_title="Residual (Actual − Predicted)", height=320)
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="card card-purple" style="border-left:3px solid #7c3aed">
          <span class="badge badge-purple">Gradient Descent</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin:0.5rem 0 0">
            An iterative optimization algorithm that starts with random (zero) weights and
            moves "downhill" on the cost function until convergence.
          </p>
          <div class="formula-box" style="color:#a78bfa">
Cost (MSE) = (1/2n) · Σ(ŷᵢ − yᵢ)²<br><br>
Update rule (each iteration):<br>
  gradient = (1/n) · Xᵀ · (X·W − y)<br>
  W ← W − α · gradient<br><br>
α (learning rate) = 0.01,  iterations = 1000
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score",      f"{m_gd['r2']:.4f}")
        c2.metric("MAE",           f"{m_gd['mae']:.4f}")
        c3.metric("Final Cost",    f"{cost_hist[-1]:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(y=cost_hist, mode="lines",
                                 line=dict(color="#7c3aed", width=2.5), name="Cost"))
        apply_theme(fig, "Cost Function Convergence — Gradient Descent (1000 iterations)")
        fig.update_layout(xaxis_title="Iteration", yaxis_title="Cost (MSE/2)", height=350)
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_all, y=y_gd, mode="markers",
                                  marker=dict(color="#7c3aed", size=3, opacity=0.3), name="GD Predictions"))
        fig2.add_trace(go.Scatter(x=[y_all.min(),y_all.max()], y=[y_all.min(),y_all.max()],
                                  mode="lines", line=dict(color="#f72585",dash="dash",width=2), name="Perfect"))
        apply_theme(fig2, "Actual vs Predicted — Gradient Descent")
        fig2.update_layout(xaxis_title="Actual Rating", yaxis_title="Predicted Rating", height=380)
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("### Normal Equation vs Gradient Descent — Side by Side")
        metrics_rows = [
            ("R² Score",  m_ne["r2"],   m_gd["r2"],   "higher is better"),
            ("MAE",       m_ne["mae"],  m_gd["mae"],  "lower is better"),
            ("RMSE",      m_ne["rmse"], m_gd["rmse"], "lower is better"),
        ]
        for label, ne_v, gd_v, note in metrics_rows:
            c1,c2,c3 = st.columns([1,1,1])
            c1.metric(f"NE — {label}",  f"{ne_v:.4f}")
            c2.metric(f"GD — {label}",  f"{gd_v:.4f}", f"diff: {gd_v-ne_v:+.4f}")
            c3.markdown(f"<div style='padding-top:1.2rem;color:#64748b;font-size:0.82rem'>{note}</div>", unsafe_allow_html=True)

        st.markdown("""
        <div class="card card-gold" style="margin-top:1rem">
          <span class="badge badge-gold">Why the difference?</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin:0.5rem 0 0">
            Normal Equation finds the <strong style="color:#fbbf24">exact mathematical optimum</strong> in one step.
            Gradient Descent with only 1000 iterations and lr=0.01 gets <strong>close but not identical</strong> —
            it needs more iterations or a better learning rate to fully converge. In practice, Gradient Descent
            scales better to very large datasets where Normal Equation's matrix inversion becomes too expensive.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Overlay scatter
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_all, y=y_ne, mode="markers",
                                 marker=dict(color="#00d4ff",size=3,opacity=0.2), name="Normal Eq."))
        fig.add_trace(go.Scatter(x=y_all, y=y_gd, mode="markers",
                                 marker=dict(color="#7c3aed",size=3,opacity=0.2), name="Gradient Desc."))
        fig.add_trace(go.Scatter(x=[40,95], y=[40,95], mode="lines",
                                 line=dict(color="#f72585",dash="dash",width=2), name="Perfect"))
        apply_theme(fig, "Overlay — Normal Equation (cyan) vs Gradient Descent (purple)")
        fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted", height=420)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.markdown("### Feature Weights — What Does the Model Value?")
        coef_df = pd.DataFrame({"Feature": list(coefs.keys()), "Weight": list(coefs.values())})
        coef_df["Abs_W"] = coef_df["Weight"].abs()
        coef_df = coef_df.sort_values("Weight")
        colors = ["#f72585" if w < 0 else "#00d4ff" for w in coef_df["Weight"]]
        fig = go.Figure(go.Bar(x=coef_df["Weight"], y=coef_df["Feature"],
                               orientation="h", marker=dict(color=colors, line=dict(color="#0b1120",width=1))))
        apply_theme(fig, "Feature Coefficients (after StandardScaler)")
        fig.update_layout(height=520, yaxis=dict(autorange="reversed"))
        fig.add_vline(x=0, line_color="#64748b", line_dash="dash")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="card card-accent">
          <span class="badge badge-cyan">How to read this</span>
          <p style="color:#94a3b8;font-size:0.85rem;line-height:1.7;margin:0.5rem 0 0">
            Because features were <strong style="color:#00d4ff">standardized (z-scored)</strong> before regression,
            the weights are comparable. A <strong>larger positive weight</strong> = that feature strongly
            <em>increases</em> the predicted overall rating. A <strong>negative weight</strong> = that feature
            inversely predicts overall rating (e.g., high age reduces rating when other factors are held constant).
          </p>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# 🎮 PREDICT A PLAYER
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🎮  Predict a Player":
    st.markdown('<div class="hero-title" style="font-size:2.2rem">Predict a Player\'s Rating</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter player attributes → our Normal Equation model predicts the overall rating</div>', unsafe_allow_html=True)
    st.markdown("---")

    feat_info = {
        "age":                    (15, 45, 25, "Age (years)"),
        "value_eur":              (0, 200_000_000, 5_000_000, "Market Value (€)"),
        "wage_eur":               (0, 600_000, 10_000, "Weekly Wage (€)"),
        "movement_reactions":     (1, 99, 65, "Reactions"),
        "mentality_composure":    (1, 99, 65, "Composure"),
        "attacking_short_passing":(1, 99, 65, "Short Passing"),
        "skill_ball_control":     (1, 99, 65, "Ball Control"),
        "mentality_vision":       (1, 99, 60, "Vision"),
        "mentality_positioning":  (1, 99, 60, "Positioning"),
        "attacking_finishing":    (1, 99, 55, "Finishing"),
        "skill_dribbling":        (1, 99, 65, "Dribbling"),
        "skill_long_passing":     (1, 99, 60, "Long Passing"),
        "power_stamina":          (1, 99, 65, "Stamina"),
        "attacking_volleys":      (1, 99, 55, "Volleys"),
        "skill_curve":            (1, 99, 55, "Curve"),
        "power_shot_power":       (1, 99, 65, "Shot Power"),
        "power_long_shots":       (1, 99, 55, "Long Shots"),
        "mentality_aggression":   (1, 99, 55, "Aggression"),
        "attacking_crossing":     (1, 99, 55, "Crossing"),
        "skill_fk_accuracy":      (1, 99, 50, "FK Accuracy"),
    }

    st.markdown("#### ⚡ Quick Templates")
    preset_col = st.columns(4)
    presets = {
        "World Class Striker": dict(age=27, value_eur=120_000_000, wage_eur=350_000, movement_reactions=92, mentality_composure=90, attacking_short_passing=85, skill_ball_control=88, mentality_vision=84, mentality_positioning=93, attacking_finishing=93, skill_dribbling=87, skill_long_passing=75, power_stamina=80, attacking_volleys=85, skill_curve=82, power_shot_power=90, power_long_shots=84, mentality_aggression=77, attacking_crossing=78, skill_fk_accuracy=75),
        "Elite Midfielder":    dict(age=26, value_eur=90_000_000,  wage_eur=250_000, movement_reactions=88, mentality_composure=88, attacking_short_passing=91, skill_ball_control=90, mentality_vision=92, mentality_positioning=85, attacking_finishing=75, skill_dribbling=88, skill_long_passing=89, power_stamina=85, attacking_volleys=72, skill_curve=80, power_shot_power=80, power_long_shots=82, mentality_aggression=72, attacking_crossing=83, skill_fk_accuracy=78),
        "Solid Defender":      dict(age=28, value_eur=30_000_000,  wage_eur=80_000,  movement_reactions=75, mentality_composure=75, attacking_short_passing=73, skill_ball_control=72, mentality_vision=65, mentality_positioning=60, attacking_finishing=42, skill_dribbling=65, skill_long_passing=72, power_stamina=80, attacking_volleys=45, skill_curve=55, power_shot_power=72, power_long_shots=58, mentality_aggression=82, attacking_crossing=60, skill_fk_accuracy=55),
        "Young Prospect":      dict(age=19, value_eur=5_000_000,   wage_eur=8_000,   movement_reactions=67, mentality_composure=60, attacking_short_passing=68, skill_ball_control=70, mentality_vision=65, mentality_positioning=63, attacking_finishing=62, skill_dribbling=71, skill_long_passing=60, power_stamina=72, attacking_volleys=55, skill_curve=60, power_shot_power=65, power_long_shots=58, mentality_aggression=55, attacking_crossing=60, skill_fk_accuracy=52),
    }

    selected_preset = None
    for i, (pname, pvals) in enumerate(presets.items()):
        if preset_col[i].button(pname, use_container_width=True):
            selected_preset = pvals
            st.session_state["preset"] = pvals

    if "preset" not in st.session_state:
        st.session_state["preset"] = {}

    if selected_preset:
        st.session_state["preset"] = selected_preset

    preset = st.session_state.get("preset", {})

    st.markdown("---")
    st.markdown("#### 🎛️ Input Player Attributes")
    col1, col2 = st.columns(2)
    inputs = {}

    feature_list = list(feat_info.keys())
    half = len(feature_list) // 2
    for i, feat in enumerate(feature_list):
        mn, mx, dv, label = feat_info[feat]
        default = preset.get(feat, dv)
        col = col1 if i < half else col2
        if feat in ["value_eur", "wage_eur"]:
            inputs[feat] = col.number_input(label, min_value=mn, max_value=mx, value=int(default), step=500_000 if feat=="value_eur" else 1000)
        else:
            inputs[feat] = col.slider(label, mn, mx, int(default))

    st.markdown("---")
    if st.button("⚡  PREDICT OVERALL RATING", use_container_width=True):
        x_input = np.array([[inputs[f] for f in FEATURES]])
        x_scaled = scaler.transform(x_input)
        x_b = np.c_[np.ones(1), x_scaled]
        prediction = float(x_b @ W_ne)
        prediction = np.clip(prediction, 40, 99)

        def rating_color(r):
            if r >= 85: return "#fbbf24"
            elif r >= 75: return "#00d4ff"
            elif r >= 65: return "#10b981"
            else: return "#64748b"

        def rating_label(r):
            if r >= 90: return "World Class ⭐"
            elif r >= 85: return "Elite Player 🔥"
            elif r >= 80: return "Professional ✅"
            elif r >= 75: return "Good Player 👍"
            elif r >= 65: return "Average 📊"
            else: return "Developing 📈"

        clr = rating_color(prediction)
        lbl = rating_label(prediction)

        c1, c2, c3 = st.columns([1,1,1])
        with c2:
            st.markdown(f"""
            <div class="player-card" style="border:2px solid {clr};box-shadow:0 0 30px {clr}40">
              <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#64748b;letter-spacing:0.15em;text-transform:uppercase;margin-bottom:0.5rem">Predicted Rating</div>
              <div class="player-rating" style="color:{clr};text-shadow:0 0 20px {clr}80">{prediction:.0f}</div>
              <div style="font-family:'Exo 2',sans-serif;font-size:1rem;color:{clr};margin-top:0.5rem;font-weight:600">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        percentile = (np.sum(df_out["overall"].dropna() <= prediction) / len(df_out["overall"].dropna()) * 100)
        c1, c2, c3 = st.columns(3)
        c1.metric("Predicted Overall", f"{prediction:.1f}")
        c2.metric("Percentile",        f"Top {100-percentile:.0f}%")
        c3.metric("Category",          lbl.split()[0])

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_out["overall"].dropna(), nbinsx=35, histnorm="probability density",
                                   marker_color="#1a2744", marker_line=dict(color="#0b1120",width=0.5), name="All Players"))
        fig.add_vline(x=prediction, line_color=clr, line_width=3,
                      annotation_text=f"Your Player: {prediction:.0f}", annotation_font_color=clr, annotation_font_size=14)
        apply_theme(fig, "Where Does Your Player Rank?")
        fig.update_layout(height=320, xaxis_title="Overall Rating", yaxis_title="Density")
        st.plotly_chart(fig, use_container_width=True)

        # Top 5 similar players
        df_out_c = df_out.dropna(subset=["overall"])
        df_out_c["diff"] = (df_out_c["overall"] - prediction).abs()
        similar = df_out_c.nsmallest(5,"diff")[["short_name","club_name","position_group","overall"]]
        similar.columns = ["Name","Club","Position","Overall"]
        st.markdown("#### 🤝 Most Similar Players in the Dataset")
        st.dataframe(similar.reset_index(drop=True), use_container_width=True)
