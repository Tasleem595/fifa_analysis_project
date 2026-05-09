import os
import warnings
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import binom, poisson, norm, probplot
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH  = os.path.join(BASE_DIR, "fifa_cleaned.csv")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FIFA 22 Analytics", page_icon="⚽",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Exo+2:wght@300;400;600;700;800&family=DM+Sans:wght@300;400;500&family=Space+Mono:wght@400;700&display=swap');

:root{--bg:#05080f;--card:#0b1120;--cyan:#00d4ff;--pink:#f72585;
      --purple:#7c3aed;--gold:#fbbf24;--green:#10b981;--text:#e2e8f0;
      --muted:#64748b;--border:rgba(0,212,255,0.12);}

html,body,.stApp{background:var(--bg)!important;color:var(--text);font-family:'DM Sans',sans-serif;}
#MainMenu,footer{visibility:hidden;}
header{visibility:visible!important;}

/* ── FIX: prevent top crop from sidebar toggle ── */
.block-container{
    padding-top:4.2rem!important;
    padding-bottom:2rem!important;
    max-width:1400px!important;
}

[data-testid="collapsedControl"]{
    top:0.6rem!important;
    background:rgba(0,212,255,.14)!important;
    border:1px solid rgba(0,212,255,.25)!important;
    border-radius:999px!important;
    box-shadow:0 0 16px rgba(0,212,255,.14)!important;
    z-index:9999!important;
    position:fixed!important;
}
[data-testid="collapsedControl"]:hover{
    background:rgba(0,212,255,.22)!important;
    border-color:rgba(0,212,255,.4)!important;
}

/* ════════════════════════════════════════════════
   SIDEBAR — Complete Professional Redesign
   ════════════════════════════════════════════════ */
[data-testid="stSidebar"]{
    background:linear-gradient(160deg,#010509 0%,#04091a 30%,#060f22 60%,#08142c 100%)!important;
    border-right:1px solid rgba(0,212,255,0.18)!important;
    box-shadow:4px 0 40px rgba(0,0,0,0.6),inset -1px 0 0 rgba(0,212,255,0.06)!important;
    padding-top:0!important;
}
[data-testid="stSidebar"]>div{padding-top:0!important;}
[data-testid="stSidebar"] *{color:var(--text)!important;}

/* Sidebar inner scrollable area */
[data-testid="stSidebar"] section[data-testid="stSidebarContent"]{
    padding:1rem 0.85rem 1.5rem!important;
    display:flex;flex-direction:column;gap:0;
}

/* ── Radio group wrapper ── */
[data-testid="stSidebar"] .stRadio>label{display:none!important;}
[data-testid="stSidebar"] .stRadio>div{
    background:transparent!important;
    border:none!important;
    padding:0!important;
    border-radius:0!important;
    display:flex;flex-direction:column;gap:2px!important;
}

/* ── Each nav item ── */
[data-testid="stSidebar"] .stRadio label{
    display:flex!important;align-items:center!important;
    padding:0.78rem 1rem!important;
    border-radius:10px!important;
    cursor:pointer!important;
    transition:all .22s ease!important;
    font-weight:500!important;
    font-size:0.88rem!important;
    letter-spacing:0.01em!important;
    background:transparent!important;
    border:1px solid transparent!important;
    color:#94a3b8!important;
    margin:0!important;
    border-left:3px solid transparent!important;
}
[data-testid="stSidebar"] .stRadio label:hover{
    background:rgba(0,212,255,0.06)!important;
    border-left-color:rgba(0,212,255,0.35)!important;
    color:#cbd5e1!important;
    transform:translateX(3px)!important;
    border-top-color:transparent!important;
    border-right-color:transparent!important;
    border-bottom-color:transparent!important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"]{
    background:linear-gradient(90deg,rgba(0,212,255,0.14) 0%,rgba(124,58,237,0.07) 100%)!important;
    border-left:3px solid #00d4ff!important;
    border-top-color:rgba(0,212,255,0.1)!important;
    border-right-color:rgba(0,212,255,0.06)!important;
    border-bottom-color:rgba(0,212,255,0.08)!important;
    color:#00d4ff!important;
    box-shadow:0 2px 16px rgba(0,212,255,0.1),inset 0 0 20px rgba(0,212,255,0.04)!important;
    transform:translateX(3px)!important;
}
[data-testid="stSidebar"] .stRadio [aria-checked="true"] p,
[data-testid="stSidebar"] .stRadio [aria-checked="true"] span{
    color:#00d4ff!important;font-weight:700!important;
}

/* Hide the radio dot */
[data-testid="stSidebar"] .stRadio [type="radio"]{display:none!important;}
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] > div:first-child{
    display:none!important;
}

/* ── Sidebar divider ── */
[data-testid="stSidebar"] hr{
    border-color:rgba(0,212,255,0.10)!important;
    margin:.8rem 0!important;
}

/* ── Sidebar markdown ── */
[data-testid="stSidebar"] .stMarkdown{padding-left:.1rem!important;}

/* ════════════════════════
   MAIN CONTENT COMPONENTS
   ════════════════════════ */
[data-testid="metric-container"]{
    background:var(--card)!important;border:1px solid var(--border)!important;
    border-radius:12px!important;padding:.9rem 1.1rem!important;
}
[data-testid="stMetricValue"]{color:var(--cyan)!important;font-family:'Space Mono',monospace!important;font-size:1.7rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.07em;}
[data-testid="stMetricDelta"]>div{font-family:'Space Mono',monospace!important;font-size:.78rem!important;}

h1,h2,h3{font-family:'Exo 2',sans-serif!important;}
h1{color:var(--cyan)!important;font-weight:800!important;}
h2,h3{color:var(--text)!important;font-weight:700!important;}
hr{border-color:var(--border)!important;margin:1.2rem 0!important;}

.stTabs [data-baseweb="tab-list"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:10px!important;padding:6px!important;gap:10px!important;flex-wrap:wrap!important;}
.stTabs [data-baseweb="tab"]{color:var(--muted)!important;font-family:'Exo 2',sans-serif!important;font-size:.84rem!important;border-radius:8px!important;margin:0 4px!important;padding:.55rem .9rem!important;min-width:max-content!important;}
.stTabs [data-baseweb="tab"]:hover{background:rgba(255,255,255,.04)!important;}
.stTabs [aria-selected="true"]{background:linear-gradient(135deg,var(--cyan),#0090b3)!important;color:#000!important;font-weight:700!important;}

.stSelectbox>div>div,.stMultiSelect>div>div{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;}
.stSlider [data-testid="stThumb"]{background:var(--cyan)!important;}
.stSlider [data-testid="stTrackFill"]{background:var(--cyan)!important;}
input[type="number"],.stNumberInput input{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:8px!important;color:var(--text)!important;}
.stButton>button{background:linear-gradient(135deg,var(--cyan),#0090b3)!important;color:#000!important;font-family:'Exo 2',sans-serif!important;font-weight:700!important;border:none!important;border-radius:10px!important;padding:.55rem 2rem!important;font-size:.95rem!important;}
.stButton>button:hover{transform:translateY(-1px);box-shadow:0 4px 20px rgba(0,212,255,.3)!important;}
.stAlert{background:var(--card)!important;border-radius:10px!important;}
.stDataFrame{background:var(--card)!important;border-radius:10px!important;}

.card{background:var(--card);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:.8rem;}
.ca{border-left:3px solid var(--cyan);}
.cp{border-left:3px solid var(--pink);}
.cg{border-left:3px solid var(--green);}
.cx{border-left:3px solid var(--purple);}

.hero-title{font-family:'Exo 2',sans-serif;font-size:2.8rem;font-weight:800;line-height:1.1;
  background:linear-gradient(135deg,var(--cyan) 0%,var(--purple) 55%,var(--pink) 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;}
.hero-sub{font-family:'Space Mono',monospace;font-size:.72rem;color:var(--muted);letter-spacing:.15em;text-transform:uppercase;margin-top:.35rem;}

.badge{display:inline-block;padding:2px 11px;border-radius:20px;font-family:'Space Mono',monospace;font-size:.62rem;letter-spacing:.1em;text-transform:uppercase;margin-bottom:.5rem;}
.bc{background:rgba(0,212,255,.1);border:1px solid var(--cyan);color:var(--cyan);}
.bp{background:rgba(247,37,133,.1);border:1px solid var(--pink);color:var(--pink);}
.bg{background:rgba(16,185,129,.1);border:1px solid var(--green);color:var(--green);}
.bx{background:rgba(124,58,237,.1);border:1px solid var(--purple);color:var(--purple);}

.formula{background:#0a1020;border:1px solid rgba(124,58,237,.3);border-radius:8px;padding:.8rem 1.2rem;
  font-family:'Space Mono',monospace;font-size:.82rem;color:#a78bfa;margin:.6rem 0;}

.rating-card{background:linear-gradient(135deg,#0b1120 60%,#0f1a2e);border:2px solid var(--cyan);
  border-radius:16px;padding:1.5rem;text-align:center;box-shadow:0 0 30px rgba(0,212,255,.15);}
.rating-num{font-family:'Exo 2',sans-serif;font-size:5rem;font-weight:800;color:var(--cyan);
  text-shadow:0 0 20px rgba(0,212,255,.5);line-height:1;}

/* Sidebar nav section label */
.nav-section-label{
    font-family:'Space Mono',monospace!important;
    font-size:.6rem!important;
    letter-spacing:.2em!important;
    text-transform:uppercase!important;
    color:#334155!important;
    padding:.5rem 1rem .3rem!important;
    display:block!important;
}
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme helper ───────────────────────────────────────────────────
def fig_style(fig, title=None, h=400):
    current_title = getattr(getattr(fig, "layout", None), "title", None)
    current_text = getattr(current_title, "text", None) if current_title is not None else None
    final_title = title if title is not None else current_text

    layout_kwargs = dict(
        template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(11,17,32,0.6)", height=h,
        font=dict(family="DM Sans", color="#e2e8f0"),
        margin=dict(l=10, r=10, t=45 if final_title else 20, b=10),
        colorway=["#00d4ff","#f72585","#7c3aed","#fbbf24","#10b981","#fb923c"],
    )
    if final_title:
        layout_kwargs["title_text"] = final_title
        layout_kwargs["title_font"] = dict(family="Exo 2", size=14, color="#e2e8f0")

    fig.update_layout(**layout_kwargs)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.05)", zeroline=False)
    return fig

# ── Features (exactly from notebook cell 15 + cell 27 which removes preferred_foot)
FINAL_FEATURES = [
    'age', 'value_eur', 'wage_eur',
    'movement_reactions', 'mentality_composure',
    'attacking_short_passing', 'skill_ball_control',
    'mentality_vision', 'mentality_positioning',
    'attacking_finishing', 'skill_dribbling',
    'skill_long_passing', 'power_stamina',
    'attacking_volleys', 'skill_curve', 'power_shot_power',
    'power_long_shots', 'mentality_aggression',
    'attacking_crossing', 'skill_fk_accuracy'
]

# ── Numeric features for heatmap (from notebook cell 14) ──────────────────────
NUMERIC_FEATURES = [
    'age','height_cm','weight_kg','value_eur','wage_eur',
    'attacking_finishing','skill_dribbling','attacking_short_passing',
    'movement_sprint_speed','movement_reactions','skill_ball_control',
    'mentality_vision','power_stamina','power_strength',
    'attacking_heading_accuracy','skill_long_passing','attacking_volleys',
    'skill_curve','movement_acceleration','movement_agility',
    'movement_balance','power_shot_power','power_jumping','power_long_shots',
    'mentality_aggression','mentality_interceptions','mentality_positioning',
    'mentality_penalties','mentality_composure',
    'defending_standing_tackle','defending_sliding_tackle',
    'attacking_crossing','skill_fk_accuracy'
]

# ── Data loading + preprocessing — matches notebook exactly ───────────────────
@st.cache_data
def load_data():
    df = pd.read_csv(CSV_PATH, low_memory=False)

    # Cell 16: fill missing values
    df['value_eur'] = df['value_eur'].fillna(df['value_eur'].median())
    df['wage_eur']  = df['wage_eur'].fillna(df['wage_eur'].median())

    # Cell 17: remove GK
    df_outfield = df[~df['player_positions'].str.contains('GK', na=False)].copy()

    # Cell 18: IQR clipping (exact loop from notebook)
    for feature in FINAL_FEATURES:
        Q1 = df_outfield[feature].quantile(0.25)
        Q3 = df_outfield[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_fence = Q1 - 1.5 * IQR
        upper_fence = Q3 + 1.5 * IQR
        df_outfield[feature] = df_outfield[feature].clip(lower=lower_fence, upper=upper_fence)

    # Cell 20-21: positions
    df_outfield['primary_position'] = df_outfield['player_positions'].str.split(',').str[0]
    def position_group(pos):
        if pos in ['ST','CF','LW','RW']:          return 'Attacker'
        elif pos in ['CM','CAM','CDM','LM','RM','LWB','RWB']: return 'Midfielder'
        elif pos in ['CB','LB','RB']:             return 'Defender'
        else:                                      return 'Other'
    df_outfield['position_group'] = df_outfield['primary_position'].apply(position_group)

    return df, df_outfield

# ── Regression — matches notebook cells 28-41 exactly ────────────────────────
@st.cache_data
def run_regression(df_outfield):
    # Cell 28
    X = df_outfield[FINAL_FEATURES].dropna()
    y = df_outfield.loc[X.index, 'overall']
    X = X.values; y = y.values

    # Cell 29: train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)

    # Cell 30: StandardScaler
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Cell 31-32: numpy + bias column
    X_train_b = np.c_[np.ones(X_train_s.shape[0]), X_train_s]
    X_test_b  = np.c_[np.ones(X_test_s.shape[0]),  X_test_s]

    # Cell 33: Normal Equation  W = (XᵀX)⁻¹ Xᵀy
    W_ne = np.linalg.inv(X_train_b.T @ X_train_b) @ X_train_b.T @ y_train

    # Cell 34-35: NE predictions + metrics
    y_pred_ne = X_test_b @ W_ne
    ne = dict(mae=mean_absolute_error(y_test, y_pred_ne),
              rmse=np.sqrt(mean_squared_error(y_test, y_pred_ne)),
              r2=r2_score(y_test, y_pred_ne))

    # Cell 38-39: Gradient Descent (exact copy from notebook)
    def compute_cost(X, y, W):
        n = len(y)
        errors = X @ W - y
        return (1/(2*n)) * np.sum(errors**2)

    W_gd = np.zeros(X_train_b.shape[1])
    cost_history = []
    n = len(y_train)
    for i in range(1000):
        errors   = X_train_b @ W_gd - y_train
        gradient = (1/n) * (X_train_b.T @ errors)
        W_gd     = W_gd - 0.01 * gradient
        cost_history.append(compute_cost(X_train_b, y_train, W_gd))

    # Cell 41: GD metrics
    y_pred_gd = X_test_b @ W_gd
    gd = dict(mae=mean_absolute_error(y_test, y_pred_gd),
              rmse=np.sqrt(mean_squared_error(y_test, y_pred_gd)),
              r2=r2_score(y_test, y_pred_gd))

    return W_ne, W_gd, y_test, y_pred_ne, y_pred_gd, cost_history, ne, gd, scaler

# ── Load everything ────────────────────────────────────────────────────────────
df, df_out = load_data()
W_ne, W_gd, y_test, y_pred_ne, y_pred_gd, cost_hist, m_ne, m_gd, scaler = run_regression(df_out)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Professional redesign
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    # ── Branding ──
    st.markdown("""
    <div style="
        text-align:center;
        padding:1.4rem .8rem 1.2rem;
        background:linear-gradient(135deg,rgba(0,212,255,.10) 0%,rgba(124,58,237,.12) 100%);
        border-radius:14px;
        border:1px solid rgba(0,212,255,.18);
        box-shadow:0 0 30px rgba(0,212,255,.08);
        margin-bottom:.5rem;
    ">
      <div style="font-size:2.6rem;line-height:1;filter:drop-shadow(0 0 14px rgba(0,212,255,.5))">⚽</div>
      <div style="
        font-family:'Exo 2',sans-serif;font-size:1.3rem;font-weight:800;
        background:linear-gradient(135deg,#00d4ff,#7c3aed);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;background-clip:text;
        margin-top:.45rem;letter-spacing:.03em;
      ">FIFA 22</div>
      <div style="
        font-family:'Space Mono',monospace;font-size:.6rem;
        color:#475569;letter-spacing:.22em;text-transform:uppercase;margin-top:.2rem;
      ">PLAYER ANALYTICS</div>
      <div style="
        display:flex;justify-content:center;gap:.4rem;margin-top:.7rem;flex-wrap:wrap;
      ">
        <span style="background:rgba(0,212,255,.1);border:1px solid rgba(0,212,255,.2);border-radius:20px;
          padding:1px 8px;font-size:.55rem;font-family:'Space Mono',monospace;color:#00d4ff;
          letter-spacing:.08em;">19,239 PLAYERS</span>
        <span style="background:rgba(124,58,237,.1);border:1px solid rgba(124,58,237,.2);border-radius:20px;
          padding:1px 8px;font-size:.55rem;font-family:'Space Mono',monospace;color:#a78bfa;
          letter-spacing:.08em;">110 FEATURES</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Nav label ──
    st.markdown('<span class="nav-section-label">Navigation</span>', unsafe_allow_html=True)

    page = st.radio("Navigate", [
        "🏠  Overview",
        "🧹  Data Cleaning",
        "📊  EDA",
        "🔗  Correlation",
        "📐  Distributions",
        "📈  Regression",
        "🎮  Predict a Player",
    ], label_visibility="collapsed")

    st.divider()

    # ── Bottom info ──
    st.markdown("""
    <div style="padding:.4rem .2rem;display:flex;flex-direction:column;gap:.5rem;">
      <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:.6rem .9rem;">
        <div style="font-size:.6rem;font-family:'Space Mono',monospace;color:#334155;
          letter-spacing:.12em;text-transform:uppercase;margin-bottom:.3rem;">Data Source</div>
        <div style="font-size:.75rem;color:#64748b;font-family:'DM Sans',sans-serif;">
          FIFA 22 · SoFIFA Dataset</div>
      </div>
      <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
        border-radius:8px;padding:.6rem .9rem;">
        <div style="font-size:.6rem;font-family:'Space Mono',monospace;color:#334155;
          letter-spacing:.12em;text-transform:uppercase;margin-bottom:.3rem;">Stack</div>
        <div style="font-size:.75rem;color:#64748b;font-family:'DM Sans',sans-serif;">
          Streamlit · Plotly · NumPy · scikit-learn</div>
      </div>
    </div>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 🏠 OVERVIEW
# ═══════════════════════════════════════════════════════════════════
if page == "🏠  Overview":
    st.markdown('<div class="hero-title">FIFA 22<br>Player Analytics</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Probability & Statistics · EDA · Linear Regression</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    <div class="card cx">
      <span class="badge bx">Interactive Football Analytics Dashboard</span>
      <p style="color:#94a3b8;font-size:.9rem;line-height:1.8;margin:.5rem 0 0">
        This dashboard combines <b style="color:#00d4ff">Exploratory Data Analysis</b>,
        <b style="color:#7c3aed"> Probability & Statistics</b>, and
        <b style="color:#10b981">Machine Learning Regression</b> into one professional FIFA 22 analytics platform.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Total Players",      f"{len(df):,}")
    c2.metric("Outfield Players",   f"{len(df_out):,}")
    c3.metric("Nationalities",      f"{df['nationality_name'].nunique()}")
    c4.metric("Clubs",              f"{df['club_name'].nunique()}")
    c5.metric("Avg Overall Rating", f"{df['overall'].mean():.1f}")

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2 = st.columns([1.4, 1])

    with col1:
        st.markdown("### 🌍 Top 15 Nationalities by Player Count")
        nat = df['nationality_name'].value_counts().head(15).reset_index()
        nat.columns = ['Nation','Players']
        fig = px.bar(nat, x='Players', y='Nation', orientation='h',
                     color='Players', color_continuous_scale=['#1a2744','#00d4ff'])
        fig_style(fig, h=380)
        fig.update_layout(yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### ⚽ Position Groups")
        pg = df_out['position_group'].value_counts().reset_index()
        pg.columns = ['Group','Count']
        fig2 = px.pie(pg, names='Group', values='Count', hole=0.55,
                      color_discrete_sequence=['#00d4ff','#7c3aed','#f72585','#fbbf24'])
        fig_style(fig2, 'Position Group Distribution', h=380)
        fig2.update_traces(textfont_color='white', textfont_family='Space Mono', textfont_size=11)
        fig2.update_layout(legend=dict(orientation='h', y=-0.05))
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🏆 Top 20 Rated Players")
    top20 = df.nlargest(20,'overall')[['short_name','nationality_name','club_name',
                                       'player_positions','overall','potential','age','value_eur']].copy()
    top20['value_eur'] = top20['value_eur'].apply(lambda x: f"€{x/1e6:.1f}M")
    top20.columns = ['Name','Nation','Club','Positions','Overall','Potential','Age','Value']
    top20.index = range(1, 21)
    st.dataframe(top20, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════
# 🧹 DATA CLEANING
# ═══════════════════════════════════════════════════════════════════
elif page == "🧹  Data Cleaning":
    st.markdown('<div class="hero-title" style="font-size:2rem">Data Cleaning</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Missing values · Outlier detection with IQR</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔍 Missing Values", "📦 Outliers (IQR)"])

    # ── Missing Values ─────────────────────────────────────────────
    with tab1:
        miss = df.isnull().sum()
        miss_pct = (miss / len(df) * 100).round(2)
        miss_df = pd.DataFrame({'Column': miss.index, 'Missing': miss.values, 'Pct (%)': miss_pct.values})
        miss_df = miss_df[miss_df['Missing'] > 0].sort_values('Missing', ascending=False)

        col1, col2 = st.columns([1.5, 1])
        with col1:
            fig = px.bar(miss_df.head(20), x='Pct (%)', y='Column', orientation='h',
                         color='Pct (%)', color_continuous_scale=['#1a2744','#f72585'],
                         title='Top 20 Columns — % Missing')
            fig_style(fig, h=430)
            fig.update_layout(yaxis=dict(autorange='reversed'), coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### After Cleaning (key columns)")
            key = ['overall','age','value_eur','wage_eur','movement_reactions',
                   'skill_ball_control','mentality_composure']
            after = df[key].isnull().sum().reset_index()
            after.columns = ['Column','Missing']
            after['Status'] = after['Missing'].apply(lambda x: '✅ Clean' if x==0 else f'⚠️ {x}')
            st.dataframe(after, use_container_width=True, hide_index=True)

            st.markdown("""
            <div class="card ca" style="margin-top:.8rem">
              <p style="color:#94a3b8;font-size:.84rem;line-height:1.7;margin:0">
                <b style="color:#00d4ff">nation_* (96% missing)</b><br>Most players don't play for national teams — expected.<br><br>
                <b style="color:#00d4ff">value_eur / wage_eur</b><br>Filled with <b>median</b> (robust to outliers).<br><br>
                <b style="color:#00d4ff">pace / physic (11%)</b><br>GK players — removed before regression.
              </p>
            </div>
            """, unsafe_allow_html=True)

        # Missing values in final_features only (from notebook cell 15)
        st.markdown("#### Missing Values in Regression Features")
        feat_miss = df[FINAL_FEATURES].isnull().sum().reset_index()
        feat_miss.columns = ['Feature','Missing Before']
        feat_miss['Missing After'] = 0
        feat_miss['Fix'] = feat_miss.apply(
            lambda r: 'fillna(median)' if r['Feature'] in ['value_eur','wage_eur'] else 'None needed', axis=1)
        st.dataframe(feat_miss, use_container_width=True, hide_index=True)

    # ── Outliers ───────────────────────────────────────────────────
    with tab2:
        st.markdown("""
        <div class="card ca">
          <span class="badge bc">IQR Method — Exactly as in Notebook</span>
          <div class="formula">
Q1 = 25th percentile,  Q3 = 75th percentile<br>
IQR = Q3 − Q1<br>
Lower Fence = Q1 − 1.5 × IQR<br>
Upper Fence = Q3 + 1.5 × IQR<br><br>
Values outside fences → clipped (not deleted)
          </div>
        </div>
        """, unsafe_allow_html=True)

        feat_sel = st.selectbox("Choose a feature to visualize:", FINAL_FEATURES)

        raw = df_out[feat_sel].dropna()
        Q1, Q3 = raw.quantile(0.25), raw.quantile(0.75)
        IQR = Q3 - Q1
        lf, uf = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        clipped = raw.clip(lower=lf, upper=uf)
        n_out = int(((raw < lf) | (raw > uf)).sum())

        col1, col2 = st.columns(2)
        with col1:
            fig = go.Figure(go.Box(y=raw, name='Before', marker_color='#f72585', boxmean=True))
            fig_style(fig, 'Before IQR Clipping', h=350)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig2 = go.Figure(go.Box(y=clipped, name='After', marker_color='#10b981', boxmean=True))
            fig_style(fig2, 'After IQR Clipping', h=350)
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Lower Fence", f"{lf:.2f}")
        c2.metric("Upper Fence", f"{uf:.2f}")
        c3.metric("Outliers Found", f"{n_out:,}")
        c4.metric("Action", "Clipped")

        # Summary table for all features (notebook cell 18 output)
        st.markdown("#### IQR Summary — All Regression Features")
        rows = []
        for f in FINAL_FEATURES:
            col_ = df_out[f].dropna()
            q1,q3 = col_.quantile(0.25), col_.quantile(0.75)
            iqr_ = q3 - q1
            lo,hi = q1-1.5*iqr_, q3+1.5*iqr_
            n_ = int(((col_<lo)|(col_>hi)).sum())
            rows.append({'Feature':f,'Q1':round(q1,2),'Q3':round(q3,2),
                         'IQR':round(iqr_,2),'Lower Fence':round(lo,2),
                         'Upper Fence':round(hi,2),'Outliers':n_})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════════════
# 📊 EDA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown('<div class="hero-title" style="font-size:2rem">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Rating distribution · Age analysis · Descriptive statistics</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📉 Overall Rating", "🏃 Age & Physical", "📋 Descriptive Stats"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            # Notebook cell 10
            fig = px.histogram(df, x='overall', nbins=30, title='Player Overall Rating Distribution',
                               color_discrete_sequence=['#00d4ff'])
            fig_style(fig, h=380)
            fig.update_traces(marker_line_color='#0b1120', marker_line_width=1)
            mu_ = df['overall'].mean()
            fig.add_vline(x=mu_, line_color='#f72585', line_dash='dash',
                          annotation_text=f'Mean={mu_:.1f}', annotation_font_color='#f72585')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.box(df_out, x='position_group', y='overall', color='position_group',
                          title='Overall Rating by Position Group',
                          color_discrete_sequence=['#00d4ff','#f72585','#7c3aed','#fbbf24'])
            fig_style(fig2, h=380)
            fig2.update_layout(showlegend=False)
            st.plotly_chart(fig2, use_container_width=True)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Mean",   f"{df['overall'].mean():.2f}")
        c2.metric("Median", f"{df['overall'].median():.0f}")
        c3.metric("Std Dev",f"{df['overall'].std():.2f}")
        c4.metric("Range",  f"{int(df['overall'].min())} – {int(df['overall'].max())}")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            # Notebook cell 11 — age vs overall scatter
            samp = df_out.sample(min(4000, len(df_out)), random_state=42)
            fig = px.scatter(samp, x='age', y='overall', color='position_group',
                             opacity=0.55, title='Age vs Overall Rating',
                             color_discrete_sequence=['#00d4ff','#f72585','#7c3aed','#fbbf24'])
            fig.update_traces(marker_size=4)
            fig_style(fig, h=380)
            # Notebook cell 12 — correlation
            corr_age = df_out['age'].corr(df_out['overall'])
            fig.update_layout(title='Age vs Overall Rating Analysis')
            fig.add_annotation(text=f'Correlation (age, overall) = {corr_age:.3f}',
                               xref='paper', yref='paper', x=0.02, y=0.97,
                               showarrow=False, font=dict(color='#fbbf24', size=12))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            fig2 = px.histogram(df, x='age', nbins=30,
                                title='Age Distribution of All Players',
                                color_discrete_sequence=['#7c3aed'])
            fig_style(fig2, h=380)
            fig2.update_traces(marker_line_color='#0b1120', marker_line_width=1)
            fig2.add_vline(x=df['age'].mean(), line_color='#f72585', line_dash='dash',
                           annotation_text=f"Mean={df['age'].mean():.1f}y",
                           annotation_font_color='#f72585')
            st.plotly_chart(fig2, use_container_width=True)

        col1, col2 = st.columns(2)
        with col1:
            fig3 = px.scatter(samp, x='height_cm', y='weight_kg', color='position_group',
                              opacity=0.55, title='Height vs Weight',
                              color_discrete_sequence=['#00d4ff','#f72585','#7c3aed','#fbbf24'])
            fig3.update_traces(marker_size=4)
            fig_style(fig3, h=350)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            # Notebook cell 24 — preferred foot
            foot = df_out['preferred_foot'].value_counts().reset_index()
            foot.columns = ['Foot','Count']
            fig4 = px.pie(foot, names='Foot', values='Count', hole=0.5,
                          title='Preferred Foot', color_discrete_sequence=['#00d4ff','#f72585'])
            fig_style(fig4, h=350)
            st.plotly_chart(fig4, use_container_width=True)

    with tab3:
        st.markdown("#### 📋 Statistical Summary of All Players (describe table)")
        desc = df[['overall','potential','age','height_cm','weight_kg','value_eur','wage_eur']].describe().round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### 🎯 Overall Rating Statistics (describe summary)")
        c1,c2,c3,c4,c5,c6 = st.columns(6)
        od = df['overall'].describe()
        c1.metric("count", f"{od['count']:.0f}")
        c2.metric("mean",  f"{od['mean']:.2f}")
        c3.metric("std",   f"{od['std']:.2f}")
        c4.metric("min",   f"{od['min']:.0f}")
        c5.metric("median",f"{od['50%']:.0f}")
        c6.metric("max",   f"{od['max']:.0f}")


# ═══════════════════════════════════════════════════════════════════
# 🔗 CORRELATION
# ═══════════════════════════════════════════════════════════════════
elif page == "🔗  Correlation":
    st.markdown('<div class="hero-title" style="font-size:2rem">Correlation Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Which features most predict Overall Rating?</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2 = st.tabs(["🔥 Full Heatmap", "🎯 Final Features Heatmap"])

    with tab1:
        # Notebook cell 14 — big heatmap
        avail = [f for f in NUMERIC_FEATURES if f in df.columns]
        corr_big = df[avail + ['overall']].corr()
        fig = px.imshow(corr_big, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        aspect='auto', text_auto='.1f')
        fig_style(fig, 'All Features Correlation with Overall Rating', h=680)
        fig.update_traces(textfont_size=7)
        fig.update_layout(coloraxis_colorbar=dict(title='r'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        # Notebook cell 23 — final features heatmap
        corr_final = df[FINAL_FEATURES + ['overall']].corr()
        fig2 = px.imshow(corr_final, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                         text_auto='.2f', aspect='auto')
        fig_style(fig2, 'Final Regression Features — Correlation Heatmap', h=600)
        fig2.update_traces(textfont_size=9)
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Correlation of Each Feature with Overall Rating")
        cr = corr_final['overall'].drop('overall').sort_values()
        colors_ = ['#f72585' if v < 0 else '#00d4ff' for v in cr.values]
        fig3 = go.Figure(go.Bar(x=cr.values, y=cr.index, orientation='h',
                                marker=dict(color=colors_, line=dict(color='#0b1120', width=1))))
        fig_style(fig3, h=500)
        fig3.add_vline(x=0, line_color='#64748b', line_dash='dash')
        st.plotly_chart(fig3, use_container_width=True)

        


# ═══════════════════════════════════════════════════════════════════
# 📐 PROBABILITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📐  Distributions":
    st.markdown('<div class="hero-title" style="font-size:2rem">Probability Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Binomial · Poisson · Normal — Applied to FIFA 22</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Notebook cell 43
    ratings = df_out['overall'].dropna().values

    tab1, tab2, tab3 = st.tabs(["🎲 Binomial", "⚡ Poisson", "🔔 Normal"])

    # ══════════════════════════════════════════════════════════
    # BINOMIAL (notebook cells 44-45)
    # FIX: slider is placed FIRST so both graphs react to it
    # ══════════════════════════════════════════════════════════
    with tab1:
        st.markdown("""
        <div class="card ca">
          <span class="badge bc">Binomial Distribution</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            If we randomly pick <b style="color:#00d4ff">20 players</b>, what is the probability of getting
            exactly <em>k</em> elite players (overall ≥ 80)?
          </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Notebook cell 45 — exact fixed parameters ──
        ELITE_THRESHOLD = 80
        n_trials = 20
        p_elite = np.mean(ratings >= ELITE_THRESHOLD)
        binom_dist = binom(n=n_trials, p=p_elite)
        k_values = np.arange(0, n_trials + 1)
        pmf_values = binom_dist.pmf(k_values)

        # ── Fixed summary metrics ──
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("P(elite ≥80)",   f"{p_elite:.4f} ({p_elite*100:.1f}%)")
        c2.metric("P(exactly 5)",    f"{binom_dist.pmf(5):.4f}")
        c3.metric("P(at least 3)",   f"{binom_dist.sf(2):.4f}")
        c4.metric("Expected elite",  f"{binom_dist.mean():.2f}")

        st.markdown("---")

        # ── SLIDER FIRST — so charts below react to it ──
        k_q = st.slider(
            "🎯 Select k to highlight in both charts:",
            min_value=0, max_value=n_trials, value=5,
            help="Move this slider — both the PMF bar and CDF marker update instantly"
        )

        # ── k-specific metrics (shown right after slider) ──
        ck1, ck2, ck3 = st.columns(3)
        ck1.metric(f"P(X = {k_q})",  f"{binom_dist.pmf(k_q):.4f}")
        ck2.metric(f"P(X ≤ {k_q})", f"{binom_dist.cdf(k_q):.4f}")
        ck3.metric(f"P(X ≥ {k_q})", f"{binom_dist.sf(k_q - 1):.4f}")

        # ── Charts built AFTER slider is read — fully reactive ──
        # Highlight selected bar in pink, others in cyan
        bar_colors = ['#f72585' if int(k) == int(k_q) else '#00d4ff' for k in k_values]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                f'PMF — P(X = k)  [k={k_q} highlighted]',
                f'CDF — P(X ≤ k)  [k={k_q} marked]'
            ]
        )

        # PMF bar chart
        fig.add_trace(
            go.Bar(
                x=k_values, y=pmf_values,
                marker_color=bar_colors,
                marker_line=dict(color='#0b1120', width=1),
                name='PMF',
                hovertemplate='k=%{x}<br>P(X=k)=%{y:.4f}<extra></extra>'
            ),
            row=1, col=1
        )
        # Mean line on PMF
        fig.add_vline(
            x=binom_dist.mean(), line_color='#fbbf24', line_dash='dash',
            annotation_text=f"Mean={binom_dist.mean():.1f}",
            annotation_font_color='#fbbf24',
            row=1, col=1
        )
        # Selected k line on PMF
        fig.add_vline(
            x=k_q, line_color='#f72585', line_dash='dot',
            annotation_text=f"k={k_q}",
            annotation_font_color='#f72585',
            annotation_position='top right',
            row=1, col=1
        )

        # CDF line
        fig.add_trace(
            go.Scatter(
                x=k_values, y=binom_dist.cdf(k_values),
                mode='lines+markers',
                line=dict(color='#7c3aed', width=2.5),
                marker=dict(size=5, color='#7c3aed'),
                name='CDF',
                hovertemplate='k=%{x}<br>P(X≤k)=%{y:.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        # Highlighted point on CDF for selected k
        fig.add_trace(
            go.Scatter(
                x=[k_q], y=[binom_dist.cdf(k_q)],
                mode='markers',
                marker=dict(
                    color='#f72585', size=14, symbol='circle',
                    line=dict(color='white', width=2)
                ),
                name=f'k={k_q}',
                hovertemplate=f'k={k_q}<br>P(X≤{k_q})={binom_dist.cdf(k_q):.4f}<extra></extra>'
            ),
            row=1, col=2
        )
        # Horizontal dashed line at CDF value for selected k
        fig.add_hline(
            y=binom_dist.cdf(k_q),
            line_color='#f72585', line_dash='dot', line_width=1,
            annotation_text=f"P(X≤{k_q})={binom_dist.cdf(k_q):.3f}",
            annotation_font_color='#f72585',
            annotation_position='bottom right',
            row=1, col=2
        )

        fig_style(fig, h=400)
        fig.update_layout(
            showlegend=False,
            title_text=f'Binomial (n={n_trials}, p={p_elite:.3f})',
            title_font=dict(family='Exo 2', size=14)
        )
        st.plotly_chart(fig, use_container_width=True)

    # ── POISSON (notebook cells 46-47) ────────────────────────────
    with tab2:
        st.markdown("""
        <div class="card cp">
          <span class="badge bp">Poisson Distribution</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            In a batch of <b style="color:#f72585">50 players</b>, how many world-class players
            (overall ≥ 88) do we expect?
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Notebook cell 47 — exact values
        WORLD_CLASS = 88
        BATCH_SIZE  = 50
        p_wc = np.mean(ratings >= WORLD_CLASS)
        lam  = BATCH_SIZE * p_wc
        pd_dist = poisson(mu=lam)
        k_p = np.arange(0, 15)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("P(world-class ≥88)", f"{p_wc:.4f}")
        c2.metric("λ",                  f"{lam:.4f}")
        c3.metric("P(exactly 0)",        f"{pd_dist.pmf(0):.4f}")
        c4.metric("P(3 or more)",        f"{pd_dist.sf(2):.4f}")

        sim = np.random.poisson(lam, size=5000)
        fig = make_subplots(rows=1, cols=2,
                            subplot_titles=[f'Poisson PMF (λ={lam:.2f})', 'Simulation vs Theoretical'])
        fig.add_trace(go.Bar(x=k_p, y=pd_dist.pmf(k_p), marker_color='#f72585',
                             marker_line=dict(color='#0b1120', width=1)), row=1, col=1)
        fig.add_vline(x=lam, line_color='#fbbf24', line_dash='dash',
                      annotation_text=f'λ={lam:.2f}', annotation_font_color='#fbbf24', row=1, col=1)
        bins_ = np.arange(0, 15) - 0.5
        sh, _ = np.histogram(sim, bins=bins_, density=True)
        sx = (bins_[:-1] + bins_[1:]) / 2
        fig.add_trace(go.Bar(x=sx, y=sh, marker_color='#7c3aed', opacity=0.65, name='Simulated'), row=1, col=2)
        fig.add_trace(go.Scatter(x=k_p, y=pd_dist.pmf(k_p), mode='lines+markers',
                                 line=dict(color='#f72585', width=2.5), name='Theoretical'), row=1, col=2)
        fig_style(fig, h=380)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # ── NORMAL (notebook cells 48-49) ─────────────────────────────
    with tab3:
        st.markdown("""
        <div class="card cx">
          <span class="badge bx">Normal Distribution</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            Does overall_rating follow a Normal distribution? We fit Normal and compute key probabilities.
          </p>
        </div>
        """, unsafe_allow_html=True)

        # Notebook cell 49 — exact values
        ratings_all = df['overall'].dropna()
        mu, sigma = norm.fit(ratings_all)
        p_above_80  = norm.sf(80, mu, sigma)
        p_60_to_75  = norm.cdf(75, mu, sigma) - norm.cdf(60, mu, sigma)
        p_one_sigma = norm.cdf(mu+sigma, mu, sigma) - norm.cdf(mu-sigma, mu, sigma)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("μ (Mean)",          f"{mu:.4f}")
        c2.metric("σ (Std Dev)",       f"{sigma:.4f}")
        c3.metric("P(rating > 80)",    f"{p_above_80:.4f} ({p_above_80*100:.1f}%)")
        c4.metric("P(60 < r < 75)",    f"{p_60_to_75:.4f} ({p_60_to_75*100:.1f}%)")

        x_r = np.linspace(ratings_all.min()-5, ratings_all.max()+5, 300)
        pdf_ = norm.pdf(x_r, mu, sigma)
        (osm, osr), (slope, intercept, r_qq) = probplot(ratings_all, dist='norm')

        fig = make_subplots(rows=1, cols=3,
                            subplot_titles=['Histogram + Normal Fit', 'Normal CDF', 'Q-Q Plot (Normality Check)'])
        fig.add_trace(go.Histogram(x=ratings_all, nbinsx=30, histnorm='probability density',
                                   marker_color='#7c3aed', opacity=0.7,
                                   marker_line=dict(color='#0b1120', width=0.5), name='Data'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_r, y=pdf_, mode='lines',
                                 line=dict(color='#f72585', width=2.5),
                                 name=f'Normal(μ={mu:.1f}, σ={sigma:.1f})'), row=1, col=1)
        fig.add_vline(x=mu, line_color='#fbbf24', line_dash='dash',
                      annotation_text=f'Mean={mu:.1f}', annotation_font_color='#fbbf24', row=1, col=1)
        fig.add_trace(go.Scatter(x=x_r, y=norm.cdf(x_r, mu, sigma), mode='lines',
                                 line=dict(color='#7c3aed', width=2.5)), row=1, col=2)
        fig.add_hline(y=0.95, line_color='#f72585', line_dash='dot',
                      annotation_text='95th pct', annotation_font_color='#f72585', row=1, col=2)
        fig.add_hline(y=0.50, line_color='#fbbf24', line_dash='dot',
                      annotation_text='Median', annotation_font_color='#fbbf24', row=1, col=2)
        fig.add_trace(go.Scatter(x=osm, y=osr, mode='markers',
                                 marker=dict(color='#00d4ff', size=2, opacity=0.5)), row=1, col=3)
        fig.add_trace(go.Scatter(x=osm, y=slope*np.array(osm)+intercept, mode='lines',
                                 line=dict(color='#f72585', width=2),
                                 name=f'R²={r_qq**2:.4f}'), row=1, col=3)
        fig_style(fig, h=420)
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown(f"""
        <div class="card cg">
          <b style="color:#10b981">Q-Q R² = {r_qq**2:.4f}</b> — Close to 1 confirms overall rating
          follows a Normal distribution. P(within 1σ of mean) = <b style="color:#00d4ff">{p_one_sigma:.4f} ({p_one_sigma*100:.1f}%)</b> &nbsp;|&nbsp;
          95th percentile = <b style="color:#00d4ff">{norm.ppf(0.95, mu, sigma):.1f}</b> &nbsp;|&nbsp;
          5th percentile = <b style="color:#00d4ff">{norm.ppf(0.05, mu, sigma):.1f}</b>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 📈 REGRESSION
# ═══════════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    st.markdown('<div class="hero-title" style="font-size:2rem">Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Normal Equation · Gradient Descent · 80/20 train-test split</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📐 Normal Equation", "⚙️ Gradient Descent", "📊 Comparison"])

    # ── NORMAL EQUATION (notebook cells 33-36) ────────────────────
    with tab1:
        st.markdown("""
        <div class="card ca">
          <span class="badge bc">Normal Equation — Closed Form Solution</span>
          <div class="formula">W = (Xᵀ · X)⁻¹ · Xᵀ · y</div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score",  f"{m_ne['r2']:.4f}", "on test set")
        c2.metric("MAE",       f"{m_ne['mae']:.4f}", "rating points")
        c3.metric("RMSE",      f"{m_ne['rmse']:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            # Notebook cell 36 — actual vs predicted
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=y_test, y=y_pred_ne, mode='markers',
                                     marker=dict(color='#00d4ff', size=4, opacity=0.35), name='Predictions'))
            fig.add_trace(go.Scatter(x=[40,95], y=[40,95], mode='lines',
                                     line=dict(color='#f72585', dash='dash', width=2), name='Perfect'))
            fig_style(fig, 'Actual vs Predicted — Normal Equation', h=380)
            fig.update_layout(xaxis_title='Actual Rating', yaxis_title='Predicted Rating')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            residuals = y_test - y_pred_ne
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_pred_ne, y=residuals, mode='markers',
                                      marker=dict(color='#7c3aed', size=4, opacity=0.35)))
            fig2.add_hline(y=0, line_color='#f72585', line_dash='dash')
            fig_style(fig2, 'Residual Plot (Actual − Predicted)', h=380)
            fig2.update_layout(xaxis_title='Predicted', yaxis_title='Residual')
            st.plotly_chart(fig2, use_container_width=True)

    # ── GRADIENT DESCENT (notebook cells 38-41) ───────────────────
    with tab2:
        st.markdown("""
        <div class="card cx">
          <span class="badge bx">Gradient Descent — alpha=0.01 · iterations=1000</span>
          <div class="formula" style="color:#a78bfa">
Cost = (1/2n) · Σ(Xw − y)²<br>
gradient = (1/n) · Xᵀ · (Xw − y)<br>
W ← W − α · gradient
          </div>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("R² Score",   f"{m_gd['r2']:.4f}", "on test set")
        c2.metric("MAE",        f"{m_gd['mae']:.4f}")
        c3.metric("Final Cost", f"{cost_hist[-1]:.4f}")

        col1, col2 = st.columns(2)
        with col1:
            # Notebook cell 40 — cost history
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=cost_hist, mode='lines',
                                     line=dict(color='#7c3aed', width=2.5)))
            fig_style(fig, 'Cost Function — Gradient Descent (1000 iterations)', h=360)
            fig.update_layout(xaxis_title='Iteration', yaxis_title='Cost')
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            # Notebook cell 41 — actual vs predicted GD
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=y_test, y=y_pred_gd, mode='markers',
                                      marker=dict(color='#7c3aed', size=4, opacity=0.35)))
            fig2.add_trace(go.Scatter(x=[40,95], y=[40,95], mode='lines',
                                      line=dict(color='#f72585', dash='dash', width=2)))
            fig_style(fig2, 'Actual vs Predicted — Gradient Descent', h=360)
            fig2.update_layout(xaxis_title='Actual Rating', yaxis_title='Predicted Rating')
            st.plotly_chart(fig2, use_container_width=True)

    # ── COMPARISON ────────────────────────────────────────────────
    with tab3:
        st.markdown("### Normal Equation vs Gradient Descent")
        comp = pd.DataFrame({
            'Metric': ['R² Score','MAE','RMSE'],
            'Normal Equation': [f"{m_ne['r2']:.4f}", f"{m_ne['mae']:.4f}", f"{m_ne['rmse']:.4f}"],
            'Gradient Descent': [f"{m_gd['r2']:.4f}", f"{m_gd['mae']:.4f}", f"{m_gd['rmse']:.4f}"],
        })
        st.dataframe(comp, use_container_width=True, hide_index=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred_ne, mode='markers',
                                 marker=dict(color='#00d4ff', size=3, opacity=0.3), name='Normal Eq.'))
        fig.add_trace(go.Scatter(x=y_test, y=y_pred_gd, mode='markers',
                                 marker=dict(color='#7c3aed', size=3, opacity=0.3), name='Gradient Desc.'))
        fig.add_trace(go.Scatter(x=[40,95], y=[40,95], mode='lines',
                                 line=dict(color='#f72585', dash='dash', width=2), name='Perfect'))
        fig_style(fig, 'Overlay — Normal Equation (cyan) vs Gradient Descent (purple)', h=420)
        fig.update_layout(xaxis_title='Actual Rating', yaxis_title='Predicted Rating',
                          legend=dict(orientation='h', y=1.08))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="card ca" style="margin-top:.5rem">
          <p style="color:#94a3b8;font-size:.84rem;line-height:1.7;margin:0">
            <b style="color:#00d4ff">Normal Equation</b> finds the exact mathematical optimum in one step using matrix inversion.<br>
            <b style="color:#7c3aed">Gradient Descent</b> with 1000 iterations and α=0.01 gets close but doesn't fully converge — more iterations would close the gap.
          </p>
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════
# 🎮 PREDICT A PLAYER
# ═══════════════════════════════════════════════════════════════════
elif page == "🎮  Predict a Player":
    st.markdown('<div class="hero-title" style="font-size:2rem">Predict a Player\'s Rating</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Enter player attributes → Normal Equation model predicts overall rating</div>', unsafe_allow_html=True)
    st.markdown("---")

    # Slider config: feature → (min, max, default, label)
    feat_cfg = {
        'age':                    (15, 45, 25, 'Age (years)'),
        'value_eur':              (0, 200_000_000, 5_000_000, 'Market Value (€)'),
        'wage_eur':               (0, 600_000,     10_000,    'Weekly Wage (€)'),
        'movement_reactions':     (1, 99, 65, 'Reactions'),
        'mentality_composure':    (1, 99, 65, 'Composure'),
        'attacking_short_passing':(1, 99, 65, 'Short Passing'),
        'skill_ball_control':     (1, 99, 65, 'Ball Control'),
        'mentality_vision':       (1, 99, 60, 'Vision'),
        'mentality_positioning':  (1, 99, 60, 'Positioning'),
        'attacking_finishing':    (1, 99, 55, 'Finishing'),
        'skill_dribbling':        (1, 99, 65, 'Dribbling'),
        'skill_long_passing':     (1, 99, 60, 'Long Passing'),
        'power_stamina':          (1, 99, 65, 'Stamina'),
        'attacking_volleys':      (1, 99, 55, 'Volleys'),
        'skill_curve':            (1, 99, 55, 'Curve'),
        'power_shot_power':       (1, 99, 65, 'Shot Power'),
        'power_long_shots':       (1, 99, 55, 'Long Shots'),
        'mentality_aggression':   (1, 99, 55, 'Aggression'),
        'attacking_crossing':     (1, 99, 55, 'Crossing'),
        'skill_fk_accuracy':      (1, 99, 50, 'FK Accuracy'),
    }

    # Quick presets
    presets = {
        '⭐ World Class': dict(age=27,value_eur=120_000_000,wage_eur=350_000,
            movement_reactions=92,mentality_composure=90,attacking_short_passing=85,
            skill_ball_control=88,mentality_vision=84,mentality_positioning=93,
            attacking_finishing=93,skill_dribbling=87,skill_long_passing=75,
            power_stamina=80,attacking_volleys=85,skill_curve=82,power_shot_power=90,
            power_long_shots=84,mentality_aggression=77,attacking_crossing=78,skill_fk_accuracy=75),
        '💪 Solid Pro': dict(age=26,value_eur=25_000_000,wage_eur=60_000,
            movement_reactions=78,mentality_composure=76,attacking_short_passing=75,
            skill_ball_control=76,mentality_vision=72,mentality_positioning=74,
            attacking_finishing=70,skill_dribbling=73,skill_long_passing=71,
            power_stamina=78,attacking_volleys=65,skill_curve=68,power_shot_power=74,
            power_long_shots=68,mentality_aggression=68,attacking_crossing=67,skill_fk_accuracy=60),
        '🌱 Young Prospect': dict(age=19,value_eur=3_000_000,wage_eur=5_000,
            movement_reactions=65,mentality_composure=58,attacking_short_passing=66,
            skill_ball_control=68,mentality_vision=62,mentality_positioning=60,
            attacking_finishing=60,skill_dribbling=70,skill_long_passing=58,
            power_stamina=70,attacking_volleys=52,skill_curve=58,power_shot_power=63,
            power_long_shots=55,mentality_aggression=52,attacking_crossing=58,skill_fk_accuracy=50),
    }

    pc1, pc2, pc3, _ = st.columns([1,1,1,1])
    for col, (pname, pvals) in zip([pc1,pc2,pc3], presets.items()):
        if col.button(pname, use_container_width=True):
            st.session_state['preset'] = pvals

    preset = st.session_state.get('preset', {})

    st.markdown("---")
    st.markdown("#### 🎛️ Player Attributes")
    col1, col2 = st.columns(2)
    inputs = {}
    feats_left  = FINAL_FEATURES[:10]
    feats_right = FINAL_FEATURES[10:]

    for feat in feats_left:
        mn, mx, dv, label = feat_cfg[feat]
        default = int(preset.get(feat, dv))
        if feat in ('value_eur', 'wage_eur'):
            inputs[feat] = col1.number_input(label, min_value=mn, max_value=mx, value=default,
                                              step=500_000 if feat=='value_eur' else 1_000)
        else:
            inputs[feat] = col1.slider(label, mn, mx, default)

    for feat in feats_right:
        mn, mx, dv, label = feat_cfg[feat]
        default = int(preset.get(feat, dv))
        if feat in ('value_eur', 'wage_eur'):
            inputs[feat] = col2.number_input(label, min_value=mn, max_value=mx, value=default,
                                              step=500_000 if feat=='value_eur' else 1_000)
        else:
            inputs[feat] = col2.slider(label, mn, mx, default)

    st.markdown("---")
    if st.button("⚡  PREDICT OVERALL RATING", use_container_width=True):
        # Build input vector in exact feature order
        x_input = np.array([[inputs[f] for f in FINAL_FEATURES]], dtype=float)

        # Scale using the same scaler fitted on X_train
        x_scaled = scaler.transform(x_input)

        # Add bias column
        x_b = np.c_[np.ones(1), x_scaled]

        # Predict — use float(np.squeeze()) to safely convert to Python float
        prediction = float(np.squeeze(x_b @ W_ne))
        prediction = np.clip(prediction, 40, 99)

        def get_color(r):
            if r >= 85: return '#fbbf24'
            elif r >= 75: return '#00d4ff'
            elif r >= 65: return '#10b981'
            else: return '#64748b'

        def get_label(r):
            if r >= 90: return 'World Class ⭐'
            elif r >= 85: return 'Elite Player 🔥'
            elif r >= 80: return 'Professional ✅'
            elif r >= 75: return 'Good Player 👍'
            elif r >= 65: return 'Average 📊'
            else: return 'Developing 📈'

        clr = get_color(prediction)
        lbl = get_label(prediction)
        pct = float(np.mean(df_out['overall'].dropna() <= prediction) * 100)

        _, cc, _ = st.columns([1,1,1])
        with cc:
            st.markdown(f"""
            <div class="rating-card" style="border-color:{clr};box-shadow:0 0 30px {clr}30">
              <div style="font-family:'Space Mono',monospace;font-size:.68rem;color:#64748b;letter-spacing:.15em;text-transform:uppercase">Predicted Overall Rating</div>
              <div class="rating-num" style="color:{clr};text-shadow:0 0 20px {clr}60">{prediction:.0f}</div>
              <div style="font-family:'Exo 2',sans-serif;color:{clr};font-size:1rem;font-weight:600;margin-top:.4rem">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Predicted Overall", f"{prediction:.1f}")
        c2.metric("Better than",       f"{pct:.0f}% of players")
        c3.metric("Top percentile",    f"Top {100-pct:.0f}%")

        fig = go.Figure()
        fig.add_trace(go.Histogram(x=df_out['overall'].dropna(), nbinsx=35,
                                   histnorm='probability density',
                                   marker_color='#1a2744',
                                   marker_line=dict(color='#0b1120', width=0.5), name='All Players'))
        fig.add_vline(x=prediction, line_color=clr, line_width=3,
                      annotation_text=f'Your Player: {prediction:.0f}',
                      annotation_font_color=clr, annotation_font_size=14)
        fig_style(fig, 'Where Does Your Player Rank?', h=300)
        fig.update_layout(xaxis_title='Overall Rating', yaxis_title='Density')
        st.plotly_chart(fig, use_container_width=True)

        # Most similar real players
        df_c = df_out.dropna(subset=['overall']).copy()
        df_c['_diff'] = (df_c['overall'] - prediction).abs()
        sim = df_c.nsmallest(5, '_diff')[['short_name','club_name','position_group','overall']].copy()
        sim.columns = ['Name','Club','Position','Overall']
        st.markdown("#### 🤝 Most Similar Players in Dataset")
        st.dataframe(sim.reset_index(drop=True), use_container_width=True, hide_index=True)


# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#64748b;
font-family:'Space Mono',monospace;font-size:.72rem">
⚽ FIFA 22 Analytics Dashboard · Streamlit + Plotly + Machine Learning<br>
Built with Exploratory Data Analysis, Probability Distributions & Linear Regression
</div>
""", unsafe_allow_html=True)
