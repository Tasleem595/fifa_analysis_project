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

[data-testid="collapsedControl"]{
    background:rgba(0,212,255,.14)!important;
    border:1px solid rgba(0,212,255,.25)!important;
    border-radius:999px!important;
    box-shadow:0 0 16px rgba(0,212,255,.14)!important;
}
[data-testid="collapsedControl"]:hover{
    background:rgba(0,212,255,.22)!important;
    border-color:rgba(0,212,255,.4)!important;
}

.block-container{padding-top:1.2rem!important;padding-bottom:2rem!important;padding-left:1rem!important;padding-right:1rem!important;}

[data-testid="stSidebar"]{
    background:linear-gradient(180deg,#0a1428 0%,#0f1a2e 40%,#091c38 100%)!important;
    border-right:2px solid rgba(0,212,255,0.3)!important;
    box-shadow:0 0 60px rgba(0,212,255,0.15),inset -1px 0 30px rgba(0,212,255,0.05)!important;
    padding-top:0.8rem!important;
}

[data-testid="stSidebar"] *{
    color:var(--text)!important;
}

[data-testid="stSidebar"] .stRadio > div{
    background:rgba(0,50,100,0.3)!important;
    padding:0.8rem!important;
    border-radius:12px!important;
    border:1px solid rgba(0,212,255,0.25)!important;
    backdrop-filter:blur(15px)!important;
    box-shadow:0 0 20px rgba(0,212,255,0.08)!important;
}

[data-testid="stSidebar"] .stRadio label{
    margin:0.4rem 0!important;
    padding:0.85rem 1.2rem!important;
    border-radius:10px!important;
    transition:all .25s ease!important;
    font-weight:600!important;
    font-size:.95rem!important;
    background:rgba(0,212,255,0.02)!important;
    border:1.5px solid rgba(0,212,255,0.1)!important;
    cursor:pointer!important;
    color:#d1d5db!important;
}

[data-testid="stSidebar"] .stRadio label:hover{
    background:linear-gradient(135deg,rgba(0,212,255,.18),rgba(124,58,237,.14))!important;
    transform:translateX(4px) scale(1.02)!important;
    border:1.5px solid rgba(0,212,255,.35)!important;
    box-shadow:0 0 20px rgba(0,212,255,.12)!important;
    color:#00d4ff!important;
}

[data-testid="stSidebar"] .stRadio [aria-checked="true"]{
    background:linear-gradient(135deg,rgba(0,212,255,.28),rgba(124,58,237,.22))!important;
    border:1.5px solid rgba(0,212,255,.45)!important;
    box-shadow:0 0 25px rgba(0,212,255,.2)!important;
    color:#00d4ff!important;
    font-weight:700!important;
}

[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,0.1)!important;
    margin:0.8rem 0!important;
}

[data-testid="stSidebar"] .stMarkdown{
    padding-left:.1rem!important;
}

.sidebar-header{
    text-align:center;
    padding:1.2rem 1rem 1rem;
    background:linear-gradient(135deg,rgba(0,212,255,.08),rgba(124,58,237,.08));
    border-radius:14px;
    border:1px solid rgba(0,212,255,.15);
    box-shadow:0 0 20px rgba(0,212,255,.06);
    margin-bottom:1.2rem;
}

.sidebar-header .emoji{
    font-size:2.8rem;
    filter:drop-shadow(0 0 10px rgba(0,212,255,.3));
    display:block;
    margin-bottom:0.4rem;
}

.sidebar-header .title{
    font-family:'Exo 2',sans-serif;
    font-size:1.3rem;
    font-weight:800;
    color:#00d4ff;
    margin:0;
}

.sidebar-header .subtitle{
    font-family:'Space Mono',monospace;
    font-size:.65rem;
    color:#94a3b8;
    letter-spacing:.15em;
    text-transform:uppercase;
    margin-top:0.3rem;
}

.sidebar-footer{
    font-size:.65rem;
    color:#64748b;
    text-align:center;
    font-family:'Space Mono',monospace;
    line-height:1.8;
}

[data-testid="metric-container"]{background:var(--card)!important;border:1px solid var(--border)!important;border-radius:12px!important;padding:.9rem 1.1rem!important;}
[data-testid="stMetricValue"]{color:var(--cyan)!important;font-family:'Space Mono',monospace!important;font-size:1.7rem!important;font-weight:700!important;}
[data-testid="stMetricLabel"]{color:var(--muted)!important;font-size:.72rem!important;text-transform:uppercase;letter-spacing:.07em;}
[data-testid="stMetricDelta"] > div{font-family:'Space Mono',monospace!important;font-size:.78rem!important;}

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

# ── Sidebar (PROFESSIONAL REDESIGN) ────────────────────────────────────────────
with st.sidebar:
    # Professional header card
    st.markdown("""
    <div style="background:linear-gradient(135deg,#001a33 0%,#003366 50%,#004d80 100%);
    border:2px solid #00d4ff;border-radius:16px;padding:2rem 1.5rem;
    text-align:center;box-shadow:0 0 40px rgba(0,212,255,.25),inset 0 0 30px rgba(0,212,255,.08);
    margin-bottom:1.8rem;backdrop-filter:blur(10px)">
      <div style="font-size:4rem;margin-bottom:.6rem;filter:drop-shadow(0 0 15px rgba(0,212,255,.6))">⚽</div>
      <h1 style="font-family:'Exo 2',sans-serif;font-size:1.6rem;font-weight:900;
      color:#00d4ff;margin:0;text-shadow:0 0 20px rgba(0,212,255,.5)">FIFA 22</h1>
      <p style="font-family:'Space Mono',monospace;font-size:.7rem;color:#94a3b8;
      letter-spacing:.2em;text-transform:uppercase;margin:.5rem 0 0;
      text-shadow:0 0 10px rgba(0,212,255,.2)">ANALYTICS PLATFORM</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div style="height:2px;background:linear-gradient(90deg,transparent,#00d4ff,transparent);
    margin:1.5rem 0;border-radius:1px;box-shadow:0 0 15px rgba(0,212,255,.4)"></div>
    """, unsafe_allow_html=True)
    
    # Navigation section
    st.markdown("""
    <p style="font-family:'Space Mono',monospace;font-size:.65rem;color:#64748b;
    letter-spacing:.1em;text-transform:uppercase;margin:1rem 0 .8rem;">📍 NAVIGATE</p>
    """, unsafe_allow_html=True)
    
    page = st.radio("Navigate", [
        "🏠  Overview",
        "🧹  Data Cleaning",
        "📊  EDA",
        "🔗  Correlation",
        "📐  Distributions",
        "📈  Regression",
        "🎮  Predict a Player",
    ], label_visibility="collapsed")
    
    st.markdown("""
    <div style="height:2px;background:linear-gradient(90deg,transparent,#7c3aed,transparent);
    margin:1.8rem 0;border-radius:1px;box-shadow:0 0 12px rgba(124,58,237,.3)"></div>
    """, unsafe_allow_html=True)
    
    # Professional footer
    st.markdown("""
    <div style="background:linear-gradient(135deg,rgba(0,212,255,.05),rgba(124,58,237,.05));
    border:1px solid rgba(0,212,255,.2);border-radius:12px;padding:1.2rem;
    text-align:center;margin-top:2rem">
      <p style="font-family:'Space Mono',monospace;font-size:.65rem;color:#94a3b8;
      margin:0;line-height:1.9;letter-spacing:.05em">
        <span style="color:#00d4ff;font-weight:700">19,239</span> Players<br>
        <span style="color:#00d4ff;font-weight:700">110</span> Features<br>
        <span style="font-size:.6rem;color:#64748b;display:block;margin-top:.5rem">
          FIFA 22 · SoFIFA Dataset
        </span>
      </p>
    </div>
    
    <div style="padding:.8rem 0;text-align:center;border-top:1px solid rgba(0,212,255,.1);
    margin-top:1rem">
      <p style="font-family:'Exo 2',sans-serif;font-size:.6rem;color:#475569;margin:0.6rem 0 0">
        ⚡ Built with Streamlit + Plotly + ML
      </p>
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
        fig_style(fig2, 'Age Distribution of FIFA Players', h=380)
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
            fig_style(fig, h=450)
            fig.update_layout(coloraxis_showscale=False)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("#### Summary Stats")
            st.metric("Columns with Missing", len(miss_df))
            st.metric("Max % Missing", f"{miss_pct.max():.1f}%")
            st.metric("Avg % Missing", f"{miss_pct[miss_pct > 0].mean():.1f}%")

        st.markdown("#### Complete Missing Values Table")
        st.dataframe(miss_df.reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Outliers IQR ────────────────────────────────────────────────
    with tab2:
        st.markdown("#### IQR Clipping Results (on outfield players after removing GK)")
        st.markdown("For each feature in FINAL_FEATURES: Q1 - 1.5×IQR ≤ value ≤ Q3 + 1.5×IQR")

        outlier_data = []
        for feature in FINAL_FEATURES:
            Q1 = df_out[feature].quantile(0.25)
            Q3 = df_out[feature].quantile(0.75)
            IQR = Q3 - Q1
            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR
            outlier_data.append({
                'Feature': feature,
                'Q1': f"{Q1:.0f}",
                'Q3': f"{Q3:.0f}",
                'IQR': f"{IQR:.0f}",
                'Lower Bound': f"{lower:.0f}",
                'Upper Bound': f"{upper:.0f}"
            })

        outlier_df = pd.DataFrame(outlier_data)
        st.dataframe(outlier_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# 📊 EDA
# ═══════════════════════════════════════════════════════════════════
elif page == "📊  EDA":
    st.markdown('<div class="hero-title" style="font-size:2rem">Exploratory Data Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Distribution · Patterns · Feature relationships</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📈 Distributions", "📊 Relationships", "📋 Summary Statistics"])

    # ── Distributions ──────────────────────────────────────────────
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            fig1 = px.histogram(df_out, x='overall', nbinsx=40, title='Overall Rating Distribution',
                               color_discrete_sequence=['#00d4ff'],
                               labels={'overall': 'Overall Rating', 'count': 'Count'})
            fig_style(fig1, h=380)
            fig1.update_traces(marker_line=dict(color='#0b1120', width=0.5))
            st.plotly_chart(fig1, use_container_width=True)

        with col2:
            fig1b = px.histogram(df_out, x='age', nbinsx=35, title='Player Age Distribution',
                                color_discrete_sequence=['#7c3aed'])
            fig_style(fig1b, h=380)
            fig1b.update_traces(marker_line=dict(color='#0b1120', width=0.5))
            st.plotly_chart(fig1b, use_container_width=True)

        samp = df_out.sample(min(2000, len(df_out)), random_state=42)
        fig2 = px.scatter(samp, x='overall', y='potential', color='position_group',
                         title='Overall vs Potential Rating by Position',
                         color_discrete_sequence=['#00d4ff','#f72585','#7c3aed','#fbbf24'],
                         opacity=0.6)
        fig2.update_traces(marker_size=4)
        fig_style(fig2, h=380)
        fig2.add_annotation(x=0.5, y=-0.15, xref='paper', yref='paper',
                           text='<i style="color:#94a3b8;font-size:.85rem">Correlation: higher overall typically means higher potential</i>',
                           annotation_font_color='#f72585')
        st.plotly_chart(fig2, use_container_width=True)

    # ── Relationships ──────────────────────────────────────────────
    with tab2:
        col1, col2 = st.columns(2)

        with col1:
            samp = df_out.sample(min(1500, len(df_out)), random_state=42)
            fig3 = px.scatter(samp, x='height_cm', y='weight_kg', color='position_group',
                              opacity=0.55, title='Height vs Weight',
                              color_discrete_sequence=['#00d4ff','#f72585','#7c3aed','#fbbf24'])
            fig3.update_traces(marker_size=4)
            fig_style(fig3, h=350)
            st.plotly_chart(fig3, use_container_width=True)

        with col2:
            foot = df_out['preferred_foot'].value_counts().reset_index()
            foot.columns = ['Foot','Count']
            fig4 = px.pie(foot, names='Foot', values='Count', hole=0.5,
                          title='Preferred Foot', color_discrete_sequence=['#00d4ff','#f72585'])
            fig_style(fig4, h=350)
            st.plotly_chart(fig4, use_container_width=True)

    # ── Summary Statistics ─────────────────────────────────────────
    with tab3:
        st.markdown("#### 📋 Statistical Summary of All Players")
        desc = df[['overall','potential','age','height_cm','weight_kg','value_eur','wage_eur']].describe().round(2)
        st.dataframe(desc, use_container_width=True)

        st.markdown("#### 🎯 Overall Rating Statistics")
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

    tab1, tab2 = st.tabs(["🔥 Full Heatmap (Notebook Cell 14)", "🎯 Final Features Heatmap (Notebook Cell 23)"])

    with tab1:
        avail = [f for f in NUMERIC_FEATURES if f in df.columns]
        corr_big = df[avail + ['overall']].corr()
        fig = px.imshow(corr_big, color_continuous_scale='RdBu_r', zmin=-1, zmax=1,
                        aspect='auto', text_auto='.1f')
        fig_style(fig, 'All Features Correlation with Overall Rating', h=680)
        fig.update_traces(textfont_size=7)
        fig.update_layout(coloraxis_colorbar=dict(title='r'))
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
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

        st.markdown("#### Notebook Cell 12 — Individual Correlation")
        st.code(f"df['age'].corr(df['overall'])  →  {df['age'].corr(df['overall']):.4f}", language='python')

# ═══════════════════════════════════════════════════════════════════
# 📐 PROBABILITY DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════
elif page == "📐  Distributions":
    st.markdown('<div class="hero-title" style="font-size:2rem">Probability Distributions</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Binomial · Poisson · Normal — Applied to FIFA 22</div>', unsafe_allow_html=True)
    st.markdown("---")

    ratings = df_out['overall'].dropna().values

    tab1, tab2, tab3 = st.tabs(["🎲 Binomial", "⚡ Poisson", "🔔 Normal"])

    # ── BINOMIAL (FIXED: slider updates both graphs) ───────────────
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

        ELITE_THRESHOLD = 80
        n_trials = 20
        p_elite = np.mean(ratings >= ELITE_THRESHOLD)
        binom_dist = binom(n=n_trials, p=p_elite)
        k_values = np.arange(0, n_trials + 1)
        pmf_values = binom_dist.pmf(k_values)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("P(elite ≥80)",      f"{p_elite:.4f} ({p_elite*100:.1f}%)")
        c2.metric("P(exactly 5)",       f"{binom_dist.pmf(5):.4f}")
        c3.metric("P(at least 3)",      f"{binom_dist.sf(2):.4f}")
        c4.metric("Expected elite",     f"{binom_dist.mean():.2f}")

        st.markdown("---")

        # SLIDER - updates dynamically
        k_q = st.slider("Find probability for exactly k elite players:", 0, n_trials, 5, key="binom_slider")

        # BOTH GRAPHS UPDATE BASED ON SLIDER
        fig = make_subplots(rows=1, cols=2, subplot_titles=['PMF — P(X = k)', 'CDF — P(X ≤ k)'])
        
        fig.add_trace(go.Bar(x=k_values, y=pmf_values, marker_color='#00d4ff',
                             marker_line=dict(color='#0b1120', width=1), name='PMF'), row=1, col=1)
        
        # Highlight selected k on PMF
        fig.add_trace(go.Bar(x=[k_q], y=[pmf_values[k_q]], marker_color='#f72585',
                             name=f'Selected k={k_q}'), row=1, col=1)
        
        fig.add_vline(x=binom_dist.mean(), line_color='#fbbf24', line_dash='dash',
                      annotation_text=f"Mean={binom_dist.mean():.1f}",
                      annotation_font_color='#fbbf24', row=1, col=1)
        
        fig.add_trace(go.Scatter(x=k_values, y=binom_dist.cdf(k_values),
                                 mode='lines+markers',
                                 line=dict(color='#7c3aed', width=2.5),
                                 marker=dict(size=6), name='CDF'), row=1, col=2)
        
        # Highlight CDF at selected k
        fig.add_vline(x=k_q, line_color='#f72585', line_dash='dash',
                      annotation_text=f"P(X ≤ {k_q})", annotation_font_color='#f72585', row=1, col=2)
        
        fig_style(fig, h=380)
        fig.update_layout(showlegend=True,
                          title_text=f'Binomial (n={n_trials}, p={p_elite:.3f})',
                          title_font=dict(family='Exo 2', size=14))
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("#### 📊 Probability for Selected k")
        c1,c2,c3 = st.columns(3)
        c1.metric(f"P(X = {k_q})",  f"{binom_dist.pmf(k_q):.4f}")
        c2.metric(f"P(X ≤ {k_q})", f"{binom_dist.cdf(k_q):.4f}")
        c3.metric(f"P(X ≥ {k_q})", f"{binom_dist.sf(k_q-1):.4f}")

    # ── POISSON ────────────────────────────────────────────────────
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

    # ── NORMAL ─────────────────────────────────────────────────────
    with tab3:
        st.markdown("""
        <div class="card cx">
          <span class="badge bx">Normal Distribution</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            Does overall_rating follow a Normal distribution? We fit Normal and compute key probabilities.
          </p>
        </div>
        """, unsafe_allow_html=True)

        ratings_all = df['overall'].dropna()
        mu, sigma = norm.fit(ratings_all)
        p_above_80  = norm.sf(80, mu, sigma)
        p_60_to_75  = norm.cdf(75, mu, sigma) - norm.cdf(60, mu, sigma)
        p_one_sigma = norm.cdf(mu+sigma, mu, sigma) - norm.cdf(mu-sigma, mu, sigma)

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("μ (mean)",        f"{mu:.2f}")
        c2.metric("σ (std dev)",     f"{sigma:.2f}")
        c3.metric("P(X > 80)",       f"{p_above_80:.4f}")
        c4.metric("P(60 ≤ X ≤ 75)", f"{p_60_to_75:.4f}")

        x_range = np.linspace(ratings_all.min() - sigma, ratings_all.max() + sigma, 200)
        y_norm = norm.pdf(x_range, mu, sigma)

        fig = make_subplots(rows=1, cols=2, subplot_titles=['Histogram + Normal PDF', 'Q-Q Plot (Normality Check)'])
        fig.add_trace(go.Histogram(x=ratings_all, nbinsx=40, name='Data',
                                   marker_color='#00d4ff', opacity=0.6, histnorm='density'), row=1, col=1)
        fig.add_trace(go.Scatter(x=x_range, y=y_norm, mode='lines', name='Normal Fit',
                                 line=dict(color='#f72585', width=3)), row=1, col=1)

        q, r = probplot(ratings_all)
        fig.add_trace(go.Scatter(x=q[0], y=q[1], mode='markers', name='Q-Q',
                                 marker=dict(color='#00d4ff', size=5)), row=1, col=2)
        m_qqplot, b_qqplot = np.polyfit(q[0], q[1], 1)
        x_qqplot = np.array([q[0].min(), q[0].max()])
        y_qqplot = m_qqplot * x_qqplot + b_qqplot
        fig.add_trace(go.Scatter(x=x_qqplot, y=y_qqplot, mode='lines', name='fit',
                                 line=dict(color='#f72585', width=2, dash='dash')), row=1, col=2)

        fig_style(fig, h=400)
        fig.update_xaxes(title_text='Overall Rating', row=1, col=1)
        fig.update_xaxes(title_text='Theoretical Quantiles', row=1, col=2)
        fig.update_yaxes(title_text='Density', row=1, col=1)
        fig.update_yaxes(title_text='Sample Quantiles', row=1, col=2)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Normal Fit Quality")
        st.metric("P(X within ±σ)", f"{p_one_sigma:.4f} (≈68.27% if perfect)")

# ═══════════════════════════════════════════════════════════════════
# 📈 REGRESSION
# ═══════════════════════════════════════════════════════════════════
elif page == "📈  Regression":
    st.markdown('<div class="hero-title" style="font-size:2rem">Linear Regression</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Normal Equation · Gradient Descent · Predictions</div>', unsafe_allow_html=True)
    st.markdown("---")

    tab1, tab2, tab3 = st.tabs(["📊 Normal Equation", "⚙️ Gradient Descent", "📈 Predictions"])

    with tab1:
        st.markdown("""
        <div class="card ca">
          <span class="badge bc">Normal Equation</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            Closed-form solution: <code style="color:#a78bfa">W = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y</code>
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("MAE",  f"{m_ne['mae']:.4f}")
        c2.metric("RMSE", f"{m_ne['rmse']:.4f}")
        c3.metric("R²",   f"{m_ne['r2']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=y_test, y=y_pred_ne, mode='markers',
                                 marker=dict(color='#00d4ff', size=5, opacity=0.6), name='Predictions'))
        min_val, max_val = y_test.min(), y_test.max()
        fig.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                 mode='lines', line=dict(color='#f72585', dash='dash', width=2), name='Perfect Fit'))
        fig_style(fig, 'Actual vs Predicted (Normal Equation)', h=400)
        fig.update_xaxes(title_text='Actual Rating')
        fig.update_yaxes(title_text='Predicted Rating')
        st.plotly_chart(fig, use_container_width=True)

        residuals_ne = y_test - y_pred_ne
        fig2 = px.histogram(residuals_ne, nbinsx=40, title='Residuals Distribution (Normal Equation)',
                            color_discrete_sequence=['#00d4ff'])
        fig_style(fig2, h=350)
        fig2.update_layout(xaxis_title='Residual', yaxis_title='Frequency')
        st.plotly_chart(fig2, use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="card cp">
          <span class="badge bp">Gradient Descent</span>
          <p style="color:#94a3b8;font-size:.85rem;line-height:1.7;margin:.4rem 0 0">
            Iterative optimization: 1000 iterations, learning rate = 0.01
          </p>
        </div>
        """, unsafe_allow_html=True)

        c1,c2,c3 = st.columns(3)
        c1.metric("MAE",  f"{m_gd['mae']:.4f}")
        c2.metric("RMSE", f"{m_gd['rmse']:.4f}")
        c3.metric("R²",   f"{m_gd['r2']:.4f}")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=list(range(len(cost_hist))), y=cost_hist,
                                 mode='lines', line=dict(color='#7c3aed', width=2)))
        fig_style(fig, 'Cost Function Over Iterations', h=350)
        fig.update_xaxes(title_text='Iteration')
        fig.update_yaxes(title_text='Cost (MSE)')
        st.plotly_chart(fig, use_container_width=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=y_test, y=y_pred_gd, mode='markers',
                                  marker=dict(color='#7c3aed', size=5, opacity=0.6), name='Predictions'))
        min_val, max_val = y_test.min(), y_test.max()
        fig2.add_trace(go.Scatter(x=[min_val, max_val], y=[min_val, max_val],
                                  mode='lines', line=dict(color='#f72585', dash='dash', width=2), name='Perfect Fit'))
        fig_style(fig2, 'Actual vs Predicted (Gradient Descent)', h=400)
        fig2.update_xaxes(title_text='Actual Rating')
        fig2.update_yaxes(title_text='Predicted Rating')
        st.plotly_chart(fig2, use_container_width=True)

    with tab3:
        st.markdown("#### 🎯 Method Comparison")
        comparison_df = pd.DataFrame({
            'Metric': ['MAE', 'RMSE', 'R²'],
            'Normal Equation': [f"{m_ne['mae']:.4f}", f"{m_ne['rmse']:.4f}", f"{m_ne['r2']:.4f}"],
            'Gradient Descent': [f"{m_gd['mae']:.4f}", f"{m_gd['rmse']:.4f}", f"{m_gd['r2']:.4f}"]
        })
        st.dataframe(comparison_df, use_container_width=True, hide_index=True)

        st.markdown("#### 📊 Feature Importance (Top 10 by |weight|)")
        abs_weights = np.abs(W_ne[1:])
        top_idx = np.argsort(abs_weights)[::-1][:10]
        imp_df = pd.DataFrame({
            'Feature': [FINAL_FEATURES[i] for i in top_idx],
            'Weight': [W_ne[i+1] for i in top_idx],
            '|Weight|': [abs_weights[i] for i in top_idx]
        })
        st.dataframe(imp_df, use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════
# 🎮 PREDICT A PLAYER
# ═══════════════════════════════════════════════════════════════════
elif page == "🎮  Predict a Player":
    st.markdown('<div class="hero-title" style="font-size:2rem">Predict Player Overall</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Input attributes · See where they rank</div>', unsafe_allow_html=True)
    st.markdown("---")

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
        x_input = np.array([[inputs[f] for f in FINAL_FEATURES]], dtype=float)
        x_scaled = scaler.transform(x_input)
        x_b = np.c_[np.ones(1), x_scaled]
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

        df_c = df_out.dropna(subset=['overall']).copy()
        df_c['_diff'] = (df_c['overall'] - prediction).abs()
        sim = df_c.nsmallest(5, '_diff')[['short_name','club_name','position_group','overall']].copy()
        sim.columns = ['Name','Club','Position','Overall']
        st.markdown("#### 🤝 Most Similar Players in Dataset")
        st.dataframe(sim.reset_index(drop=True), use_container_width=True, hide_index=True)

st.markdown("---")
st.markdown("""
<div style="text-align:center;padding:1rem 0;color:#64748b;
font-family:'Space Mono',monospace;font-size:.72rem">
⚽ FIFA 22 Analytics Dashboard · Streamlit + Plotly + Machine Learning<br>
Built with Exploratory Data Analysis, Probability Distributions & Linear Regression
</div>
""", unsafe_allow_html=True)
