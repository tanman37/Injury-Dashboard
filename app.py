import streamlit as st
import polars as pl
import pandas as pd
import nflreadpy as nfl
import xgboost as xgb
from sklearn.metrics import roc_auc_score
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NFL Injury Risk",
    page_icon="🏈",
    layout="centered",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0d0d0d;
    color: #e8e8e8;
}

h1, h2, h3 {
    font-family: 'Bebas Neue', sans-serif;
    letter-spacing: 0.05em;
}

.stApp {
    background-color: #0d0d0d;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #111111;
    border-right: 1px solid #222;
}

/* Inputs */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background-color: #1a1a1a !important;
    border-color: #333 !important;
    color: #e8e8e8 !important;
    border-radius: 4px !important;
}

/* Number inputs */
input[type="number"], input[type="text"] {
    background-color: #1a1a1a !important;
    color: #e8e8e8 !important;
}

/* Labels */
label, .stSelectbox label, .stNumberInput label {
    color: #aaa !important;
    font-size: 0.78rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
}

/* Button */
.stButton > button {
    background-color: #c8f53a;
    color: #0d0d0d;
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1.1rem;
    letter-spacing: 0.1em;
    border: none;
    border-radius: 4px;
    padding: 0.6rem 2rem;
    width: 100%;
    transition: background-color 0.2s;
}
.stButton > button:hover {
    background-color: #d9ff4d;
    color: #0d0d0d;
}

/* Risk cards */
.risk-card {
    padding: 2rem;
    border-radius: 6px;
    text-align: center;
    margin-top: 1.5rem;
}
.risk-low    { background: #0f2e1a; border: 1px solid #1a5c30; }
.risk-medium { background: #2e2000; border: 1px solid #7a5500; }
.risk-high   { background: #2e0a0a; border: 1px solid #8b1a1a; }

.risk-pct {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 4rem;
    line-height: 1;
    margin-bottom: 0.3rem;
}
.risk-low    .risk-pct { color: #4ade80; }
.risk-medium .risk-pct { color: #fbbf24; }
.risk-high   .risk-pct { color: #f87171; }

.risk-label {
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.15em;
    color: #888;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 1.5rem 0;
}

.section-title {
    font-family: 'Bebas Neue', sans-serif;
    font-size: 1rem;
    letter-spacing: 0.12em;
    color: #666;
    margin-bottom: 0.75rem;
    margin-top: 1.25rem;
}
</style>
""", unsafe_allow_html=True)


# ── Model training (cached so it only runs once per session) ──────────────────
@st.cache_resource(show_spinner=False)
def train_models():
    seasons = list(range(2012, 2025))
    raw_snaps     = nfl.load_snap_counts(seasons)
    raw_injuries  = nfl.load_injuries(seasons)
    raw_stats     = nfl.load_player_stats(seasons)
    raw_schedules = nfl.load_schedules(seasons)

    reg_schedules = raw_schedules.filter(pl.col("game_type") == "REG")
    raw_stats     = raw_stats.rename({"player_id": "gsis_id"})

    df = raw_snaps.join(
        raw_stats,
        left_on=["player", "team", "season", "week"],
        right_on=["player_display_name", "team", "season", "week"],
        how="inner",
    )
    target_positions = ["QB", "RB", "WR", "TE"]
    df = df.filter(pl.col("position").is_in(target_positions))

    stat_cols = ["offense_snaps", "attempts", "sacks_suffered", "carries", "targets"]
    df = df.with_columns([pl.col(c).fill_null(0).cast(pl.Int64) for c in stat_cols])
    df = df.sort(["gsis_id", "season", "week"]).with_columns([
        pl.col("offense_snaps").cum_sum().over(["gsis_id", "season"]).alias("cum_snaps"),
        pl.col("attempts").cum_sum().over(["gsis_id", "season"]).alias("cum_attempts"),
        pl.col("sacks_suffered").cum_sum().over(["gsis_id", "season"]).alias("cum_sacks"),
        pl.col("carries").cum_sum().over(["gsis_id", "season"]).alias("cum_carries"),
        pl.col("targets").cum_sum().over(["gsis_id", "season"]).alias("cum_targets"),
    ])

    tz_map = {
        'ARI': 2, 'ATL': 0, 'BAL': 0, 'BUF': 0, 'CAR': 0, 'CHI': 1, 'CIN': 0, 'CLE': 0,
        'DAL': 1, 'DEN': 2, 'DET': 0, 'GB': 1,  'HOU': 1, 'IND': 0, 'JAX': 0, 'KC': 1,
        'OAK': 3, 'LV': 3,  'LAC': 3, 'LA': 3,  'MIA': 0, 'MIN': 1, 'NE': 0,  'NO': 1,
        'NYG': 0, 'NYJ': 0, 'PHI': 0, 'PIT': 0, 'SF': 3,  'SEA': 3, 'TB': 0,  'TEN': 1,
        'WAS': 0, 'SD': 3,  'STL': 1,
    }

    sched = reg_schedules.select(["game_id", "home_team", "away_team"])
    df = df.join(sched, on="game_id", how="left")
    df = df.with_columns([
        pl.col("home_team").replace(tz_map).cast(pl.Int64).alias("game_tz"),
        pl.col("team").replace(tz_map).cast(pl.Int64).alias("team_tz"),
    ])
    df = df.with_columns((pl.col("team_tz") - pl.col("game_tz")).abs().alias("tz_diff"))

    df_final = df.select([
        "gsis_id", "player", "position", "team", "season", "week", "game_id", "tz_diff",
        "offense_snaps", "attempts", "sacks_suffered", "carries", "targets",
        "cum_snaps", "cum_attempts", "cum_sacks", "cum_carries", "cum_targets",
    ])

    injury_statuses = ["Out", "Doubtful"]
    injuries = raw_injuries.filter(
        pl.col("report_status").is_in(injury_statuses)
    ).select(["gsis_id", "season", "week", "report_status"]).unique()

    df_final = df_final.with_columns((pl.col("week") + 1).alias("next_week_lookup"))

    for col_name in ["season", "week", "next_week_lookup"]:
        if col_name in df_final.columns:
            df_final = df_final.with_columns(pl.col(col_name).cast(pl.Int32))
    for col_name in ["season", "week"]:
        if col_name in injuries.columns:
            injuries = injuries.with_columns(pl.col(col_name).cast(pl.Int32))

    ml_model_df = df_final.join(
        injuries,
        left_on=["gsis_id", "season", "next_week_lookup"],
        right_on=["gsis_id", "season", "week"],
        how="left",
    )
    ml_model_df = ml_model_df.with_columns(
        pl.col("report_status").is_not_null().cast(pl.Int8).alias("injury_occurrence")
    ).drop(["next_week_lookup"])

    ml_model_df = ml_model_df.sort(["gsis_id", "season", "week"]).with_columns([
        pl.col("injury_occurrence").shift(1).over(["gsis_id", "season"]).fill_null(0).alias("injured_last_week"),
        pl.col("injury_occurrence").shift(1).rolling_sum(window_size=3).over(["gsis_id", "season"]).fill_null(0).alias("injuries_last_3_weeks"),
    ])

    train_df = ml_model_df.filter(pl.col("season") < 2024)
    test_df  = ml_model_df.filter(pl.col("season") == 2024)

    pos_features = {
        "QB": ["tz_diff", "offense_snaps", "cum_snaps", "attempts", "cum_attempts", "sacks_suffered", "cum_sacks", "injured_last_week", "injuries_last_3_weeks"],
        "RB": ["tz_diff", "offense_snaps", "cum_snaps", "carries",  "cum_carries",  "injured_last_week", "injuries_last_3_weeks"],
        "WR": ["tz_diff", "offense_snaps", "cum_snaps", "targets",  "cum_targets",  "injured_last_week", "injuries_last_3_weeks"],
        "TE": ["tz_diff", "offense_snaps", "cum_snaps", "targets",  "cum_targets",  "injured_last_week", "injuries_last_3_weeks"],
    }

    position_models = {}
    model_scores    = {}

    for pos, feats in pos_features.items():
        pos_train = train_df.filter(pl.col("position") == pos)
        pos_test  = test_df.filter(pl.col("position") == pos)

        X_tr = pos_train.select(feats)
        y_tr = pos_train.select("injury_occurrence")
        X_te = pos_test.select(feats)
        y_te = pos_test.select("injury_occurrence")

        neg       = (y_tr.to_series() == 0).sum()
        pos_count = (y_tr.to_series() == 1).sum()
        ratio     = neg / pos_count if pos_count > 0 else 1

        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            scale_pos_weight=ratio, objective="binary:logistic", tree_method="hist",
        )
        model.fit(X_tr, y_tr)

        y_prob = model.predict_proba(X_te)[:, 1]
        auc    = roc_auc_score(y_te, y_prob)

        position_models[pos] = (model, feats)
        model_scores[pos]    = round(auc, 4)

    return position_models, model_scores


# ── Helpers ───────────────────────────────────────────────────────────────────
THRESHOLDS = {
    "QB": (0.3828, 0.6906),
    "RB": (0.5548, 0.6937),
    "WR": (0.5510, 0.6721),
    "TE": (0.5346, 0.6266),
}

def risk_category(prob, position):
    low_threshold, high_threshold = THRESHOLDS[position]
    if prob < low_threshold:
        return "LOW",    "risk-low"
    elif prob < high_threshold:
        return "MEDIUM", "risk-medium"
    else:
        return "HIGH",   "risk-high"


# ── App layout ────────────────────────────────────────────────────────────────
st.markdown("<h1 style='font-size:2.8rem; margin-bottom:0;'>NFL INJURY RISK</h1>", unsafe_allow_html=True)
st.markdown("<p style='color:#555; font-size:0.85rem; margin-top:0; letter-spacing:0.08em;'>PLAYER ASSESSMENT TOOL</p>", unsafe_allow_html=True)
st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# Train models
with st.spinner("Loading models — this takes about 60 seconds on first run..."):
    position_models, model_scores = train_models()

# Position selector
position = st.selectbox("Position", ["QB", "RB", "WR", "TE"])

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Dynamic inputs based on position ─────────────────────────────────────────
st.markdown("<p class='section-title'>Cumulative Season Stats (entering this week)</p>", unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    cum_snaps = st.number_input("Cumulative Snaps", min_value=0, value=0, step=1)
with col2:
    injured_last_week    = st.selectbox("Injured Last Week?", ["No", "Yes"])
    injured_last_week_val = 1 if injured_last_week == "Yes" else 0

injuries_last_3 = st.number_input("Injuries in Last 3 Weeks (0–3)", min_value=0, max_value=3, value=0, step=1)

if position == "QB":
    col3, col4 = st.columns(2)
    with col3:
        cum_attempts = st.number_input("Cumulative Pass Attempts", min_value=0, value=0, step=1)
    with col4:
        cum_sacks = st.number_input("Cumulative Sacks Taken", min_value=0, value=0, step=1)
elif position == "RB":
    cum_carries = st.number_input("Cumulative Carries", min_value=0, value=0, step=1)
else:
    cum_targets = st.number_input("Cumulative Targets", min_value=0, value=0, step=1)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)
st.markdown("<p class='section-title'>Projected Stats for This Week</p>", unsafe_allow_html=True)

col5, col6 = st.columns(2)
with col5:
    offense_snaps = st.number_input("Projected Snaps", min_value=0, value=0, step=1)
with col6:
    tz_diff = st.number_input("Time Zone Differential (0–3)", min_value=0, max_value=3, value=0, step=1)

if position == "QB":
    col7, col8 = st.columns(2)
    with col7:
        attempts     = st.number_input("Projected Pass Attempts", min_value=0, value=0, step=1)
    with col8:
        sacks_suffered = st.number_input("Projected Sacks", min_value=0, value=0, step=1)
elif position == "RB":
    carries = st.number_input("Projected Carries", min_value=0, value=0, step=1)
else:
    targets = st.number_input("Projected Targets", min_value=0, value=0, step=1)

st.markdown("<hr class='divider'>", unsafe_allow_html=True)

# ── Predict ───────────────────────────────────────────────────────────────────
if st.button("CALCULATE INJURY RISK"):
    model, feats = position_models[position]

    if position == "QB":
        input_data = {
            "tz_diff": tz_diff, "offense_snaps": offense_snaps, "cum_snaps": cum_snaps,
            "attempts": attempts, "cum_attempts": cum_attempts,
            "sacks_suffered": sacks_suffered, "cum_sacks": cum_sacks,
            "injured_last_week": injured_last_week_val, "injuries_last_3_weeks": injuries_last_3,
        }
    elif position == "RB":
        input_data = {
            "tz_diff": tz_diff, "offense_snaps": offense_snaps, "cum_snaps": cum_snaps,
            "carries": carries, "cum_carries": cum_carries,
            "injured_last_week": injured_last_week_val, "injuries_last_3_weeks": injuries_last_3,
        }
    else:
        input_data = {
            "tz_diff": tz_diff, "offense_snaps": offense_snaps, "cum_snaps": cum_snaps,
            "targets": targets, "cum_targets": cum_targets,
            "injured_last_week": injured_last_week_val, "injuries_last_3_weeks": injuries_last_3,
        }

    input_df = pd.DataFrame([input_data])[feats]
    prob     = model.predict_proba(input_df)[:, 1][0]
    label, css_class = risk_category(prob, position)
    pct = f"{prob * 100:.1f}%"

    st.markdown(f"""
    <div class="risk-card {css_class}">
        <div class="risk-pct">{pct}</div>
        <div class="risk-label">Injury Risk — {label}</div>
    </div>
    """, unsafe_allow_html=True)