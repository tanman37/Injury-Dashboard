import nflreadpy as nfl
import pandas as pd
import polars as pl
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib

seasons = list(range(2012, 2025))
raw_snaps = nfl.load_snap_counts(seasons)
raw_injuries = nfl.load_injuries(seasons)
raw_stats = nfl.load_player_stats(seasons)
raw_schedules = nfl.load_schedules(seasons)
reg_schedules = raw_schedules.filter(pl.col("game_type") == "REG")
raw_stats = raw_stats.rename({"player_id": "gsis_id"})

df = raw_snaps.join(
    raw_stats,
    left_on=["player", "team", "season", "week"],
    right_on=["player_display_name", "team", "season", "week"],
    how="inner"
)
target_positions = ["QB", "RB", "WR", "TE"]
df = df.filter(pl.col("position").is_in(target_positions))
stat_cols = ["offense_snaps", "attempts", "sacks_suffered", "carries", "targets"]
df = df.with_columns([
    pl.col(c).fill_null(0).cast(pl.Int64) for c in stat_cols
])
df = df.sort(["gsis_id", "season", "week"]).with_columns([
    pl.col("offense_snaps").cum_sum().over(["gsis_id", "season"]).alias("cum_snaps"),
    pl.col("attempts").cum_sum().over(["gsis_id", "season"]).alias("cum_attempts"),
    pl.col("sacks_suffered").cum_sum().over(["gsis_id", "season"]).alias("cum_sacks"),
    pl.col("carries").cum_sum().over(["gsis_id", "season"]).alias("cum_carries"),
    pl.col("targets").cum_sum().over(["gsis_id", "season"]).alias("cum_targets")
])

tz_map = {
    'ARI': 2, 'ATL': 0, 'BAL': 0, 'BUF': 0, 'CAR': 0, 'CHI': 1, 'CIN': 0, 'CLE': 0,
    'DAL': 1, 'DEN': 2, 'DET': 0, 'GB': 1, 'HOU': 1, 'IND': 0, 'JAX': 0, 'KC': 1, 'OAK': 3,
    'LV': 3, 'LAC': 3, 'LA': 3, 'MIA': 0, 'MIN': 1, 'NE': 0, 'NO': 1, 'NYG': 0, 'SD': 3,
    'NYJ': 0, 'PHI': 0, 'PIT': 0, 'SF': 3, 'SEA': 3, 'TB': 0, 'TEN': 1, 'WAS': 0, 'STL': 1
}

sched = reg_schedules.select([
    "game_id", "home_team", "away_team"
])

df = df.join(
    sched,
    on="game_id",
    how="left"
)
df = df.with_columns([
    pl.col("home_team").replace(tz_map).cast(pl.Int64).alias("game_tz"),
    pl.col("team").replace(tz_map).cast(pl.Int64).alias("team_tz")
])
df = df.with_columns(
    (pl.col("team_tz") - pl.col("game_tz")).abs().alias("tz_diff")
)
df_final = df.select([
    "gsis_id", "player", "position", "team", "season", "week", "game_id", "tz_diff", 
    "offense_snaps", "attempts", "sacks_suffered", "carries", "targets", 
    "cum_snaps", "cum_attempts", "cum_sacks", "cum_carries", "cum_targets"
])

injury_statuses  = ["Out", "Doubtful"]
injuries = raw_injuries.filter(
    pl.col("report_status").is_in(injury_statuses)
).select([
    "gsis_id", "season", "week", "report_status"
]).unique()

df_final = df_final.with_columns(
    (pl.col("week")+1).alias("next_week_lookup")
)

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
    how="left"
)
ml_model_df = ml_model_df.with_columns(
    pl.col("report_status").is_not_null().cast(pl.Int8).alias("injury_occurrence")
).drop(["next_week_lookup"])
ml_model_df = ml_model_df.sort(["gsis_id", "season", "week"]).with_columns([
    pl.col("injury_occurrence")
      .shift(1)
      .over(["gsis_id", "season"])
      .fill_null(0)
      .alias("injured_last_week"),
    pl.col("injury_occurrence")
      .shift(1)
      .rolling_sum(window_size=3)
      .over(["gsis_id", "season"])
      .fill_null(0)
      .alias("injuries_last_3_weeks")
])

train_df = ml_model_df.filter(pl.col("season") < 2024)
test_df = ml_model_df.filter(pl.col("season") == 2024)

pos_features = {
    "QB": ["tz_diff", "offense_snaps", "cum_snaps", "attempts", "cum_attempts", "sacks_suffered", "cum_sacks", "injured_last_week", "injuries_last_3_weeks"],
    "RB": ["tz_diff", "offense_snaps", "cum_snaps", "carries", "cum_carries", "injured_last_week", "injuries_last_3_weeks"],
    "WR": ["tz_diff", "offense_snaps", "cum_snaps", "targets", "cum_targets", "injured_last_week", "injuries_last_3_weeks"],
    "TE": ["tz_diff", "offense_snaps", "cum_snaps", "targets", "cum_targets", "injured_last_week", "injuries_last_3_weeks"]
}

position_models = {}

for pos, feats in pos_features.items():
    pos_train = train_df.filter(pl.col("position") == pos)
    pos_test = test_df.filter(pl.col("position") == pos)

    X_tr = pos_train.select(feats)
    y_tr = pos_train.select("injury_occurrence")
    X_te = pos_test.select(feats)
    y_te = pos_test.select("injury_occurrence")

    neg = (y_tr.to_series() == 0).sum()
    pos_count = (y_tr.to_series() == 1).sum()
    ratio = neg / pos_count if pos_count > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        scale_pos_weight=ratio,
        objective="binary:logistic",
        tree_method="hist"
    )
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)
    y_prob = model.predict_proba(X_te)[:, 1]
    print(f"\n{pos} Model:")
    print(classification_report(y_te, y_pred))
    print(f"ROC-AUC: {roc_auc_score(y_te, y_prob):.4f}")

    position_models[pos] = (model, feats)
    y_prob = model.predict_proba(X_te)[:, 1]
    print(f"\n{pos} Probability Distribution:")
    print(f"  Min:    {y_prob.min():.4f}")
    print(f"  Max:    {y_prob.max():.4f}")
    print(f"  Mean:   {y_prob.mean():.4f}")
    print(f"  Median: {pd.Series(y_prob).median():.4f}")
    print(f"  80th %: {pd.Series(y_prob).quantile(0.80):.4f}")
    print(f"  90th %: {pd.Series(y_prob).quantile(0.90):.4f}")
    print(f"  95th %: {pd.Series(y_prob).quantile(0.95):.4f}")
    print(f"  99th %: {pd.Series(y_prob).quantile(0.99):.4f}")

for pos, (model, feats) in position_models.items():
    joblib.dump((model, feats), f"{pos}_injury_model.pkl")