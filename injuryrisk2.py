import nflreadpy as nfl
import pandas as pd
import polars as pl

seasons = [2021, 2022, 2023, 2024]
raw_snaps = nfl.load_snap_counts(seasons)
raw_injuries = nfl.load_injuries(seasons)
raw_stats = nfl.load_player_stats(seasons)
raw_schedules = nfl.load_schedules(seasons)
reg_schedules = raw_schedules.filter(pl.col("game_type") == "REG")
raw_stats = raw_stats.rename({"player_id": "gsis_id"})

print(raw_stats)

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
    'DAL': 1, 'DEN': 2, 'DET': 0, 'GB': 1, 'HOU': 1, 'IND': 0, 'JAX': 0, 'KC': 1,
    'LV': 3, 'LAC': 3, 'LA': 3, 'MIA': 0, 'MIN': 1, 'NE': 0, 'NO': 1, 'NYG': 0,
    'NYJ': 0, 'PHI': 0, 'PIT': 0, 'SF': 3, 'SEA': 3, 'TB': 0, 'TEN': 1, 'WAS': 0
}

sched = raw_schedules.select([
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

print(df)

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

ml_model_df = df_final.join(
    injuries,
    left_on=["gsis_id", "season", "next_week_lookup"],
    right_on=["gsis_id", "season", "week"],
    how="left"
)

ml_model_df = ml_model_df.with_columns(
    pl.col("report_status").is_not_null().cast(pl.Int8).alias("injury_occurrence")
).drop(["next_week_lookup"])

import xgboost as xgb

pos_map = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}

def get_dmatrix_direct(df_subset):
    X = df_subset.select([
        pl.col("position").replace(pos_map).cast(pl.Int32),
        "tz_diff", "offense_snaps", "cum_snaps",
        "attempts", "cum_attempts", "sacks_suffered", "cum_sacks",
        "carries", "cum_carries", "targets", "cum_targets"
    ])
    y = df_subset.select("injury_occurrence")
    return xgb.DMatrix(X.to_arrow(), label=y.to_arrow())

train_data = ml_model_df.filter(pl.col("season") < 2024)
test_data = ml_model_df.filter(pl.col("season") == 2024)

dtrain = get_dmatrix_direct(train_data)
dtest = get_dmatrix_direct(test_data)

# 3. Parameters (Optimized for Positional Categories)
params = {
    "objective": "binary:logistic",
    "tree_method": "hist", 
    "max_depth": 4,
    "eta": 0.1,
}

# 4. Train the "Brain"
bst = xgb.train(params, dtrain, num_boost_round=100)

# 5. Predict and Results
preds = bst.predict(dtest)
df_results = test_data.with_columns(
    pl.Series(preds).alias("injury_prob")
)

print(df_results.select(["player", "position", "week", "injury_prob"]).sort("injury_prob", descending=True).head(10))