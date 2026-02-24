NFL Player Injury Risk Dashboard
Live App: https://huggingface.co/spaces/tanman37/nfl-injury-dashboard
A machine learning pipeline that predicts week-to-week injury risk for NFL skill position players (QB, RB, WR, TE) using XGBoost classifiers trained on historical NFL data from 2012–2023, with a Streamlit-based interactive frontend designed for coaching staff use.

Overview
This project was built to demonstrate applied sports analytics — specifically the application of tree-based ML methods to player health and availability, one of the highest-leverage decision areas in football operations. 
The end product is a coach-facing tool that takes a player's cumulative season workload and projected weekly usage (specific statistics in inputs dependent on position) as inputs and returns a calibrated injury risk probability and risk category (Low / Medium / High).

Data Sources
All data is sourced via the nflreadpy library:

Snap counts (2012–2024) — primary workload signal
Player stats — position-specific usage (pass attempts, carries, targets, sacks)
Injury reports — structured Out/Doubtful designations used as the target label
Schedules — used to derive timezone differential as a travel fatigue proxy


Feature Engineering
Features were engineered to capture three injury risk dimensions:
Workload accumulation — cumulative season snap counts, attempts, carries, and targets track the physical toll of a full season. Single-game volume captures acute workload spikes.
Travel fatigue — timezone differential between a player's home market and the game location serves as a proxy for travel stress.
Injury history — binary flag for prior week injury status and a rolling 3-week injury count, derived by lagging the target variable within each player-season. These were the highest-signal features added during development.

Modeling Approach
Four separate XGBoost classifiers were trained — one per position — rather than a single model with position as a categorical feature. This architecture ensures position-specific feature sets and prevents cross-position feature interactions (e.g., pass attempts influencing RB predictions).
Position features for QB: tz_diff (time zone change), snaps, cum_snaps (cumulative snaps), attempts, cum_attempts (cumulative passing atttempts), sacks, cum_sacks (cumulative sacks), injury history
                  for RB: tz_diff, snaps, cum_snaps, carries, cum_carries, injury history
                  for WR: tz_diff, snaps, cum_snaps, targets, cum_targets, injury history
                  for TE: tz_diff, snaps, cum_snaps, targets, cum_targets, injury history
Class imbalance (approximately 3–4% injury rate per player-week) is addressed via position-specific scale_pos_weight values calculated from each position's training data distribution rather than a global ratio.
Train/test split: 2012–2023 seasons for training, 2024 season held out for evaluation.

Model Performance
Positional ROC-AUC:
  QB: 0.6791
  WR: 0.6721
  TE: 0.5853
  RB: 0.5635
ROC-AUC was prioritized over accuracy given the class imbalance — a model predicting "no injury" every week achieves 96%+ accuracy but provides no decision-making value. Risk thresholds for the Low/Medium/High categories were calibrated to each position's empirical probability distribution on the 2024 test set rather than using fixed global cutoffs.

Iterative Development
The repository reflects an honest iterative development process across three files:
injuryrisk.py — Initial proof of concept. Single XGBoost model across all positions with a hardcoded class imbalance ratio. Established the core data pipeline joining snap counts, player stats, injury reports, and schedule data.
injuryrisk2.py — Introduced position-specific separate models and interaction constraints.
injury_risk_model.py — Final production model. Added lagged injury history features (prior week injury status, rolling 3-week injury count), position-specific class imbalance ratios, and model persistence via joblib. This version powers the live app.

Frontend
The interactive dashboard is built in Streamlit and deployed on Hugging Face Spaces. The app dynamically renders position-relevant input fields based on the coach's position selection, trains the models on startup using @st.cache_resource to avoid retraining on every prediction, and returns a probability estimate alongside a color-coded risk category.
Live app: https://huggingface.co/spaces/tanman37/nfl-injury-dashboard

Tech Stack

Python — Polars, Pandas, XGBoost, scikit-learn, Streamlit
Data — nflreadpy
Deployment — Hugging Face Spaces (Docker/Streamlit)
Version control — Git / GitHub


Limitations and Next Steps
The current model is intentionally scoped as a proof of concept. Meaningful next steps include:

Age — injury risk increases nonlinearly with age and is derivable from existing player data
Days rest — short weeks (Thursday games) are a documented injury risk factor derivable from schedule data already in the pipeline
Snap percentage — normalizing snap counts relative to team totals would produce a stronger workload signal than raw counts
Position-specific contact exposure — RB injury prediction in particular is limited by the fact that acute contact injuries are poorly predicted by workload history alone


About
Built as a demonstration of applied sports analytics and ML engineering skills. Developed from scratch in approximately one week with limited prior data science background — reflecting the kind of growth mindset, self-directed learning, and commitment to continuous improvement central to contributing effectively in a fast-moving football analytics environment.
