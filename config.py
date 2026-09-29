from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RAW = DATA / 'raw'
PROCESSED = DATA / 'processed'
MODELS = DATA / 'models'
LOGS = DATA / 'logs'
for p in (RAW, PROCESSED, MODELS, LOGS): p.mkdir(parents=True, exist_ok=True)

FEATURES = [
    'elo_diff_pre','elo_prob_home','home_gf_5','home_ga_5','away_gf_5','away_ga_5',
    'home_ppg_5','away_ppg_5','home_home_gf_5','home_home_ga_5','away_away_gf_5','away_away_ga_5',
    'home_matches_before','away_matches_before','home_rest_days','away_rest_days',
    'form_diff_5','goal_diff_form_5','rest_diff','experience_diff',
]
MIN_MATCHES = 5
SEED = 42
