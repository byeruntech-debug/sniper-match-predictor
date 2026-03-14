# Football Match Prediction System
Last Updated: 2026-03-14 02:30

## Akurasi Tervalidasi
- Bundesliga Sniper (Backtest): 72.1% | Coverage 20% | Brier 0.5243
- La Liga Sniper (Blind Test): 75.0% | Coverage 30% | Brier 0.2002
- EPL Rolling Backtest: 62.1% | Coverage 28% | Brier 0.2073

## Quick Start (Sesi Colab Baru)
  from google.colab import drive
  drive.mount('/content/drive')
  BASE = '/content/drive/MyDrive/Football_Project'
  exec(open(BASE + '/scripts/football_predictor_full.py').read())
  exec(open(BASE + '/scripts/football_predictor_v9_sniper.py').read())
  predict_match_v9('Barcelona', 'Real Madrid')

## Sniper Filter Rules
  1. Consensus 3/3 model setuju
  2. Draw danger: d_prob < 0.28
  3. Home Win: Elo gap > 80, home_wr > 35%, away unbeaten < 3
  4. Away Win: Elo gap > 150, away goals 2/3 laga, home GA > 0.8

## Struktur Folder
  scripts/   — v9_sniper.py (MAIN), log_tracker.py, predictor_full.py
  data/      — laliga2526/, epl/, players2526/, injuries/, fifa25/
  notebooks/ — football_simulation_v3.ipynb
  outputs/   — match_analysis.png, epl_backtest_results.csv
  archive/   — v5, v6, v7 untuk referensi
  report/    — portfolio document