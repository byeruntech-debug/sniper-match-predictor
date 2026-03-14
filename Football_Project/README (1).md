# Sniper Match Predictor — Cross-League Analytics

> **Multi-league football prediction system** using Dixon-Coles, Elo Ratings, and a 4-layer Sniper filter.
> Validated across three leagues covering the full spectrum of modern football.

![Dashboard](portfolio_dashboard.png)

---

## Performance Matrix

| Liga | Sniper Accuracy | Coverage | Brier Score | Karakteristik |
|------|:-:|:-:|:-:|---|
| La Liga 🇪🇸 | **75.0%** | 30% | 0.2002 | Low-scoring / Posesif |
| EPL 🏴󠁧󠁢󠁥󠁮󠁧󠁿 | **62.1%** | 28% | 0.2073 | High-intensity / Chaos |
| Bundesliga 🇩🇪 | **68.4%** | 32% | 0.4869* | High-scoring / Transisi |

> *Bundesliga Brier dihitung dari SNIPER predictions saja (n=76). Nilai lebih tinggi karena overconfidence bias pada tim besar di musim anomali (Leverkusen invincible 23/24). Dokumentasi lengkap di [Failure Analysis](#failure-analysis).

---

## Arsitektur Model

```
Data Sources                 Model Engine                  Sniper Filter
─────────────               ──────────────────            ──────────────────
La Liga 2526  ──┐           ┌─ Dixon-Coles ──────┐        L1: Konsensus 3/3
EPL 2223-2425 ──┤──────────►│  Poisson MLE       ├──────► L2: Draw danger
Bundesliga    ──┤           │  λ_home / μ_away   │        L3: Elo gap filter
FIFA25 ratings──┤           ├─ Elo Rating ───────┤        L4: Trap stadium
Players 2526  ──┤           │  K=32, HFA +50     │              │
Injuries data ──┘           ├─ Gemini AI ────────┤              ▼
                            │  Contextual LLM    │        SNIPER / HOLD / SKIP
                            └────────────────────┘
```

---

## The Technical Journey

### Phase 1 — Baseline (V4–V5)
Dimulai dengan Monte Carlo simulation sederhana menggunakan statistik raw (xG, shots on target). Akurasi awal ~44% — nyaris sama dengan random guessing. Identifikasi bahwa statistik agregat tidak mampu menangkap *dynamics* pertandingan individual.

### Phase 2 — Sniper Filter & Consensus Engine (V6–V9)

Implementasi **Consensus Engine** yang hanya memicu sinyal "Sniper" ketika 3 model independen sepakat:

1. **Dixon-Coles** — Probability calibration berbasis distribusi Poisson dengan koreksi low-score (`τ` correction). Parameter attack/defense per tim, di-fit via MLE.
2. **Elo Rating** — Long-term team strength hierarchy. K=32, home advantage +50 Elo points. Walk-forward update setelah setiap laga.
3. **Gemini AI** — Contextual filter: UCL/Copa fatigue, "defensive fortress" home advantage, injury impact assessment.

**4-layer Sniper Filter:**
```
L1  Konsensus 3/3      — ketiga model harus setuju pada outcome yang sama
L2  Draw danger        — d_prob < 0.28 (La Liga) / 0.30 (Bundesliga)
L3  Elo gap            — minimal gap > 80 (home) atau > 150 (away)
L4  Trap stadium       — clean sheet rate vs big teams < 35%
```

### Phase 3 — Cross-League Stress Test & Domain Adaptation

Model di-deploy ke Bundesliga **tanpa tuning** untuk mengukur transferability.

**Initial transfer result (Phase 3a):**
- Brier Score: **0.6632** → lebih buruk dari random guessing
- 100% prediksi SNIPER adalah home_win → systematic bias
- Root cause: Elo cold-start (semua tim = 1500) + Dixon-Coles tidak pernah melihat data Bundesliga

**German Calibration (Phase 3b):**
```python
# Burn-in: 306 laga 22/23 untuk memanaskan Elo
for match in bundesliga_2223:
    update_elo(match)   # Bayern: 1500 → 1630, Schalke: 1500 → 1446

# Retrain Dixon-Coles dengan data Bundesliga lokal
dc_bundesliga = fit_dixon_coles(bundesliga_2223)
# h_adv = 0.3439, rho = -0.1493
# Bayern attack: +0.5417, Schalke defense: -0.5158

# Backtest pada 23/24
results = backtest_walkforward(bundesliga_2324, dc_bundesliga, elo_warm)
# SNIPER: 68.4% accuracy | 76 laga | coverage 32%
# Away Win: 16 laga (56.2%) — bias home_win terselesaikan
```

---

## Failure Analysis

Model menunjukkan **overconfidence bias** pada tim besar Bundesliga di musim 23/24:

| Pertandingan | Prediksi | Aktual | Brier |
|---|---|---|:-:|
| Bayern Munich vs Werder Bremen | home_win (P=85.5%) | away_win | 1.6485 |
| Dortmund vs Stuttgart | home_win (P=78.2%) | away_win | 1.4897 |
| RB Leipzig vs Bochum | home_win (P=79.1%) | draw | 1.3591 |

**Root causes yang teridentifikasi:**

1. **Regime shift** — Musim 23/24 adalah anomali historis: Leverkusen invincible season pertama dalam sejarah Bundesliga. Parameter Dixon-Coles yang di-fit dari 22/23 tidak bisa mengantisipasi pergeseran kekuatan tim.

2. **Draw underestimation di high-scoring leagues** — Dixon-Coles mengasumsikan distribusi gol independen. Di Bundesliga, tim tamu tetap menyerang saat tertinggal, meningkatkan frekuensi comeback yang tidak tertangkap oleh parameter `ρ` standar.

3. **Missing promoted teams** — Darmstadt dan Heidenheim (21.6% laga) tidak ada di training set, memaksa model skip laga-laga yang potensial menjadi upset.

**Takeaway:** Fitur berbasis sejarah (Elo + DC) memerlukan komponen **live form** dan **market sentiment** untuk mendeteksi regime shift secara real-time.

---

## Dataset

| File | Deskripsi | Ukuran |
|------|-----------|--------|
| `laliga_2025_2026_stats.csv` | La Liga 25/26 match results (270 laga) | 146 KB |
| `epl_2223/2324/2425.csv` | EPL 3 musim, 380 laga/musim | ~530 KB |
| `bundesliga_2223/2324/2425.csv` | Bundesliga 3 musim, 306 laga/musim | ~300 KB |
| `players_data-2025_2026.csv` | Stats 2695 pemain top Eropa | 1.0 MB |
| `male_players.csv` | FIFA 25 ratings 16,161 pemain | 5.2 MB |
| `full_dataset_thesis.csv` | Injury data 2020-2025, 15,603 baris | 1.6 MB |

**Sources:** [football-data.co.uk](https://www.football-data.co.uk) · [FBref](https://fbref.com) · [EA Sports FC 25](https://www.ea.com/games/ea-sports-fc)

---

## Struktur Proyek

```
Football_Project/
├── scripts/
│   ├── football_predictor_full.py      # Engine utama (Dixon-Coles + Elo + Gemini)
│   ├── football_predictor_v9_sniper.py # 4-layer Sniper filter
│   └── football_log_tracker.py         # Prediction log + equity curve
├── data/
│   ├── laliga2526/                      # Match results La Liga
│   ├── epl/                             # EPL 3 musim
│   ├── bundesliga/                      # Bundesliga 3 musim
│   ├── players2526/                     # Player stats 2025/26
│   ├── injuries/                        # Injury dataset 2020-2025
│   └── fifa25/                          # FIFA 25 player ratings
├── notebooks/
│   └── football_simulation_v3.ipynb    # Development notebook lengkap
├── outputs/
│   ├── epl_backtest_results.csv         # 760 EPL predictions
│   ├── bundesliga_backtest_results.csv  # 306 Bundesliga predictions
│   ├── bundesliga_calibrated_results.csv# Post-calibration results
│   ├── portfolio_dashboard.png          # Visualisasi perbandingan 3 liga
│   └── match_analysis.png
├── report/
│   ├── football_portfolio_report.docx   # Main portfolio document
│   └── bundesliga_technical_report.docx # Bundesliga failure analysis
└── archive/                             # V4–V7 untuk referensi
```

---

## Quick Start

```python
from google.colab import drive
drive.mount('/content/drive')

BASE = '/content/drive/MyDrive/Football_Project'

# Load semua cell
exec(open(f'{BASE}/scripts/football_predictor_full.py').read())
exec(open(f'{BASE}/scripts/football_predictor_v9_sniper.py').read())

# Prediksi
predict_match_v9('Barcelona', 'Real Madrid')
# → SNIPER: home_win | P(H)=0.623 P(D)=0.201 P(A)=0.176 | Confidence: Tinggi
```

---

## Roadmap — V10

- [ ] **Live form feature** — rata-rata gol & poin 5 laga terakhir sebagai fitur dinamis
- [ ] **Elo decay mechanism** — rating menurun eksponensial saat tim tidak aktif
- [ ] **Promoted team bootstrapping** — Elo awal dari rata-rata divisi asal
- [ ] **Platt scaling calibration** — post-hoc probability calibration untuk Brier fix
- [ ] **Bundesliga blind test 24/25** — validasi 50 laga live sebagai penutup portofolio
- [ ] **Automated weekly predictor** — fetch jadwal → run Sniper → log ke Drive

---

## Validated Metrics Summary

```
Model        : Dixon-Coles + Elo Rating + Gemini AI (Consensus 3/3)
Filter       : V9 Sniper (4-layer)
Validation   : Walk-forward backtest (no data leakage)

La Liga      : 75.0% accuracy | 30% coverage | Brier 0.2002 | n≈90 laga
EPL          : 62.1% accuracy | 28% coverage | Brier 0.2073 | n=213 laga
Bundesliga   : 68.4% accuracy | 32% coverage | Brier 0.4869 | n=76 laga
               (post German Calibration — Dixon-Coles retrained on BL data)
```

---

*Built with Python · Pandas · SciPy · Matplotlib · Google Colab · Gemini API*
