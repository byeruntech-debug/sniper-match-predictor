#!/usr/bin/env python3
"""
Football Match Predictor — La Liga
Versi : 6.0 FULL REBUILD
Upgrade: Elo Rating + xG Split Home/Away + Dixon-Coles + Poisson Calibration
         + FIFA25 Player Strength + Injury Impact

CARA PAKAI:
  Tempel cell-cell di bawah ini SETELAH Cell 10 di notebook football_predictor_v5.py
  Jalankan semua cell dari atas ke bawah, lalu ganti Cell 9 dengan Cell 16 untuk prediksi.
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 11 — Load Semua Data Source
# ══════════════════════════════════════════════════════════════════════
# CELL 11 — Load & Merge Semua Data Source
import pandas as pd
import numpy as np
import warnings, os, math
from scipy.optimize import minimize
from scipy.stats import poisson
from collections import defaultdict
warnings.filterwarnings("ignore")

# ── 1. La Liga match results (270 laga real) ──────────────────────────
LALIGA_CSV  = "/content/laliga2526/laliga_2025_2026_stats.csv"
df_matches  = pd.read_csv(LALIGA_CSV, parse_dates=["Date"])
df_matches  = df_matches.sort_values("Date").reset_index(drop=True)

# Kolom wajib yang kita pakai
COLS_NEEDED = ["Date","HomeTeam","AwayTeam","FTHG","FTAG","FTR",
               "HS","AS","HST","AST","B365H","B365D","B365A"]
df_matches  = df_matches[COLS_NEEDED].dropna(subset=["FTHG","FTAG","FTR"])

print(f"✅ Match data: {len(df_matches)} laga | "
      f"{df_matches['Date'].min().date()} s/d {df_matches['Date'].max().date()}")

# ── 2. Player data (FIFA 25 ratings) ──────────────────────────────────
FIFA_CSV   = "/content/fifa25/male_players.csv"
try:
    df_fifa = pd.read_csv(FIFA_CSV, low_memory=False)
    fifa_cols = ["club_name","short_name","player_positions","overall","pace",
                 "shooting","passing","defending","physic"]
    df_fifa   = df_fifa[[c for c in fifa_cols if c in df_fifa.columns]]
    print(f"✅ FIFA25 data: {len(df_fifa)} pemain dari {df_fifa['club_name'].nunique()} klub")
except Exception as e:
    df_fifa = pd.DataFrame()
    print(f"⚠️  FIFA25 tidak tersedia: {e}")

# ── 3. Player stats La Liga 2526 ───────────────────────────────────────
PLAYERS_CSV = "/content/players2526/players_data-2025_2026.csv"
try:
    df_players = pd.read_csv(PLAYERS_CSV, low_memory=False)
    print(f"✅ Players2526 data: {len(df_players)} baris, {df_players.shape[1]} kolom")
    print(f"   Kolom sample: {list(df_players.columns[:8])}")
except Exception as e:
    df_players = pd.DataFrame()
    print(f"⚠️  Players2526 tidak tersedia: {e}")

# ── 4. Injury data ────────────────────────────────────────────────────
INJURY_CSV = "/content/injuries/full_dataset_thesis - 1.csv"
try:
    df_injuries = pd.read_csv(INJURY_CSV, low_memory=False)
    print(f"✅ Injury data: {len(df_injuries)} baris")
    print(f"   Kolom: {list(df_injuries.columns[:6])}")
except Exception as e:
    df_injuries = pd.DataFrame()
    print(f"⚠️  Injury data tidak tersedia: {e}")

print("\n✅ CELL 11 SELESAI — Semua data loaded!")


# ══════════════════════════════════════════════════════════════════════
# CELL 12 — Elo Rating System (Dynamic)
# ══════════════════════════════════════════════════════════════════════
# CELL 12 — Elo Rating System
class EloRatingSystem:
    """
    Elo Rating dinamis yang di-fit dari data historis La Liga.

    Keunggulan vs static rating:
    - Rating berubah setelah SETIAP pertandingan
    - Menang lawan tim kuat = naik lebih banyak (K * expected_score)
    - Otomatis encode form & relative strength
    - Home advantage dimasukkan langsung ke expected score
    """

    def __init__(self, k=32, home_advantage=65, initial_rating=1500):
        self.K               = k
        self.home_advantage  = home_advantage   # poin tambah ke home team
        self.initial_rating  = initial_rating
        self.ratings         = defaultdict(lambda: initial_rating)
        self.history         = []               # log semua perubahan rating

    def expected_score(self, rating_a, rating_b):
        """Probabilitas menang A vs B (logistic curve)."""
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))

    def update(self, home_team, away_team, home_goals, away_goals):
        """Update rating setelah 1 pertandingan."""
        r_h = self.ratings[home_team] + self.home_advantage
        r_a = self.ratings[away_team]

        e_h = self.expected_score(r_h, r_a)
        e_a = 1 - e_h

        # Actual score: 1=menang, 0.5=seri, 0=kalah
        if home_goals > away_goals:
            s_h, s_a = 1.0, 0.0
        elif home_goals == away_goals:
            s_h, s_a = 0.5, 0.5
        else:
            s_h, s_a = 0.0, 1.0

        delta_h = self.K * (s_h - e_h)
        delta_a = self.K * (s_a - e_a)

        old_h = self.ratings[home_team]
        old_a = self.ratings[away_team]

        self.ratings[home_team] += delta_h
        self.ratings[away_team] += delta_a

        self.history.append({
            "home_team"      : home_team,
            "away_team"      : away_team,
            "home_goals"     : home_goals,
            "away_goals"     : away_goals,
            "elo_home_before": old_h,
            "elo_away_before": old_a,
            "elo_home_after" : self.ratings[home_team],
            "elo_away_after" : self.ratings[away_team],
            "delta_home"     : delta_h,
            "delta_away"     : delta_a,
        })

    def fit(self, df):
        """Fit Elo dari seluruh dataframe match (urut tanggal)."""
        print("📊 Fitting Elo ratings dari data historis...")
        for _, row in df.iterrows():
            self.update(row["HomeTeam"], row["AwayTeam"],
                        int(row["FTHG"]), int(row["FTAG"]))
        ratings_sorted = sorted(self.ratings.items(), key=lambda x: -x[1])
        print(f"✅ Elo fitted! Top 5 tim:")
        for name, r in ratings_sorted[:5]:
            print(f"   {name:25s}: {r:.0f}")
        return self

    def get_win_probs(self, home_team, away_team):
        """
        Probabilitas H/D/A menggunakan Elo.
        Draw diestimasi: semakin dekat rating → draw lebih mungkin.
        """
        r_h = self.ratings[home_team] + self.home_advantage
        r_a = self.ratings[away_team]
        e_h = self.expected_score(r_h, r_a)

        rating_diff = abs(r_h - r_a)
        draw_prob   = max(0.18, 0.30 - 0.002 * rating_diff)

        win_h  = e_h       * (1 - draw_prob)
        win_a  = (1 - e_h) * (1 - draw_prob)

        total  = win_h + draw_prob + win_a
        return {
            "home_win" : round(win_h   / total, 4),
            "draw"     : round(draw_prob / total, 4),
            "away_win" : round(win_a   / total, 4),
        }

    def get_ratings_df(self):
        return pd.DataFrame([
            {"team": t, "elo": r} for t, r in
            sorted(self.ratings.items(), key=lambda x: -x[1])
        ])


# Fit Elo dari 80% data training
TRAIN_SIZE  = int(len(df_matches) * 0.80)
df_train    = df_matches.iloc[:TRAIN_SIZE]
df_val      = df_matches.iloc[TRAIN_SIZE:]

elo_system  = EloRatingSystem(k=32, home_advantage=65)
elo_system.fit(df_train)

print(f"\n📋 Split data:")
print(f"   Training : {len(df_train)} laga")
print(f"   Validasi : {len(df_val)} laga  ← untuk evaluasi akurasi")
print("\n✅ CELL 12 SELESAI!")


# ══════════════════════════════════════════════════════════════════════
# CELL 13 — Dixon-Coles Model
# ══════════════════════════════════════════════════════════════════════
# CELL 13 — Dixon-Coles Model
class DixonColesModel:
    """
    Dixon-Coles (1997) — gold standard prediksi sepakbola.

    Keunggulan vs Poisson biasa:
    1. Attack/defense parameter per tim di-fit dari data real
    2. Koreksi tau untuk skor rendah (0-0, 1-0, 0-1, 1-1)
       → mengatasi draw blindness utama
    3. Time-decay: laga lama bobot lebih kecil
    4. xG split home/away per tim otomatis
    """

    def __init__(self, xi=0.0018):
        self.xi       = xi
        self.params   = None
        self.teams    = []
        self.team_idx = {}

    def _time_weight(self, date, ref_date):
        days = (ref_date - date).days
        return math.exp(-self.xi * days)

    def _tau(self, lam_h, lam_a, x, y, rho):
        """Koreksi Dixon-Coles untuk skor rendah."""
        if x == 0 and y == 0:
            return 1 - lam_h * lam_a * rho
        elif x == 1 and y == 0:
            return 1 + lam_a * rho
        elif x == 0 and y == 1:
            return 1 + lam_h * rho
        elif x == 1 and y == 1:
            return 1 - rho
        else:
            return 1.0

    def _neg_log_likelihood(self, params, data, weights):
        n_teams  = len(self.teams)
        attack   = {t: params[i]           for i, t in enumerate(self.teams)}
        defense  = {t: params[i + n_teams] for i, t in enumerate(self.teams)}
        home_adv = params[2 * n_teams]
        rho      = params[2 * n_teams + 1]

        nll = 0
        for i, row in enumerate(data):
            h, a, gh, ga = row
            lam_h = math.exp(attack[h] + defense[a] + home_adv)
            lam_a = math.exp(attack[a] + defense[h])

            tau = self._tau(lam_h, lam_a, gh, ga, rho)
            if tau <= 0:
                tau = 1e-9

            log_lik = (weights[i] *
                       (math.log(tau) +
                        math.log(poisson.pmf(gh, lam_h) + 1e-9) +
                        math.log(poisson.pmf(ga, lam_a) + 1e-9)))
            nll -= log_lik

        return nll

    def fit(self, df):
        print("🔧 Fitting Dixon-Coles model (butuh ~30 detik)...")

        self.teams    = sorted(list(set(df["HomeTeam"]) | set(df["AwayTeam"])))
        self.team_idx = {t: i for i, t in enumerate(self.teams)}
        n             = len(self.teams)
        ref_date      = df["Date"].max()

        data    = [(r["HomeTeam"], r["AwayTeam"], int(r["FTHG"]), int(r["FTAG"]))
                   for _, r in df.iterrows()]
        weights = [self._time_weight(r["Date"], ref_date) for _, r in df.iterrows()]

        x0 = ([0.1] * n + [-0.1] * n + [0.3, -0.1])

        constraints = [{"type": "eq", "fun": lambda p: sum(p[:n])}]

        result = minimize(
            self._neg_log_likelihood,
            x0,
            args=(data, weights),
            method="L-BFGS-B",
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9},
        )

        self.params    = result.x
        self._attack   = {t: self.params[i]     for i, t in enumerate(self.teams)}
        self._defense  = {t: self.params[i + n] for i, t in enumerate(self.teams)}
        self._home_adv = self.params[2 * n]
        self._rho      = self.params[2 * n + 1]

        atk_sorted = sorted(self._attack.items(), key=lambda x: -x[1])
        print(f"\n✅ Dixon-Coles fitted! home_adv={self._home_adv:.3f} | rho={self._rho:.3f}")
        print("   Top 5 Attack Strength:")
        for name, v in atk_sorted[:5]:
            print(f"     {name:25s}: {v:+.3f}")
        return self

    def predict_proba(self, home_team, away_team, max_goals=8):
        """Matrix skor + H/D/A probabilities."""
        if (self.params is None or
                home_team not in self.teams or
                away_team not in self.teams):
            lam_h, lam_a = 1.35, 1.05
            rho      = self._rho      if self.params is not None else -0.1
            home_adv = self._home_adv if self.params is not None else 0.3
        else:
            lam_h = math.exp(
                self._attack[home_team] + self._defense[away_team] + self._home_adv
            )
            lam_a = math.exp(
                self._attack[away_team] + self._defense[home_team]
            )
            rho = self._rho

        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        for gh in range(max_goals + 1):
            for ga in range(max_goals + 1):
                tau = self._tau(lam_h, lam_a, gh, ga, rho)
                score_matrix[gh, ga] = (
                    tau * poisson.pmf(gh, lam_h) * poisson.pmf(ga, lam_a)
                )

        score_matrix /= score_matrix.sum()

        home_win = float(np.sum(np.tril(score_matrix, -1)))
        draw     = float(np.sum(np.diag(score_matrix)))
        away_win = float(np.sum(np.triu(score_matrix, 1)))

        flat_idx   = np.argsort(score_matrix.ravel())[::-1][:6]
        top_scores = [
            ((int(i // (max_goals+1)), int(i % (max_goals+1))),
             float(score_matrix.ravel()[i]))
            for i in flat_idx
        ]

        return {
            "home_win"    : round(home_win, 4),
            "draw"        : round(draw,     4),
            "away_win"    : round(away_win, 4),
            "lambda_home" : round(lam_h, 3),
            "lambda_away" : round(lam_a, 3),
            "top_scores"  : top_scores,
            "score_matrix": score_matrix,
        }

    def get_strength_df(self):
        return pd.DataFrame([
            {
                "team"       : t,
                "attack_str" : round(self._attack[t], 3),
                "defense_str": round(self._defense[t], 3),
                "net_strength": round(self._attack[t] - self._defense[t], 3),
            }
            for t in self.teams
        ]).sort_values("net_strength", ascending=False).reset_index(drop=True)


dc_model = DixonColesModel(xi=0.0018)
dc_model.fit(df_train)

print("\n📊 Strength Table (Top 10):")
print(dc_model.get_strength_df().head(10).to_string(index=False))
print("\n✅ CELL 13 SELESAI!")


# ══════════════════════════════════════════════════════════════════════
# CELL 14 — FIFA25 Strength Modifier + Injury Impact
# ══════════════════════════════════════════════════════════════════════
# CELL 14 — Player Strength & Injury Impact
class PlayerStrengthEngine:
    """
    Hitung team strength dari FIFA25 + pengaruh absensi pemain.

    Logika:
    - Squad strength = rata-rata overall 11 pemain terbaik
    - Injury impact = (rating pemain / squad_strength) * bobot posisi
    - GK cedera → penalti terbesar (25% dari pertandingan)
    """

    POSITION_WEIGHT = {
        "GK": 0.25, "CB": 0.12, "LB": 0.08, "RB": 0.08,
        "DM": 0.10, "CM": 0.10, "AM": 0.12, "LW": 0.08,
        "RW": 0.08, "ST": 0.15, "CF": 0.13,
    }

    # Nama klub FIFA → nama di match CSV
    CLUB_MAP = {
        "FC Barcelona"              : "Barcelona",
        "Real Madrid CF"            : "Real Madrid",
        "Club Atlético de Madrid"   : "Atletico Madrid",
        "Athletic Club"             : "Athletic Club",
        "Real Sociedad de Fútbol"   : "Real Sociedad",
        "RC Celta de Vigo"          : "Celta Vigo",
        "RCD Espanyol de Barcelona" : "Espanyol",
        "Rayo Vallecano de Madrid"  : "Rayo Vallecano",
        "CA Osasuna"                : "Osasuna",
        "RCD Mallorca"              : "Mallorca",
        "Deportivo Alavés"          : "Alaves",
        "Real Betis Balompié"       : "Real Betis",
        "Girona FC"                 : "Girona",
        "Villarreal CF"             : "Villarreal",
        "Sevilla FC"                : "Sevilla",
        "Valencia CF"               : "Valencia",
        "Getafe CF"                 : "Getafe",
    }

    def __init__(self, df_fifa):
        self.df_fifa     = df_fifa
        self.squad_cache = {}
        # Buat reverse map: nama CSV → nama FIFA
        self._reverse_map = {v: k for k, v in self.CLUB_MAP.items()}

    def get_squad_strength(self, team_name):
        if self.df_fifa.empty:
            return 75.0
        if team_name in self.squad_cache:
            return self.squad_cache[team_name]

        # Coba cari dengan nama asli atau mapping
        fifa_name = self._reverse_map.get(team_name, team_name)
        mask = (self.df_fifa["club_name"].str.contains(fifa_name, case=False, na=False) |
                self.df_fifa["club_name"].str.contains(team_name, case=False, na=False))
        squad = self.df_fifa[mask].nlargest(11, "overall")

        strength = round(float(squad["overall"].mean()), 2) if not squad.empty else 75.0
        self.squad_cache[team_name] = strength
        return strength

    def get_injury_penalty(self, team_name, absent_players=None):
        """Returns float 0.0–0.35 (semakin besar = semakin berpengaruh)."""
        if not absent_players or self.df_fifa.empty:
            return 0.0

        squad_strength = self.get_squad_strength(team_name)
        total_penalty  = 0.0

        players = (absent_players if isinstance(absent_players, list)
                   else list(absent_players.keys()))

        for player in players:
            mask   = self.df_fifa["short_name"].str.contains(str(player), case=False, na=False)
            p_data = self.df_fifa[mask]

            if p_data.empty:
                p_rating, p_pos = 75.0, "CM"
            else:
                p_row    = p_data.iloc[0]
                p_rating = float(p_row.get("overall", 75))
                raw_pos  = str(p_row.get("player_positions", "CM")).split(",")[0].strip()
                p_pos    = raw_pos if raw_pos in self.POSITION_WEIGHT else "CM"

            rel_impact    = (p_rating / squad_strength) * self.POSITION_WEIGHT.get(p_pos, 0.10)
            total_penalty += rel_impact
            print(f"     ⚕️  {player} ({p_pos}, {p_rating:.0f}): -xG {rel_impact*100:.1f}%")

        return round(min(total_penalty, 0.35), 3)

    def get_strength_modifier(self, home_team, away_team):
        """Multiplier xG dari selisih FIFA squad strength (range 0.90–1.10)."""
        if self.df_fifa.empty:
            return 1.0, 1.0

        s_h = self.get_squad_strength(home_team)
        s_a = self.get_squad_strength(away_team)
        avg = (s_h + s_a) / 2

        mod_h = 1.0 + (s_h - avg) / avg * 0.5
        mod_a = 1.0 + (s_a - avg) / avg * 0.5

        return (round(max(0.90, min(1.10, mod_h)), 4),
                round(max(0.90, min(1.10, mod_a)), 4))


pse = PlayerStrengthEngine(df_fifa)

print("🧪 Test squad strength:")
for team in ["Barcelona", "Real Madrid", "Atletico Madrid", "Girona", "Osasuna"]:
    s = pse.get_squad_strength(team)
    print(f"   {team:25s}: {s:.1f}")

print("\n✅ CELL 14 SELESAI!")


# ══════════════════════════════════════════════════════════════════════
# CELL 15 — Ensemble Predictor V6 (Elo 30% + DC 45% + MC 25%)
# ══════════════════════════════════════════════════════════════════════
# CELL 15 — Ensemble Predictor V6
class EnsemblePredictor:
    """
    Weighted ensemble dari 3 model:
      Dixon-Coles  45% — terbaik untuk distribusi gol & draw
      Elo Rating   30% — encode kekuatan relatif & form dinamis
      Monte Carlo  25% — capture fatigue, PPDA, set piece, volatility

    Perbaikan utama vs V5:
      1. Draw Threshold: |home-away| < DRAW_MARGIN → prediksi Draw
      2. Confidence dari konsensus 3 model, bukan cuma max prob
      3. FIFA25 strength modifier pada xG
      4. Injury penalty mengurangi xG tim secara proporsional
    """

    DRAW_MARGIN  = 0.08   # jika |home-away| < 8% → dorong ke draw
    WEIGHTS_DC   = 0.45
    WEIGHTS_ELO  = 0.30
    WEIGHTS_MC   = 0.25

    def __init__(self, dc_model, elo_system, mc_simulator, fe, pse):
        self.dc  = dc_model
        self.elo = elo_system
        self.mc  = mc_simulator
        self.fe  = fe
        self.pse = pse

    def _get_mc_probs(self, home_team, away_team, home_data, away_data,
                      days_rest_h, days_rest_a, ppda_h, ppda_a,
                      setpiece_h, setpiece_a, absent_h, absent_a):

        inj_h = self.pse.get_injury_penalty(home_team, absent_h)
        inj_a = self.pse.get_injury_penalty(away_team, absent_a)

        home_p = self.fe.build_profile(
            home_team, home_data, days_rest=days_rest_h,
            is_home=True, ppda=ppda_h, setpiece=setpiece_h
        )
        away_p = self.fe.build_profile(
            away_team, away_data, days_rest=days_rest_a,
            is_home=False, ppda=ppda_a, setpiece=setpiece_a
        )

        # Terapkan injury penalty
        if inj_h > 0:
            home_p["form"]["xg_avg"] *= (1 - inj_h)
            print(f"   ⚕️  {home_team} xG -{inj_h*100:.1f}% karena cedera")
        if inj_a > 0:
            away_p["form"]["xg_avg"] *= (1 - inj_a)
            print(f"   ⚕️  {away_team} xG -{inj_a*100:.1f}% karena cedera")

        # FIFA25 strength modifier
        mod_h, mod_a = self.pse.get_strength_modifier(home_team, away_team)
        home_p["form"]["xg_avg"] *= mod_h
        away_p["form"]["xg_avg"] *= mod_a

        sim = self.mc.run(home_p, away_p)
        return {
            "home_win": sim["home_win_pct"] / 100,
            "draw"    : sim["draw_pct"]     / 100,
            "away_win": sim["away_win_pct"] / 100,
        }, sim, home_p, away_p

    def predict(self, home_team, away_team,
                home_data=None, away_data=None,
                days_rest_h=5, days_rest_a=5,
                ppda_h=10.5, ppda_a=11.0,
                setpiece_h=5.0, setpiece_a=5.0,
                absent_h=None, absent_a=None,
                ctx=None):

        print(f"\n{'='*60}")
        print(f"  ⚽ ENSEMBLE V6: {home_team} vs {away_team}")
        print(f"{'='*60}")

        ctx = ctx or {}

        # ── Model 1: Dixon-Coles ───────────────────────────────────
        dc_probs = self.dc.predict_proba(home_team, away_team)
        print(f"\n[1/3] Dixon-Coles  H={dc_probs['home_win']*100:.1f}% "
              f"D={dc_probs['draw']*100:.1f}% A={dc_probs['away_win']*100:.1f}%")
        print(f"      λ_home={dc_probs['lambda_home']} | λ_away={dc_probs['lambda_away']}")

        # ── Model 2: Elo ───────────────────────────────────────────
        elo_probs = self.elo.get_win_probs(home_team, away_team)
        elo_home  = self.elo.ratings.get(home_team, 1500)
        elo_away  = self.elo.ratings.get(away_team, 1500)
        print(f"\n[2/3] Elo Rating   H={elo_probs['home_win']*100:.1f}% "
              f"D={elo_probs['draw']*100:.1f}% A={elo_probs['away_win']*100:.1f}%")
        print(f"      {home_team}: {elo_home:.0f} | {away_team}: {elo_away:.0f}")

        # ── Model 3: Monte Carlo ───────────────────────────────────
        print(f"\n[3/3] Monte Carlo:")
        if home_data is not None and away_data is not None:
            mc_probs, sim_result, home_p, away_p = self._get_mc_probs(
                home_team, away_team, home_data, away_data,
                days_rest_h, days_rest_a, ppda_h, ppda_a,
                setpiece_h, setpiece_a, absent_h, absent_a
            )
            print(f"      H={mc_probs['home_win']*100:.1f}% "
                  f"D={mc_probs['draw']*100:.1f}% A={mc_probs['away_win']*100:.1f}%")
        else:
            mc_probs   = dc_probs
            sim_result = None
            home_p, away_p = None, None
            print("      ⚠️  Dilewati (gunakan DC sebagai fallback)")

        # ── Weighted Ensemble ──────────────────────────────────────
        h = (self.WEIGHTS_DC  * dc_probs["home_win"] +
             self.WEIGHTS_ELO * elo_probs["home_win"] +
             self.WEIGHTS_MC  * mc_probs["home_win"])

        d = (self.WEIGHTS_DC  * dc_probs["draw"] +
             self.WEIGHTS_ELO * elo_probs["draw"] +
             self.WEIGHTS_MC  * mc_probs["draw"])

        a = (self.WEIGHTS_DC  * dc_probs["away_win"] +
             self.WEIGHTS_ELO * elo_probs["away_win"] +
             self.WEIGHTS_MC  * mc_probs["away_win"])

        total = h + d + a
        h /= total; d /= total; a /= total

        # ── Draw Threshold (perbaikan draw blindness) ──────────────
        if abs(h - a) < self.DRAW_MARGIN:
            boost = self.DRAW_MARGIN * 0.7
            h    -= boost * (h / (h + a + 1e-9))
            a    -= boost * (a / (h + a + 1e-9))
            d    += boost
            total2 = h + d + a
            h /= total2; d /= total2; a /= total2

        # ── Final prediction ───────────────────────────────────────
        probs = {"home_win": h, "draw": d, "away_win": a}
        pred  = max(probs, key=probs.get)

        # ── Confidence dari konsensus 3 model ─────────────────────
        votes     = [max(dc_probs, key=dc_probs.get),
                     max(elo_probs, key=elo_probs.get),
                     max(mc_probs, key=mc_probs.get)]
        consensus  = votes.count(pred) / 3
        confidence = round(max(h, d, a) * 0.6 + consensus * 0.4, 4)
        conf_label = ("Tinggi" if confidence > 0.58 else
                      ("Sedang" if confidence > 0.46 else "Rendah"))

        top_scores = dc_probs["top_scores"]

        print(f"\n{'─'*60}")
        print(f"  ENSEMBLE RESULT")
        print(f"  🏠 Home Win  : {h*100:.1f}%")
        print(f"  🤝 Draw      : {d*100:.1f}%")
        print(f"  ✈️  Away Win  : {a*100:.1f}%")
        print(f"  🎯 Prediksi  : {pred.upper()} | {conf_label} ({confidence*100:.1f}%)")
        print(f"  🏆 Top Skor  : {top_scores[0][0][0]}-{top_scores[0][0][1]} "
              f"({top_scores[0][1]*100:.1f}%)")
        print(f"  🤝 Konsensus : {votes.count(pred)}/3 model setuju")
        print(f"{'='*60}")

        return {
            "prediction"      : pred,
            "home_win_prob"   : round(h, 4),
            "draw_prob"       : round(d, 4),
            "away_win_prob"   : round(a, 4),
            "confidence"      : confidence,
            "confidence_label": conf_label,
            "model_consensus" : votes.count(pred),
            "model_votes"     : votes,
            "dc_probs"        : dc_probs,
            "elo_probs"       : elo_probs,
            "mc_probs"        : mc_probs,
            "top_scores"      : top_scores,
            "sim_result"      : sim_result,
            "home_profile"    : home_p,
            "away_profile"    : away_p,
            "elo_home"        : elo_home,
            "elo_away"        : elo_away,
        }


# Inisialisasi Ensemble V6
ensemble = EnsemblePredictor(
    dc_model     = dc_model,
    elo_system   = elo_system,
    mc_simulator = sim_engine,   # dari Cell 5 (v5)
    fe           = fe,            # dari Cell 4 (v5)
    pse          = pse,
)

print("\n✅ CELL 15 SELESAI — EnsemblePredictor V6 siap!")


# ══════════════════════════════════════════════════════════════════════
# CELL 16 — Validasi + predict_match()
# ══════════════════════════════════════════════════════════════════════
# CELL 16 — Validasi Akurasi & Fungsi predict_match()

def validate_model(df_val, max_laga=None, verbose=False):
    """Bandingkan prediksi V6 vs hasil nyata di df_val."""
    results = []
    n       = min(len(df_val), max_laga) if max_laga else len(df_val)

    print(f"\n{'='*55}")
    print(f"  🧪 VALIDASI {n} LAGA — ENSEMBLE V6")
    print(f"{'='*55}")

    for _, row in df_val.head(n).iterrows():
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]
        actual = {"H": "home_win", "D": "draw", "A": "away_win"}[row["FTR"]]

        try:
            res    = ensemble.predict(h_team, a_team)
            correct = res["prediction"] == actual
            results.append({
                "match"     : f"{h_team} vs {a_team}",
                "actual"    : actual,
                "predicted" : res["prediction"],
                "correct"   : correct,
                "confidence": res["confidence_label"],
                "consensus" : res["model_consensus"],
            })
            if verbose:
                status = "✅" if correct else "❌"
                print(f"  {status} {h_team:20s} vs {a_team:20s} "
                      f"| Pred:{res['prediction']:10s} Aktual:{actual}")
        except Exception as e:
            if verbose:
                print(f"  ⚠️  Error {h_team} vs {a_team}: {e}")
            continue

    if not results:
        print("  ❌ Tidak ada hasil!")
        return pd.DataFrame()

    df_res   = pd.DataFrame(results)
    accuracy = df_res["correct"].mean() * 100

    print(f"\n  Akurasi per confidence level:")
    for level in ["Tinggi", "Sedang", "Rendah"]:
        sub = df_res[df_res["confidence"] == level]
        if len(sub):
            acc = sub["correct"].mean() * 100
            print(f"    {level:8s}: {acc:.1f}% ({sub['correct'].sum()}/{len(sub)})")

    print(f"\n  Akurasi per konsensus model:")
    for vote in [3, 2, 1]:
        sub = df_res[df_res["consensus"] == vote]
        if len(sub):
            acc = sub["correct"].mean() * 100
            print(f"    {vote}/3 model setuju: {acc:.1f}% ({sub['correct'].sum()}/{len(sub)})")

    print(f"\n  Distribusi Prediksi vs Aktual:")
    for label in ["home_win", "draw", "away_win"]:
        p = (df_res["predicted"] == label).sum()
        a = (df_res["actual"]    == label).sum()
        print(f"    {label:12s}: Pred={p:3d} | Aktual={a:3d}")

    print(f"\n  {'─'*40}")
    print(f"  Total: {len(df_res)} | Benar: {df_res['correct'].sum()} | Akurasi: {accuracy:.1f}%")
    print(f"  Benchmark industri : 55-62%")
    status = "✅ DI ATAS" if accuracy >= 55 else "⚠️  DI BAWAH"
    print(f"  Status             : {status}")
    print(f"{'='*55}")
    return df_res


def predict_match(home_team, away_team,
                  days_rest_h=5, days_rest_a=5,
                  ppda_h=None, ppda_a=None,
                  setpiece_h=5.0, setpiece_a=5.0,
                  absent_h=None, absent_a=None,
                  league="La Liga", show_report=True):
    """
    One-liner prediksi pertandingan La Liga.

    Contoh:
      predict_match("Barcelona", "Real Madrid")
      predict_match("Atletico Madrid", "Sevilla", absent_h=["Griezmann"])
      predict_match("Girona", "Villarreal", days_rest_h=3, days_rest_a=6)
      predict_match("Osasuna", "Getafe", ppda_h=11.5, ppda_a=10.2)
    """
    # Ambil PPDA dari tabel estimasi (v5) jika tidak diisi manual
    if ppda_h is None:
        key_h  = next((k for k, v in STANDINGS_TO_AF.items()
                       if v.lower() == home_team.lower()), None)
        ppda_h = PPDA_ESTIMATED.get(key_h, 11.0)

    if ppda_a is None:
        key_a  = next((k for k, v in STANDINGS_TO_AF.items()
                       if v.lower() == away_team.lower()), None)
        ppda_a = PPDA_ESTIMATED.get(key_a, 11.0)

    ctx = {
        "league"  : league,
        "injuries": (f"Absen {home_team}: {absent_h or '-'} | "
                     f"{away_team}: {absent_a or '-'}"),
    }

    result = ensemble.predict(
        home_team, away_team,
        home_data   = DATA,   # dari Cell 8 (v5)
        away_data   = DATA,
        days_rest_h = days_rest_h,
        days_rest_a = days_rest_a,
        ppda_h      = ppda_h,
        ppda_a      = ppda_a,
        setpiece_h  = setpiece_h,
        setpiece_a  = setpiece_a,
        absent_h    = absent_h,
        absent_a    = absent_a,
        ctx         = ctx,
    )

    if show_report and result.get("sim_result") and result.get("home_profile"):
        from IPython.display import display, Markdown
        rpt = reporter.generate(
            result["home_profile"],
            result["away_profile"],
            result["sim_result"],
            {
                "prediction"      : result["prediction"],
                "home_win_prob"   : result["home_win_prob"],
                "draw_prob"       : result["draw_prob"],
                "away_win_prob"   : result["away_win_prob"],
                "confidence"      : result["confidence"],
                "confidence_label": result["confidence_label"],
            },
            ctx,
        )
        display(Markdown(rpt))

    return result


# ── Jalankan validasi ──────────────────────────────────────────────────
print("🚀 Memulai validasi V6...\n")
df_validation = validate_model(df_val, max_laga=50, verbose=False)

print("\n✅ CELL 16 SELESAI!")
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CARA PAKAI predict_match():

  # Prediksi dasar
  predict_match("Barcelona", "Real Madrid")

  # Dengan absensi pemain
  predict_match("Atletico Madrid", "Getafe",
                days_rest_h=3,
                absent_h=["Griezmann", "Morata"])

  # Dengan semua parameter
  predict_match("Girona", "Osasuna",
                days_rest_h=4, days_rest_a=6,
                ppda_h=12.4, ppda_a=11.5,
                setpiece_h=6.0, setpiece_a=4.5)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
