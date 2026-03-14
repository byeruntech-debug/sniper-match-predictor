#!/usr/bin/env python3
"""
Football Match Predictor — La Liga v4.0
Auto-exported dari football_simulation_v3.ipynb
"""

# ── Cell 1 ────────────────────────────────────────
# CELL 1 — Install Library
!pip install pandas numpy requests beautifulsoup4 lxml scikit-learn matplotlib seaborn google-genai -q
print("✅ CELL 1 SELESAI!")


# ── Cell 2 ────────────────────────────────────────
# CELL 2 — Konfigurasi
import os, warnings, json, re, math
import pandas as pd
import numpy as np
from collections import Counter
warnings.filterwarnings("ignore")

GEMINI_API_KEY = ""          # ← isi API key Gemini kamu
N_SIMULATIONS  = 10_000
FORM_WINDOW    = 5
CURRENT_SEASON = 2024

print("✅ CELL 2 SELESAI!")


# ── Cell 3 ────────────────────────────────────────
# CELL 3 — Data Collector
import requests
from bs4 import BeautifulSoup

class UnderstatCollector:
    BASE_URL = "https://understat.com"
    HEADERS  = {"User-Agent": "Mozilla/5.0"}

    def __init__(self, league="EPL", season=2024):
        self.league  = league
        self.season  = season

    def get_data(self):
        print("📥 Mengambil data Understat...")
        url = f"{self.BASE_URL}/league/{self.league}/{self.season}"
        try:
            resp = requests.get(url, headers=self.HEADERS, timeout=15)
            soup = BeautifulSoup(resp.content, "lxml")
            teams_data = {}
            for script in soup.find_all("script"):
                if "teamsData" in str(script):
                    raw = re.search(r"teamsData\s*=\s*JSON\.parse\(\'(.+?)\'\)", str(script))
                    if raw:
                        decoded    = raw.group(1).encode("utf-8").decode("unicode_escape")
                        teams_data = json.loads(decoded)
                        break
            rows = []
            for tid, tdata in teams_data.items():
                history = tdata.get("history", [])
                if history:
                    rows.append({
                        "team"   : tdata.get("title",""),
                        "xPTS"   : round(sum(h.get("xpts",0) for h in history),2),
                        "PTS"    : sum(h.get("pts",0) for h in history),
                        "xG"     : round(sum(h.get("xG",0)  for h in history),2),
                        "xGA"    : round(sum(h.get("xGA",0) for h in history),2),
                        "matches": len(history),
                        "history": history,
                    })
            df = pd.DataFrame(rows)
            print(f"✅ {len(df)} tim berhasil diambil!")
            return df if not df.empty else self._demo_data()
        except Exception as e:
            print(f"⚠️  Gagal online ({e}) → pakai data demo")
            return self._demo_data()

    def _demo_data(self):
        #  name               pts  xpts  xg/g  xga/g  shoot_skill  gk_skill  volatility
        teams = [
            ("Arsenal",          65, 58, 1.90, 0.90,  1.05, 1.08, 0.10),
            ("Manchester City",  63, 61, 1.80, 0.80,  1.10, 1.12, 0.08),
            ("Liverpool",        60, 57, 1.70, 1.00,  1.08, 1.05, 0.12),
            ("Chelsea",          48, 45, 1.40, 1.30,  0.98, 0.97, 0.20),
            ("Manchester United",40, 44, 1.20, 1.50,  0.95, 0.93, 0.25),
            ("Tottenham",        44, 42, 1.50, 1.40,  1.02, 0.95, 0.22),
            ("Newcastle",        50, 47, 1.50, 1.10,  1.00, 1.03, 0.15),
            ("Aston Villa",      52, 49, 1.60, 1.20,  1.03, 1.00, 0.18),
        ]
        rows = []
        for name, pts, xpts, xg_pg, xga_pg, sk, gk, vol in teams:
            history = []
            rng = np.random.default_rng(abs(hash(name)) % (2**32))
            for _ in range(28):
                w     = rng.random()
                pts_m = 3 if w > 0.55 else (1 if w > 0.30 else 0)
                history.append({
                    "xG"    : round(float(rng.normal(xg_pg, 0.40)), 2),
                    "xGA"   : round(float(rng.normal(xga_pg,0.30)), 2),
                    "scored": int(rng.poisson(xg_pg * sk)),
                    "missed": int(rng.poisson(xga_pg / gk)),
                    "pts"   : pts_m,
                    "xpts"  : round(float(rng.normal(1.0, 0.30)), 2),
                    "minutes_played": int(rng.integers(70, 96)),
                })
            rows.append({
                "team"         : name,
                "PTS"          : pts,
                "xPTS"         : xpts,
                "xG"           : round(xg_pg * 28, 1),
                "xGA"          : round(xga_pg * 28, 1),
                "matches"      : 28,
                "shoot_skill"  : sk,
                "gk_skill"     : gk,
                "volatility"   : vol,
                "history"      : history,
            })
        return pd.DataFrame(rows)

print("✅ CELL 3 SELESAI!")


# ── Cell 4 ────────────────────────────────────────
# CELL 4 — Feature Engineering (UPGRADED)
# Perbaikan: Shooting Skill, Dynamic Fatigue, Time-Decay Form
class FeatureEngineer:
    """
    PERBAIKAN 1: Shooting Skill & GK Overperformance
    PERBAIKAN 2: Dynamic Fatigue Scaling (otomatis dari menit bermain)
    PERBAIKAN 4: Volatility Index
    PERBAIKAN 6: Time-Decay Weighting untuk form
    """

    def __init__(self, form_window=5):
        self.form_window = form_window

    # --- PERBAIKAN 2: Dynamic fatigue dari history menit bermain ---
    def auto_fatigue(self, history, days_rest=4):
        """Hitung fatigue otomatis dari menit bermain 30 hari terakhir."""
        recent = history[-4:] if len(history) >= 4 else history  # ~4 laga terakhir ≈ 30 hari
        total_min = sum(m.get("minutes_played", 85) for m in recent)
        max_min   = 4 * 96
        fatigue_from_minutes = min(total_min / max_min, 1.0)
        fatigue_from_rest    = max(0.0, 1.0 - days_rest / 7.0)
        return round(fatigue_from_minutes * 0.6 + fatigue_from_rest * 0.4, 3)

    # --- PERBAIKAN 6: Time-Decay Weighting ---
    def calculate_form(self, history, window=5):
        """Form dengan bobot eksponensial — laga terbaru lebih berpengaruh."""
        recent = history[-window:] if len(history) >= window else history
        if not recent:
            return {
                "wins":2,"draws":1,"losses":2,
                "xg_avg":1.3,"xga_avg":1.3,
                "weighted_form":1.2,"form_string":"WDLWL",
                "efficiency_ratio":1.0,
            }
        n = len(recent)
        # Decay: laga terbaru bobot terbesar (exponential)
        raw_weights = [math.exp(0.3 * i) for i in range(n)]
        total_w     = sum(raw_weights)
        weights     = [w / total_w for w in raw_weights]

        goals_scored   = [m.get("scored", m.get("xG", 1.2)) for m in recent]
        goals_conceded = [m.get("missed", m.get("xGA",1.2)) for m in recent]
        xg_vals        = [m.get("xG",  1.2) for m in recent]
        xga_vals       = [m.get("xGA", 1.2) for m in recent]
        pts_vals       = [m.get("pts",   0) for m in recent]

        # Rasio efisiensi: gol nyata vs xG
        actual_scored = sum(goals_scored)
        expected_scored = sum(xg_vals)
        efficiency = actual_scored / expected_scored if expected_scored > 0 else 1.0

        return {
            "wins"           : sum(1 for m in recent if m.get("pts")==3),
            "draws"          : sum(1 for m in recent if m.get("pts")==1),
            "losses"         : sum(1 for m in recent if m.get("pts")==0),
            "xg_avg"         : round(float(np.average(xg_vals,  weights=weights)), 3),
            "xga_avg"        : round(float(np.average(xga_vals, weights=weights)), 3),
            "weighted_form"  : round(float(np.average(pts_vals,  weights=weights)), 3),
            "form_string"    : "".join(["W" if m.get("pts")==3 else "D" if m.get("pts")==1 else "L" for m in recent]),
            "efficiency_ratio": round(efficiency, 3),  # > 1.0 = overperform xG
        }

    def build_profile(self, name, data, days_rest=5, is_home=True,
                      ppda=10.5, setpiece=5.0, override_fatigue=None):
        print(f"  🔧 Building profile: {name}")
        mask = data["team"].str.contains(name, case=False, na=False)
        row  = data[mask]
        history = row.iloc[0]["history"] if not row.empty else []
        form    = self.calculate_form(history)

        xpts = float(row.iloc[0]["xPTS"]) if not row.empty else 40.0
        pts  = float(row.iloc[0]["PTS"])  if not row.empty else 40.0

        # PERBAIKAN 1: Shooting Skill & GK Skill dari data
        shoot_skill = float(row.iloc[0]["shoot_skill"]) if ("shoot_skill" in row.columns and not row.empty) else form["efficiency_ratio"]
        gk_skill    = float(row.iloc[0]["gk_skill"])    if ("gk_skill"    in row.columns and not row.empty) else 1.0
        volatility  = float(row.iloc[0]["volatility"])  if ("volatility"  in row.columns and not row.empty) else 0.15

        # PERBAIKAN 2: Fatigue otomatis
        fatigue = override_fatigue if override_fatigue is not None else self.auto_fatigue(history, days_rest)

        print(f"     Form: {form['form_string']} | xG: {form['xg_avg']} | xGA: {form['xga_avg']}")
        print(f"     Shooting Skill: {shoot_skill:.2f} | GK Skill: {gk_skill:.2f} | Fatigue: {fatigue:.3f} | Volatility: {volatility:.2f}")

        return {
            "team_name"      : name,
            "is_home"        : is_home,
            "form"           : form,
            "xpts"           : xpts,
            "actual_pts"     : pts,
            "luck_factor"    : round(pts - xpts, 2),
            "fatigue_index"  : fatigue,
            "home_advantage" : 0.12 if is_home else 0.0,
            "ppda"           : ppda,
            "set_piece_danger": setpiece,
            "shoot_skill"    : shoot_skill,   # BARU
            "gk_skill"       : gk_skill,      # BARU
            "volatility"     : volatility,    # BARU
        }

print("✅ CELL 4 SELESAI!")


# ── Cell 5 ────────────────────────────────────────
# CELL 5 — Monte Carlo Simulator (UPGRADED)
# Perbaikan: Time-Slice (6 segmen), Volatility Index, Game State
class MonteCarloSimulator:
    """
    PERBAIKAN 3: Time-Slice Game State (6 segmen x 15 menit)
    PERBAIKAN 4: Volatility Index (kejutan sepakbola)
    PERBAIKAN 1: Shooting Skill & GK Overperformance
    """

    N_SEGMENTS = 6   # 6 x 15 menit = 90 menit

    def __init__(self, n=10_000):
        self.n   = n
        self.rng = np.random.default_rng(42)

    def _base_lambda(self, atk, dfn, segment=0):
        """
        Lambda (expected goals per segmen) disesuaikan per segmen.
        Segmen awal: tim press lebih agresif.
        Segmen akhir: tim unggul cenderung bertahan (ditangani di run()).
        """
        # Dasar dari xG tim
        lam = atk["form"]["xg_avg"] / self.N_SEGMENTS

        # Koreksi fatigue — lebih terasa di segmen akhir
        fatigue_effect = atk["fatigue_index"] * (0.05 + 0.05 * (segment / self.N_SEGMENTS))
        lam -= fatigue_effect

        # Home advantage
        lam += atk["home_advantage"] / self.N_SEGMENTS

        # PPDA: tim dengan pressing lebih baik (PPDA rendah) lebih dominan
        lam += max(0, (dfn["ppda"] - atk["ppda"]) * 0.01)

        # Set piece
        lam += atk["set_piece_danger"] * 0.003

        # Luck normalization
        lam -= 0.005 * atk["luck_factor"]

        # PERBAIKAN 1: Shooting Skill
        lam *= atk["shoot_skill"]

        # PERBAIKAN 1: GK Skill lawan (mengurangi xG yang masuk gawang)
        lam /= dfn["gk_skill"]

        # Batas bawah & atas
        return max(0.03, min(0.9, lam))

    def _simulate_match(self, home, away):
        """Simulasi satu pertandingan dengan Time-Slice."""
        h_goals = 0
        a_goals = 0

        for seg in range(self.N_SEGMENTS):
            # PERBAIKAN 3: Jika salah satu tim unggul, tim itu bertahan lebih
            goal_diff = h_goals - a_goals
            h_state_adj = -0.015 * max(goal_diff, 0)    # unggul → lebih defensif
            a_state_adj = -0.015 * max(-goal_diff, 0)

            lh = max(0.01, self._base_lambda(home, away, seg) + h_state_adj)
            la = max(0.01, self._base_lambda(away, home, seg) + a_state_adj)

            # PERBAIKAN 4: Volatility — tingkatkan std dev Poisson via Negative Binomial
            vol = (home["volatility"] + away["volatility"]) / 2
            r_h = max(0.1, lh / max(vol, 0.001))
            r_a = max(0.1, la / max(vol, 0.001))
            p_h = r_h / (r_h + 1)
            p_a = r_a / (r_a + 1)

            h_goals += int(self.rng.negative_binomial(r_h, p_h))
            a_goals += int(self.rng.negative_binomial(r_a, p_a))

        return h_goals, a_goals

    def run(self, home, away):
        print(f"\n🎲 Simulasi {self.n:,}x Time-Slice: {home['team_name']} vs {away['team_name']}")
        pairs = [self._simulate_match(home, away) for _ in range(self.n)]

        counter = Counter(pairs)
        hg_arr  = np.array([p[0] for p in pairs])
        ag_arr  = np.array([p[1] for p in pairs])

        hw_pct = round(sum(h > a for h, a in pairs) / self.n * 100, 1)
        d_pct  = round(sum(h == a for h, a in pairs) / self.n * 100, 1)
        aw_pct = round(sum(h < a for h, a in pairs) / self.n * 100, 1)
        btts   = round(sum(h > 0 and a > 0 for h, a in pairs) / self.n * 100, 1)
        ov25   = round(sum(h + a > 2.5 for h, a in pairs) / self.n * 100, 1)

        print(f"  🏠 Home Menang : {hw_pct}%")
        print(f"  🤝 Seri        : {d_pct}%")
        print(f"  ✈️  Away Menang : {aw_pct}%")
        print(f"  ⚽ BTTS: {btts}% | Over 2.5: {ov25}%")
        top = counter.most_common(1)[0][0]
        print(f"  🎯 Skor Paling Mungkin: {top[0]}-{top[1]}")

        return {
            "home_team"         : home["team_name"],
            "away_team"         : away["team_name"],
            "home_win_pct"      : hw_pct,
            "draw_pct"          : d_pct,
            "away_win_pct"      : aw_pct,
            "btts_pct"          : btts,
            "over25_pct"        : ov25,
            "avg_home_goals"    : round(float(np.mean(hg_arr)), 2),
            "avg_away_goals"    : round(float(np.mean(ag_arr)), 2),
            "std_home_goals"    : round(float(np.std(hg_arr)), 2),
            "std_away_goals"    : round(float(np.std(ag_arr)), 2),
            "most_likely_scores": counter.most_common(5),
            "n_simulations"     : self.n,
        }

print("✅ CELL 5 SELESAI!")


# ── Cell 6 ────────────────────────────────────────
# CELL 6 — Match Predictor
class MatchPredictor:
    def predict(self, home, away, sim):
        hp = sim["home_win_pct"] / 100
        dp = sim["draw_pct"]     / 100
        ap = sim["away_win_pct"] / 100

        adj = (
            -(home["luck_factor"] - away["luck_factor"]) * 0.01
            + (away["fatigue_index"] - home["fatigue_index"]) * 0.02
            + (away["ppda"] - home["ppda"]) * 0.01
        )
        hp = max(0.05, min(0.90, hp + adj))
        ap = max(0.05, min(0.90, ap - adj))
        total = hp + dp + ap
        hp /= total; dp /= total; ap /= total

        probs  = [hp, dp, ap]
        labels = ["home_win", "draw", "away_win"]
        pred   = labels[int(np.argmax(probs))]
        conf   = max(probs)
        conf_l = "Tinggi" if conf > 0.55 else ("Sedang" if conf > 0.45 else "Rendah")

        print(f"\n🎯 PREDIKSI: {pred.upper()} | Keyakinan: {conf_l} ({conf*100:.1f}%)")
        return {
            "prediction"     : pred,
            "home_win_prob"  : round(hp, 3),
            "draw_prob"      : round(dp, 3),
            "away_win_prob"  : round(ap, 3),
            "confidence"     : round(conf, 3),
            "confidence_label": conf_l,
        }

print("✅ CELL 6 SELESAI!")


# ── Cell 7 ────────────────────────────────────────
# CELL 7 — Report Generator (UPGRADED)
# Perbaikan: Template-Based Fallback jika API mati
class ReportGenerator:
    """
    PERBAIKAN 5: Template-Based Fallback jika API mati.
    """

    def __init__(self):
        self.client = None
        self._init_gemini()

    def _init_gemini(self):
        try:
            from google import genai
            self.client = genai.Client(api_key=GEMINI_API_KEY)
            print("✅ Gemini AI siap!")
        except Exception as e:
            print(f"⚠️  Gemini tidak tersedia ({e}) → akan pakai template fallback")

    def generate(self, home, away, sim, pred, ctx=None):
        print("\n📝 Generating laporan...")
        ctx = ctx or {}
        if self.client:
            try:
                return self._gemini_report(home, away, sim, pred, ctx)
            except Exception as e:
                print(f"⚠️  Gemini error ({e}) → fallback ke template")
        return self._template_report(home, away, sim, pred, ctx)

    def _gemini_report(self, home, away, sim, pred, ctx):
        from google import genai
        prompt = self._build_prompt(home, away, sim, pred, ctx)
        resp   = self.client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
        print("✅ Laporan Gemini selesai!")
        return resp.text

    def _build_prompt(self, home, away, sim, pred, ctx):
        return f"""
Kamu adalah analis sepakbola profesional. Buat laporan analisis pertandingan komprehensif bahasa Indonesia.

TIM KANDANG : {home["team_name"]} | TIM TAMU: {away["team_name"]}
Liga: {ctx.get("league","?")} | Wasit: {ctx.get("referee","TBD")} | Cuaca: {ctx.get("weather","Normal")}

SIMULASI MONTE CARLO TIME-SLICE ({sim["n_simulations"]:,}x):
- Home Win: {sim["home_win_pct"]}% | Draw: {sim["draw_pct"]}% | Away Win: {sim["away_win_pct"]}%
- BTTS: {sim["btts_pct"]}% | Over 2.5: {sim["over25_pct"]}%
- Rata-rata Gol: {sim["avg_home_goals"]} - {sim["avg_away_goals"]}  (SD: {sim["std_home_goals"]} vs {sim["std_away_goals"]})
- Top 5 Skor: {sim["most_likely_scores"]}

{home["team_name"]} (KANDANG):
Form({home["form"]["form_string"]}) xG:{home["form"]["xg_avg"]} xGA:{home["form"]["xga_avg"]}
xPTS:{home["xpts"]} Luck:{home["luck_factor"]:+.1f} Fatigue:{home["fatigue_index"]:.3f}
PPDA:{home["ppda"]} SetPiece:{home["set_piece_danger"]}
Shooting Skill:{home["shoot_skill"]:.2f} GK Skill:{home["gk_skill"]:.2f} Volatility:{home["volatility"]:.2f}
Efficiency Ratio:{home["form"]["efficiency_ratio"]:.2f}

{away["team_name"]} (TAMU):
Form({away["form"]["form_string"]}) xG:{away["form"]["xg_avg"]} xGA:{away["form"]["xga_avg"]}
xPTS:{away["xpts"]} Luck:{away["luck_factor"]:+.1f} Fatigue:{away["fatigue_index"]:.3f}
PPDA:{away["ppda"]} SetPiece:{away["set_piece_danger"]}
Shooting Skill:{away["shoot_skill"]:.2f} GK Skill:{away["gk_skill"]:.2f} Volatility:{away["volatility"]:.2f}
Efficiency Ratio:{away["form"]["efficiency_ratio"]:.2f}

PREDIKSI: {pred["prediction"].upper()} | Keyakinan: {pred["confidence_label"]}
Cedera: {ctx.get("injuries","-")} | H2H: {ctx.get("h2h","-")}

Tulis laporan komprehensif:
## 7. STATISTIK PERBANDINGAN MUSIM
## 8. FAKTOR KUNCI (3-5 faktor, urutkan dari paling berpengaruh)
## 9. ANALISIS KONTEKSTUAL
## 10. PERFORMA TERKINI (Time-Decay Form)
## 11. PRODUKTIVITAS GOL (Shooting Skill vs GK Overperformance)
## 12. ANALISIS TAKTIK (Game State & Time-Slice)
## 13. MOTIVASI & KONTEKS
## 14. KONDISI EKSTERNAL
## 15. CLASH OF STYLES & VOLATILITY
---
## 🎯 RINGKASAN AKHIR
**Prediksi Utama:** | **Skor Alternatif:** | **Keyakinan:** | **Faktor Pembeda:** | **Catatan:**
"""

    # PERBAIKAN 5: Template fallback — tetap informatif tanpa AI
    def _template_report(self, home, away, sim, pred, ctx):
        h = home["team_name"]
        a = away["team_name"]
        top3 = sim["most_likely_scores"][:3]
        fav  = h if pred["prediction"] == "home_win" else (a if pred["prediction"] == "away_win" else "—")

        form_h = home["form"]
        form_a = away["form"]

        # Analisis shooting skill
        sk_comment = ""
        if home["shoot_skill"] > 1.05:
            sk_comment += f"  - {h} overperform xG (shooting skill {home['shoot_skill']:.2f}) → gol aktual lebih banyak dari expected.\n"
        if away["gk_skill"] > 1.05:
            sk_comment += f"  - Kiper {a} di atas rata-rata (GK skill {away['gk_skill']:.2f}) → menekan peluang lawan.\n"

        # Fatigue comment
        fat_h  = home["fatigue_index"]
        fat_a  = away["fatigue_index"]
        fat_comment = f"  - {h} fatigue: {fat_h:.3f} | {a} fatigue: {fat_a:.3f}"
        if abs(fat_h - fat_a) > 0.15:
            tim_lelah = h if fat_h > fat_a else a
            fat_comment += f" → {tim_lelah} terlihat lebih lelah, waspada di segmen akhir."

        # Volatility
        vol_avg = (home["volatility"] + away["volatility"]) / 2
        vol_comment = ("Pertandingan ini tergolong TINGGI volatilitas — kejutan mungkin terjadi."
                       if vol_avg > 0.18 else "Pertandingan relatif bisa diprediksi secara statistik.")

        report = f"""
# 📊 LAPORAN ANALISIS: {h} vs {a}
**Liga:** {ctx.get("league","?")} | **Wasit:** {ctx.get("referee","TBD")} | **Cuaca:** {ctx.get("weather","Normal")}

---
## 7. STATISTIK PERBANDINGAN
| Statistik | {h} | {a} |
|---|---|---|
| Form (5 laga) | {form_h["form_string"]} | {form_a["form_string"]} |
| xG avg | {form_h["xg_avg"]} | {form_a["xg_avg"]} |
| xGA avg | {form_h["xga_avg"]} | {form_a["xga_avg"]} |
| Shooting Skill | {home["shoot_skill"]:.2f} | {away["shoot_skill"]:.2f} |
| GK Skill | {home["gk_skill"]:.2f} | {away["gk_skill"]:.2f} |
| Fatigue Index | {fat_h:.3f} | {fat_a:.3f} |
| PPDA | {home["ppda"]} | {away["ppda"]} |
| Luck Factor | {home["luck_factor"]:+.1f} | {away["luck_factor"]:+.1f} |

---
## 8. FAKTOR KUNCI
1. **Shooting Skill vs GK Overperformance**
{sk_comment if sk_comment else "  - Kedua tim mendekati rata-rata dalam efisiensi penyelesaian."}
2. **Kelelahan (Fatigue)**
{fat_comment}
3. **Game State & Time-Slice**
  - Simulasi membagi 90 menit menjadi 6 segmen. Tim unggul cenderung bertahan di segmen akhir.
4. **Volatility**
  - {vol_comment}
5. **Lucky vs Unlucky**
  - {h} luck factor: {home["luck_factor"]:+.1f} | {a}: {away["luck_factor"]:+.1f} (negatif = overperform luck)

---
## 10. PERFORMA TERKINI (Time-Decay)
- **{h}:** {form_h["form_string"]} | Weighted Form: {form_h["weighted_form"]} | Efisiensi gol: {form_h["efficiency_ratio"]:.2f}
- **{a}:** {form_a["form_string"]} | Weighted Form: {form_a["weighted_form"]} | Efisiensi gol: {form_a["efficiency_ratio"]:.2f}

*Weighted form menggunakan exponential decay — laga terbaru bobot 3x lebih besar dari laga awal window.*

---
## 11. PRODUKTIVITAS GOL
- Avg gol simulasi: **{sim["avg_home_goals"]}** ({h}) vs **{sim["avg_away_goals"]}** ({a})
- Standar deviasi: {sim["std_home_goals"]} vs {sim["std_away_goals"]}
- BTTS: **{sim["btts_pct"]}%** | Over 2.5: **{sim["over25_pct"]}%**

---
## 🎯 RINGKASAN AKHIR
**Prediksi Utama:** {fav if fav != "—" else "Seri"} ({pred["prediction"].replace("_"," ").title()})
**Probabilitas:** 🏠 {sim["home_win_pct"]}% | 🤝 {sim["draw_pct"]}% | ✈️ {sim["away_win_pct"]}%
**Top Skor:** {" | ".join([f"{s[0][0]}-{s[0][1]} ({s[1]/sim['n_simulations']*100:.1f}%)" for s in top3])}
**Keyakinan:** {pred["confidence_label"]} ({pred["confidence"]*100:.1f}%)
**Catatan:** {ctx.get("injuries","-")} | H2H: {ctx.get("h2h","-")}
"""
        print("✅ Laporan template selesai!")
        return report

print("✅ CELL 7 SELESAI!")


# ── Cell 8 ────────────────────────────────────────
# CELL 8 — Setup & Load Data
from IPython.display import display, Markdown
import math

fe          = FeatureEngineer(form_window=FORM_WINDOW)
sim_engine  = MonteCarloSimulator(n=N_SIMULATIONS)
predictor   = MatchPredictor()
reporter    = ReportGenerator()

collector = UnderstatCollector(league="EPL", season=CURRENT_SEASON)
DATA      = collector.get_data()
if DATA.empty or "team" not in DATA.columns:
    DATA = collector._demo_data()

print("✅ CELL 8 SELESAI!")
print(f"Tim tersedia: {list(DATA['team'].values)}")


# ── Cell 9 ────────────────────────────────────────
# CELL 9 — ⚽ JALANKAN ANALISIS
# Ubah HOME_TEAM dan AWAY_TEAM sesuai pertandingan
# ✏️ UBAH NAMA TIM DI SINI
HOME_TEAM = "Arsenal"
AWAY_TEAM = "Chelsea"

CTX = {
    "league"  : "Premier League",
    "referee" : "Michael Oliver",
    "weather" : "Berawan 12°C",
    "injuries": "Saka 75% fit, Havertz siap",
    "h2h"     : "Arsenal menang 3 dari 5 H2H terakhir",
}

# days_rest = hari recovery sejak laga terakhir
# ppda      = semakin kecil = pressing lebih agresif
# setpiece  = rating bahaya bola mati 1-10
home = fe.build_profile(HOME_TEAM, DATA, days_rest=6, is_home=True,  ppda=9.2,  setpiece=6.5)
away = fe.build_profile(AWAY_TEAM, DATA, days_rest=3, is_home=False, ppda=11.8, setpiece=5.8)

sim    = sim_engine.run(home, away)
pred   = predictor.predict(home, away, sim)
report = reporter.generate(home, away, sim, pred, CTX)

display(Markdown(report))


# ── Cell 10 ────────────────────────────────────────
# CELL 10 — Visualisasi Lengkap (6 Panel)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

fig = plt.figure(figsize=(18, 10))
fig.suptitle(f"⚽ {sim["home_team"]} vs {sim["away_team"]}", fontsize=16, fontweight="bold", y=0.98)
gs  = gridspec.GridSpec(2, 3, figure=fig, hspace=0.4, wspace=0.35)

# --- 1. Win Probability ---
ax1 = fig.add_subplot(gs[0, 0])
vals = [sim["home_win_pct"], sim["draw_pct"], sim["away_win_pct"]]
lbls = [f"{sim["home_team"]}\nMenang", "Seri", f"{sim["away_team"]}\nMenang"]
colors = ["#e74c3c", "#95a5a6", "#3498db"]
bars = ax1.bar(lbls, vals, color=colors, edgecolor="white", linewidth=1.2)
for b, v in zip(bars, vals):
    ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.8, f"{v}%",
             ha="center", fontweight="bold", fontsize=11)
ax1.set_title("Probabilitas Hasil", fontweight="bold")
ax1.set_ylim(0, max(vals) + 15)
ax1.grid(axis="y", alpha=0.3)

# --- 2. Top Skor ---
ax2 = fig.add_subplot(gs[0, 1])
scores = sim["most_likely_scores"][:6]
ax2.bar([f"{s[0][0]}-{s[0][1]}" for s in scores],
        [s[1] / N_SIMULATIONS * 100 for s in scores],
        color="#1abc9c", edgecolor="white")
ax2.set_title("Top 6 Skor (%)", fontweight="bold")
ax2.tick_params(axis="x", rotation=30)
ax2.grid(axis="y", alpha=0.3)

# --- 3. Radar / Perbandingan Tim ---
ax3 = fig.add_subplot(gs[0, 2])
cats = ["xG Atk", "Defense", "Form", "Stamina", "Finishing", "GK"]
hv = [
    min(home["form"]["xg_avg"] / 2.5, 1.0),
    max(0, 1 - home["form"]["xga_avg"] / 2.5),
    home["form"]["weighted_form"] / 3,
    1 - home["fatigue_index"],
    min(home["shoot_skill"] - 0.8, 1.0),
    min(home["gk_skill"] - 0.8, 1.0),
]
av = [
    min(away["form"]["xg_avg"] / 2.5, 1.0),
    max(0, 1 - away["form"]["xga_avg"] / 2.5),
    away["form"]["weighted_form"] / 3,
    1 - away["fatigue_index"],
    min(away["shoot_skill"] - 0.8, 1.0),
    min(away["gk_skill"] - 0.8, 1.0),
]
x = np.arange(len(cats))
ax3.bar(x - 0.2, hv, 0.35, label=sim["home_team"], color="#e74c3c", alpha=0.85)
ax3.bar(x + 0.2, av, 0.35, label=sim["away_team"], color="#3498db", alpha=0.85)
ax3.set_xticks(x); ax3.set_xticklabels(cats, fontsize=8, rotation=15)
ax3.set_title("Perbandingan 6 Dimensi", fontweight="bold")
ax3.legend(fontsize=8); ax3.grid(axis="y", alpha=0.3)
ax3.set_ylim(0, 1.2)

# --- 4. Distribusi Total Gol ---
ax4 = fig.add_subplot(gs[1, 0])
pairs_sample = [sim_engine._simulate_match(home, away) for _ in range(2000)]
total_goals  = [h + a for h, a in pairs_sample]
bins = range(0, max(total_goals) + 2)
ax4.hist(total_goals, bins=bins, color="#9b59b6", edgecolor="white", align="left")
ax4.axvline(2.5, color="red", linestyle="--", label="Over/Under 2.5")
ax4.set_title("Distribusi Total Gol", fontweight="bold")
ax4.set_xlabel("Total Gol"); ax4.legend(fontsize=8); ax4.grid(alpha=0.3)

# --- 5. Volatility & Standar Deviasi ---
ax5 = fig.add_subplot(gs[1, 1])
metrics  = ["BTTS %", "Over 2.5 %", "SD Gol (Home)", "SD Gol (Away)"]
vals5    = [sim["btts_pct"], sim["over25_pct"],
            sim["std_home_goals"] * 10, sim["std_away_goals"] * 10]
ax5.barh(metrics, vals5, color=["#f39c12","#e67e22","#e74c3c","#3498db"])
ax5.set_title("Market & Volatility", fontweight="bold")
ax5.grid(axis="x", alpha=0.3)
for i, v in enumerate(vals5):
    ax5.text(v + 0.5, i, f"{v:.1f}", va="center", fontsize=9)

# --- 6. Shooting Skill vs GK ---
ax6 = fig.add_subplot(gs[1, 2])
teams   = [sim["home_team"], sim["away_team"]]
sk_vals = [home["shoot_skill"], away["shoot_skill"]]
gk_vals = [home["gk_skill"],   away["gk_skill"]]
x6 = np.arange(len(teams))
ax6.bar(x6 - 0.2, sk_vals, 0.35, label="Shooting Skill", color="#e67e22")
ax6.bar(x6 + 0.2, gk_vals, 0.35, label="GK Skill",       color="#27ae60")
ax6.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, label="Baseline")
ax6.set_xticks(x6); ax6.set_xticklabels(teams)
ax6.set_title("Shooting vs GK Skill", fontweight="bold")
ax6.legend(fontsize=8); ax6.grid(axis="y", alpha=0.3)
ax6.set_ylim(0.85, 1.20)

plt.savefig("match_analysis.png", dpi=150, bbox_inches="tight")
plt.show()

print(f"""
{"="*50}
🏠 {sim["home_team"]} Menang : {sim["home_win_pct"]}%
🤝 Seri                    : {sim["draw_pct"]}%
✈️  {sim["away_team"]} Menang : {sim["away_win_pct"]}%
⚽ BTTS: {sim["btts_pct"]}% | Over 2.5: {sim["over25_pct"]}%
🎯 {pred["prediction"].upper()} | {pred["confidence_label"]}
🏆 Top Skor: {sim["most_likely_scores"][0][0][0]}-{sim["most_likely_scores"][0][0][1]}
{"="*50}
""")


