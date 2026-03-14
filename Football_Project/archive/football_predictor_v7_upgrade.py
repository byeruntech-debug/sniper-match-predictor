#!/usr/bin/env python3
"""
Football Match Predictor — La Liga
Versi : 7.0 — Meta-Model Stacking + Dynamic Draw Margin + Team-Specific HFA

CARA PAKAI:
  Tempel SETELAH semua cell V6 sudah berjalan (Cell 11-16 + patch).
  Jalankan Cell 17, 18, 19 secara berurutan.
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 17 — Feature Extraction untuk Meta-Model
# ══════════════════════════════════════════════════════════════════════
# CELL 17 — Bangun Dataset untuk Meta-Model Stacking
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")

print("🔧 Membangun dataset meta-model dari training data...\n")

def extract_features(home_team, away_team, dc_model, elo_system,
                     df_matches=None, team_hfa=None):
    """
    Ekstrak 15 fitur dari output 3 model untuk Meta-Model.

    Fitur:
      [0-2]  DC probabilities (home_win, draw, away_win)
      [3-5]  Elo probabilities
      [6]    Elo rating difference (home - away)
      [7]    DC lambda_home
      [8]    DC lambda_away
      [9]    DC total expected goals (lambda_h + lambda_a)
      [10]   DC lambda difference (home - away)
      [11]   |home_win - away_win| dari DC (confidence gap)
      [12]   |home_win - away_win| dari Elo
      [13]   Team-specific HFA home team (0.0-1.0)
      [14]   Team-specific HFA away team (0.0-1.0)
    """
    # DC
    dc_full = dc_model.predict_proba(home_team, away_team)
    dc_p    = {k: dc_full[k] for k in ["home_win","draw","away_win"]}

    # Home bias correction (sama seperti V6)
    dc_p["home_win"] *= 0.88
    dc_p["away_win"] *= 1.08
    dc_p["draw"]     *= 1.05
    t = sum(dc_p.values())
    dc_p = {k: v/t for k, v in dc_p.items()}

    lam_h = dc_full["lambda_home"]
    lam_a = dc_full["lambda_away"]

    # Elo
    elo_p    = elo_system.get_win_probs(home_team, away_team)
    elo_home = elo_system.ratings.get(home_team, 1500)
    elo_away = elo_system.ratings.get(away_team, 1500)

    # Team-specific HFA
    hfa_h = team_hfa.get(home_team, 0.45) if team_hfa else 0.45
    hfa_a = team_hfa.get(away_team, 0.45) if team_hfa else 0.45

    features = [
        dc_p["home_win"],                          # 0
        dc_p["draw"],                              # 1
        dc_p["away_win"],                          # 2
        elo_p["home_win"],                         # 3
        elo_p["draw"],                             # 4
        elo_p["away_win"],                         # 5
        (elo_home - elo_away) / 400,               # 6 — normalized elo diff
        lam_h,                                     # 7
        lam_a,                                     # 8
        lam_h + lam_a,                             # 9 — total xG
        lam_h - lam_a,                             # 10 — xG difference
        abs(dc_p["home_win"] - dc_p["away_win"]),  # 11 — DC confidence gap
        abs(elo_p["home_win"] - elo_p["away_win"]),# 12 — Elo confidence gap
        hfa_h,                                     # 13
        hfa_a,                                     # 14
    ]
    return features


# ── Hitung Team-Specific HFA dari data historis ───────────────────────
print("📊 Menghitung Team-Specific HFA dari data historis...")

team_home_pts  = {}
team_total_pts = {}

for _, row in df_train.iterrows():
    h, a = row["HomeTeam"], row["AwayTeam"]
    ftr  = row["FTR"]

    # Home points
    if ftr == "H":
        h_pts, a_pts = 3, 0
    elif ftr == "D":
        h_pts, a_pts = 1, 1
    else:
        h_pts, a_pts = 0, 3

    team_home_pts[h]   = team_home_pts.get(h, 0)   + h_pts
    team_total_pts[h]  = team_total_pts.get(h, 0)  + h_pts
    team_total_pts[a]  = team_total_pts.get(a, 0)  + a_pts

# HFA = proporsi poin yang diraih di kandang
team_hfa = {}
for team in team_total_pts:
    total = team_total_pts[team]
    home  = team_home_pts.get(team, 0)
    team_hfa[team] = round(home / total, 3) if total > 0 else 0.45

# Tampilkan top/bottom HFA
hfa_sorted = sorted(team_hfa.items(), key=lambda x: -x[1])
print("\n  Top 5 HFA (paling kuat di kandang):")
for t, v in hfa_sorted[:5]:
    print(f"    {t:25s}: {v:.3f}")
print("\n  Bottom 5 HFA (paling lemah di kandang):")
for t, v in hfa_sorted[-5:]:
    print(f"    {t:25s}: {v:.3f}")


# ── Build training dataset untuk meta-model ───────────────────────────
print("\n📦 Membangun training dataset...")

X_train_meta = []
y_train_meta = []
label_map    = {"H": 0, "D": 1, "A": 2}  # 0=home_win, 1=draw, 2=away_win

for _, row in df_train.iterrows():
    try:
        feats = extract_features(
            row["HomeTeam"], row["AwayTeam"],
            dc_model, elo_system, df_train, team_hfa
        )
        X_train_meta.append(feats)
        y_train_meta.append(label_map[row["FTR"]])
    except Exception as e:
        continue

X_train_meta = np.array(X_train_meta)
y_train_meta = np.array(y_train_meta)

print(f"✅ Training dataset: {len(X_train_meta)} sampel x {X_train_meta.shape[1]} fitur")
print(f"   Distribusi: Home={sum(y_train_meta==0)} | "
      f"Draw={sum(y_train_meta==1)} | Away={sum(y_train_meta==2)}")

print("\n✅ CELL 17 SELESAI!")


# ══════════════════════════════════════════════════════════════════════
# CELL 18 — Train Meta-Model + Dynamic Draw Margin
# ══════════════════════════════════════════════════════════════════════
# CELL 18 — Train Meta-Model (Stacking) + Dynamic Draw Margin
print("🤖 Training Meta-Model (Logistic Regression + Platt Scaling)...\n")

# ── Scaler ────────────────────────────────────────────────────────────
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_train_meta)

# ── Base model: Logistic Regression ───────────────────────────────────
# class_weight='balanced' membantu Draw yang under-represented
base_lr = LogisticRegression(
    C=0.5,
    max_iter=1000,
    class_weight="balanced",   # kunci untuk draw
    multi_class="multinomial",
    solver="lbfgs",
    random_state=42,
)

# ── Platt Scaling via CalibratedClassifierCV ──────────────────────────
# cv=5 fold cross-validation untuk kalibrasi probabilitas
meta_model = CalibratedClassifierCV(base_lr, cv=5, method="sigmoid")
meta_model.fit(X_scaled, y_train_meta)

# In-sample accuracy
y_pred_train = meta_model.predict(X_scaled)
train_acc    = accuracy_score(y_train_meta, y_pred_train) * 100
print(f"✅ Meta-model trained! In-sample accuracy: {train_acc:.1f}%")

# Cek feature importance (dari base estimator rata-rata)
try:
    coef_avg = np.mean([est.base_estimator.coef_
                        for est in meta_model.calibrated_classifiers_], axis=0)
    feat_names = [
        "DC_home","DC_draw","DC_away",
        "Elo_home","Elo_draw","Elo_away",
        "Elo_diff","lam_home","lam_away","lam_total",
        "lam_diff","DC_gap","Elo_gap","HFA_home","HFA_away"
    ]
    print("\n  Top fitur per kelas (abs coef):")
    for cls_idx, cls_name in enumerate(["HOME_WIN","DRAW","AWAY_WIN"]):
        top_idx = np.argsort(np.abs(coef_avg[cls_idx]))[::-1][:4]
        top_f   = [(feat_names[i], round(coef_avg[cls_idx][i],3))
                   for i in top_idx]
        print(f"    {cls_name}: {top_f}")
except Exception:
    print("  (Feature importance tidak tersedia untuk model terkalibrasi)")


# ── Dynamic Draw Margin ───────────────────────────────────────────────
def get_dynamic_draw_margin(lam_h, lam_a):
    """
    Margin seri dinamis berdasarkan total expected goals.
    Laga defensif (xG total rendah) → margin lebih lebar → lebih sering seri.
    Laga ofensif (xG total tinggi) → margin lebih sempit → cari pemenang.

    Threshold:
      xG_total < 2.0  → margin 0.16 (laga sangat defensif)
      xG_total < 2.5  → margin 0.12 (laga seimbang)
      xG_total < 3.2  → margin 0.10 (laga normal)
      xG_total >= 3.2 → margin 0.07 (laga ofensif)
    """
    total_xg = lam_h + lam_a
    if total_xg < 2.0:
        return 0.16
    elif total_xg < 2.5:
        return 0.12
    elif total_xg < 3.2:
        return 0.10
    else:
        return 0.07


print(f"\n📐 Dynamic Draw Margin examples:")
for xg in [1.6, 2.1, 2.8, 3.5]:
    m = get_dynamic_draw_margin(xg/2, xg/2)
    print(f"   xG_total={xg:.1f} → margin={m}")

print("\n✅ CELL 18 SELESAI!")


# ══════════════════════════════════════════════════════════════════════
# CELL 19 — EnsemblePredictor V7 + Validasi Final
# ══════════════════════════════════════════════════════════════════════
# CELL 19 — Predictor V7 + Validasi
import types

def predict_v7(self, home_team, away_team,
               home_data=None, away_data=None,
               days_rest_h=5, days_rest_a=5,
               ppda_h=10.5, ppda_a=11.0,
               setpiece_h=5.0, setpiece_a=5.0,
               absent_h=None, absent_a=None,
               ctx=None):
    """
    V7: Meta-Model Stacking + Dynamic Draw Margin + Team-Specific HFA.

    Pipeline:
      1. Ekstrak 15 fitur dari DC + Elo
      2. Meta-model prediksi probabilitas (Logistic Regression terkalibrasi)
      3. Monte Carlo untuk konteks fatigue/PPDA/injury
      4. Ensemble final: meta 55% + MC 25% + fallback Elo 20%
      5. Dynamic draw margin berdasarkan total xG
    """

    def best_outcome(p):
        return max(["home_win","draw","away_win"], key=lambda k: p[k])

    ctx     = ctx or {}
    verbose = ctx.get("verbose", False)

    if verbose:
        print(f"\n{'='*60}")
        print(f"  ⚽ ENSEMBLE V7: {home_team} vs {away_team}")
        print(f"{'='*60}")

    # ── Step 1: Ekstrak fitur ──────────────────────────────────────
    try:
        feats   = extract_features(home_team, away_team,
                                   self.dc, self.elo, None, team_hfa)
        dc_full = self.dc.predict_proba(home_team, away_team)
        lam_h   = dc_full["lambda_home"]
        lam_a   = dc_full["lambda_away"]
    except Exception as e:
        if verbose: print(f"  ⚠️  Feature extraction error: {e}")
        # Fallback ke V6
        dc_full = self.dc.predict_proba(home_team, away_team)
        feats   = None
        lam_h   = dc_full["lambda_home"]
        lam_a   = dc_full["lambda_away"]

    # ── Step 2: Meta-model probabilities ──────────────────────────
    if feats is not None:
        try:
            X_test   = scaler.transform([feats])
            meta_proba = meta_model.predict_proba(X_test)[0]
            # meta_model kelas: 0=home_win, 1=draw, 2=away_win
            meta_p = {
                "home_win": float(meta_proba[0]),
                "draw"    : float(meta_proba[1]),
                "away_win": float(meta_proba[2]),
            }
        except Exception as e:
            if verbose: print(f"  ⚠️  Meta-model error: {e}")
            meta_p = None
    else:
        meta_p = None

    # Fallback jika meta-model gagal
    if meta_p is None:
        dc_p = {k: dc_full[k] for k in ["home_win","draw","away_win"]}
        dc_p["home_win"] *= 0.88; dc_p["away_win"] *= 1.08; dc_p["draw"] *= 1.05
        t = sum(dc_p.values()); dc_p = {k: v/t for k, v in dc_p.items()}
        meta_p = dc_p

    # Elo sebagai stabilizer
    elo_p    = self.elo.get_win_probs(home_team, away_team)
    elo_home = self.elo.ratings.get(home_team, 1500)
    elo_away = self.elo.ratings.get(away_team, 1500)

    if verbose:
        print(f"\n  Meta : H={meta_p['home_win']*100:.1f}% "
              f"D={meta_p['draw']*100:.1f}% A={meta_p['away_win']*100:.1f}%")
        print(f"  Elo  : H={elo_p['home_win']*100:.1f}% "
              f"D={elo_p['draw']*100:.1f}% A={elo_p['away_win']*100:.1f}%")
        print(f"  λ_home={lam_h:.3f} | λ_away={lam_a:.3f} | "
              f"xG_total={lam_h+lam_a:.3f}")

    # ── Step 3: Monte Carlo ────────────────────────────────────────
    sim_result = None
    home_p = away_p = None
    mc_p   = dict(meta_p)

    if home_data is not None and away_data is not None:
        try:
            inj_h = self.pse.get_injury_penalty(home_team, absent_h)
            inj_a = self.pse.get_injury_penalty(away_team, absent_a)

            home_p = self.fe.build_profile(
                home_team, home_data, days_rest=days_rest_h,
                is_home=True, ppda=ppda_h, setpiece=setpiece_h)
            away_p = self.fe.build_profile(
                away_team, away_data, days_rest=days_rest_a,
                is_home=False, ppda=ppda_a, setpiece=setpiece_a)

            if inj_h > 0: home_p["form"]["xg_avg"] *= (1 - inj_h)
            if inj_a > 0: away_p["form"]["xg_avg"] *= (1 - inj_a)

            mod_h, mod_a = self.pse.get_strength_modifier(home_team, away_team)
            home_p["form"]["xg_avg"] *= mod_h
            away_p["form"]["xg_avg"] *= mod_a

            sim_result = self.mc.run(home_p, away_p)
            mc_p = {
                "home_win": sim_result["home_win_pct"] / 100,
                "draw"    : sim_result["draw_pct"]     / 100,
                "away_win": sim_result["away_win_pct"] / 100,
            }
            if verbose:
                print(f"  MC   : H={mc_p['home_win']*100:.1f}% "
                      f"D={mc_p['draw']*100:.1f}% A={mc_p['away_win']*100:.1f}%")
        except Exception as e:
            if verbose: print(f"  ⚠️  MC error: {e}")

    # ── Step 4: Weighted Ensemble ──────────────────────────────────
    # Meta-model = 55% (paling dipercaya karena sudah terkalibrasi)
    # MC         = 25% (tambah konteks situasional)
    # Elo        = 20% (stabilizer global strength)
    h = 0.55*meta_p["home_win"] + 0.25*mc_p["home_win"] + 0.20*elo_p["home_win"]
    d = 0.55*meta_p["draw"]     + 0.25*mc_p["draw"]     + 0.20*elo_p["draw"]
    a = 0.55*meta_p["away_win"] + 0.25*mc_p["away_win"] + 0.20*elo_p["away_win"]
    t = h+d+a; h/=t; d/=t; a/=t

    # ── Step 5: Dynamic Draw Margin ───────────────────────────────
    draw_margin = get_dynamic_draw_margin(lam_h, lam_a)

    if abs(h - a) < draw_margin:
        boost = draw_margin * 0.65
        s     = h + a + 1e-9
        h    -= boost * (h / s)
        a    -= boost * (a / s)
        d    += boost
        t2    = h + d + a
        h /= t2; d /= t2; a /= t2

    # ── Final prediction ───────────────────────────────────────────
    final = {"home_win": h, "draw": d, "away_win": a}
    pred  = max(final, key=final.get)

    # Confidence dari konsensus 3 sumber
    votes     = [best_outcome(meta_p), best_outcome(elo_p), best_outcome(mc_p)]
    consensus = votes.count(pred) / 3
    conf      = round(max(h, d, a) * 0.6 + consensus * 0.4, 4)
    conf_l    = "Tinggi" if conf > 0.58 else ("Sedang" if conf > 0.46 else "Rendah")

    # Top scores dari DC
    top_scores = []
    for item in dc_full.get("top_scores", []):
        try:
            top_scores.append(((int(item[0][0]), int(item[0][1])), float(item[1])))
        except: continue
    if not top_scores:
        top_scores = [((1,1),0.10),((1,0),0.09),((2,0),0.08)]

    if verbose:
        print(f"\n  {'─'*50}")
        print(f"  Draw margin: {draw_margin} (xG_total={lam_h+lam_a:.2f})")
        print(f"  🏠 {h*100:.1f}% | 🤝 {d*100:.1f}% | ✈️  {a*100:.1f}%")
        print(f"  🎯 {pred.upper()} | {conf_l} ({conf*100:.1f}%) | "
              f"Konsensus {votes.count(pred)}/3")
        print(f"  🏆 Top Skor: {top_scores[0][0][0]}-{top_scores[0][0][1]} "
              f"({top_scores[0][1]*100:.1f}%)")
        print(f"  HFA: {home_team}={team_hfa.get(home_team,0.45):.3f} | "
              f"{away_team}={team_hfa.get(away_team,0.45):.3f}")
        print(f"{'='*60}")

    return {
        "prediction"      : pred,
        "home_win_prob"   : round(h, 4),
        "draw_prob"       : round(d, 4),
        "away_win_prob"   : round(a, 4),
        "confidence"      : conf,
        "confidence_label": conf_l,
        "model_consensus" : votes.count(pred),
        "model_votes"     : votes,
        "meta_probs"      : meta_p,
        "elo_probs"       : elo_p,
        "mc_probs"        : mc_p,
        "top_scores"      : top_scores,
        "sim_result"      : sim_result,
        "home_profile"    : home_p,
        "away_profile"    : away_p,
        "elo_home"        : elo_home,
        "elo_away"        : elo_away,
        "draw_margin_used": draw_margin,
        "lambda_home"     : lam_h,
        "lambda_away"     : lam_a,
    }


# Patch ke ensemble
ensemble.predict = types.MethodType(predict_v7, ensemble)
print("✅ Ensemble V7 siap!\n")


# ── Validasi V7 ───────────────────────────────────────────────────────
def validate_v7(df_val, max_laga=None, verbose=False):
    results = []
    n       = min(len(df_val), max_laga) if max_laga else len(df_val)

    print(f"\n{'='*57}")
    print(f"  🧪 VALIDASI {n} LAGA — ENSEMBLE V7")
    print(f"{'='*57}")

    draw_margins_used = []

    for _, row in df_val.head(n).iterrows():
        h_team = row["HomeTeam"]
        a_team = row["AwayTeam"]
        actual = {"H":"home_win","D":"draw","A":"away_win"}[row["FTR"]]

        try:
            res     = ensemble.predict(
                h_team, a_team,
                home_data = DATA,
                away_data = DATA,
                ctx       = {"verbose": False}
            )
            correct = res["prediction"] == actual
            draw_margins_used.append(res["draw_margin_used"])

            results.append({
                "match"       : f"{h_team} vs {a_team}",
                "actual"      : actual,
                "predicted"   : res["prediction"],
                "correct"     : correct,
                "confidence"  : res["confidence_label"],
                "consensus"   : res["model_consensus"],
                "draw_margin" : res["draw_margin_used"],
                "lam_total"   : round(res["lambda_home"] + res["lambda_away"], 3),
            })

            if verbose:
                s = "✅" if correct else "❌"
                print(f"  {s} {h_team:18s} vs {a_team:18s} | "
                      f"Pred:{res['prediction']:10s} Aktual:{actual:10s} | "
                      f"margin={res['draw_margin_used']}")
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

    print(f"\n  Akurasi per konsensus:")
    for vote in [3, 2, 1]:
        sub = df_res[df_res["consensus"] == vote]
        if len(sub):
            acc = sub["correct"].mean() * 100
            print(f"    {vote}/3 model: {acc:.1f}% ({sub['correct'].sum()}/{len(sub)})")

    print(f"\n  Distribusi Prediksi vs Aktual:")
    for label in ["home_win", "draw", "away_win"]:
        p = (df_res["predicted"] == label).sum()
        a_cnt = (df_res["actual"] == label).sum()
        diff  = p - a_cnt
        arrow = "🔼" if diff > 2 else ("🔽" if diff < -2 else "✅")
        print(f"    {label:12s}: Pred={p:3d} | Aktual={a_cnt:3d} {arrow}")

    # Draw margin breakdown
    print(f"\n  Draw margin yang dipakai:")
    for m in sorted(set(draw_margins_used)):
        sub = df_res[df_res["draw_margin"] == m]
        acc = sub["correct"].mean() * 100 if len(sub) else 0
        print(f"    margin={m}: {len(sub)} laga | akurasi={acc:.1f}%")

    print(f"\n  {'─'*45}")
    print(f"  Total: {len(df_res)} | Benar: {df_res['correct'].sum()} "
          f"| Akurasi: {accuracy:.1f}%")

    # Perbandingan dengan V6
    v6_acc = 56.0
    delta  = accuracy - v6_acc
    arrow  = "📈" if delta > 0 else "📉"
    print(f"  vs V6 (56.0%)  : {arrow} {delta:+.1f}%")
    print(f"  Benchmark      : 55-62%")
    status = "✅ DI ATAS" if accuracy >= 55 else "⚠️  DI BAWAH"
    print(f"  Status         : {status}")
    print(f"{'='*57}")
    return df_res


print("🚀 Validasi V7...\n")
df_v7 = validate_v7(df_val, max_laga=50, verbose=False)

print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CARA PAKAI V7:

  predict_match("Barcelona", "Real Madrid")

  # Dengan context verbose (tampilkan semua detail)
  result = ensemble.predict(
      "Atletico Madrid", "Real Madrid",
      home_data=DATA, away_data=DATA,
      ctx={"verbose": True}
  )

  # Gunakan hanya laga dengan konsensus 3/3
  # → akurasi tertinggi (target 65%+)
  if result["model_consensus"] == 3:
      print("HIGH CONFIDENCE:", result["prediction"])
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
