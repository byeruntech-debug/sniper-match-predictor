#!/usr/bin/env python3
"""
Football Match Predictor — La Liga
Versi : 9.0 — THE ELITE SNIPER
Upgrade dari V8: Away Filter + Trap Stadium + Squad Depth Check

CARA PAKAI:
  Tempel setelah semua cell V6/V7/V8 berjalan.
  Jalankan Cell 20 saja.
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 20 — V9 Sniper Refinement
# ══════════════════════════════════════════════════════════════════════
# CELL 20 — V9 Elite Sniper

import pandas as pd
import numpy as np
from collections import defaultdict

print("🔧 Building V9 Elite Sniper...\n")

# ── 1. Hitung "Home Dog Power" dari data historis ─────────────────────
# Tim underdog yang sering clean sheet di kandang vs tim besar
def build_home_dog_power(df_matches):
    """
    Hitung: seberapa sering tim underdog (lower Elo) menahan
    tim besar di kandang sendiri.
    Score: clean sheet kandang vs top-5 tim per season.
    """
    # Top 5 tim berdasarkan total win
    top_teams = (df_matches[df_matches["FTR"]=="H"]["HomeTeam"]
                 .value_counts().head(5).index.tolist() +
                 df_matches[df_matches["FTR"]=="A"]["AwayTeam"]
                 .value_counts().head(5).index.tolist())
    top_teams = list(set(top_teams))

    home_dog_cs = defaultdict(int)   # clean sheet vs top tim
    home_dog_games = defaultdict(int)

    for _, row in df_matches.iterrows():
        if row["AwayTeam"] in top_teams:
            home_dog_games[row["HomeTeam"]] += 1
            # Clean sheet = tim tamu tidak cetak gol
            if int(row["FTAG"]) == 0:
                home_dog_cs[row["HomeTeam"]] += 1

    # Rate clean sheet vs big teams
    power = {}
    for team in home_dog_games:
        games = home_dog_games[team]
        cs    = home_dog_cs[team]
        power[team] = round(cs / games, 3) if games >= 2 else 0.0

    return power, top_teams

home_dog_power, top_5_teams = build_home_dog_power(df_train)

# Tampilkan "trap stadiums"
trap_sorted = sorted(home_dog_power.items(), key=lambda x: -x[1])
print("🏟️  Trap Stadiums (clean sheet rate vs big teams):")
for team, rate in trap_sorted[:8]:
    flag = " ⚠️  TRAP" if rate >= 0.35 else ""
    print(f"   {team:25s}: {rate:.3f}{flag}")

TRAP_THRESHOLD = 0.35   # CS rate >= 35% vs big teams = trap stadium


# ── 2. Hitung recent clean sheet kandang ─────────────────────────────
def get_home_clean_sheets(team, df_matches, n=3):
    """Berapa clean sheet dalam n laga kandang terakhir."""
    home_m = df_matches[df_matches["HomeTeam"] == team].tail(n)
    if home_m.empty:
        return 0
    return int((home_m["FTAG"] == 0).sum())


# ── 3. Sniper Filter V9 ───────────────────────────────────────────────
def sniper_filter_v9(result, home_team, away_team,
                     df_matches, elo_system,
                     home_dog_power, top_5_teams,
                     verbose=False):
    """
    Filter 4 lapis V9.

    Layer 1 — Konsensus 3/3 (wajib)
    Layer 2 — Draw danger: prob_draw > 0.28 → HOLD
    Layer 3 — Prediksi-specific:
        HOME: Elo_diff>80, home_wr>0.35, away_unbeaten<3
        AWAY: Elo_diff<-60 (away lebih kuat),
              + Trap Stadium check (jika home CS rate vs big >=0.35 → HOLD)
              + Recent home CS check (2+ CS terakhir → HOLD)
              + Elo diff harus > 100 untuk lolos trap
        DRAW: draw_prob > 0.35 dan dominan
    Layer 4 — Away Extra: away_prob harus > home_prob + 0.10
    """
    pred      = result["prediction"]
    consensus = result["model_consensus"]
    h_prob    = result["home_win_prob"]
    d_prob    = result["draw_prob"]
    a_prob    = result["away_win_prob"]

    reasons = []

    # Layer 1
    if consensus < 3:
        return False, f"Konsensus {consensus}/3", "SKIP"

    # Layer 2: Draw danger
    if d_prob > 0.28:
        return False, f"Draw danger (d={d_prob:.2f})", "HOLD"

    elo_home = elo_system.ratings.get(home_team, 1500)
    elo_away = elo_system.ratings.get(away_team, 1500)
    elo_diff = elo_home - elo_away   # positif = home lebih kuat

    if pred == "home_win":
        # Sama seperti V8 — sudah 90%, jangan diubah
        home_form = get_home_form(home_team, df_matches, n=5)
        away_form = get_away_form(away_team, df_matches, n=3)

        if elo_diff < 80:
            return False, f"Elo gap kecil ({elo_diff:.0f}<80)", "HOLD"
        if home_form["win_rate"] < 0.35:
            return False, f"Home WR rendah ({home_form['win_rate']:.2f})", "HOLD"
        if away_form["unbeaten"] >= 3:
            return False, f"Away unbeaten {away_form['unbeaten']}", "HOLD"

        return True, (f"✅ HOME Sniper | Elo+{elo_diff:.0f} "
                      f"HWR={home_form['win_rate']:.2f}"), "SNIPER"

    elif pred == "away_win":
        # Away harus lebih kuat dari home
        if elo_diff > -60:
            return False, f"Away tidak cukup kuat (Elo_diff={elo_diff:.0f})", "HOLD"

        # Layer 4: Away prob harus jauh lebih tinggi dari home
        if a_prob < h_prob + 0.10:
            return False, (f"Away margin kecil "
                           f"(a={a_prob:.2f} vs h={h_prob:.2f})"), "HOLD"

        # Trap Stadium check
        h_dog_rate = home_dog_power.get(home_team, 0.0)
        if h_dog_rate >= TRAP_THRESHOLD:
            # Tim tamu harus punya Elo dominan untuk tembus trap
            if abs(elo_diff) < 150:
                return False, (f"Trap stadium ({home_team} CS={h_dog_rate:.2f}) "
                               f"+ Elo gap tidak cukup besar ({abs(elo_diff):.0f}<150)"), "HOLD"
            else:
                reasons.append(f"Trap override (Elo gap={abs(elo_diff):.0f}≥150)")

        # Recent home clean sheet check
        home_cs_recent = get_home_clean_sheets(home_team, df_matches, n=3)
        if home_cs_recent >= 2:
            return False, (f"Home {home_cs_recent} CS dalam 3 laga kandang terakhir "
                           f"— defensive fortress"), "HOLD"

        reason_str = (f"✅ AWAY Sniper | Elo{elo_diff:.0f} "
                      f"a={a_prob:.2f} h={h_prob:.2f}")
        if reasons:
            reason_str += f" [{', '.join(reasons)}]"
        return True, reason_str, "SNIPER"

    elif pred == "draw":
        if d_prob > 0.35 and d_prob > h_prob and d_prob > a_prob:
            return True, f"✅ DRAW Sniper (d={d_prob:.2f})", "SNIPER"
        return False, f"Draw tidak dominan (d={d_prob:.2f})", "HOLD"

    return False, "Unknown", "SKIP"


# ── 4. Validasi V9 ────────────────────────────────────────────────────
print("\n🚀 Validasi V9 Sniper...\n")

all_v9     = []
sniper_v9  = []
hold_v9    = []

for _, row in df_val.head(50).iterrows():
    h_team = row["HomeTeam"]
    a_team = row["AwayTeam"]
    actual = {"H":"home_win","D":"draw","A":"away_win"}[row["FTR"]]

    try:
        res = ensemble.predict(h_team, a_team,
                               home_data=DATA, away_data=DATA,
                               ctx={"verbose": False})

        is_sniper, reason, tier = sniper_filter_v9(
            res, h_team, a_team, df_train,
            elo_system, home_dog_power, top_5_teams
        )

        correct = res["prediction"] == actual
        entry = {
            "match"    : f"{h_team} vs {a_team}",
            "actual"   : actual,
            "predicted": res["prediction"],
            "correct"  : correct,
            "tier"     : tier,
            "reason"   : reason,
            "d_prob"   : res["draw_prob"],
            "h_prob"   : res["home_win_prob"],
            "a_prob"   : res["away_win_prob"],
            "consensus": res["model_consensus"],
        }
        all_v9.append(entry)
        if tier == "SNIPER": sniper_v9.append(entry)
        elif tier == "HOLD": hold_v9.append(entry)
    except Exception as e:
        continue

df_all9    = pd.DataFrame(all_v9)
df_sniper9 = pd.DataFrame(sniper_v9) if sniper_v9 else pd.DataFrame()
df_hold9   = pd.DataFrame(hold_v9)   if hold_v9   else pd.DataFrame()

# ── Laporan ───────────────────────────────────────────────────────────
acc_all9 = df_all9["correct"].mean()*100

print(f"{'='*60}")
print(f"  📊 SEMUA LAGA: {acc_all9:.1f}% "
      f"({df_all9['correct'].sum()}/{len(df_all9)})")
print(f"{'='*60}")
for label in ["home_win","draw","away_win"]:
    p = (df_all9["predicted"]==label).sum()
    a = (df_all9["actual"]==label).sum()
    arrow = "🔼" if p-a>2 else ("🔽" if a-p>2 else "✅")
    print(f"  {label:12s}: Pred={p:3d} | Aktual={a:3d} {arrow}")

if not df_sniper9.empty:
    acc_s9 = df_sniper9["correct"].mean()*100
    print(f"\n{'='*60}")
    print(f"  🎯 SNIPER V9: {acc_s9:.1f}% "
          f"({df_sniper9['correct'].sum()}/{len(df_sniper9)} laga)")
    print(f"  Coverage: {len(df_sniper9)}/50 = {len(df_sniper9)/50*100:.0f}%")
    print(f"{'='*60}")

    print(f"\n  Breakdown per prediksi:")
    for label in ["home_win","draw","away_win"]:
        sub = df_sniper9[df_sniper9["predicted"]==label]
        if len(sub):
            acc = sub["correct"].mean()*100
            print(f"    {label:12s}: {acc:.1f}% "
                  f"({sub['correct'].sum()}/{len(sub)})")

    print(f"\n  Detail laga Sniper:")
    for _, r in df_sniper9.iterrows():
        s = "✅" if r["correct"] else "❌"
        print(f"    {s} {r['match']:38s} "
              f"Pred:{r['predicted']:10s} Aktual:{r['actual']}")

if not df_hold9.empty:
    acc_h9 = df_hold9["correct"].mean()*100
    print(f"\n  ⏸️  HOLD ({len(df_hold9)} laga): {acc_h9:.1f}%")
    # Tunjukkan alasan HOLD terbanyak
    from collections import Counter
    hold_reasons = [r["reason"].split("(")[0].strip()
                    for r in hold_v9]
    print("  Top alasan HOLD:")
    for reason, count in Counter(hold_reasons).most_common(5):
        print(f"    {count}x {reason}")

acc_s9 = df_sniper9["correct"].mean()*100 if not df_sniper9.empty else 0
delta_v8 = acc_s9 - 76.5
delta_v6 = acc_s9 - 56.0

print(f"\n{'='*60}")
print(f"  🏆 RINGKASAN PERJALANAN")
print(f"{'='*60}")
print(f"  V5 (baseline)  : 44.0%")
print(f"  V6             : 56.0%  (+12.0%)")
print(f"  V7.1 Sniper    : 64.3%  (+8.3%)")
print(f"  V8 Sniper      : 76.5%  (+12.2%)")
print(f"  V9 Sniper      : {acc_s9:.1f}%  ({delta_v8:+.1f}% vs V8)")
print(f"  Coverage       : {len(df_sniper9)}/50 laga "
      f"({len(df_sniper9)/50*100:.0f}%)")
status = ("🏆 ELITE 80%!" if acc_s9 >= 80 else
          ("✅ TARGET MET" if acc_s9 >= 75 else "📈 PROGRESS"))
print(f"  Status         : {status}")
print(f"{'='*60}")

# ── 5. Fungsi predict_match() final dengan Sniper tier ───────────────
def predict_match_v9(home_team, away_team,
                     days_rest_h=5, days_rest_a=5,
                     ppda_h=None, ppda_a=None,
                     absent_h=None, absent_a=None,
                     league="La Liga"):
    """
    One-liner prediksi dengan Sniper tier otomatis.

    Output:
      SNIPER  → prediksi sangat reliable (target 80%+)
      HOLD    → prediksi ada tapi perlu perhatian ekstra
      SKIP    → lewati, tidak cukup data/konsensus
    """
    if ppda_h is None:
        key_h  = next((k for k, v in STANDINGS_TO_AF.items()
                       if v.lower() == home_team.lower()), None)
        ppda_h = PPDA_ESTIMATED.get(key_h, 11.0)
    if ppda_a is None:
        key_a  = next((k for k, v in STANDINGS_TO_AF.items()
                       if v.lower() == away_team.lower()), None)
        ppda_a = PPDA_ESTIMATED.get(key_a, 11.0)

    res = ensemble.predict(
        home_team, away_team,
        home_data=DATA, away_data=DATA,
        days_rest_h=days_rest_h, days_rest_a=days_rest_a,
        ppda_h=ppda_h, ppda_a=ppda_a,
        absent_h=absent_h, absent_a=absent_a,
        ctx={"verbose": False},
    )

    is_sniper, reason, tier = sniper_filter_v9(
        res, home_team, away_team, df_train,
        elo_system, home_dog_power, top_5_teams
    )

    tier_emoji = {"SNIPER":"🎯","HOLD":"⏸️ ","SKIP":"⛔"}[tier]

    print(f"\n{'='*55}")
    print(f"  ⚽ {home_team} vs {away_team}")
    print(f"  🏠 {res['home_win_prob']*100:.1f}% | "
          f"🤝 {res['draw_prob']*100:.1f}% | "
          f"✈️  {res['away_win_prob']*100:.1f}%")
    print(f"  🎯 {res['prediction'].upper()} | "
          f"Konsensus {res['model_consensus']}/3")
    print(f"  🏆 Top Skor: "
          f"{res['top_scores'][0][0][0]}-{res['top_scores'][0][0][1]} "
          f"({res['top_scores'][0][1]*100:.1f}%)")
    print(f"  {tier_emoji} Tier: {tier}")
    print(f"  📋 {reason}")
    print(f"{'='*55}")

    return {**res, "tier": tier, "sniper_reason": reason}


print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CARA PAKAI V9:

  predict_match_v9("Barcelona", "Real Madrid")
  predict_match_v9("Real Madrid", "Getafe")
  predict_match_v9("Atletico Madrid", "Sevilla",
                    absent_h=["Griezmann"])

  Tier output:
    🎯 SNIPER → percaya penuh (target 80%+)
    ⏸️  HOLD   → ada sinyal tapi tidak pasti
    ⛔ SKIP   → lewati
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
