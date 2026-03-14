#!/usr/bin/env python3
"""
Football Prediction Log Tracker
Mencatat prediksi live, audit performa, equity curve

CARA PAKAI:
  1. Jalankan setelah semua cell V8/V11 berjalan
  2. Log prediksi sebelum pertandingan:
       log_prediction("Barcelona", "Real Madrid")
  3. Setelah pertandingan selesai, update hasil:
       update_result("Barcelona", "Real Madrid", "H")
  4. Lihat performa:
       show_dashboard()
"""

# ══════════════════════════════════════════════════════════════════════
# CELL 21 — Prediction Log Tracker
# ══════════════════════════════════════════════════════════════════════
# CELL 21 — Log Tracker

import pandas as pd
import numpy as np
import json, os
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.stats import binom

# ── Storage: Google Drive ─────────────────────────────────────────────
LOG_PATH = "/content/drive/MyDrive/football_predictions_log.json"

def _load_log():
    """Load log dari file JSON."""
    if os.path.exists(LOG_PATH):
        with open(LOG_PATH, "r") as f:
            return json.load(f)
    return {"predictions": [], "metadata": {"created": str(datetime.now()),
                                              "version": "V8-Hybrid"}}

def _save_log(data):
    """Simpan log ke file JSON."""
    # Buat folder jika belum ada
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    with open(LOG_PATH, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"✅ Log tersimpan: {LOG_PATH}")


def log_prediction(home_team, away_team,
                   match_date=None,
                   league="La Liga",
                   days_rest_h=5, days_rest_a=5,
                   absent_h=None, absent_a=None,
                   notes=""):
    """
    Log prediksi SEBELUM pertandingan dimainkan.

    Args:
        home_team   : nama tim kandang
        away_team   : nama tim tamu
        match_date  : tanggal pertandingan (str "YYYY-MM-DD"), default hari ini
        league      : nama liga
        notes       : catatan konteks (cuaca, motivasi, dll)

    Returns:
        dict prediksi + entry_id untuk update_result nanti
    """
    if match_date is None:
        match_date = datetime.now().strftime("%Y-%m-%d")

    # Jalankan prediksi
    res = predict_clean(home_team, away_team, df_true_train)
    is_sniper, reason, tier = sniper_v8_clean(
        res, home_team, away_team, df_true_train, elo_clean
    )

    tier_emoji = {"SNIPER": "🎯", "HOLD": "⏸️ ", "SKIP": "⛔"}[tier]

    entry = {
        "id"          : f"{match_date}_{home_team}_{away_team}".replace(" ","_"),
        "date"        : match_date,
        "league"      : league,
        "home_team"   : home_team,
        "away_team"   : away_team,
        "prediction"  : res["prediction"],
        "home_prob"   : res["home_win_prob"],
        "draw_prob"   : res["draw_prob"],
        "away_prob"   : res["away_win_prob"],
        "confidence"  : res["confidence"],
        "consensus"   : res["model_consensus"],
        "tier"        : tier,
        "tier_reason" : reason,
        "top_score"   : (f"{res['top_scores'][0][0][0]}-"
                         f"{res['top_scores'][0][0][1]}"
                         if res.get("top_scores") else "?"),
        "top_score_pct": round(res["top_scores"][0][1]*100, 1)
                         if res.get("top_scores") else 0,
        "actual_result": None,   # diisi setelah pertandingan
        "correct"     : None,
        "notes"       : notes,
        "logged_at"   : str(datetime.now()),
        "updated_at"  : None,
    }

    # Simpan ke log
    data = _load_log()
    # Cek duplikat
    existing_ids = [p["id"] for p in data["predictions"]]
    if entry["id"] in existing_ids:
        print(f"⚠️  Prediksi {entry['id']} sudah ada di log!")
        return entry

    data["predictions"].append(entry)
    _save_log(data)

    # Tampilkan summary
    print(f"\n{'='*55}")
    print(f"  📝 PREDIKSI DICATAT")
    print(f"  ⚽ {home_team} vs {away_team} | {match_date}")
    print(f"  🏠 {res['home_win_prob']*100:.1f}% | "
          f"🤝 {res['draw_prob']*100:.1f}% | "
          f"✈️  {res['away_win_prob']*100:.1f}%")
    print(f"  🎯 Prediksi: {res['prediction'].upper()}")
    print(f"  🏆 Top Skor: {entry['top_score']} ({entry['top_score_pct']}%)")
    print(f"  {tier_emoji} Tier: {tier}")
    print(f"  📋 {reason}")
    print(f"  🆔 ID: {entry['id']}")
    print(f"{'='*55}")
    print(f"  → Setelah pertandingan, jalankan:")
    print(f"    update_result('{home_team}', '{away_team}', 'H/D/A')")

    return entry


def update_result(home_team, away_team, result_ftr,
                  match_date=None, score="?-?"):
    """
    Update hasil aktual setelah pertandingan selesai.

    Args:
        home_team  : nama tim kandang (sama persis dengan saat log)
        away_team  : nama tim tamu
        result_ftr : 'H' (home win), 'D' (draw), 'A' (away win)
        score      : skor aktual, misal "2-1"
    """
    if result_ftr not in ["H", "D", "A"]:
        print("❌ result_ftr harus 'H', 'D', atau 'A'")
        return

    actual_map = {"H": "home_win", "D": "draw", "A": "away_win"}
    actual     = actual_map[result_ftr]

    if match_date is None:
        match_date = datetime.now().strftime("%Y-%m-%d")

    entry_id = f"{match_date}_{home_team}_{away_team}".replace(" ","_")

    data = _load_log()
    updated = False
    for p in data["predictions"]:
        if p["id"] == entry_id or (
            p["home_team"] == home_team and
            p["away_team"] == away_team and
            p["actual_result"] is None
        ):
            p["actual_result"] = actual
            p["correct"]       = (p["prediction"] == actual)
            p["score_actual"]  = score
            p["updated_at"]    = str(datetime.now())
            updated = True

            result_emoji = "✅" if p["correct"] else "❌"
            print(f"\n{result_emoji} UPDATE: {home_team} vs {away_team}")
            print(f"   Prediksi : {p['prediction'].upper()} | "
                  f"Aktual: {actual.upper()} | Skor: {score}")
            print(f"   Tier     : {p['tier']}")
            break

    if not updated:
        print(f"⚠️  Prediksi tidak ditemukan: {home_team} vs {away_team}")
        print(f"   Cek log dengan: show_log()")
        return

    _save_log(data)


def show_dashboard(last_n=None):
    """
    Tampilkan dashboard performa lengkap dengan equity curve.

    Args:
        last_n: tampilkan N prediksi terakhir saja (None = semua)
    """
    data  = _load_log()
    preds = [p for p in data["predictions"] if p["actual_result"] is not None]

    if not preds:
        print("⚠️  Belum ada prediksi yang sudah diupdate hasilnya.")
        print("   Gunakan update_result() setelah pertandingan selesai.")
        return

    if last_n:
        preds = preds[-last_n:]

    df = pd.DataFrame(preds)

    # Hitung metrics
    sniper   = df[df["tier"] == "SNIPER"]
    hold     = df[df["tier"] == "HOLD"]

    acc_all    = df["correct"].mean()*100    if len(df)      else 0
    acc_sniper = sniper["correct"].mean()*100 if len(sniper) else 0
    acc_hold   = hold["correct"].mean()*100   if len(hold)   else 0

    # Confidence interval sniper
    if len(sniper) >= 5:
        n_s  = len(sniper)
        k_s  = sniper["correct"].sum()
        ci_l = binom.ppf(0.025, n_s, max(k_s/n_s, 0.01)) / n_s * 100
        ci_h = binom.ppf(0.975, n_s, min(k_s/n_s, 0.99)) / n_s * 100
        ci_str = f"[{ci_l:.1f}%–{ci_h:.1f}%]"
    else:
        ci_str = "(min 5 laga untuk CI)"

    # ── Print laporan teks ─────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  📊 FOOTBALL PREDICTION TRACKER DASHBOARD")
    print(f"  Total prediksi selesai: {len(df)}")
    print(f"{'='*60}")

    print(f"\n  AKURASI KESELURUHAN")
    print(f"  Semua laga   : {acc_all:.1f}% ({df['correct'].sum()}/{len(df)})")
    print(f"  🎯 Sniper    : {acc_sniper:.1f}% "
          f"({sniper['correct'].sum() if len(sniper) else 0}/{len(sniper)}) "
          f"{ci_str}")
    print(f"  ⏸️  Hold     : {acc_hold:.1f}% "
          f"({hold['correct'].sum() if len(hold) else 0}/{len(hold)})")

    print(f"\n  SNIPER BREAKDOWN")
    for label in ["home_win","draw","away_win"]:
        sub = sniper[sniper["prediction"]==label] if len(sniper) else pd.DataFrame()
        if len(sub):
            acc = sub["correct"].mean()*100
            print(f"  {label:12s}: {acc:.1f}% ({sub['correct'].sum()}/{len(sub)})")

    print(f"\n  DISTRIBUSI SNIPER")
    if len(sniper):
        for label in ["home_win","draw","away_win"]:
            p = (sniper["prediction"]==label).sum()
            a = (sniper["actual_result"]==label).sum()
            print(f"  {label:12s}: Pred={p} | Aktual={a}")

    # Recent form (last 10 sniper)
    if len(sniper) >= 5:
        recent = sniper.tail(10)
        recent_acc = recent["correct"].mean()*100
        form_str = "".join(["✅" if r else "❌"
                             for r in recent["correct"].values])
        print(f"\n  RECENT FORM (last {len(recent)} Sniper): {form_str}")
        print(f"  Recent accuracy: {recent_acc:.1f}%")

    # ── Plot equity curve ──────────────────────────────────────────
    if len(sniper) >= 3:
        fig = plt.figure(figsize=(16, 8))
        fig.suptitle("⚽ Football Prediction Tracker",
                     fontsize=14, fontweight="bold")
        gs = gridspec.GridSpec(2, 3, figure=fig,
                               hspace=0.45, wspace=0.35)

        # 1. Equity Curve (Sniper)
        ax1 = fig.add_subplot(gs[0, :2])
        cumulative = sniper["correct"].cumsum().values
        x = range(1, len(sniper)+1)
        random_line = [i * 0.5 for i in x]   # 50% baseline
        benchmark   = [i * 0.62 for i in x]  # 62% benchmark

        ax1.fill_between(x, random_line, cumulative,
                         where=[c >= r for c, r in
                                zip(cumulative, random_line)],
                         alpha=0.15, color="#27ae60", label="Above random")
        ax1.fill_between(x, random_line, cumulative,
                         where=[c < r for c, r in
                                zip(cumulative, random_line)],
                         alpha=0.15, color="#e74c3c")
        ax1.plot(x, cumulative, "b-o", markersize=5,
                 linewidth=2, label="Actual wins")
        ax1.plot(x, random_line, "gray", linestyle="--",
                 linewidth=1, label="50% baseline")
        ax1.plot(x, benchmark, "orange", linestyle="--",
                 linewidth=1, label="62% benchmark")
        ax1.set_title("Equity Curve — Sniper Predictions",
                      fontweight="bold")
        ax1.set_xlabel("Laga ke-")
        ax1.set_ylabel("Total benar")
        ax1.legend(fontsize=8)
        ax1.grid(alpha=0.3)

        # 2. Akurasi Rolling (window 10)
        ax2 = fig.add_subplot(gs[0, 2])
        if len(sniper) >= 10:
            rolling = sniper["correct"].rolling(10).mean() * 100
            ax2.plot(rolling.values, "purple", linewidth=2)
            ax2.axhline(75, color="green", linestyle="--",
                        linewidth=1, label="Target 75%")
            ax2.axhline(62, color="orange", linestyle="--",
                        linewidth=1, label="Benchmark 62%")
            ax2.axhline(50, color="gray", linestyle="--",
                        linewidth=1, label="Random 50%")
            ax2.set_title("Rolling Accuracy (10 laga)",
                          fontweight="bold")
            ax2.set_ylim(0, 110)
            ax2.legend(fontsize=7)
            ax2.grid(alpha=0.3)
        else:
            ax2.text(0.5, 0.5, f"Min 10 laga\n(saat ini {len(sniper)})",
                     ha="center", va="center",
                     transform=ax2.transAxes, fontsize=11)
            ax2.set_title("Rolling Accuracy", fontweight="bold")

        # 3. Distribusi prediksi
        ax3 = fig.add_subplot(gs[1, 0])
        labels = ["Home\nWin", "Draw", "Away\nWin"]
        pred_c = [(sniper["prediction"]==l).sum()
                  for l in ["home_win","draw","away_win"]]
        act_c  = [(sniper["actual_result"]==l).sum()
                  for l in ["home_win","draw","away_win"]]
        x3 = np.arange(3)
        ax3.bar(x3-0.2, pred_c, 0.35, label="Pred",
                color="#3498db", alpha=0.85)
        ax3.bar(x3+0.2, act_c,  0.35, label="Aktual",
                color="#e74c3c", alpha=0.85)
        ax3.set_xticks(x3)
        ax3.set_xticklabels(labels)
        ax3.set_title("Pred vs Aktual (Sniper)",
                      fontweight="bold")
        ax3.legend(fontsize=8)
        ax3.grid(axis="y", alpha=0.3)

        # 4. Akurasi per tier
        ax4 = fig.add_subplot(gs[1, 1])
        tiers = ["SNIPER", "HOLD"]
        accs  = [acc_sniper, acc_hold]
        colors = ["#27ae60" if a >= 62 else "#e74c3c" for a in accs]
        bars = ax4.bar(tiers, accs, color=colors, edgecolor="white")
        ax4.axhline(62, color="orange", linestyle="--",
                    linewidth=1, label="Benchmark 62%")
        ax4.axhline(50, color="gray", linestyle="--",
                    linewidth=1, label="Random 50%")
        for b, v in zip(bars, accs):
            ax4.text(b.get_x() + b.get_width()/2,
                     b.get_height() + 1,
                     f"{v:.1f}%", ha="center",
                     fontweight="bold", fontsize=11)
        ax4.set_title("Akurasi per Tier", fontweight="bold")
        ax4.set_ylim(0, 115)
        ax4.legend(fontsize=8)
        ax4.grid(axis="y", alpha=0.3)

        # 5. Trend akurasi (cumulative)
        ax5 = fig.add_subplot(gs[1, 2])
        cum_acc = [sniper["correct"].iloc[:i+1].mean()*100
                   for i in range(len(sniper))]
        ax5.plot(range(1, len(sniper)+1), cum_acc,
                 "b-", linewidth=2, label="Sniper accuracy")
        ax5.axhline(75, color="green", linestyle="--",
                    linewidth=1, label="Target 75%")
        ax5.axhline(62, color="orange", linestyle="--",
                    linewidth=1, label="Benchmark 62%")
        ax5.fill_between(range(1, len(sniper)+1), 62, cum_acc,
                         where=[c >= 62 for c in cum_acc],
                         alpha=0.1, color="green")
        ax5.set_title("Cumulative Accuracy Trend",
                      fontweight="bold")
        ax5.set_ylim(0, 110)
        ax5.legend(fontsize=7)
        ax5.grid(alpha=0.3)

        plt.savefig("/content/prediction_dashboard.png",
                    dpi=150, bbox_inches="tight")
        plt.show()
        print("✅ Dashboard disimpan: /content/prediction_dashboard.png")

    return df


def show_log(tier=None, last_n=20):
    """Tampilkan log prediksi (termasuk yang belum ada hasilnya)."""
    data  = _load_log()
    preds = data["predictions"]

    if tier:
        preds = [p for p in preds if p.get("tier") == tier]

    preds = preds[-last_n:]

    print(f"\n{'='*70}")
    print(f"  📋 LOG PREDIKSI (last {len(preds)})")
    print(f"{'='*70}")

    for p in preds:
        status = ("✅" if p["correct"] else
                  ("❌" if p["correct"] is False else "⏳"))
        tier_e = {"SNIPER":"🎯","HOLD":"⏸️ ","SKIP":"⛔"}.get(p["tier"],"❓")
        actual = p["actual_result"] or "pending"
        print(f"  {status} {p['date']} | {p['home_team']:18s} vs "
              f"{p['away_team']:18s} | "
              f"Pred:{p['prediction']:10s} Actual:{actual:10s} | "
              f"{tier_e}{p['tier']}")

    pending = sum(1 for p in data["predictions"]
                  if p["actual_result"] is None)
    if pending:
        print(f"\n  ⏳ {pending} prediksi menunggu hasil aktual")


def export_report():
    """Export log ke CSV untuk analisis lanjutan."""
    data  = _load_log()
    preds = [p for p in data["predictions"]
             if p["actual_result"] is not None]

    if not preds:
        print("⚠️  Belum ada data untuk diekspor.")
        return

    df = pd.DataFrame(preds)
    path = "/content/prediction_report.csv"
    df.to_csv(path, index=False)
    print(f"✅ Exported {len(df)} prediksi ke {path}")
    return df


print("✅ CELL 21 SELESAI — Log Tracker siap!")
print("""
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  CARA PAKAI LOG TRACKER:

  # 1. Log prediksi SEBELUM pertandingan
  log_prediction("Barcelona", "Real Madrid",
                  match_date="2026-03-16",
                  notes="El Clasico, Barcelona butuh menang")

  # 2. Update hasil SETELAH pertandingan
  update_result("Barcelona", "Real Madrid", "H", score="2-1")

  # 3. Lihat dashboard + equity curve
  show_dashboard()

  # 4. Lihat semua log
  show_log()

  # 5. Export ke CSV
  export_report()

  # Tips: Mount Google Drive dulu agar log tidak hilang
  # from google.colab import drive
  # drive.mount('/content/drive')
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
""")
