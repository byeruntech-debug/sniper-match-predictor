#!/usr/bin/env python3
"""
Football Match Predictor — La Liga
Versi: 4.0 (xG split + PPDA + H2H + Window 10)
"""

# ══════════════════════════════════════════════════════
#  IMPORTS
# ══════════════════════════════════════════════════════
import os, re, json, time, types, unicodedata
import requests
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from datetime import datetime, timedelta
from collections import Counter

# ══════════════════════════════════════════════════════
#  CLASS DEFINITIONS
# ══════════════════════════════════════════════════════


# ══════════════════════════════════════════════════════
#  KONFIGURASI — GANTI SESUAI KEBUTUHAN
# ══════════════════════════════════════════════════════

FD_API_KEY       = "GANTI_DENGAN_FD_API_KEY"
API_FOOTBALL_KEY = "GANTI_DENGAN_API_FOOTBALL_KEY"
GEMINI_KEY       = "GANTI_DENGAN_GEMINI_KEY"

# ══════════════════════════════════════════════════════
#  MAPPING DATA
# ══════════════════════════════════════════════════════

STANDINGS_TO_AF = {'FC Barcelona': 'Barcelona', 'Real Madrid CF': 'Real Madrid', 'Club Atlético de Madrid': 'Atletico Madrid', 'Villarreal CF': 'Villarreal', 'Real Betis Balompié': 'Real Betis', 'RC Celta de Vigo': 'Celta Vigo', 'RCD Espanyol de Barcelona': 'Espanyol', 'Real Sociedad de Fútbol': 'Real Sociedad', 'Getafe CF': 'Getafe', 'Athletic Club': 'Athletic Club', 'CA Osasuna': 'Osasuna', 'Valencia CF': 'Valencia', 'Rayo Vallecano de Madrid': 'Rayo Vallecano', 'Sevilla FC': 'Sevilla', 'Girona FC': 'Girona', 'Deportivo Alavés': 'Alaves', 'Elche CF': 'Elche', 'RCD Mallorca': 'Mallorca', 'Levante UD': 'Levante', 'Real Oviedo': 'Oviedo'}

TEAM_ID_MAP = {'Barcelona': 529, 'Atletico Madrid': 530, 'Athletic Club': 531, 'Valencia': 532, 'Villarreal': 533, 'Las Palmas': 534, 'Sevilla': 536, 'Leganes': 537, 'Celta Vigo': 538, 'Espanyol': 540, 'Real Madrid': 541, 'Alaves': 542, 'Real Betis': 543, 'Getafe': 546, 'Girona': 547, 'Real Sociedad': 548, 'Valladolid': 720, 'Osasuna': 727, 'Rayo Vallecano': 728, 'Mallorca': 798, 'Elche': 797, 'Levante': 539, 'Oviedo': 718}

PPDA_ESTIMATED = {'FC Barcelona': 11.23, 'Real Madrid CF': 11.28, 'Club Atlético de Madrid': 10.41, 'Villarreal CF': 10.85, 'Real Betis Balompié': 10.69, 'RC Celta de Vigo': 10.05, 'RCD Espanyol de Barcelona': 11.92, 'Real Sociedad de Fútbol': 12.12, 'Getafe CF': 10.2, 'Athletic Club': 9.66, 'CA Osasuna': 11.52, 'Valencia CF': 10.27, 'Rayo Vallecano de Madrid': 9.9, 'Sevilla FC': 11.41, 'Girona FC': 12.37, 'Deportivo Alavés': 13.35, 'Elche CF': 14.62, 'RCD Mallorca': 11.47, 'Levante UD': 11.75, 'Real Oviedo': 11.4}


# ══════════════════════════════════════════════════════
#  FUNGSI HELPER
# ══════════════════════════════════════════════════════

def normalize(s):
    return unicodedata.normalize("NFD", s).encode("ascii", "ignore").decode().lower()

def find_team(keyword):
    kw = normalize(keyword)
    matches = [name for name in standings.keys() if kw in normalize(name)]
    if not matches:
        print(f"❌ Tidak ditemukan: '{keyword}'")
    return matches

def get_ppda(team_name):
    return PPDA_ESTIMATED.get(team_name, 11.0)

def get_team_id_af(standings_name):
    af_name = STANDINGS_TO_AF.get(standings_name)
    if not af_name:
        return None
    return TEAM_ID_MAP.get(af_name)

def apply_h2h_adjustment(hp, dp, ap, h2h, weight=0.15):
    if not h2h or h2h["total"] < 3:
        return hp, dp, ap
    hp_adj = hp * (1 - weight) + h2h["home_win_rate"] * weight
    dp_adj = dp * (1 - weight) + h2h["draw_rate"]     * weight
    ap_adj = ap * (1 - weight) + h2h["away_win_rate"] * weight
    total  = hp_adj + dp_adj + ap_adj
    return round(hp_adj/total, 3), round(dp_adj/total, 3), round(ap_adj/total, 3)


# ══════════════════════════════════════════════════════
#  FUNGSI H2H
# ══════════════════════════════════════════════════════

def get_h2h_apifootball(home_name, away_name, last_n=10):
    home_id = get_team_id_af(home_name)
    away_id = get_team_id_af(away_name)
    if not home_id or not away_id:
        return None
    resp = requests.get(
        f"https://v3.football.api-sports.io/fixtures/headtohead?h2h={home_id}-{away_id}",
        headers={"x-apisports-key": API_FOOTBALL_KEY}, timeout=10
    )
    fixtures = resp.json().get("response", [])
    finished = [f for f in fixtures
                if f["fixture"]["status"]["short"] == "FT"
                and f["league"]["id"] == 140]
    finished.sort(key=lambda x: x["fixture"]["date"])
    recent = finished[-last_n:]
    home_wins = away_wins = draws = 0
    h2h_processed = []
    for f in recent:
        hn  = f["teams"]["home"]["name"]
        hs  = f["goals"]["home"]
        as_ = f["goals"]["away"]
        if hs is None: continue
        is_home = hn == STANDINGS_TO_AF.get(home_name)
        if hs > as_:
            result = "home_win" if is_home else "away_win"
            if is_home: home_wins += 1
            else:       away_wins += 1
        elif hs < as_:
            result = "away_win" if is_home else "home_win"
            if is_home: away_wins += 1
            else:       home_wins += 1
        else:
            result = "draw"
            draws += 1
        h2h_processed.append({
            "date": f["fixture"]["date"][:10],
            "is_home": is_home,
            "gf": hs if is_home else as_,
            "ga": as_ if is_home else hs,
            "result": result
        })
    total = max(home_wins + away_wins + draws, 1)
    return {
        "matches"      : h2h_processed,
        "home_wins"    : home_wins,
        "away_wins"    : away_wins,
        "draws"        : draws,
        "total"        : total,
        "home_win_rate": round(home_wins / total, 3),
        "draw_rate"    : round(draws     / total, 3),
        "away_win_rate": round(away_wins / total, 3),
    }


# ══════════════════════════════════════════════════════
#  FUNGSI UTAMA
# ══════════════════════════════════════════════════════

def predict_match(home_name, away_name):
    if home_name not in standings:
        candidates = find_team(home_name)
        if not candidates: return None
        home_name = candidates[0]
        print(f"  ℹ️  Home → '{home_name}'")
    if away_name not in standings:
        candidates = find_team(away_name)
        if not candidates: return None
        away_name = candidates[0]
        print(f"  ℹ️  Away → '{away_name}'")

    ppda_home = get_ppda(home_name)
    ppda_away = get_ppda(away_name)

    print(f"\n{'═'*54}")
    print(f"  ⚽ {home_name}  vs  {away_name}")
    print(f"{'═'*54}")

    home_p = xg_collector_2025.build_real_profile_xg(
        home_name, standings, is_home=True,  ppda=ppda_home, setpiece=5.0)
    away_p = xg_collector_2025.build_real_profile_xg(
        away_name, standings, is_home=False, ppda=ppda_away, setpiece=5.0)

    if not home_p or not away_p:
        print("❌ Profil gagal dibangun.")
        return None

    sim_r  = sim_engine.run(home_p, away_p)
    pred_r = predictor.predict(home_p, away_p, sim_r)

    hp = pred_r["home_win_prob"]
    dp = pred_r["draw_prob"]
    ap = pred_r["away_win_prob"]

    print(f"\n📋 Mengambil data H2H...")
    h2h = get_h2h_apifootball(home_name, away_name)
    if h2h and h2h["total"] >= 3:
        hp, dp, ap = apply_h2h_adjustment(hp, dp, ap, h2h)
        print(f"  H2H ({h2h['total']} laga): 🏠{h2h['home_wins']}W 🤝{h2h['draws']}D ✈️{h2h['away_wins']}W")
        print(f"  Setelah H2H → H:{hp*100:.1f}% D:{dp*100:.1f}% A:{ap*100:.1f}%")
    else:
        print(f"  ⚠️  H2H tidak cukup data")

    probs  = [hp, dp, ap]
    labels = ["home_win", "draw", "away_win"]
    pred   = labels[probs.index(max(probs))]
    conf   = max(probs)
    conf_l = "Tinggi" if conf > 0.55 else ("Sedang" if conf > 0.45 else "Rendah")

    icon      = {"home_win":"🏠 HOME WIN","draw":"🤝 DRAW","away_win":"✈️  AWAY WIN"}[pred]
    conf_icon = {"Tinggi":"🟢","Sedang":"🟡","Rendah":"🔴"}[conf_l]

    print(f"\n{'─'*54}")
    print(f"  HASIL PREDIKSI FINAL")
    print(f"{'─'*54}")
    print(f"  {icon}")
    print(f"  {conf_icon} Keyakinan : {conf_l} ({conf*100:.1f}%)")
    print(f"{'─'*54}")
    print(f"  🏠 Home Win : {hp*100:.1f}%  {{'█' * int(hp*100/5)}}")
    print(f"  🤝 Draw     : {dp*100:.1f}%  {{'█' * int(dp*100/5)}}")
    print(f"  ✈️  Away Win : {ap*100:.1f}%  {{'█' * int(ap*100/5)}}")
    print(f"{'═'*54}\n")

    return {"prediction": pred, "home_win_prob": hp,
            "draw_prob": dp, "away_win_prob": ap,
            "confidence": conf, "confidence_label": conf_l}


# ══════════════════════════════════════════════════════
#  INISIALISASI — jalankan sekali di awal sesi
# ══════════════════════════════════════════════════════

def init():
    global fd_collector, xg_collector_2025, sim_engine
    global predictor, fe, standings, PPDA_ESTIMATED

    fd_collector       = FootballDataCollector(api_key=FD_API_KEY)
    xg_collector_2025  = UnderstatXGCollectorV2()
    sim_engine         = MonteCarloSimulator()
    predictor          = MatchPredictor()
    fe                 = FeatureEngineer()

    # Patch method v3 (xG home/away split + window 10)
    xg_collector_2025.build_xg_history      = types.MethodType(build_xg_history_v2,      xg_collector_2025)
    xg_collector_2025.build_real_profile_xg = types.MethodType(build_real_profile_xg_v3, xg_collector_2025)

    # Load standings
    data     = fd_collector._get("/competitions/PD/standings")
    table    = data["standings"][0]["table"]
    standings = {
        row["team"]["name"]: {
            "team_id" : row["team"]["id"],
            "position": row["position"],
            "pts"     : row["points"],
            "played"  : row["playedGames"],
            "won"     : row["won"],
            "draw"    : row["draw"],
            "lost"    : row["lost"],
            "gf"      : row["goalsFor"],
            "ga"      : row["goalsAgainst"],
            "gd"      : row["goalDifference"],
            "xg_est"  : round(row["goalsFor"]     / max(row["playedGames"], 1), 3),
            "xga_est" : round(row["goalsAgainst"] / max(row["playedGames"], 1), 3),
        }
        for row in table
    }

    # Refresh PPDA
    PPDA_ESTIMATED = {team: estimate_ppda_from_profile(team) for team in standings}

    print("✅ Sistem siap!")
    print(f"   Tim loaded : {len(standings)}")
    print(f"   PPDA ready : {len(PPDA_ESTIMATED)}")
    return True


if __name__ == "__main__":
    init()
    # Contoh penggunaan:
    # predict_match("atletico", "getafe")
    # predict_match("barcelona", "real madrid")
