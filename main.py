#!/usr/bin/env python3
"""Monte Carlo rank forecaster for Roblox's Up-and-Coming top 10.

Entry for make.fun bounty roblox-up-and-coming-forecast-01.
Reads the chart history at $MAKEFUN_INPUT_DIR/history.ndjson (or /input)
and writes $MAKEFUN_OUTPUT_DIR/forecast.json (or /output/forecast.json):
per-place probabilities for the top 10 plus an ordered top10 guess.

Method: simulate the chart 72 hours forward. Each game keeps a score
(-rank). Every hour the score drifts by a shrunk rank-velocity momentum
(rising games keep rising) minus a small age decay for games older than
2 days, plus Gaussian noise. Ranks each hour are the score order.
Probabilities are the simulated frequencies over 1000 seeded runs.

Tuned on the practice history (4 backtest cutoffs, 10 reads each):
mean log score -36.55 vs house baseline -45.01 (official practice
baseline -43.51). Parameters: sigma 2.5, drift_shrink 0.237,
age_decay 0.02, 15 named games per place, seed 12345.

Standard library only, no network, deterministic: the same history
always produces the same file byte for byte.
"""

import json
import math
import os
import random
from datetime import datetime

PLACES = 10
HOURS = 72
SIMS = 1000
SIGMA = 2.5
DRIFT_SHRINK = 0.237
AGE_DECAY = 0.02
NAMED_PER_PLACE = 15
SEED = 12345


def parse_ts(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def load_reads(input_dir):
    """Read history.ndjson into [{at, order, tt}], oldest first."""
    reads = []
    with open(os.path.join(input_dir, "history.ndjson")) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            uc = r["up_and_coming"]
            order = [x["universe_id"] for x in sorted(uc, key=lambda z: z["rank"])]
            tt = None
            if r.get("top_trending"):
                tt = {x["universe_id"]: x["rank"] for x in r["top_trending"]}
            reads.append({"at": parse_ts(r["at"]), "order": order, "tt": tt})
    return reads


def mc_forecast(hist):
    """Monte Carlo rank forecast. hist: list of reads (oldest first)."""
    rng = random.Random(SEED)
    last = hist[-1]
    chart = last["order"]
    rank0 = {u: k + 1 for k, u in enumerate(chart)}

    # Rank velocity: rank change per hour over the last two intervals, shrunk.
    rhist = {u: [] for u in chart}
    for r in hist:
        for k, u in enumerate(r["order"]):
            if u in rhist:
                rhist[u].append((r["at"], k + 1))
    vel = {}
    for u in chart:
        h = rhist[u]
        if len(h) >= 2:
            vs = []
            for j in range(max(0, len(h) - 2), len(h) - 1):
                dt = (h[j + 1][0] - h[j][0]).total_seconds() / 3600
                if dt > 0:
                    vs.append((h[j + 1][1] - h[j][1]) / dt)
            v = sum(vs) / len(vs) if vs else 0.0
        else:
            v = 0.0
        vel[u] = -DRIFT_SHRINK * v

    firstseen = {}
    for r in hist:
        for u in r["order"]:
            if u not in firstseen:
                firstseen[u] = r["at"]

    counts = {k: {} for k in range(1, PLACES + 1)}
    rank_sum = {u: 0.0 for u in chart}
    rank_cnt = {u: 0 for u in chart}

    for _ in range(SIMS):
        score = {u: -float(rank0[u]) for u in chart}
        drift = {}
        for u in chart:
            d = vel[u]
            age = (last["at"] - firstseen[u]).total_seconds() / 86400
            d -= AGE_DECAY * max(0.0, age - 2.0)
            drift[u] = d
        for _ in range(HOURS):
            for u in chart:
                score[u] += drift[u] + rng.gauss(0.0, SIGMA)
            ranked = sorted(chart, key=lambda u: -score[u])
            for k in range(PLACES):
                g = ranked[k]
                counts[k + 1][g] = counts[k + 1].get(g, 0) + 1
            for k, u in enumerate(ranked):
                rank_sum[u] += k + 1
                rank_cnt[u] += 1

    total = HOURS * SIMS
    probs = []
    for k in range(1, PLACES + 1):
        top = sorted(counts[k].items(), key=lambda kv: -kv[1])[:NAMED_PER_PLACE]
        rows = [
            {"universe_id": g, "place": k, "p": round(cnt / total, 6)}
            for g, cnt in top
        ]
        # Keep each place's probabilities summing to at most 1.
        s = sum(row["p"] for row in rows)
        if s > 1.0 and rows:
            biggest = max(rows, key=lambda row: row["p"])
            biggest["p"] = round(biggest["p"] - (s - 1.0), 6)
        probs.extend(rows)

    exprank = {u: (rank_sum[u] / rank_cnt[u] if rank_cnt[u] else 999) for u in chart}
    top10 = sorted(chart, key=lambda u: exprank[u])[:PLACES]
    return {"probabilities": probs, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    hist = load_reads(input_dir)
    forecast = mc_forecast(hist)
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "forecast.json"), "w") as f:
        json.dump(forecast, f)


if __name__ == "__main__":
    main()
