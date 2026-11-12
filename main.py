#!/usr/bin/env python3
"""Monte Carlo rank forecaster for Roblox's Up-and-Coming top 10.

Entry for make.fun bounty roblox-up-and-coming-forecast-01.
Reads the chart history at $MAKEFUN_INPUT_DIR/history.ndjson (or /input)
and writes $MAKEFUN_OUTPUT_DIR/forecast.json (or /output/forecast.json):
per-place probabilities for the top 10 plus an ordered top10 guess.

How it works: simulate the chart 72 hours forward. Every game keeps a
score (its negative rank). Each hour the score drifts by a shrunk
rank-velocity momentum (games that are climbing keep climbing) minus a
small age decay for games first seen more than 2 days ago, plus Gaussian
noise. Ranks each hour are the order of the scores. Probabilities are the
simulated frequencies over 1000 seeded runs; the top10 is ordered by
expected rank across all simulated hours.

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

# Forecast knobs, tuned on the practice history.
PLACES = 10
HOURS = 72
SIMS = 1000
SIGMA = 2.5
DRIFT_SHRINK = 0.237
AGE_DECAY = 0.02
NAMED_PER_PLACE = 15
SEED = 12345


def parse_ts(text):
    """Parse an ISO timestamp, accepting a trailing Z for UTC."""
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def load_reads(input_dir):
    """Read history.ndjson into a list of reads, oldest first.

    Each read is a dict with 'at' (datetime), 'order' (universe ids in
    Up-and-Coming rank order) and 'tt' (Top Trending ranks by id, or None).
    """
    reads = []
    path = os.path.join(input_dir, "history.ndjson")
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            up = record["up_and_coming"]
            order = [x["universe_id"] for x in sorted(up, key=lambda z: z["rank"])]
            trending = None
            if record.get("top_trending"):
                trending = {x["universe_id"]: x["rank"] for x in record["top_trending"]}
            reads.append({"at": parse_ts(record["at"]), "order": order, "tt": trending})
    return reads


def rank_velocity(hist, chart):
    """Per-game rank drift per hour, from the last two intervals.

    For each game on the current chart, take the rank change per hour over
    the last two intervals (at most), average them, and shrink by
    DRIFT_SHRINK. A negative return means the game is climbing, so the
    drift added to its score is positive.
    """
    traces = {u: [] for u in chart}
    for read in hist:
        for pos, uid in enumerate(read["order"]):
            if uid in traces:
                traces[uid].append((read["at"], pos + 1))
    drift = {}
    for uid in chart:
        trace = traces[uid]
        if len(trace) >= 2:
            steps = []
            start = max(0, len(trace) - 2)
            for j in range(start, len(trace) - 1):
                hours = (trace[j + 1][0] - trace[j][0]).total_seconds() / 3600
                if hours > 0:
                    steps.append((trace[j + 1][1] - trace[j][1]) / hours)
            mean_step = sum(steps) / len(steps) if steps else 0.0
        else:
            mean_step = 0.0
        drift[uid] = -DRIFT_SHRINK * mean_step
    return drift


def first_seen(hist):
    """Earliest read time for each universe id in the history."""
    seen = {}
    for read in hist:
        for uid in read["order"]:
            if uid not in seen:
                seen[uid] = read["at"]
    return seen


def mc_forecast(hist):
    """Monte Carlo rank forecast. hist: list of reads (oldest first)."""
    rng = random.Random(SEED)
    last = hist[-1]
    chart = last["order"]
    rank0 = {u: pos + 1 for pos, u in enumerate(chart)}
    vel = rank_velocity(hist, chart)
    seen = first_seen(hist)

    counts = {k: {} for k in range(1, PLACES + 1)}
    rank_sum = {u: 0.0 for u in chart}
    rank_cnt = {u: 0 for u in chart}

    for _ in range(SIMS):
        score = {u: -float(rank0[u]) for u in chart}
        drift = {}
        for u in chart:
            d = vel[u]
            age_days = (last["at"] - seen[u]).total_seconds() / 86400
            d -= AGE_DECAY * max(0.0, age_days - 2.0)
            drift[u] = d
        for _ in range(HOURS):
            for u in chart:
                score[u] += rng.gauss(0.0, SIGMA) + drift[u]
            ranked = sorted(chart, key=lambda u: -score[u])
            for k in range(PLACES):
                game = ranked[k]
                counts[k + 1][game] = counts[k + 1].get(game, 0) + 1
            for k, u in enumerate(ranked):
                rank_sum[u] += 1 + k
                rank_cnt[u] += 1

    total = SIMS * HOURS
    probs = []
    for k in range(1, PLACES + 1):
        best = sorted(counts[k].items(), key=lambda kv: -kv[1])[:NAMED_PER_PLACE]
        rows = [
            {"universe_id": g, "place": k, "p": round(cnt / total, 6)}
            for g, cnt in best
        ]
        # Keep each place's probabilities summing to at most 1.
        s = sum(row["p"] for row in rows)
        if s > 1.0 and rows:
            biggest = max(rows, key=lambda row: row["p"])
            biggest["p"] = round(biggest["p"] - (s - 1.0), 6)
        probs.extend(rows)

    expected = {u: (rank_sum[u] / rank_cnt[u] if rank_cnt[u] else 999) for u in chart}
    top10 = sorted(chart, key=lambda u: expected[u])[:PLACES]
    return {"probabilities": probs, "top10": top10}


def main():
    input_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    output_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    hist = load_reads(input_dir)
    forecast = mc_forecast(hist)
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "forecast.json")
    with open(out_path, "w") as handle:
        json.dump(forecast, handle)


if __name__ == "__main__":
    main()
