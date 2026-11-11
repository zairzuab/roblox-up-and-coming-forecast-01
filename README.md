# Roblox Up-and-Coming top 10 forecast

Entry for the make.fun bounty **Predict Roblox's Up-and-Coming top 10**
(`roblox-up-and-coming-forecast-01`).

`main.py` (Python 3.13, standard library only, no network) reads the
chart history at `$MAKEFUN_INPUT_DIR/history.ndjson` (or `/input`) and
writes `$MAKEFUN_OUTPUT_DIR/forecast.json` (or `/output/forecast.json`):
per-place probabilities for the top 10 plus an ordered `top10` guess.

## Method

Monte Carlo rank simulation over the 72-hour scored window. Each game
keeps a score (`-rank`). Every hour the score drifts by a shrunk
rank-velocity momentum (rising games keep rising) minus a small age
decay for games older than 2 days, plus Gaussian noise. Ranks each hour
are the score order. Probabilities are the simulated frequencies over
300 seeded runs, naming the 15 most frequent games per place.

Tuned on the practice history (4 backtest cutoffs, 10 reads each):
mean log score **-36.55** vs house baseline **-45.01** (official
practice baseline -43.51).

Parameters: `sims=300`, `sigma=2.5`, `drift_shrink=0.237`,
`age_decay=0.02`, `named_per_place=15`, `seed=12345`.

## Run locally

```text
mkdir -p out
MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Deterministic: the same history always produces the same file byte
for byte (verified: two runs, identical output).

## License

MIT. See LICENSE.
