# lineup_sim

Baseball lineup-optimization project. **Hypothesis:** power hitters may score more runs batting
*higher* in the order (where a walk/extra-base hit is more valuable) rather than in the traditional
3-4-5 spots, with contact hitters behind them to move runners over. The plan is to (1) cluster hitters
by plate approach from Statcast data, (2) identify which teams already bat power-up vs. traditional,
and (3) run Monte Carlo lineup simulations to test the hypothesis. Full write-up in `instructions.md`.

Season under study: **2026** (in progress; data pulled through ~2026-06-30).

## Pipeline & files (run in this order)
1. **`pull_hitter_data.py`** → `hitter_stats_2026.csv`. Pulls 2026 regular-season pitch-by-pitch data
   from Baseball Savant via `pybaseball`, aggregates per batter. Output columns: `batter, name, pa`
   + five approach stats, **all on a 0–100 scale**: `k_pct, bb_pct, chase_pct, whiff_pct, barrel_pct`.
2. **`explore_hitters.ipynb`** — light EDA (distributions, outliers, correlations) on the PA ≥ 150 set.
3. **`cluster_hitters.ipynb`** → `hitter_clusters_2026.csv`. K-means / DBSCAN / GMM clustering on the
   standardized stats. Output adds `kmeans_label, gmm_label, dbscan_label, archetype` (power/contact).
4. **`pull_lineup_data.py`** → `hitter_lineup_2026.csv`. Recovers each hitter's modal batting-order slot
   and **team** from the cached Statcast pull (Statcast has no lineup column; it is reconstructed from
   `at_bat_number` order per game). Columns: `batter, name, team, primary_slot, games_started, slot_share,
   g_slot1..g_slot9`. Used to assemble each team's real 9-man lineup (the optimization baseline).
5. **`pull_outcome_data.py`** → `hitter_outcomes_2026.csv`. Per-PA **outcome probabilities** the simulator
   needs (the approach stats can't drive a sim). Source is season counting stats from **Baseball Reference**
   (`batting_stats_bref`), *not* Statcast — FanGraphs `batting_stats` is currently HTTP-403, and SB/CS/sac/
   GDP are hard to reconstruct from pitch data. Ten `p_*` columns partition the PA (sum to 1): `p_1b, p_2b,
   p_3b, p_hr, p_bb` (BB+HBP+interference, taken as the residual), `p_k, p_dp, p_out, p_sf, p_sh`; plus
   `sb_rate, cs_rate` (per time-reached-first) and `avg, obp` for validation. Filtered to PA ≥ 150.
6. **`lineup_sim.py`** (+ `test_lineup_sim.py`) — the `LineupSimulator` class: a base-out state machine that
   plays innings/games one PA at a time from a batter's outcome vector. Runner advancement on hits is
   governed by `DEFAULT_ADVANCEMENT` (the tuning knobs), calibrated so a league-average lineup scores
   ~4.4 R/G. Stolen bases/sacrifices come from the per-batter rates. `python test_lineup_sim.py` runs the
   transition unit tests without needing pytest.
7. **`optimize_lineups.py`** → `optimal_lineups_2026.csv`. Per team, greedy pairwise-swap hill-climb (with
   random restarts, common-random-numbers seed) over the 9! order space to find the max-runs lineup, then a
   big independent re-sim of baseline vs. best with a Welch t-test. The team's real order is always a
   candidate, so `best_rpg ≥ baseline_rpg`.
8. **`analyze_optimal_lineups.ipynb`** — reruns the `explore_lineups.ipynb` slot-profile toolkit (mean-by-slot,
   heatmap, boxplots, eta²/ANOVA/Spearman/silhouette) on the **optimized** lineups and lays them next to the
   real ones, to see if the run-maximizing orders have a prototype and whether power moves up. Uses the raw
   approach stats, *not* cluster labels (clustering found no usable archetypes).
9. **`dashboard.py`** — Streamlit front-end over `optimal_lineups_2026.csv`. Alphabetical team picker → real
   vs. optimized lineup side by side as custom HTML **lineup cards** (per-hitter OBP/HR%/barrel%/BB%/K% chips
   heat-shaded by percentile vs. the qualified population — Savant-style red=hot/blue=cold, **K% inverted** —
   plus a ▲/▼ "moved" badge), the runs/game gain **gated on `p_value < 0.05`** (no gain claimed when not
   significant). Dark "Baseball Savant" theme in **`.streamlit/config.toml`** (red `#d22d49` accent). Pure CSV
   read, no simulation at runtime. Run: `streamlit run dashboard.py`.

## Key decisions & conventions
- **Clustering/analysis population = PA ≥ 150** (≈285 players). Drops low-sample pitchers/bench noise.
- **Barrel%** is pulled as Savant's official figure (`statcast_batter_exitvelo_barrels(...).brl_percent`,
  barrels per batted-ball event) and merged by batter id — *not* recomputed from pitch data.
- The other four stats are computed from pitch-by-pitch: K%/BB% per PA; whiff% = swing-and-miss / swings;
  chase% = out-of-zone swings / out-of-zone pitches (zone > 9). PA = unique `(game_pk, at_bat_number)`.
- **Gotcha:** in bulk `statcast()` output, `player_name` is the **pitcher**, not the batter. Group by the
  `batter` MLBAM id; resolve names via `playerid_reverse_lookup`.

## Key finding (clustering)
The five stats form a **continuum, not discrete archetypes.** All three algorithms agree: K-means' best
silhouette is a weak ~0.28 at k=2, GMM's BIC favors a single Gaussian, DBSCAN finds one dense blob + a
few outliers. PCA shows ~2 underlying axes (PC1+PC2 ≈ 83%): a **power axis** (K%/whiff%/barrel% move
together) and a largely independent **discipline axis** (walk% vs chase%, r≈-0.73). The only meaningful
split is power-leaning vs contact-leaning, driven by the power axis. **Implication:** for the simulation,
treat "power" as a position on a spectrum (e.g. PC1 or a barrel/whiff index), not a hard class.

## Simulation model & caveats
- **Each PA is drawn independently** from the batter's own outcome distribution — no count, pitcher, park,
  or lineup-protection effects. Batting order matters here *only through sequencing* (who bats with runners
  on), which is exactly the effect being tested.
- **Raw individual rates** (no regression to league mean), per the project decision — rare events (3B, SB)
  are therefore noisy for lower-PA hitters.
- **Order effects are small** (~0.1–0.3 R/G between a good and a poor order). Deltas from `optimize_lineups.py`
  are directional, not precise; the hill-climb is heuristic (local optimum, not a proven global best).
- Two teams (e.g. STL, SEA) have < 9 qualified hitters; their 9th slot is padded with a league-average
  batter (`n_padded` column flags this).

## Next steps (not yet built)
- Team-level analysis: flag which 2026 lineups already bat power-up vs. traditional (partly answered by
  `analyze_optimal_lineups.ipynb`, which contrasts real vs. optimized power-by-slot).
- Optional: validate/tune `DEFAULT_ADVANCEMENT` against real team run totals; add pitcher/handedness effects.

## Environment / running
- Python 3.11. Installed: `pybaseball` 2.2.7, pandas, numpy, scikit-learn 1.8.0, matplotlib, seaborn, scipy,
  `streamlit` 1.55 (dashboard). Pinned in `requirements.txt` (`pip install -r requirements.txt`).
- `notebook`/`jupyterlab`/`nbconvert` are **not** installed — run `.ipynb` files via **VS Code's** Jupyter
  support (Run All). To validate a notebook headless, exec its code cells with the matplotlib `Agg` backend.
- `pybaseball` caching is enabled (cache at `~/.pybaseball/cache`); the season pull is only fetched once.
- The `*_2026.csv` data products are committed intentionally (in-progress-season snapshots, slow to re-pull).
