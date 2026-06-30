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

## Next steps (not yet built)
- Team-level analysis: flag which 2026 lineups already bat power-up vs. traditional.
- Monte Carlo lineup simulation comparing real vs. power-up orderings (runs/game, significance test).

## Environment / running
- Python 3.11. Installed: `pybaseball` 2.2.7, pandas, numpy, scikit-learn 1.8.0, matplotlib, seaborn, scipy.
- `notebook`/`jupyterlab`/`nbconvert` are **not** installed — run `.ipynb` files via **VS Code's** Jupyter
  support (Run All). To validate a notebook headless, exec its code cells with the matplotlib `Agg` backend.
- `pybaseball` caching is enabled (cache at `~/.pybaseball/cache`); the season pull is only fetched once.
- The `*_2026.csv` data products are committed intentionally (in-progress-season snapshots, slow to re-pull).
