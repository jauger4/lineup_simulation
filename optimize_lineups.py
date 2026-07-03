"""Search each team's batting-order space for the highest run-scoring lineup.

Fifth step of the lineup-optimization project (see instructions.md). For every team we take its
nine regular hitters (from hitter_lineup_2026.csv) with their real-life batting order as the
baseline, then hill-climb over pairwise swaps -- re-simulating with lineup_sim.py -- to find the
order that scores the most runs per game. 9! = 362,880 orders is too many to simulate
exhaustively at high game counts, so this is a greedy search with random restarts, not a proof
of global optimality.

Two game-count settings:
  * N_SEARCH  -- cheaper sims used to compare candidates during the hill-climb. Every candidate
    is scored with the SAME simulator seed (common random numbers) so comparisons are paired and
    reproducible, which keeps the tiny order-to-order differences from drowning in Monte Carlo
    noise.
  * N_FINAL   -- a big, independent re-simulation of the baseline vs. the best order found, used
    for the reported runs/game and the significance test.

The baseline order is always one of the candidates, so the reported "best" can never score below
the team's real order.

Caveat, stated plainly in the output: real batting-order effects are small (roughly 0.1-0.3
R/G between a good and a poor order), and this model treats every PA as independent of count,
pitcher, and lineup protection. Treat the deltas as directional, not precise.

Run:  python optimize_lineups.py
"""

from itertools import combinations

import numpy as np
import pandas as pd
from scipy import stats

from lineup_sim import LineupSimulator, Batter, league_average_batter

OUTCOMES_CSV = "hitter_outcomes_2026.csv"
LINEUP_CSV = "hitter_lineup_2026.csv"
OUTPUT_CSV = "optimal_lineups_2026.csv"

N_SEARCH = 5000        # games per candidate during the hill-climb
N_FINAL = 30000        # games for the final baseline-vs-best comparison
RESTARTS = 2           # random-restart hill climbs beyond the from-baseline climb
EVAL_SEED = 7          # common-random-numbers seed reused for every search evaluation
SLOTS = [f"g_slot{s}" for s in range(1, 10)]


def load_data():
    outcomes = pd.read_csv(OUTCOMES_CSV)
    lineup = pd.read_csv(LINEUP_CSV)
    batters = {int(r["batter"]): Batter.from_row(r) for _, r in outcomes.iterrows()}
    avg = league_average_batter(outcomes)
    return outcomes, lineup, batters, avg


def build_team_lineup(team_rows, batters, avg):
    """Return (batter_ids, Batter list, n_padded) for one team in real-life batting order.

    Only hitters with outcome data are eligible. Slots 1-9 are filled greedily: for each slot
    take the still-unused eligible hitter who started there most often. Any slot left unfilled
    (teams with fewer than nine qualified hitters) gets a league-average placeholder.
    """
    eligible = team_rows[team_rows["batter"].isin(batters)].copy()
    order_ids = []
    used = set()
    for slot in SLOTS:
        pool = eligible[~eligible["batter"].isin(used)]
        if pool.empty:
            break
        pick = int(pool.loc[pool[slot].idxmax(), "batter"])
        order_ids.append(pick)
        used.add(pick)

    n_padded = 0
    while len(order_ids) < 9:
        order_ids.append(None)          # placeholder -> league-average batter
        n_padded += 1

    lineup = [batters[b] if b is not None else avg for b in order_ids]
    return order_ids, lineup, n_padded


def mean_runs(lineup, n):
    """Mean runs/game, always with the common-random-numbers seed (paired comparisons)."""
    return LineupSimulator(seed=EVAL_SEED).simulate_games(lineup, n).mean()


def hill_climb(lineup, order):
    """Greedy pairwise-swap hill climb over index permutations of `lineup`.

    `order` and the return value are permutations of range(9): position -> baseline slot index.
    """
    current = list(order)
    cur_score = mean_runs([lineup[k] for k in current], N_SEARCH)
    improved = True
    while improved:
        improved = False
        best_swap, best_score = None, cur_score
        for i, j in combinations(range(9), 2):
            cand = current.copy()
            cand[i], cand[j] = cand[j], cand[i]
            score = mean_runs([lineup[k] for k in cand], N_SEARCH)
            if score > best_score + 1e-9:
                best_score, best_swap = score, (i, j)
        if best_swap is not None:
            i, j = best_swap
            current[i], current[j] = current[j], current[i]
            cur_score, improved = best_score, True
    return current


def optimize_team(order_ids, lineup, rng):
    """Hill-climb (with restarts) and confirm the winner against the baseline at N_FINAL."""
    idx = list(range(9))
    candidates = [idx, hill_climb(lineup, idx)]
    for _ in range(RESTARTS):
        start = idx.copy()
        rng.shuffle(start)
        candidates.append(hill_climb(lineup, start))

    # De-duplicate candidate orderings, then pick the best on a big fresh simulation.
    seen, uniq = set(), []
    for perm in candidates:
        key = tuple(perm)
        if key not in seen:
            seen.add(key)
            uniq.append(perm)

    # Point estimates: score every unique candidate on the SAME big paired sample. Because the
    # baseline is always a candidate, the selected best can never score below it here.
    scored = [(perm, mean_runs([lineup[k] for k in perm], N_FINAL)) for perm in uniq]
    best_perm, best_rpg = max(scored, key=lambda t: t[1])
    baseline_rpg = next(rpg for perm, rpg in scored if perm == idx)

    # Significance: independent large samples (fresh seeds) so the Welch test is not run on the
    # very data the winner was selected on.
    base_runs = LineupSimulator(seed=EVAL_SEED + 1).simulate_games(lineup, N_FINAL)
    best_runs = LineupSimulator(seed=EVAL_SEED + 2).simulate_games(
        [lineup[k] for k in best_perm], N_FINAL
    )
    _, p = stats.ttest_ind(best_runs, base_runs, equal_var=False)

    best_ids = [order_ids[k] for k in best_perm]
    return {
        "baseline_rpg": round(float(baseline_rpg), 4),
        "best_rpg": round(float(best_rpg), 4),
        "delta_rpg": round(float(best_rpg - baseline_rpg), 4),
        "p_value": round(float(p), 5),
        "best_order_slots": best_perm,          # baseline slot each best-lineup position came from
        "best_ids": best_ids,
    }


def main():
    outcomes, lineup_df, batters, avg = load_data()
    id_to_name = dict(zip(outcomes["batter"], outcomes["name"]))
    rng = np.random.default_rng(EVAL_SEED)

    rows = []
    for team, grp in lineup_df.groupby("team"):
        order_ids, lineup, n_padded = build_team_lineup(grp, batters, avg)
        res = optimize_team(order_ids, lineup, rng)

        base_names = [id_to_name.get(b, "LeagueAvg") for b in order_ids]
        best_names = [id_to_name.get(b, "LeagueAvg") for b in res["best_ids"]]
        rows.append({
            "team": team,
            "baseline_rpg": res["baseline_rpg"],
            "best_rpg": res["best_rpg"],
            "delta_rpg": res["delta_rpg"],
            "p_value": res["p_value"],
            "n_padded": n_padded,
            "baseline_order": "|".join(base_names),
            "best_order": "|".join(best_names),
            "baseline_ids": "|".join(str(b) for b in order_ids),
            "best_ids": "|".join(str(b) for b in res["best_ids"]),
            "best_from_slots": "|".join(str(s + 1) for s in res["best_order_slots"]),
        })
        print(f"{team}: baseline {res['baseline_rpg']:.3f} -> best {res['best_rpg']:.3f} "
              f"(+{res['delta_rpg']:.3f} R/G, p={res['p_value']:.4f})"
              + (f"  [padded {n_padded}]" if n_padded else ""))

    out = pd.DataFrame(rows).sort_values("delta_rpg", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")

    sig = (out["p_value"] < 0.05).sum()
    print(f"\nSaved {OUTPUT_CSV}. Mean gain {out['delta_rpg'].mean():.3f} R/G; "
          f"{sig}/{len(out)} teams' improvement is significant at p<0.05.")
    print("Caveat: order effects are small and PAs are modelled independently -- deltas are "
          "directional, not exact.")


if __name__ == "__main__":
    main()
