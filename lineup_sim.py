"""Base-out Monte Carlo simulator for a baseball lineup.

Core of the lineup-optimization project (see instructions.md §Monte Carlo). Given nine
batters, each with a per-plate-appearance outcome distribution (from hitter_outcomes_2026.csv),
this plays out innings one PA at a time on a base-out state machine and returns runs scored.
Downstream code (optimize_lineups.py) reorders the nine batters and re-simulates to find the
order that scores the most.

Model / scope:
  * Each PA is drawn independently from the batter's own outcome distribution -- there are no
    count/pitcher/park/protection effects. Batting order matters here only through *sequencing*
    (who bats with runners on), which is exactly the effect the project is testing.
  * Runner advancement on hits is partly probabilistic, governed by DEFAULT_ADVANCEMENT. Those
    few constants are the simulator's main tuning knobs; they are set so a league-average lineup
    scores a realistic ~4.4 runs / 9-inning game (see calibrate() / tests).
  * Stolen bases and sacrifices are modelled from per-batter rates. A steal is rolled the moment
    a batter reaches first with second base open, using that batter's sb_rate / cs_rate.

Outcomes (indices used throughout) partition the PA and match hitter_outcomes_2026.csv:
    1b, 2b, 3b, hr, bb, k, dp, out, sf, sh
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

import numpy as np

# Outcome vocabulary, aligned with the p_* columns of hitter_outcomes_2026.csv.
OUTCOMES = ("1b", "2b", "3b", "hr", "bb", "k", "dp", "out", "sf", "sh")
O_1B, O_2B, O_3B, O_HR, O_BB, O_K, O_DP, O_OUT, O_SF, O_SH = range(10)
PROB_COLS = tuple(f"p_{o}" for o in OUTCOMES)

# Runner-advancement probabilities on base hits. These are the calibration knobs; the defaults
# roughly match league baserunning and put a league-average lineup near 4.4 R/G.
DEFAULT_ADVANCEMENT = {
    "from2_scores_on_1b": 0.68,   # runner on 2nd scores on a single (else stops at 3rd)
    "from1_to3_on_1b": 0.42,      # runner on 1st reaches 3rd on a single (else stops at 2nd)
    "from1_scores_on_2b": 0.55,   # runner on 1st scores on a double (else stops at 3rd)
}


@dataclass
class Batter:
    """One hitter's PA outcome distribution plus steal rates.

    `cdf` is the cumulative distribution over OUTCOMES, precomputed for fast sampling.
    """

    name: str
    probs: tuple            # probability per OUTCOMES index (sums to ~1)
    sb_rate: float          # P(steal 2nd | on first, 2nd open)
    cs_rate: float          # P(caught stealing | on first, 2nd open)
    cdf: tuple = field(init=False)

    def __post_init__(self):
        total = 0.0
        cdf = []
        for p in self.probs:
            total += p
            cdf.append(total)
        cdf[-1] = 1.0           # guard against float drift so sampling always resolves
        self.cdf = tuple(cdf)

    @classmethod
    def from_row(cls, row):
        """Build from a hitter_outcomes_2026.csv row (a pandas Series or mapping)."""
        return cls(
            name=row["name"],
            probs=tuple(float(row[c]) for c in PROB_COLS),
            sb_rate=float(row["sb_rate"]),
            cs_rate=float(row["cs_rate"]),
        )


def league_average_batter(outcomes_df, name="LeagueAvg"):
    """A single Batter whose rates are the PA-weighted league means (for calibration)."""
    w = outcomes_df["pa"]
    probs = tuple((outcomes_df[c] * w).sum() / w.sum() for c in PROB_COLS)
    sb = (outcomes_df["sb_rate"] * w).sum() / w.sum()
    cs = (outcomes_df["cs_rate"] * w).sum() / w.sum()
    return Batter(name=name, probs=probs, sb_rate=sb, cs_rate=cs)


class LineupSimulator:
    """Simulates innings/games for a 9-batter lineup on a base-out state machine."""

    def __init__(self, seed=None, advance=None):
        self._rand = random.Random(seed)
        a = DEFAULT_ADVANCEMENT if advance is None else advance
        self.p_from2_scores_on_1b = a["from2_scores_on_1b"]
        self.p_from1_to3_on_1b = a["from1_to3_on_1b"]
        self.p_from1_scores_on_2b = a["from1_scores_on_2b"]

    # -- single plate appearance -------------------------------------------------------------
    def simulate_pa(self, batter, bases, outs):
        """Resolve one PA. `bases` is a mutable [on1, on2, on3] list; returns (bases, outs, runs).

        The list is mutated in place and also returned, so callers can thread state through a
        loop cheaply.
        """
        rnd = self._rand.random
        u = rnd()
        cdf = batter.cdf
        o = 0
        while u > cdf[o]:
            o += 1

        b1, b2, b3 = bases
        runs = 0

        if o == O_OUT or o == O_K:
            outs += 1
            return bases, outs, runs

        if o == O_1B:
            # Batter to first; existing runners advance (lead runner first to avoid collisions).
            n1, n2, n3 = True, False, False
            if b3:
                runs += 1
            if b2:
                if rnd() < self.p_from2_scores_on_1b:
                    runs += 1
                else:
                    n3 = True
            if b1:
                if rnd() < self.p_from1_to3_on_1b and not n3:
                    n3 = True
                else:
                    n2 = True
            bases[0], bases[1], bases[2] = n1, n2, n3
            outs += self._maybe_steal(batter, bases)
            return bases, outs, runs

        if o == O_2B:
            n2, n3 = True, False
            if b3:
                runs += 1
            if b2:
                runs += 1
            if b1:
                if rnd() < self.p_from1_scores_on_2b:
                    runs += 1
                else:
                    n3 = True
            bases[0], bases[1], bases[2] = False, n2, n3
            return bases, outs, runs

        if o == O_3B:
            runs += b1 + b2 + b3
            bases[0], bases[1], bases[2] = False, False, True
            return bases, outs, runs

        if o == O_HR:
            runs += 1 + b1 + b2 + b3
            bases[0], bases[1], bases[2] = False, False, False
            return bases, outs, runs

        if o == O_BB:
            # Force advances only: runners move only when pushed by the batter behind them.
            if b1:
                if b2:
                    if b3:
                        runs += 1        # bases loaded: runner on third forced home
                    bases[2] = True      # runner 2 -> 3
                bases[1] = True          # runner 1 -> 2
            bases[0] = True              # batter -> 1
            outs += self._maybe_steal(batter, bases)
            return bases, outs, runs

        if o == O_DP:
            # Grounded double play: needs a runner on first and a spare out.
            if b1 and outs < 2:
                outs += 2
                bases[0] = False         # lead runner (on first) and batter both retired
            else:
                outs += 1                # no one to double off -> ordinary out
            return bases, outs, runs

        if o == O_SF:
            # Sac fly only "works" with a runner on third and fewer than two outs.
            if b3 and outs < 2:
                runs += 1
                bases[2] = False
            outs += 1
            return bases, outs, runs

        # O_SH: sacrifice bunt -- give up an out to move every runner up one base.
        if outs < 2:
            if b3:
                runs += 1
            bases[0], bases[1], bases[2] = False, b1, b2
        outs += 1
        return bases, outs, runs

    def _maybe_steal(self, batter, bases):
        """Roll a steal attempt when the batter is on first with second open. Returns added outs."""
        if bases[0] and not bases[1]:
            u = self._rand.random()
            if u < batter.sb_rate:
                bases[0], bases[1] = False, True          # safe at second
            elif u < batter.sb_rate + batter.cs_rate:
                bases[0] = False                          # caught stealing
                return 1
        return 0

    # -- innings and games -------------------------------------------------------------------
    def simulate_inning(self, lineup, start_idx):
        """Bat until three outs. Returns (runs, next_batter_index)."""
        outs = 0
        runs = 0
        bases = [False, False, False]
        idx = start_idx
        pa = self.simulate_pa
        while outs < 3:
            _, outs, r = pa(lineup[idx], bases, outs)
            runs += r
            idx = idx + 1 if idx < 8 else 0
        return runs, idx

    def simulate_game(self, lineup, innings=9):
        """One game: sum runs over `innings`, carrying the batting-order pointer across innings."""
        runs = 0
        idx = 0
        inning = self.simulate_inning
        for _ in range(innings):
            r, idx = inning(lineup, idx)
            runs += r
        return runs

    def simulate_games(self, lineup, n, innings=9):
        """Return an array of per-game runs over `n` games (caller takes mean/std/CI)."""
        game = self.simulate_game
        return np.fromiter((game(lineup, innings) for _ in range(n)), dtype=np.int32, count=n)
