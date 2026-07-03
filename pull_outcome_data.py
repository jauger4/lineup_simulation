"""Build each 2026 hitter's per-plate-appearance outcome probabilities for the simulator.

Fourth data step of the lineup-optimization project (see instructions.md). The clustering
work established that hitter approach is a continuum, so the project now moves to the Monte
Carlo simulation. A base-out simulator needs, for every hitter, the probability that a plate
appearance ends in each discrete outcome (single, double, ..., strikeout, double play, sac).
The approach stats in hitter_stats_2026.csv do not carry that; this table does.

Source: season counting stats from Baseball Reference via pybaseball.batting_stats_bref.
FanGraphs (batting_stats) is currently blocked (HTTP 403), and reconstructing SB/CS/sac/GDP
from the Statcast pitch data is unreliable, so bref is the pragmatic source here. It already
carries an mlbID column, so the result merges by MLBAM id with the other *_2026.csv products
(no name join needed). bref also collapses traded players to one row per player.

Each row's ten p_* columns partition the plate appearance (they sum to 1):
    PA = AB + BB + HBP + SF + SH
       = (1B + 2B + 3B + HR) + SO + GDP + other_batted_outs + (BB + HBP) + SF + SH
so p_1b + p_2b + p_3b + p_hr + p_bb + p_k + p_dp + p_out + p_sf + p_sh == 1.
    - p_bb folds hit-by-pitch (and the rare catcher-interference reach) into walks; all are
      force-advance-only on-base outcomes. It is taken as the residual 1 - (all other p_*) so
      the row partitions exactly.
    - p_out is non-strikeout, non-GDP batted-ball outs (AB - H - SO - GDP); this also absorbs
      reached-on-error, a small simplification.
Stolen bases are expressed as a rate *per time reached first base* (~1B + BB + HBP), which is
how the simulator applies them: roll a steal attempt when a batter is standing on first.

Run:  python pull_outcome_data.py
"""

import pandas as pd
from pybaseball import batting_stats_bref, cache

cache.enable()

SEASON = 2026
MIN_PA = 150                 # project-wide qualified-hitter cutoff (see CLAUDE.md)
OUTPUT_CSV = "hitter_outcomes_2026.csv"


def fix_name(s):
    r"""Repair bref names that arrive as a bytes-repr string (e.g. 'Hern\xc3\xa1ndez').

    The '\xHH' pieces are the literal UTF-8 bytes of an accented character; rebuild the byte
    sequence and decode it. Plain-ASCII names pass through untouched.
    """
    if not isinstance(s, str) or "\\x" not in s:
        return s
    out = bytearray()
    i = 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s) and s[i + 1] == "x":
            out.append(int(s[i + 2:i + 4], 16))
            i += 4
        else:
            out.append(ord(s[i]))
            i += 1
    try:
        return out.decode("utf-8")
    except UnicodeDecodeError:
        return s


def pull_season_stats(season):
    """Pull season batting counting stats, keep major-league qualified hitters."""
    df = batting_stats_bref(season)
    df = df[df["Lev"].str.startswith("Maj")].copy()   # drop any minor-league lines
    df = df[df["PA"] >= MIN_PA].copy()
    df["Name"] = df["Name"].map(fix_name)
    return df


def compute_outcome_probs(df):
    """Turn counting stats into per-PA outcome probabilities that partition the PA."""
    pa = df["PA"]
    singles = df["H"] - df["2B"] - df["3B"] - df["HR"]
    other_outs = df["AB"] - df["H"] - df["SO"] - df["GDP"]   # batted outs minus DPs

    out = pd.DataFrame({
        "batter": df["mlbID"].astype(int),
        "name": df["Name"],
        "team": df["Tm"],
        "pa": pa,
        "avg": df["BA"],
        "obp": df["OBP"],
        "p_1b": singles / pa,
        "p_2b": df["2B"] / pa,
        "p_3b": df["3B"] / pa,
        "p_hr": df["HR"] / pa,
        "p_k": df["SO"] / pa,
        "p_dp": df["GDP"] / pa,
        "p_out": other_outs / pa,
        "p_sf": df["SF"] / pa,
        "p_sh": df["SH"] / pa,
    })
    # BB (+HBP +interference) is the residual so the row partitions the PA exactly.
    non_bb = ["p_1b", "p_2b", "p_3b", "p_hr", "p_k", "p_dp", "p_out", "p_sf", "p_sh"]
    out["p_bb"] = 1.0 - out[non_bb].sum(axis=1)

    # Force-advance-only walks (excl. interference) drive the on-first steal opportunity.
    walks = df["BB"] + df["HBP"]

    # Steal rates are per time reached first base, the state where the sim applies them.
    times_on_first = singles + walks
    out["sb_rate"] = (df["SB"] / times_on_first).where(times_on_first > 0, 0.0)
    out["cs_rate"] = (df["CS"] / times_on_first).where(times_on_first > 0, 0.0)

    cols = [
        "batter", "name", "team", "pa", "avg", "obp",
        "p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_k", "p_dp", "p_out", "p_sf", "p_sh",
        "sb_rate", "cs_rate",
    ]
    return out[cols].reset_index(drop=True)


def main():
    df = pull_season_stats(SEASON)
    out = compute_outcome_probs(df)

    prob_cols = ["p_1b", "p_2b", "p_3b", "p_hr", "p_bb", "p_k", "p_dp", "p_out", "p_sf", "p_sh"]
    row_sums = out[prob_cols].sum(axis=1)
    assert (row_sums.sub(1.0).abs() < 1e-9).all(), "outcome probabilities must sum to 1 per row"

    out = out.sort_values("pa", ascending=False).reset_index(drop=True)
    out.to_csv(OUTPUT_CSV, index=False, encoding="utf-8")
    print(f"Hitters: {len(out)} | saved to {OUTPUT_CSV}")
    print(out.head(12).to_string(index=False))


if __name__ == "__main__":
    main()
