"""Derive each 2026 hitter's typical batting-order slot from Statcast data.

Second data-pull step of the lineup-optimization project (see instructions.md). The
clustering step found no discrete hitter archetypes, so instead of inventing classes we
let real MLB lineup usage define the template: this script tags every plate appearance
with the batting-order slot it came from, then reduces to each hitter's most-frequent
(modal) starting slot for the season.

There is no batting-order column in Statcast, but it is recoverable from the same
pitch-by-pitch pull used by pull_hitter_data.py (cached, so this is a cache hit, not a
re-download): within a game, a team's plate appearances in at_bat_number order cycle
through the lineup, so the first 9 distinct batters are lineup spots 1-9 for that game.
The 2026 universal DH means slot 9 is a real hitter (pitchers do not bat).

Output columns: batter, name, primary_slot, games_started, slot_share, g_slot1..g_slot9.

Run:  python pull_lineup_data.py
"""

import pandas as pd
from pybaseball import statcast, playerid_reverse_lookup, cache

# Cache the (large) season pull so it is only fetched from Savant once. Same call as
# pull_hitter_data.py, so this reuses the cached data rather than re-downloading.
cache.enable()

SEASON = 2026
START = "2026-03-01"          # before opening day; non-regular games filtered out below
END = "2026-06-30"
OUTPUT_CSV = "hitter_lineup_2026.csv"

SLOTS = range(1, 10)          # a batting order has 9 spots


def pull_pitch_data(start, end):
    """Pull all pitch-by-pitch data in the date range, keeping only regular-season games."""
    df = statcast(start_dt=start, end_dt=end)
    df = df[df["game_type"] == "R"].copy()
    return df


def assign_slots(df):
    """Tag each plate appearance with its batting-order slot (1-9), per game and team.

    The home team bats in the bottom of the inning, the away team in the top. Within a
    (game, batting team), the first 9 distinct batters in at_bat_number order are the
    starting lineup spots 1-9; later batters are repeats or substitutions and are dropped.
    """
    df["batting_team"] = df["home_team"].where(df["inning_topbot"] == "Bot", df["away_team"])

    # One row per plate appearance (at_bat_number is unique within a game).
    pa = (
        df.drop_duplicates(["game_pk", "at_bat_number"])
        .sort_values(["game_pk", "batting_team", "at_bat_number"])
    )

    # First appearance of each batter within a (game, team), in lineup order.
    first_app = pa.drop_duplicates(["game_pk", "batting_team", "batter"]).copy()
    first_app["slot"] = first_app.groupby(["game_pk", "batting_team"]).cumcount() + 1

    # Keep only the starting nine.
    return first_app[first_app["slot"] <= 9][["batter", "game_pk", "slot"]]


def summarize_slots(starters):
    """Reduce per-game slot assignments to one row per batter with the modal slot."""
    # Games at each (batter, slot).
    counts = starters.groupby(["batter", "slot"]).size().rename("g").reset_index()

    # Wide games-per-slot table (g_slot1..g_slot9), zero-filled.
    wide = (
        counts.pivot(index="batter", columns="slot", values="g")
        .reindex(columns=SLOTS)
        .fillna(0)
        .astype(int)
    )
    wide.columns = [f"g_slot{s}" for s in wide.columns]

    # Modal slot = slot with the most starts (ties break to the lower slot).
    primary = (
        counts.loc[counts.groupby("batter")["g"].idxmax()]
        .set_index("batter")["slot"]
        .rename("primary_slot")
    )

    out = wide.join(primary)
    out["games_started"] = wide.sum(axis=1)
    out["slot_share"] = (
        out.apply(lambda r: r[f"g_slot{int(r['primary_slot'])}"], axis=1) / out["games_started"]
    )
    return out.reset_index()


def add_names(summary):
    """Resolve batter MLBAM ids to readable names (bulk player_name is the pitcher)."""
    lookup = playerid_reverse_lookup(summary["batter"].tolist(), key_type="mlbam")
    lookup["name"] = (
        lookup["name_first"].str.title() + " " + lookup["name_last"].str.title()
    )
    lookup = lookup[["key_mlbam", "name"]].rename(columns={"key_mlbam": "batter"})
    return summary.merge(lookup, on="batter", how="left")


def main():
    df = pull_pitch_data(START, END)
    starters = assign_slots(df)
    summary = summarize_slots(starters)
    summary = add_names(summary)

    cols = (
        ["batter", "name", "primary_slot", "games_started", "slot_share"]
        + [f"g_slot{s}" for s in SLOTS]
    )
    summary = summary[cols].sort_values("games_started", ascending=False).reset_index(drop=True)

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Hitters: {len(summary)} | saved to {OUTPUT_CSV}")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
