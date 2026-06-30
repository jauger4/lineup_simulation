"""Pull 2026 Statcast pitch-by-pitch data and build a per-hitter stats table.

This is the first step of the lineup-optimization project (see instructions.md). It
produces the per-hitter table used for clustering hitters by plate approach. The five
clustering stats are strikeout%, walk%, chase%, whiff%, and barrel%, plus a plate-
appearance (PA) count so unqualified hitters can be filtered out in a later step.

Data is pulled from Baseball Savant via pybaseball. The resulting table is written to
CSV so it can be loaded for exploration (e.g. in a Jupyter notebook) without re-pulling.

Run:  python pull_hitter_data.py
"""

from pybaseball import (
    statcast,
    statcast_batter_exitvelo_barrels,
    playerid_reverse_lookup,
    cache,
)

# Cache the (large) season pull so it is only fetched from Savant once.
cache.enable()

SEASON = 2026
START = "2026-03-01"          # before opening day; non-regular games filtered out below
END = "2026-06-30"
OUTPUT_CSV = "hitter_stats_2026.csv"

# A pitch is a "swing" if the batter offered at it (contact or miss).
SWING_DESCRIPTIONS = {
    "foul",
    "foul_bunt",
    "foul_tip",
    "hit_into_play",
    "swinging_strike",
    "swinging_strike_blocked",
    "missed_bunt",
}
# A "whiff" is a swing and miss. Foul tips count as a swing but not a miss (Savant convention).
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked"}
STRIKEOUT_EVENTS = {"strikeout", "strikeout_double_play"}


def pull_pitch_data(start, end):
    """Pull all pitch-by-pitch data in the date range, keeping only regular-season games."""
    df = statcast(start_dt=start, end_dt=end)
    df = df[df["game_type"] == "R"].copy()
    return df


def summarize_hitters(df):
    """Group pitches by batter and compute PA, K%, BB%, chase%, and whiff%."""
    df["is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["out_of_zone"] = df["zone"] > 9            # zones 1-9 are in-zone; 11-14 are out
    df["chase_swing"] = df["is_swing"] & df["out_of_zone"]
    df["is_k"] = df["events"].isin(STRIKEOUT_EVENTS)
    df["is_bb"] = df["events"] == "walk"

    summary = df.groupby("batter").agg(
        swings=("is_swing", "sum"),
        whiffs=("is_whiff", "sum"),
        out_of_zone_pitches=("out_of_zone", "sum"),
        chase_swings=("chase_swing", "sum"),
        strikeouts=("is_k", "sum"),
        walks=("is_bb", "sum"),
    )

    # A plate appearance is one unique (game, at-bat) pairing.
    pa = (
        df.drop_duplicates(["batter", "game_pk", "at_bat_number"])
        .groupby("batter")
        .size()
        .rename("pa")
    )
    summary = summary.join(pa)

    # Scale to 0-100 to match Savant's Barrel% convention.
    summary["k_pct"] = 100 * summary["strikeouts"] / summary["pa"]
    summary["bb_pct"] = 100 * summary["walks"] / summary["pa"]
    summary["chase_pct"] = 100 * summary["chase_swings"] / summary["out_of_zone_pitches"]
    summary["whiff_pct"] = 100 * summary["whiffs"] / summary["swings"]

    return summary.reset_index()


def add_barrel_pct(summary, season):
    """Merge in Savant's official Barrel% (barrels / batted-ball event) by batter id."""
    barrels = statcast_batter_exitvelo_barrels(season, minBBE=0)
    barrels = barrels[["player_id", "brl_percent"]].rename(
        columns={"player_id": "batter", "brl_percent": "barrel_pct"}
    )
    return summary.merge(barrels, on="batter", how="left")


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
    summary = summarize_hitters(df)
    summary = add_barrel_pct(summary, SEASON)
    summary = add_names(summary)

    cols = ["batter", "name", "pa", "k_pct", "bb_pct", "chase_pct", "whiff_pct", "barrel_pct"]
    summary = summary[cols].sort_values("pa", ascending=False).reset_index(drop=True)

    summary.to_csv(OUTPUT_CSV, index=False)
    print(f"Hitters: {len(summary)} | saved to {OUTPUT_CSV}")
    print(summary.head(15).to_string(index=False))


if __name__ == "__main__":
    main()
