"""Interactive dashboard: real vs. optimized batting order, per team.

Final stage of the lineup-optimization project (see instructions.md / CLAUDE.md). This is a thin
Streamlit front-end over `optimal_lineups_2026.csv` (produced by optimize_lineups.py): pick a team,
see its real lineup next to the run-maximizing order the Monte Carlo search found, and how many more
runs per game that buys -- but only claim a gain when the difference is statistically significant.

No simulation runs here; everything is a read of the precomputed CSVs, so the app is instant.

Run:  streamlit run dashboard.py
"""

import pandas as pd
import streamlit as st

# --- config / formatting seams (tweak here when refining the look) --------------------------------
OPT_CSV = "optimal_lineups_2026.csv"
OUTCOMES_CSV = "hitter_outcomes_2026.csv"
STATS_CSV = "hitter_stats_2026.csv"
SIG_ALPHA = 0.05          # p-value threshold to claim a real run gain
GAMES_PER_SEASON = 162

st.set_page_config(page_title="Lineup Optimizer", page_icon="⚾", layout="wide")


@st.cache_data
def load_data():
    """Load the optimizer output and a per-hitter stat table keyed by MLBAM id."""
    opt = pd.read_csv(OPT_CSV)
    outcomes = pd.read_csv(OUTCOMES_CSV)[["batter", "name", "obp", "p_hr"]]
    stats = pd.read_csv(STATS_CSV)[["batter", "barrel_pct"]]
    hitters = outcomes.merge(stats, on="batter", how="left").set_index("batter")
    return opt, hitters


def parse_pipe(value):
    """Split a pipe-joined CSV field into a list of strings."""
    return str(value).split("|")


def build_lineup_frame(ids, hitters, from_slots=None):
    """One row per batting slot with the hitter's name and key stats.

    `ids` is the list of MLBAM ids in slot order ('None' marks a league-average padding slot).
    When `from_slots` is given (the optimized lineup), add a 'Moved' column showing where each
    hitter batted in the real order.
    """
    rows = []
    for pos, bid in enumerate(ids, start=1):
        row = {"Slot": pos, "Hitter": "League avg", "OBP": None, "HR%": None, "Barrel%": None}
        if bid != "None" and int(bid) in hitters.index:
            h = hitters.loc[int(bid)]
            row.update({"Hitter": h["name"], "OBP": h["obp"],
                        "HR%": h["p_hr"] * 100, "Barrel%": h["barrel_pct"]})
        if from_slots is not None:
            src = from_slots[pos - 1]           # real slot this hitter came from
            shift = src - pos                   # + = moved up the order, - = moved down
            row["Moved"] = "—" if shift == 0 else (f"↑{shift} (was {src})" if shift > 0
                                                   else f"↓{-shift} (was {src})")
        rows.append(row)
    return pd.DataFrame(rows)


COLUMN_CONFIG = {
    "Slot": st.column_config.NumberColumn(width="small"),
    "OBP": st.column_config.NumberColumn(format="%.3f"),
    "HR%": st.column_config.NumberColumn(format="%.1f"),
    "Barrel%": st.column_config.NumberColumn(format="%.1f"),
}


# --- app -----------------------------------------------------------------------------------------
opt, hitters = load_data()

st.title("⚾ Batting-order optimizer — 2026")
st.caption("Pick a team to see its real lineup vs. the order our Monte Carlo simulation scores highest.")

team = st.selectbox("Team", sorted(opt["team"]), index=0)
row = opt[opt["team"] == team].iloc[0]
significant = row["p_value"] < SIG_ALPHA

# Verdict + gated run gain.
if significant:
    st.success(
        f"**Optimized order scores +{row['delta_rpg']:.3f} runs/game** "
        f"(~{row['delta_rpg'] * GAMES_PER_SEASON:.0f} runs over 162) — "
        f"statistically significant (p = {row['p_value']:.4f})."
    )
else:
    st.info(
        f"The best order found adds only +{row['delta_rpg']:.3f} R/G, which is **not** statistically "
        f"distinguishable from the real order (p = {row['p_value']:.3f}). No reliable gain — this "
        f"team's real lineup is already about as good as it gets."
    )

# Headline numbers.
m1, m2, m3 = st.columns(3)
m1.metric("Real R/G", f"{row['baseline_rpg']:.3f}")
m2.metric("Optimized R/G", f"{row['best_rpg']:.3f}",
          delta=f"{row['delta_rpg']:+.3f}" if significant else None)
m3.metric("Significance (p-value)", f"{row['p_value']:.4f}",
          delta="significant" if significant else "not significant", delta_color="off")

# Side-by-side lineups.
left, right = st.columns(2)
with left:
    st.subheader("Real lineup")
    real_df = build_lineup_frame(parse_pipe(row["baseline_ids"]), hitters)
    st.dataframe(real_df, hide_index=True, column_config=COLUMN_CONFIG, width="stretch")
with right:
    st.subheader("Optimized lineup")
    from_slots = [int(s) for s in parse_pipe(row["best_from_slots"])]
    best_df = build_lineup_frame(parse_pipe(row["best_ids"]), hitters, from_slots=from_slots)
    st.dataframe(best_df, hide_index=True, column_config=COLUMN_CONFIG, width="stretch")

if row["n_padded"] > 0:
    st.caption(
        f"Note: {team} had fewer than 9 qualified hitters (PA ≥ 150); "
        f"{int(row['n_padded'])} slot(s) padded with a league-average batter."
    )

# Quick visual of the two run totals.
st.subheader("Runs per game")
st.bar_chart(pd.DataFrame({"Runs/game": [row["baseline_rpg"], row["best_rpg"]]},
                          index=["Real", "Optimized"]))

st.caption(
    "Model note: each plate appearance is simulated independently (no count, pitcher, or "
    "lineup-protection effects), so batting-order gains are small and directional."
)
