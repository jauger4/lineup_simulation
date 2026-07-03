"""Interactive dashboard: real vs. optimized batting order, per team.

Final stage of the lineup-optimization project (see instructions.md / CLAUDE.md). This is a thin
Streamlit front-end over `optimal_lineups_2026.csv` (produced by optimize_lineups.py): pick a team,
see its real lineup next to the run-maximizing order the Monte Carlo search found, and how many more
runs per game that buys -- but only claim a gain when the difference is statistically significant.

No simulation runs here; everything is a read of the precomputed CSVs, so the app is instant.

Styling: a dark "Baseball Savant" look (theme in .streamlit/config.toml) with custom HTML lineup
cards. Each hitter's stat chips are heat-shaded by percentile vs. the qualified hitter population --
red = hot, blue = cold (K% is inverted, since a high strikeout rate is a negative).

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

# Savant-style diverging heat scale: cold (blue) -> neutral slate -> hot (red).
HEAT_COLD = (54, 97, 173)
HEAT_MID = (107, 114, 128)
HEAT_HOT = (210, 45, 73)
CHIP_PLAIN_BG = "#222c44"

# The stat a user can switch between; one chip is shown per hitter for the selected stat.
# (label, column, format, invert_heat): invert = higher is worse, so it shades cold (K%).
STAT_SPECS = [
    ("OBP", "obp", "{:.3f}", False),
    ("Barrel%", "barrel_pct", "{:.1f}", False),
    ("BB%", "bb_pct", "{:.1f}", False),
    ("K%", "k_pct", "{:.1f}", True),
]
STAT_BY_LABEL = {label: (col, fmt, invert) for label, col, fmt, invert in STAT_SPECS}
STAT_LABELS = [label for label, *_ in STAT_SPECS]
HEAT_COLS = [col for _, col, _, _ in STAT_SPECS]

st.set_page_config(page_title="Lineup Lab", page_icon="⚾", layout="wide")


@st.cache_data
def load_data():
    """Load the optimizer output and a per-hitter stat table keyed by MLBAM id."""
    opt = pd.read_csv(OPT_CSV)
    outcomes = pd.read_csv(OUTCOMES_CSV)[["batter", "name", "obp"]]
    stats = pd.read_csv(STATS_CSV)[["batter", "barrel_pct", "bb_pct", "k_pct"]]
    hitters = outcomes.merge(stats, on="batter", how="left").set_index("batter")
    return opt, hitters


@st.cache_data
def percentiles(hitters):
    """Percentile rank (0-1) of each heat stat across the full population, keyed by batter id."""
    return {col: hitters[col].rank(pct=True) for col in HEAT_COLS}


def parse_pipe(value):
    """Split a pipe-joined CSV field into a list of strings."""
    return str(value).split("|")


def _lerp(a, b, t):
    return tuple(round(a[i] + (b[i] - a[i]) * t) for i in range(3))


def heat_color(pct, invert=False):
    """Hex color on the blue->slate->red scale for a percentile in [0, 1]."""
    if pct is None or pd.isna(pct):
        return CHIP_PLAIN_BG
    if invert:
        pct = 1.0 - pct
    if pct < 0.5:
        r, g, b = _lerp(HEAT_COLD, HEAT_MID, pct / 0.5)
    else:
        r, g, b = _lerp(HEAT_MID, HEAT_HOT, (pct - 0.5) / 0.5)
    return f"#{r:02x}{g:02x}{b:02x}"


def stat_chip(label, text, bg):
    return (f'<span class="chip" style="background:{bg}">'
            f'<span class="lbl">{label}</span>{text}</span>')


def hitter_chip(bid, hitters, pct, stat_label):
    """The single heat-shaded chip for one hitter and the currently selected stat.

    'None' ids (League-avg padding) get a neutral placeholder chip.
    """
    col, fmt, invert = STAT_BY_LABEL[stat_label]
    if bid == "None" or int(bid) not in hitters.index:
        return stat_chip(stat_label, "—", CHIP_PLAIN_BG)
    h = hitters.loc[int(bid)]
    value = h[col]
    bg = heat_color(pct[col].get(int(bid)), invert)
    return stat_chip(stat_label, fmt.format(value), bg)


def hitter_name(bid, hitters):
    if bid == "None" or int(bid) not in hitters.index:
        return "League avg"
    return hitters.loc[int(bid), "name"]


def move_badge(shift):
    if shift > 0:
        return f'<span class="mv up">▲{shift}</span>'
    if shift < 0:
        return f'<span class="mv down">▼{-shift}</span>'
    return '<span class="mv same">—</span>'


def render_lineup_card(team, kind, ids, hitters, pct, stat_label, from_slots=None):
    """Build the HTML for one lineup card (kind = 'real' or 'optimized').

    Each hitter row shows a single heat-shaded chip for `stat_label` (the stat the user
    picked in the switcher), keeping the card readable.
    """
    title_cls = "opt" if kind == "optimized" else "real"
    title = "OPTIMIZED LINEUP" if kind == "optimized" else "REAL LINEUP"
    parts = [f'<div class="card"><div class="card-title {title_cls}">{title}'
             f'<small>{team}</small></div>']
    for pos, bid in enumerate(ids, start=1):
        badge = ""
        if from_slots is not None:
            src = from_slots[pos - 1]
            badge = move_badge(src - pos)
        parts.append(
            f'<div class="lrow"><div class="slot">{pos}</div>'
            f'<div class="pname">{hitter_name(bid, hitters)}</div>'
            f'<div class="chips">{hitter_chip(bid, hitters, pct, stat_label)}</div>{badge}</div>'
        )
    parts.append("</div>")
    return "".join(parts)


CARD_CSS = """
<style>
.block-container {padding-top: 2.2rem;}
#MainMenu, footer {visibility: hidden;}
.lab-header {border-bottom: 3px solid #d22d49; padding: 6px 0 10px; margin-bottom: 14px;}
.lab-header h1 {margin: 0; font-size: 1.9rem; font-weight: 800; letter-spacing: .02em; color: #fff;}
.lab-header span {color: #d22d49;}
.lab-header p {margin: 2px 0 0; color: #9aa6be; font-size: .9rem;}
.card {background:#141a2b;border:1px solid #263149;border-radius:12px;overflow:hidden;margin-bottom:10px;}
.card-title {padding:10px 14px;font-weight:800;font-size:1.0rem;letter-spacing:.05em;color:#fff;}
.card-title.real {background:#2a3350;}
.card-title.opt {background:#d22d49;}
.card-title small {display:block;font-weight:600;font-size:.68rem;opacity:.85;letter-spacing:.1em;margin-top:1px;}
.lrow {display:flex;align-items:center;gap:10px;padding:7px 12px;border-top:1px solid #1e2740;}
.slot {flex:0 0 30px;height:30px;line-height:28px;text-align:center;border-radius:50%;
       background:#0e1220;border:1px solid #33405f;font-weight:700;color:#c9d3e6;}
.pname {flex:1 1 auto;font-weight:600;color:#eef1f8;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;}
.chips {flex:0 0 auto;display:flex;gap:4px;}
.chip {min-width:46px;text-align:center;border-radius:6px;padding:3px 6px;font-size:.74rem;font-weight:600;color:#fff;}
.chip .lbl {display:block;font-size:.56rem;font-weight:600;opacity:.8;letter-spacing:.03em;}
.mv {flex:0 0 34px;text-align:right;font-size:.8rem;font-weight:700;}
.mv.up{color:#3ddc84;} .mv.down{color:#ff6b6b;} .mv.same{color:#5c6780;}
.legend {display:flex;align-items:center;gap:10px;color:#9aa6be;font-size:.8rem;margin:2px 0 6px;}
.legend .bar {flex:0 0 200px;height:12px;border-radius:6px;
              background:linear-gradient(90deg,#3661ad,#6b7280,#d22d49);}
</style>
"""


# --- app -----------------------------------------------------------------------------------------
opt, hitters = load_data()
pct = percentiles(hitters)

st.markdown(CARD_CSS, unsafe_allow_html=True)
st.markdown(
    '<div class="lab-header"><h1>⚾ LINEUP <span>LAB</span> — 2026</h1>'
    '<p>Real lineup vs. the run-maximizing order our Monte Carlo simulation found.</p></div>',
    unsafe_allow_html=True,
)

team = st.selectbox("Team", sorted(opt["team"]), index=0)
stat_label = st.segmented_control(
    "Stat", STAT_LABELS, default=STAT_LABELS[0], selection_mode="single",
    help="Switch which stat the heat-shaded chip next to each hitter shows.",
)
if stat_label is None:            # user cleared the selection; fall back to the first stat
    stat_label = STAT_LABELS[0]
row = opt[opt["team"] == team].iloc[0]
significant = row["p_value"] < SIG_ALPHA

# Headline numbers.
m1, m2, m3 = st.columns(3)
m1.metric("Real R/G", f"{row['baseline_rpg']:.3f}")
m2.metric("Optimized R/G", f"{row['best_rpg']:.3f}",
          delta=f"{row['delta_rpg']:+.3f}" if significant else None)
m3.metric("Significance (p-value)", f"{row['p_value']:.4f}",
          delta="significant" if significant else "not significant", delta_color="off")

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

# Color legend for the selected stat's heat scale.
inverted_note = " (K%: high strikeouts = cold)" if stat_label == "K%" else ""
st.markdown(
    f'<div class="legend"><span>cold</span><div class="bar"></div><span>hot &nbsp;·&nbsp; '
    f'{stat_label} percentile vs. league{inverted_note}</span></div>',
    unsafe_allow_html=True,
)

# Side-by-side lineup cards.
left, right = st.columns(2)
with left:
    st.markdown(
        render_lineup_card(team, "real", parse_pipe(row["baseline_ids"]), hitters, pct, stat_label),
        unsafe_allow_html=True,
    )
with right:
    from_slots = [int(s) for s in parse_pipe(row["best_from_slots"])]
    st.markdown(
        render_lineup_card(team, "optimized", parse_pipe(row["best_ids"]), hitters, pct, stat_label,
                           from_slots=from_slots),
        unsafe_allow_html=True,
    )

if row["n_padded"] > 0:
    st.caption(
        f"Note: {team} had fewer than 9 qualified hitters (PA ≥ 150); "
        f"{int(row['n_padded'])} slot(s) padded with a league-average batter."
    )

st.caption(
    "Model note: each plate appearance is simulated independently (no count, pitcher, or "
    "lineup-protection effects), so batting-order gains are small and directional."
)
