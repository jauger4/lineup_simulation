"""Unit tests for the base-out transitions in lineup_sim.py.

Runnable two ways:
    pytest test_lineup_sim.py
    python  test_lineup_sim.py      (falls back to a tiny built-in runner if pytest is absent)

The tests pin down the deterministic parts of simulate_pa (bases-loaded homer scores four, a
double play with a man on first removes two, a sac fly scores from third, etc.). Probabilistic
runner advancement is checked only where the outcome is forced (e.g. HR/triple clear the bases).
"""

from lineup_sim import (
    LineupSimulator, Batter, OUTCOMES,
    O_1B, O_2B, O_3B, O_HR, O_BB, O_K, O_DP, O_OUT, O_SF, O_SH,
)


def _batter_forcing(outcome_idx):
    """A batter who always produces the given outcome (probability 1.0 on that index)."""
    probs = [0.0] * len(OUTCOMES)
    probs[outcome_idx] = 1.0
    return Batter(name=OUTCOMES[outcome_idx], probs=tuple(probs), sb_rate=0.0, cs_rate=0.0)


def _pa(outcome_idx, bases, outs):
    """Run one deterministic PA and return (bases, outs, runs)."""
    sim = LineupSimulator(seed=0)
    return sim.simulate_pa(_batter_forcing(outcome_idx), list(bases), outs)


def test_home_run_bases_loaded_scores_four():
    bases, outs, runs = _pa(O_HR, [True, True, True], 0)
    assert runs == 4
    assert bases == [False, False, False]
    assert outs == 0


def test_solo_home_run():
    bases, outs, runs = _pa(O_HR, [False, False, False], 1)
    assert runs == 1 and bases == [False, False, False] and outs == 1


def test_triple_clears_bases():
    bases, outs, runs = _pa(O_3B, [True, False, True], 0)
    assert runs == 2 and bases == [False, False, True]


def test_double_play_removes_two_outs_and_lead_runner():
    bases, outs, runs = _pa(O_DP, [True, False, False], 0)
    assert outs == 2 and bases == [False, False, False] and runs == 0


def test_double_play_without_runner_on_first_is_ordinary_out():
    bases, outs, runs = _pa(O_DP, [False, True, False], 0)
    assert outs == 1 and bases == [False, True, False]


def test_double_play_with_two_outs_is_ordinary_out():
    bases, outs, runs = _pa(O_DP, [True, False, False], 2)
    assert outs == 3 and bases == [True, False, False]


def test_sac_fly_scores_from_third():
    bases, outs, runs = _pa(O_SF, [False, False, True], 1)
    assert runs == 1 and outs == 2 and bases == [False, False, False]


def test_sac_fly_without_runner_on_third_is_just_an_out():
    bases, outs, runs = _pa(O_SF, [True, False, False], 0)
    assert runs == 0 and outs == 1 and bases == [True, False, False]


def test_sac_bunt_advances_every_runner_one_base():
    bases, outs, runs = _pa(O_SH, [True, True, False], 0)
    assert outs == 1 and bases == [False, True, True] and runs == 0


def test_walk_forces_runner_home_only_when_loaded():
    # Bases loaded: batter walk forces the run in.
    bases, outs, runs = _pa(O_BB, [True, True, True], 0)
    assert runs == 1 and bases == [True, True, True]
    # Runner on second only: no force, runner holds.
    bases, outs, runs = _pa(O_BB, [False, True, False], 0)
    assert runs == 0 and bases == [True, True, False]


def test_strikeout_and_generic_out_just_add_an_out():
    for o in (O_K, O_OUT):
        bases, outs, runs = _pa(o, [True, False, True], 1)
        assert outs == 2 and runs == 0 and bases == [True, False, True]


def test_single_batter_reaches_first_runner_on_third_scores():
    # No prior runner on first/second -> no probabilistic branch; man on third scores.
    bases, outs, runs = _pa(O_1B, [False, False, True], 0)
    assert runs == 1 and bases[0] is True and bases[2] is False


def test_probabilities_partition_the_pa():
    b = _batter_forcing(O_1B)
    assert abs(b.cdf[-1] - 1.0) < 1e-12


def _run_standalone():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"  ok  {fn.__name__}")
    print(f"\n{passed}/{len(fns)} tests passed")


if __name__ == "__main__":
    _run_standalone()
