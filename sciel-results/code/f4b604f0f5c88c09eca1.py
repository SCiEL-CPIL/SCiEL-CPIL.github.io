"""Difference triangle set (4,7) solver.
We keep the greedy-ruler family but make it much stronger:
- deterministic fast greedy baseline for a valid fallback,
- local-search / hill-climbing over the set of rows to reduce the largest entry,
- simulated-annealing style restarts within the same family,
- explicit wall-clock deadline guard.
"""
import time
import random
import math

_DEADLINE = 140.0  # seconds; safety margin under the 150 s kill
_START = None


def _elapsed():
    return time.monotonic() - _START


def _row_diffs(row):
    # return the set of positive differences within a row
    d = set()
    n = len(row)
    for i in range(n):
        ai = row[i]
        for j in range(i + 1, n):
            d.add(row[j] - ai)
    return d


def _all_diffs(rows):
    used = set()
    for r in rows:
        used |= _row_diffs(r)
    return used


def _is_valid(rows):
    used = set()
    for r in rows:
        rd = _row_diffs(r)
        if rd & used:
            return False
        used |= rd
    return True


def _max_entry(rows):
    return max(max(r) for r in rows)


def _greedy_row(used, k):
    row = [0]
    x = 0
    while len(row) < k + 1:
        x += 1
        new = {x - a for a in row}
        if not (new & used):
            row.append(x)
            used |= new
    return row


def _greedy_baseline(n_rows, k):
    used = set()
    rows = []
    for _ in range(n_rows):
        rows.append(_greedy_row(used, k))
    return rows


def _perturb_row(row, k, rng):
    """Randomly perturb a row, keeping it strictly increasing starting at 0,
    and keeping values bounded loosely by the current max plus a slack."""
    n = len(row)
    cur_max = row[-1]
    # decide a new target max in a window
    slack = max(5, cur_max // 10)
    new_max = max(k, min(cur_max + slack, cur_max + 2 * slack))
    new_max = max(new_max, k)
    # sample k-1 interior points strictly between 0 and new_max
    if k >= 1:
        interiors = sorted(rng.sample(range(1, new_max), min(k - 1, max(0, new_max - 1))))
    else:
        interiors = []
    new_row = [0] + interiors
    if len(new_row) < k + 1:
        # pad up to new_max if needed (shouldn't happen if new_max > k)
        # but ensure length
        x = new_row[-1] if new_row else 0
        while len(new_row) < k + 1:
            x += 1
            new_row.append(x)
    return new_row[:k + 1]


def _local_search(rows, n_rows, k, rng, time_limit_frac):
    """Hill-climbing / simulated annealing on the set of rows.
    Objective: minimize max entry subject to validity.
    Moves: replace one row with a perturbation; accept if valid and improves or equal (with SA noise)."""
    best_rows = [list(r) for r in rows]
    best_score = _max_entry(best_rows)
    current_rows = [list(r) for r in rows]
    current_score = _max_entry(current_rows)

    # simple SA: temperature decays; we only accept valid states.
    t0 = 10.0
    t = t0
    cooling = 0.999
    steps = 0
    max_steps = 200000
    time_budget = time_limit_frac * 120.0

    while steps < max_steps and _elapsed() < _DEADLINE - 5:
        # pick a row to perturb
        idx = rng.randrange(n_rows)
        new_row = _perturb_row(current_rows[idx], k, rng)
        trial = [list(r) for r in current_rows]
        trial[idx] = new_row
        if _is_valid(trial):
            trial_score = _max_entry(trial)
            delta = trial_score - current_score
            if delta <= 0 or (t > 0.01 and rng.random() < math.exp(-delta / t)):
                current_rows = trial
                current_score = trial_score
                if current_score < best_score:
                    best_score = current_score
                    best_rows = [list(r) for r in current_rows]
        else:
            # invalid: reject
            pass
        t *= cooling
        steps += 1

    return best_rows, best_score


def _try_improve_with_orderings(n_rows, k, base_rows, rng, time_budget):
    """Try different row orderings for the greedy construction; keep best."""
    best = [list(r) for r in base_rows]
    best_score = _max_entry(best)
    # try a few random permutations of row construction order
    orderings = [list(range(n_rows))]
    for _ in range(min(20, max(1, int(time_budget / 0.2)))):
        perm = list(range(n_rows))
        rng.shuffle(perm)
        orderings.append(perm)

    for perm in orderings:
        if _elapsed() > _DEADLINE - 5:
            break
        used = set()
        rows = [None] * n_rows
        for pos in perm:
            rows[pos] = _greedy_row(used, k)
        if _is_valid(rows):
            sc = _max_entry(rows)
            if sc < best_score:
                best_score = sc
                best = [list(r) for r in rows]
    return best, best_score


def construct(n_rows, k):
    global _START
    _START = time.monotonic()
    rng = random.Random(12345)

    # 1) fast greedy baseline
    base_rows = _greedy_baseline(n_rows, k)
    base_score = _max_entry(base_rows)

    # 2) try row-order permutations (still greedy family)
    best_rows, best_score = _try_improve_with_orderings(n_rows, k, base_rows, rng, 2.0)

    # 3) local search / SA to reduce max entry
    if _elapsed() < _DEADLINE - 10:
        ls_rows, ls_score = _local_search(base_rows, n_rows, k, rng, 0.9)
        if ls_score < best_score:
            best_score = ls_score
            best_rows = ls_rows

    # 4) a couple of fresh restarts with randomized greedy seeds if time permits
    restarts = 0
    while _elapsed() < _DEADLINE - 8 and restarts < 5:
        # randomize by perturbing the greedy baseline a bit before LS
        seed_rows = [list(r) for r in base_rows]
        for _ in range(n_rows):
            if rng.random() < 0.3:
                seed_rows[_] = _perturb_row(seed_rows[_], k, rng)
        if not _is_valid(seed_rows):
            # repair by re-greedy for invalid rows
            used = set()
            for i in range(n_rows):
                if not _is_valid([seed_rows[i]] + [seed_rows[j] for j in range(i)]):
                    pass
            # simpler: just use base
            seed_rows = [list(r) for r in base_rows]
        if _elapsed() < _DEADLINE - 10:
            ls_rows, ls_score = _local_search(seed_rows, n_rows, k, rng, 0.5)
            if ls_score < best_score:
                best_score = ls_score
                best_rows = ls_rows
        restarts += 1

    # ensure validity of the returned best
    if not _is_valid(best_rows):
        best_rows = _greedy_baseline(n_rows, k)

    return [list(r) for r in best_rows]
