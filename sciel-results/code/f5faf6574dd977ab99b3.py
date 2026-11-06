"""Small inclusion-maximal Sidon sets in [1,256].

Strategy: build a valid Sidon set, then minimise its size while keeping it
inclusion-maximal.  Local moves are:
  * delete an element and re-greedy the missing elements (shrinks size),
  * swap an element for a missing one (keeps size, may enable later deletes).
A time budget with a safety margin guarantees a valid result is returned.
"""

import time
import random

N = 256
DEADLINE = 55.0          # seconds, well under the 60 s wall-clock limit
RNG = random.Random(20240611)


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------
def _sums_of(selected):
    """Return the set of all unordered pair sums a+b (a<=b) for a Sidon set."""
    sums = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            sums.add(a + selected[j])
    return sums


def _is_sidon(selected):
    """True iff all unordered pair sums a+b (a<=b) are distinct."""
    seen = set()
    for i, a in enumerate(selected):
        for j in range(i, len(selected)):
            s = a + selected[j]
            if s in seen:
                return False
            seen.add(s)
    return True


def _blocked_by(selected, sums):
    """Return a bool list: blocked[x] is True iff adding x to selected
    would create a duplicate sum (i.e. x is NOT addable)."""
    blocked = [False] * (N + 1)
    for x in range(1, N + 1):
        # candidate new sums: x + a for a in selected, plus 2*x
        # a duplicate occurs iff any of these is already in `sums`
        # (and the new sums among themselves are always distinct because
        #  x+a != x+b for a!=b, and 2x != x+a for a != x; and x is not
        #  in selected so x+a != 2x)
        for a in selected:
            if (x + a) in sums:
                blocked[x] = True
                break
        if not blocked[x] and (2 * x) in sums:
            blocked[x] = True
    return blocked


def _greedy_extend(selected, sums, order):
    """Add every addable element from `order` (a permutation of [1..N]
    minus selected) in that order, preserving the Sidon property.
    Returns (new_selected, new_sums)."""
    sel = list(selected)
    sm = set(sums)
    sel_set = set(sel)
    for x in order:
        if x in sel_set:
            continue
        ok = True
        for a in sel:
            if (x + a) in sm:
                ok = False
                break
        if ok and (2 * x) in sm:
            ok = False
        if ok:
            for a in sel:
                sm.add(x + a)
            sm.add(2 * x)
            sel.append(x)
            sel_set.add(x)
    return sel, sm


def _greedy_full(order):
    """Greedy build from scratch over `order`."""
    sel = []
    sm = set()
    sel_set = set()
    for x in order:
        ok = True
        for a in sel:
            if (x + a) in sm:
                ok = False
                break
        if ok and (2 * x) in sm:
            ok = False
        if ok:
            for a in sel:
                sm.add(x + a)
            sm.add(2 * x)
            sel.append(x)
            sel_set.add(x)
    return sel, sm


def _addable_count(selected, sums):
    """Number of missing elements that could be added (deficit from maximal)."""
    sel_set = set(selected)
    cnt = 0
    for x in range(1, N + 1):
        if x in sel_set:
            continue
        ok = True
        for a in selected:
            if (x + a) in sums:
                ok = False
                break
        if ok and (2 * x) in sums:
            ok = False
        if ok:
            cnt += 1
    return cnt


# ---------------------------------------------------------------------------
# Local search
# ---------------------------------------------------------------------------
def _local_search(selected, sums, deadline):
    """Minimise |selected| while keeping it inclusion-maximal.

    Returns (best_selected, best_sums, best_size).
    """
    best = list(selected)
    best_sums = set(sums)
    best_size = len(best)
    best_addable = _addable_count(best, best_sums)
    # we only accept states that are inclusion-maximal (addable == 0)
    if best_addable != 0:
        # repair: greedy-extend to restore maximality
        order = [x for x in range(1, N + 1) if x not in set(best)]
        RNG.shuffle(order)
        best, best_sums = _greedy_extend(best, best_sums, order)
        best_size = len(best)
        best_addable = _addable_count(best, best_sums)

    no_improve = 0
    while time.time() < deadline:
        sel_set = set(best)
        candidates = list(best)
        RNG.shuffle(candidates)
        improved = False

        # ---- deletion moves (shrink size) ----
        for x in candidates:
            if time.time() >= deadline:
                break
            trial = [a for a in best if a != x]
            trial_sums = _sums_of(trial)
            # must stay a valid Sidon set (it will, as a subset)
            # then check maximality: if addable == 0, it's maximal
            addable = _addable_count(trial, trial_sums)
            if addable == 0 and len(trial) < best_size:
                best = trial
                best_sums = trial_sums
                best_size = len(best)
                improved = True
                break
        if improved:
            no_improve = 0
            continue

        # ---- swap moves (keep size, may enable later deletions) ----
        for x in candidates:
            if time.time() >= deadline:
                break
            # try replacing x with some missing y
            trial_base = [a for a in best if a != x]
            trial_base_sums = _sums_of(trial_base)
            # candidate y values: missing elements
            ys = [y for y in range(1, N + 1) if y not in sel_set]
            RNG.shuffle(ys)
            for y in ys:
                if time.time() >= deadline:
                    break
                trial = trial_base + [y]
                # quick Sidon check
                ok = True
                for a in trial_base:
                    if (y + a) in trial_base_sums:
                        ok = False
                        break
                if ok and (2 * y) in trial_base_sums:
                    ok = False
                if not ok:
                    continue
                trial_sums = set(trial_base_sums)
                for a in trial_base:
                    trial_sums.add(y + a)
                trial_sums.add(2 * y)
                addable = _addable_count(trial, trial_sums)
                if addable == 0 and len(trial) < best_size:
                    best = trial
                    best_sums = trial_sums
                    best_size = len(best)
                    improved = True
                    break
                # accept a swap that keeps size but is maximal (for diversity)
                if addable == 0 and len(trial) == best_size:
                    # small chance to accept to escape local minima
                    if RNG.random() < 0.05:
                        best = trial
                        best_sums = trial_sums
                        improved = True
                        break
        if improved:
            no_improve = 0
            continue

        no_improve += 1
        if no_improve >= 30:
            break
    return best, best_sums, best_size


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def construct():
    start = time.time()
    deadline = start + DEADLINE

    # ---- Phase 1: build a valid Sidon set via greedy on several orders ----
    best_sel, best_sums, best_size = None, None, N + 1
    orders = []
    # ascending
    orders.append(list(range(1, N + 1)))
    # descending
    orders.append(list(range(N, 0, -1)))
    # a few random orders
    for _ in range(6):
        o = list(range(1, N + 1))
        RNG.shuffle(o)
        orders.append(o)

    for o in orders:
        if time.time() >= deadline:
            break
        sel, sm = _greedy_full(o)
        if len(sel) < best_size:
            best_sel, best_sums, best_size = sel, sm, len(sel)

    if best_sel is None:
        # fallback: simple ascending greedy
        best_sel, best_sums = _greedy_full(list(range(1, N + 1)))
        best_size = len(best_sel)

    # ---- Phase 2: local search to shrink while keeping maximal ----
    best_sel, best_sums, best_size = _local_search(best_sel, best_sums, deadline)

    # ---- Phase 3: a few more random restarts if time remains ----
    while time.time() < deadline - 1.0:
        o = list(range(1, N + 1))
        RNG.shuffle(o)
        sel, sm = _greedy_full(o)
        if len(sel) < best_size:
            sel, sm, sz = _local_search(sel, sm, min(deadline, time.time() + 8.0))
            if sz < best_size:
                best_sel, best_sums, best_size = sel, sm, sz

    # ---- final safety: ensure validity and maximality ----
    if not _is_sidon(best_sel):
        # rebuild from scratch with ascending order
        best_sel, best_sums = _greedy_full(list(range(1, N + 1)))
    # ensure maximality
    addable = _addable_count(best_sel, best_sums)
    if addable != 0:
        order = [x for x in range(1, N + 1) if x not in set(best_sel)]
        RNG.shuffle(order)
        best_sel, best_sums = _greedy_extend(best_sel, best_sums, order)

    return best_sel
