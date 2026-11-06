# EVOLVE-BLOCK-START
import time
import random
import math
import numpy as np

_N = 192
_K = 3
_MAX = 48
_DEADLINE = 75.0  # seconds; evaluator kills at 90s, keep margin

# ---- Precomputation (once) ----
# triples[a] = list of (b,c,d) distinct, all != a, with a | (b+c+d)
_TRIPLES = []
for a in range(1, _N + 1):
    lst = []
    for b in range(1, _N + 1):
        if b == a:
            continue
        for c in range(b + 1, _N + 1):
            if c == a:
                continue
            for d in range(c + 1, _N + 1):
                if d == a:
                    continue
                if (b + c + d) % a == 0:
                    lst.append((b, c, d))
    _TRIPLES.append(lst)

# sums_by_mod[m][s] = True iff some triple of distinct elements in [1..N]
# (each != m) sums to s.  Used for fast conflict queries.
_SUMS_BY_MOD = []
for m in range(1, _N + 1):
    arr = np.zeros(3 * _N + 1, dtype=bool)
    for b in range(1, _N + 1):
        if b == m:
            continue
        for c in range(b + 1, _N + 1):
            if c == m:
                continue
            for d in range(c + 1, _N + 1):
                if d == m:
                    continue
                arr[b + c + d] = True
    _SUMS_BY_MOD.append(arr)


def _has_conflict(a, present):
    """True if a divides the sum of any 3 distinct elements of `present` (excluding a)."""
    arr = _SUMS_BY_MOD[a]
    for s in range(a, 3 * _N + 1, a):
        if arr[s]:
            return True
    return False


def _greedy(present, excluded, order):
    """Greedy add along `order`, skipping elements in `excluded`."""
    sel = list(present)
    sel_set = set(sel)
    for x in order:
        if x in excluded or x in sel_set:
            continue
        if _has_conflict(x, sel_set):
            continue
        sel.append(x)
        sel_set.add(x)
        if len(sel) >= _MAX:
            break
    return sel


def _valid(sel):
    """Full validity check (used only for final verification)."""
    s = set(sel)
    for a in sel:
        if _has_conflict(a, s):
            return False
    return True


def _local_search(sel, deadline, rng):
    """Multi-pass add/remove/swap local search. Returns best found."""
    best = list(sel)
    best_len = len(best)
    sel_set = set(best)

    def try_add(x):
        nonlocal best, best_len, sel_set
        if x in sel_set or _has_conflict(x, sel_set):
            return
        sel_set.add(x)
        best.append(x)
        best_len += 1
        if best_len >= _MAX:
            return

    def try_remove(x):
        nonlocal best, best_len, sel_set
        if x not in sel_set:
            return
        sel_set.remove(x)
        best.remove(x)
        best_len -= 1

    def try_swap(out, x):
        nonlocal best, best_len, sel_set
        if out not in sel_set or x in sel_set:
            return
        sel_set.discard(out)
        if _has_conflict(x, sel_set):
            sel_set.add(out)
            return
        sel_set.add(x)
        best[best.index(out)] = x
        best_len = len(best)

    order = list(range(_N, 0, -1))
    for _ in range(200):
        if time.time() >= deadline:
            break
        improved = False
        # Pass 1: add
        for x in order:
            if time.time() >= deadline:
                break
            if len(best) >= _MAX:
                break
            if x in sel_set or _has_conflict(x, sel_set):
                continue
            sel_set.add(x)
            best.append(x)
            improved = True
            if len(best) >= _MAX:
                break
        # Pass 2: remove + re-add (random order)
        if not improved and time.time() < deadline:
            cur = list(best)
            rng.shuffle(cur)
            for x in cur:
                if time.time() >= deadline:
                    break
                try_remove(x)
                # try to re-add a different element
                for y in order:
                    if len(best) >= _MAX:
                        break
                    if y in sel_set or _has_conflict(y, sel_set):
                        continue
                    sel_set.add(y)
                    best.append(y)
                    improved = True
                    break
                else:
                    # couldn't add anything, put x back
                    sel_set.add(x)
                    best.append(x)
        # Pass 3: swaps
        if not improved and time.time() < deadline:
            cur = list(best)
            rng.shuffle(cur)
            for out in cur:
                if time.time() >= deadline:
                    break
                for x in order:
                    if len(best) >= _MAX:
                        break
                    if x in sel_set:
                        continue
                    # test swap
                    sel_set.discard(out)
                    if _has_conflict(x, sel_set):
                        sel_set.add(out)
                        continue
                    sel_set.add(x)
                    best[best.index(out)] = x
                    improved = True
                    break
                if time.time() >= deadline:
                    break
        if not improved:
            break
    return best


def construct():
    t0 = time.time()
    deadline = t0 + _DEADLINE

    rng = random.Random(42)
    best_overall = []
    best_len = 0

    # Phase 1: deterministic greedy (descending)
    sel = _greedy([], set(), list(range(_N, 0, -1)))
    if len(sel) > best_len:
        best_overall = list(sel)
        best_len = len(sel)

    # Phase 2: local search on greedy result
    if time.time() < deadline:
        improved = _local_search(list(sel), deadline, rng)
        if len(improved) > best_len:
            best_overall = improved
            best_len = len(improved)

    # Phase 3: random restarts with different orderings
    while time.time() < deadline - 2.0:
        # random ordering biased toward larger elements
        order = list(range(1, _N + 1))
        # sort by random key with bias toward larger
        order.sort(key=lambda x: rng.random() + 0.3 * x / _N, reverse=True)

        sel = _greedy([], set(), order)
        if len(sel) > best_len:
            best_overall = list(sel)
            best_len = len(sel)

        if time.time() < deadline - 1.0:
            improved = _local_search(list(sel), deadline, rng)
            if len(improved) > best_len:
                best_overall = improved
                best_len = len(improved)

    # Final verification
    if not _valid(best_overall):
        # Fallback: simple greedy
        best_overall = _greedy([], set(), list(range(_N, 0, -1)))

    return best_overall[:_MAX]
# EVOLVE-BLOCK-END
