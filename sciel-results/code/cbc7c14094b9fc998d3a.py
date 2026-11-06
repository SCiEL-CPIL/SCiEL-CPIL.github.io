"""Maximum-size dissociated reciprocal subset of [1..22].

Structural change vs. the parent:
  * Exact arithmetic via integers scaled by L = lcm(1..22) instead of Fraction.
    A reciprocal 1/a becomes L//a (an int); a subset sum is a sum of ints.
    Distinctness of reciprocal sums is equivalent to distinctness of these
    integer sums (multiplication by the positive integer L is injective).
  * Multi-start construction: many candidate orderings (descending, ascending,
    interleaved, random) are used to greedily build a dissociated set, then a
    bounded exact beam/branch extension tries to grow each seed to maximum size.
  * A global deadline guard returns the best verified set found.

This is a different modelling family (integer bitmask + multi-start beam) rather
than a single Fraction branch-and-bound, which is much faster per node and lets
the time budget explore many more construction orders.
"""
import time
import math
import random
from itertools import combinations

_T0 = time.time()
_BUDGET = 82.0            # wall-clock guard (evaluator kills at 90s; keep margin)
_N = 22

# L = lcm(1..22); 1/a -> L//a (exact integer)
_L = 1
for _i in range(1, _N + 1):
    _L = _L // math.gcd(_L, _i) * _i
_RECIP = {a: _L // a for a in range(1, _N + 1)}


def _is_dissociated(elements):
    """Independent verification: all 2^k reciprocal sub-sums of `elements` distinct."""
    sums = {0}
    for e in elements:
        v = _RECIP[e]
        shifted = {s + v for s in sums}
        if shifted & sums:
            return False
        sums |= shifted
    return True


def _can_add(sums_set, v):
    """True iff adding integer reciprocal v keeps every sum distinct from every other."""
    # v is addable iff (v - s) is not in sums_set for any s in sums_set
    return all((v - s) not in sums_set for s in sums_set)


def _build_greedy(order, max_size=None):
    """Greedy: walk `order`, include if it preserves dissociation."""
    cur = {0}
    chosen = []
    for a in order:
        v = _RECIP[a]
        if _can_add(cur, v):
            cur |= {s + v for s in cur}
            chosen.append(a)
            if max_size and len(chosen) >= max_size:
                break
    return chosen


def _extend_bounded(seed, order, time_limit):
    """Bounded DFS extension of `seed` (already dissociated) using `order` as the
    remaining candidate list, trying to maximise cardinality. Returns best set."""
    start = time.time()
    # Build initial sums from seed
    cur = {0}
    for a in seed:
        v = _RECIP[a]
        cur |= {s + v for s in cur}
    base_size = len(seed)
    best = [base_size, list(seed)]

    # Candidates not in seed, in the given order
    seed_set = set(seed)
    remaining = [a for a in order if a not in seed_set]

    def dfs(idx, cur_set, chosen):
        if time.time() - start > time_limit:
            return
        if len(chosen) + len(remaining) - idx <= best[0]:
            return
        if idx == len(remaining):
            if len(chosen) > best[0]:
                best[0] = len(chosen)
                best[1] = list(chosen)
            return
        a = remaining[idx]
        v = _RECIP[a]
        # Include if safe
        if _can_add(cur_set, v):
            new_cur = cur_set | {s + v for s in cur_set}
            chosen.append(a)
            dfs(idx + 1, new_cur, chosen)
            chosen.pop()
        # Exclude
        dfs(idx + 1, cur_set, chosen)

    dfs(0, cur, list(seed))
    return best[1]


def _orders():
    """Generate a diverse set of construction orderings."""
    rng = random.Random(12345)
    base = list(range(1, _N + 1))
    orders = []
    # Descending (big reciprocals first) - strongest early distinctness
    orders.append(list(range(_N, 0, -1)))
    # Ascending
    orders.append(base)
    # Interleaved: alternate high/low
    hi = list(range(_N, 0, -1))
    lo = list(range(1, _N + 1))
    inter = []
    i = j = 0
    while i < len(hi) or j < len(lo):
        if i < len(hi):
            inter.append(hi[i]); i += 1
        if j < len(lo):
            inter.append(lo[j]); j += 1
    orders.append(inter)
    # Reverse interleaved
    orders.append(list(reversed(inter)))
    # Many random shuffles
    for _ in range(200):
        o = base[:]
        rng.shuffle(o)
        orders.append(o)
    return orders


def _verified_baseline():
    seed = [1, 2, 3, 5, 7, 11, 13, 17, 19]
    return seed if _is_dissociated(seed) else [1]


def construct():
    """Return a nonempty subset of [1..22] maximising dissociated cardinality."""
    fallback = _verified_baseline()
    best = [len(fallback), list(fallback)]

    def consider(cand):
        if cand and _is_dissociated(cand) and len(cand) > best[0]:
            best[0] = len(cand)
            best[1] = list(cand)

    orders = _orders()
    for order in orders:
        if time.time() - _T0 > _BUDGET:
            break
        # Greedy build
        g = _build_greedy(order)
        consider(g)
        # Bounded extension of the greedy seed (small time slice per seed)
        if time.time() - _T0 < _BUDGET - 5.0:
            ext = _extend_bounded(g, order, time_limit=1.5)
            consider(ext)
        # Also try extending the baseline with this order
        if time.time() - _T0 < _BUDGET - 5.0:
            ext2 = _extend_bounded(fallback, order, time_limit=1.0)
            consider(ext2)

    # Final safety: never return invalid
    if not best[1] or not _is_dissociated(best[1]):
        best[1] = [1]
    return best[1]
