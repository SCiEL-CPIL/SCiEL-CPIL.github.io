"""Maximum weight independent set on conflict graph for pairwise-coprime selection."""
# EVOLVE-BLOCK-START
import time
from math import gcd
from fractions import Fraction

# Precompute prime factors for 1..511
def _prime_factors(n):
    pf = []
    d = 2
    while d * d <= n:
        while n % d == 0:
            pf.append(d)
            n //= d
        d += 1
    if n > 1:
        pf.append(n)
    return pf

# Precompute weights and prime factor sets
_N = 512
_WEIGHTS = [Fraction(1, _N - a) for a in range(_N)]  # index a -> weight 1/(512-a)
_PF = [set(_prime_factors(a)) for a in range(_N)]
# Conflict: two numbers conflict if they share a prime factor
# Build conflict as: for each vertex, the set of vertices it conflicts with (via shared prime)
# We'll store per-vertex conflict set lazily or precompute adjacency
_ADJ = [set() for _ in range(_N)]
for a in range(1, _N):
    for b in range(a + 1, _N):
        if _PF[a] & _PF[b]:
            _ADJ[a].add(b)
            _ADJ[b].add(a)

# Order vertices by weight descending (highest weight = smallest a)
# We want to consider high-weight vertices first for good pruning
_ORDERED = sorted(range(1, _N), key=lambda a: -_WEIGHTS[a])
# Reindex: position in search
_POS = [0] * _N
for i, a in enumerate(_ORDERED):
    _POS[a] = i

# For branch and bound, we need for each vertex in ordered position,
# the set of later-position vertices it conflicts with (to remove when we include it)
# and the set of remaining candidates.

def _build_search():
    """Build ordered conflict lists for efficient branch-and-bound."""
    n = _N - 1
    # conflict_after[i] = set of positions j > i such that vertex _ORDERED[i] conflicts with _ORDERED[j]
    conflict_after = [set() for _ in range(n)]
    for i in range(n):
        ai = _ORDERED[i]
        for j in range(i + 1, n):
            aj = _ORDERED[j]
            if aj in _ADJ[ai]:
                conflict_after[i].add(j)
    return conflict_after

_CONFLICT_AFTER = _build_search()

def _fractional_bound(remaining, conflict_after):
    """Upper bound: sum of weights of all remaining vertices (relaxation).
    A tighter bound: fractional knapsack where including a vertex removes its
    conflicts. Simplest valid upper bound = sum of all remaining weights.
    We can improve by noting that among remaining vertices, we can't take
    conflicting pairs, but a clean fractional relaxation is complex.
    Use sum of weights as bound (valid but loose).
    """
    total = Fraction(0)
    for i in remaining:
        total += _WEIGHTS[_ORDERED[i]]
    return total

def _solve_bnb(deadline):
    n = _N - 1
    best = [Fraction(0)]
    best_set = [set()]
    # remaining is a set of positions available
    # We do DFS: at each step pick a vertex to include or exclude.
    # To keep it tractable, we use a greedy ordering and bound.

    # Use a recursive search with memoization-light approach.
    # State: frozenset of remaining positions. But that's huge.
    # Instead, do iterative deepening with a strong greedy lower bound.

    # First get a good initial solution via greedy
    greedy_set = _greedy()
    greedy_val = sum((_WEIGHTS[_ORDERED[i]] for i in greedy_set), Fraction(0))
    best[0] = greedy_val
    best_set[0] = set(greedy_set)

    # Now try to improve via branch and bound on subsets.
    # We'll do a DFS where we decide include/exclude for each vertex in order,
    # maintaining the set of still-available vertices.
    # Use recursion with a stack to avoid recursion depth issues.

    # For efficiency, represent remaining as a bitmask (n <= 511, use Python int)
    # full mask
    FULL = (1 << n) - 1

    # Precompute for each position i: bitmask of positions j>i that conflict with i
    conflict_mask = [0] * n
    for i in range(n):
        m = 0
        for j in _CONFLICT_AFTER[i]:
            m |= (1 << j)
        conflict_mask[i] = m

    # Precompute weight as Fraction; for bound we need sum of weights of set bits.
    # We'll compute bound on the fly by iterating set bits.

    # DFS stack: (mask, chosen_positions_tuple)
    # To bound, we compute sum of weights of remaining = mask.
    # We can precompute prefix? No, mask is arbitrary.

    # For speed, compute weight-sum of mask by iterating bits.
    def mask_weight(mask):
        s = Fraction(0)
        m = mask
        while m:
            i = (m & -m).bit_length() - 1
            s += _WEIGHTS[_ORDERED[i]]
            m &= m - 1
        return s

    # To make bound tighter: the fractional bound is just mask_weight (all remaining).
    # A better bound: we can't take all; but computing a tight bound is expensive.
    # Use: best possible from remaining = mask_weight (loose).
    # Prune if current_value + mask_weight <= best.

    # DFS
    stack = [(FULL, tuple())]
    # We want to explore include-branch first (greedy order) to find good solutions early.
    # At each node, pick the smallest-position vertex in mask as decision point? 
    # Actually pick the first (lowest index) vertex in mask.
    while stack:
        if time.time() > deadline:
            break
        mask, chosen = stack.pop()
        if not mask:
            # leaf
            val = sum((_WEIGHTS[_ORDERED[i]] for i in chosen), Fraction(0))
            if val > best[0]:
                best[0] = val
                best_set[0] = set(chosen)
            continue
        # Find lowest set bit
        i = (mask & -mask).bit_length() - 1
        # Two branches: exclude i, include i
        # Exclude: mask without i
        mask_excl = mask & ~(1 << i)
        # Include: remove i and its conflicts
        mask_incl = mask & ~(1 << i) & ~conflict_mask[i]
        # Bound for exclude branch
        cur_val = sum((_WEIGHTS[_ORDERED[k]] for k in chosen), Fraction(0))
        # We push include first so it's explored first (LIFO)
        # Push exclude, then include (include popped first)
        # Bound check: if cur_val + mask_weight(mask_excl) <= best, skip exclude
        if cur_val + mask_weight(mask_excl) > best[0]:
            stack.append((mask_excl, chosen))
        if cur_val + mask_weight(mask_incl) > best[0]:
            stack.append((mask_incl, chosen + (i,)))

    return best_set[0]

def _greedy():
    """Greedy: pick highest weight vertex, remove conflicts, repeat."""
    n = _N - 1
    FULL = (1 << n) - 1
    mask = FULL
    chosen = []
    while mask:
        # pick lowest index (highest weight) in mask
        i = (mask & -mask).bit_length() - 1
        chosen.append(i)
        mask &= ~(1 << i)
        mask &= ~_conflict_mask_global[i]
    return chosen

_conflict_mask_global = [0] * (_N - 1)
for _i in range(_N - 1):
    _m = 0
    for _j in _CONFLICT_AFTER[_i]:
        _m |= (1 << _j)
    _conflict_mask_global[_i] = _m

def _local_search(start_set, deadline):
    """Try 1-for-k swaps to improve."""
    chosen = set(start_set)
    val = sum((_WEIGHTS[_ORDERED[i]] for i in chosen), Fraction(0))
    n = _N - 1
    FULL = (1 << n) - 1
    chosen_mask = 0
    for i in chosen:
        chosen_mask |= (1 << i)
    improved = True
    while improved and time.time() < deadline:
        improved = False
        # Try adding a vertex not in chosen that doesn't conflict
        for i in range(n):
            if chosen_mask & (1 << i):
                continue
            if chosen_mask & _conflict_mask_global[i]:
                continue
            # Check conflicts with all chosen
            ok = True
            for j in chosen:
                if _ADJ[_ORDERED[i]] and _ORDERED[i] in _ADJ[_ORDERED[j]]:
                    ok = False
                    break
            if not ok:
                continue
            new_val = val + _WEIGHTS[_ORDERED[i]]
            if new_val > val:
                chosen.add(i)
                chosen_mask |= (1 << i)
                val = new_val
                improved = True
        # Try removing a vertex to allow adding two better ones (1-for-2)
        for i in list(chosen):
            # try removing i
            chosen.discard(i)
            cm2 = chosen_mask & ~(1 << i)
            val2 = val - _WEIGHTS[_ORDERED[i]]
            # now try to add two vertices
            added = []
            for j in range(n):
                if cm2 & (1 << j):
                    continue
                if cm2 & _conflict_mask_global[j]:
                    continue
                if len(added) == 2:
                    break
                ok = True
                for k in chosen:
                    if _ORDERED[j] in _ADJ[_ORDERED[k]]:
                        ok = False
                        break
                for k in added:
                    if _ORDERED[j] in _ADJ[_ORDERED[k]]:
                        ok = False
                        break
                if ok:
                    added.append(j)
            if len(added) == 2:
                new_val = val2 + _WEIGHTS[_ORDERED[added[0]]] + _WEIGHTS[_ORDERED[added[1]]]
                if new_val > val:
                    chosen.update(added)
                    cm2 |= (1 << added[0]) | (1 << added[1])
                    val = new_val
                    chosen_mask = cm2
                    improved = True
                    continue
            # revert
            chosen.add(i)
            chosen_mask = cm2 | (1 << i)
            val = val2 + _WEIGHTS[_ORDERED[i]]
    return chosen

def construct():
    start = time.time()
    deadline = start + 145.0  # safety margin before 150s kill
    # Phase 1: greedy
    greedy = _greedy()
    # Phase 2: local search
    ls_set = _local_search(greedy, deadline)
    # Phase 3: branch and bound if time remains
    if time.time() < deadline - 5.0:
        try:
            bnb_set = _solve_bnb(deadline)
            # compare
            bnb_val = sum((_WEIGHTS[_ORDERED[i]] for i in bnb_set), Fraction(0))
            ls_val = sum((_WEIGHTS[_ORDERED[i]] for i in ls_set), Fraction(0))
            if bnb_val > ls_val:
                ls_set = bnb_set
        except Exception:
            pass
    # Convert positions back to values
    result = [_ORDERED[i] for i in sorted(ls_set)]
    return result
# EVOLVE-BLOCK-END
