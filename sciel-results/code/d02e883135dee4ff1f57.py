"""Combined: precomputed gcd matrix + multi-greedy + swap-based local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd
from itertools import combinations

_TIME_LIMIT = 84.0
_DEADLINE = None

# Precompute gcd matrix for all pairs in [1, 128]
_N = 128
_gcd_matrix = [[0] * (_N + 1) for _ in range(_N + 1)]
for _i in range(1, _N + 1):
    for _j in range(_i, _N + 1):
        _g = gcd(_i, _j)
        _gcd_matrix[_i][_j] = _g
        _gcd_matrix[_j][_i] = _g


def _can_add(S, x):
    """Check if adding x to S keeps the set triple-free using precomputed gcd matrix."""
    for i in range(len(S)):
        a = S[i]
        ga = _gcd_matrix[a][x]
        for j in range(i + 1, len(S)):
            b = S[j]
            gb = _gcd_matrix[b][x]
            if ga == gb:
                if _gcd_matrix[a][b] == ga:
                    return False
    return True


def _is_valid(S):
    """Full validation check."""
    S_list = list(S)
    for i in range(len(S_list)):
        a = S_list[i]
        for j in range(i + 1, len(S_list)):
            b = S_list[j]
            g_ab = _gcd_matrix[a][b]
            for k in range(j + 1, len(S_list)):
                c = S_list[k]
                if _gcd_matrix[a][c] == g_ab and _gcd_matrix[b][c] == g_ab:
                    return False
    return True


def _greedy(order, deadline):
    """Greedy: iterate order, add if valid."""
    S = []
    Sset = set()
    for x in order:
        if time.time() > deadline:
            break
        if x in Sset:
            continue
        if _can_add(S, x):
            S.append(x)
            Sset.add(x)
    return S


def _local_search(S, deadline, rng):
    """Enhanced local search with multi-swap and polish passes."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))

    # Phase 1: try to add any missing element
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
                break

    # Phase 2: aggressive swap-based local search with multiple strategies
    if time.time() < deadline - 3:
        for _iter in range(30):
            if time.time() > deadline - 3:
                break
            progress = False
            for idx in range(len(S)):
                if time.time() > deadline - 3:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                # Try to find a swap that enables at least one new addition
                best_swap = None
                best_size = len(S2)
                for x in universe:
                    if time.time() > deadline - 3:
                        break
                    if x in S2set:
                        continue
                    if _can_add(S2, x):
                        # This swap is net zero, but check if it enables more
                        S3 = S2 + [x]
                        S3set = set(S3)
                        # Try to add more (up to 2 additional)
                        added = []
                        for y in universe:
                            if y in S3set:
                                continue
                            if _can_add(S3, y):
                                S3.append(y)
                                S3set.add(y)
                                added.append(y)
                                if len(added) >= 2:
                                    break
                        if added:
                            best_swap = (removed, x, added)
                            best_size = len(S3)
                            break
                        elif len(S3) > best_size:
                            best_swap = (removed, x, None)
                            best_size = len(S3)
                if best_swap:
                    r, x, extra = best_swap
                    S = S2 + [x]
                    Sset = set(S)
                    if extra:
                        for y in extra:
                            S.append(y)
                            Sset.add(y)
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: final add pass
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
                break

    # Phase 4: polish pass - try removing each element and re-adding optimally
    if time.time() < deadline - 1:
        polish_improved = True
        while polish_improved:
            if time.time() > deadline - 1:
                break
            polish_improved = False
            for idx in range(len(S)):
                if time.time() > deadline - 1:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                # Try all possible additions to S2
                best_addition = None
                best_size = len(S2)
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add(S2, x):
                        if len(S2) + 1 > best_size:
                            best_addition = x
                            best_size = len(S2) + 1
                if best_addition is not None and best_addition != removed:
                    S = S2 + [best_addition]
                    Sset = set(S)
                    polish_improved = True
                    break

    # Phase 5: final add pass after polish
    improved = True
    while improved:
        if time.time() > deadline:
            break
        improved = False
        for x in universe:
            if time.time() > deadline:
                break
            if x in Sset:
                continue
            if _can_add(S, x):
                S.append(x)
                Sset.add(x)
                improved = True
                break

    return S


def _multi_greedy_local(deadline):
    rng = random.Random(42)
    universe = list(range(1, 129))
    best = []

    # Generate diverse orderings
    orders = [
        list(range(1, 129)),
        list(range(128, 0, -1)),
        [x for x in range(1, 129) if x % 2 == 1] + [x for x in range(1, 129) if x % 2 == 0],
        [x for x in range(1, 129) if x % 2 == 0] + [x for x in range(1, 129) if x % 2 == 1],
        sorted(range(1, 129), key=lambda x: -x.bit_count()),
        sorted(range(1, 129), key=lambda x: x.bit_count()),
        [x for x in range(1, 129) if x % 3 != 0] + [x for x in range(1, 129) if x % 3 == 0],
        [x for x in range(1, 129) if x % 3 == 0] + [x for x in range(1, 129) if x % 3 != 0],
        sorted(range(1, 129), key=lambda x: (x % 2, x)),
        sorted(range(1, 129), key=lambda x: (-x.bit_count(), x)),
    ]
    # Add random orderings
    for _ in range(12):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 10:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:6]:
        if time.time() > deadline - 5:
            break
        improved = _local_search(cand, deadline, rng)
        if len(improved) > len(best):
            best = improved

    return best


def construct():
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    try:
        result = _multi_greedy_local(_DEADLINE)
    except Exception:
        # Fallback: simple ascending greedy with precomputed matrix
        result = []
        for x in range(1, 129):
            ok = True
            for i in range(len(result)):
                a = result[i]
                ga = _gcd_matrix[a][x]
                for j in range(i + 1, len(result)):
                    b = result[j]
                    if ga == _gcd_matrix[b][x] == _gcd_matrix[a][b]:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                result.append(x)
    if not result:
        result = [1]
    return result
# EVOLVE-BLOCK-END
