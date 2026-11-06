"""Combined: precomputed gcd matrix + multi-greedy + swap-based local search."""
# EVOLVE-BLOCK-START
import time
import random
from math import gcd

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

# Precompute prime factorization for each number
_prime_factors = [set() for _ in range(_N + 1)]
for _n in range(2, _N + 1):
    _tmp = _n
    _p = 2
    while _p * _p <= _tmp:
        if _tmp % _p == 0:
            _prime_factors[_n].add(_p)
            while _tmp % _p == 0:
                _tmp //= _p
        _p += 1
    if _tmp > 1:
        _prime_factors[_n].add(_tmp)


def _can_add(S, x):
    """Check if adding x to S keeps the set triple-free using precomputed gcd matrix."""
    n = len(S)
    if n < 2:
        return True
    gcd_to_x = [_gcd_matrix[S[i]][x] for i in range(n)]
    for i in range(n):
        ga = gcd_to_x[i]
        for j in range(i + 1, n):
            gb = gcd_to_x[j]
            if ga == gb and _gcd_matrix[S[i]][S[j]] == ga:
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


def _add_all_possible(S, Sset, universe, deadline):
    """Add all elements that can be added, in order."""
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
    return S, Sset


def _local_search(S, deadline, rng):
    """Enhanced local search with 1-swap, 2-swap, and polish passes."""
    S = list(S)
    Sset = set(S)
    universe = list(range(1, 129))
    d = deadline - 2  # safety margin

    # Phase 1: add all possible
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 2: 1-swap local search with greedy fill
    if time.time() < d - 5:
        for _iter in range(50):
            if time.time() > d - 5:
                break
            progress = False
            for idx in range(len(S)):
                if time.time() > d - 5:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                # Try each possible replacement
                best_gain = 0
                best_x = None
                best_added = None
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add(S2, x):
                        S3 = S2 + [x]
                        S3set = set(S3)
                        # Greedily add more
                        added = []
                        for y in universe:
                            if y in S3set:
                                continue
                            if _can_add(S3, y):
                                S3.append(y)
                                S3set.add(y)
                                added.append(y)
                        gain = len(added)
                        if gain > best_gain:
                            best_gain = gain
                            best_x = x
                            best_added = added
                            if gain >= 3:
                                break
                if best_x is not None and best_gain > 0:
                    S = S2 + [best_x]
                    Sset = set(S)
                    for y in best_added:
                        S.append(y)
                        Sset.add(y)
                    progress = True
                    break
            if not progress:
                break

    # Phase 3: add all possible again
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 4: 2-swap local search (remove 2, add 2+)
    if time.time() < d - 3:
        for _iter in range(20):
            if time.time() > d - 3:
                break
            progress = False
            n = len(S)
            for i in range(n):
                if time.time() > d - 3:
                    break
                for j in range(i + 1, n):
                    if time.time() > d - 3:
                        break
                    S2 = S[:i] + S[i+1:j] + S[j+1:]
                    S2set = set(S2)
                    best_gain = 0
                    best_pair = None
                    for x in universe:
                        if time.time() > d - 3:
                            break
                        if x in S2set:
                            continue
                        if _can_add(S2, x):
                            S3 = S2 + [x]
                            S3set = set(S3)
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
                            gain = len(added)
                            if gain > best_gain:
                                best_gain = gain
                                best_pair = (x, added)
                                if gain >= 3:
                                    break
                    if best_pair is not None and best_gain > 0:
                        x, added = best_pair
                        S = S2 + [x]
                        Sset = set(S)
                        for y in added:
                            S.append(y)
                            Sset.add(y)
                        progress = True
                        break
                if progress:
                    break
            if not progress:
                break

    # Phase 5: final add pass
    S, Sset = _add_all_possible(S, Sset, universe, d)

    # Phase 6: polish - try replacing each element with a better one
    if time.time() < d - 1:
        polish_improved = True
        while polish_improved:
            if time.time() > d - 1:
                break
            polish_improved = False
            for idx in range(len(S)):
                if time.time() > d - 1:
                    break
                removed = S[idx]
                S2 = S[:idx] + S[idx + 1:]
                S2set = set(S2)
                # Find best single replacement
                best_x = None
                for x in universe:
                    if x in S2set:
                        continue
                    if _can_add(S2, x):
                        best_x = x
                        break
                if best_x is not None and best_x != removed:
                    S = S2 + [best_x]
                    Sset = set(S)
                    polish_improved = True
                    break

    # Phase 7: final add
    S, Sset = _add_all_possible(S, Sset, universe, d)

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
        # Prime-heavy first: numbers with many distinct prime factors are more constrained
        sorted(range(1, 129), key=lambda x: -len(_prime_factors[x])),
        # Smooth numbers last
        sorted(range(1, 129), key=lambda x: max(_prime_factors[x]) if _prime_factors[x] else 0),
        # Numbers with few prime factors first (more flexible)
        sorted(range(1, 129), key=lambda x: len(_prime_factors[x])),
    ]
    # Add random orderings
    for _ in range(16):
        o = universe[:]
        rng.shuffle(o)
        orders.append(o)

    # Run greedy on each ordering
    candidates = []
    for order in orders:
        if time.time() > deadline - 15:
            break
        cand = _greedy(order, deadline)
        candidates.append(cand)
        if len(cand) > len(best):
            best = cand

    # Local search on top candidates
    candidates.sort(key=len, reverse=True)
    for cand in candidates[:8]:
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
