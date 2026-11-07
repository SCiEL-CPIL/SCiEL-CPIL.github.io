"""Large sets with distinct subset products.

Key equivalence: a set S of integers in [2,40] has all 2^|S| subset products
distinct  <=>  the prime-exponent vectors of S are linearly independent over Z.
For integer vectors, Z-independence = Q-independence = real-independence, so we
test linear independence with exact Fraction Gaussian elimination.

Strategy: greedy lower bound, then exact branch-and-bound to find the true
maximum cardinality, with a strict deadline guard.
"""
# EVOLVE-BLOCK-START
import time
import sys
from fractions import Fraction

_START = time.time()
_DEADLINE = _START + 55.0  # safety margin under the 60s evaluator limit

# All primes in [2, 40]
_PRIMES = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37]
_P = len(_PRIMES)

# Exponent vector of each integer 2..40 (tuple over _PRIMES)
_VECTORS = []
for _x in range(2, 41):
    _v = [0] * _P
    for _i, _p in enumerate(_PRIMES):
        while _x % _p == 0:
            _v[_i] += 1
            _x //= _p
    _VECTORS.append(tuple(_v))
_N = len(_VECTORS)  # 39
_IDX_TO_VAL = [i + 2 for i in range(_N)]


def _independent(vec, basis):
    """If vec is linearly independent of basis, return the extended basis; else None.

    basis: list of (pivot_col, [Fraction,...]) in increasing pivot order, each row
    reduced to 0 in all earlier pivot columns and with pivot entry exactly 1."""
    v = [Fraction(x) for x in vec]
    for pv, brow in basis:
        f = v[pv]
        if f != 0:
            for j in range(_P):
                v[j] -= f * brow[j]
    pv = next((j for j in range(_P) if v[j] != 0), None)
    if pv is None:
        return None
    lead = v[pv]
    v = [x / lead for x in v]
    return basis + [(pv, v)]


def _is_prime(n):
    if n < 2:
        return False
    i = 2
    while i * i <= n:
        if n % i == 0:
            return False
        i += 1
    return True


def _pf_count(x):
    c = 0
    d = 2
    while d * d <= x:
        while x % d == 0:
            c += 1
            x //= d
        d += 1
    if x > 1:
        c += 1
    return c


def construct():
    sys.setrecursionlimit(100000)

    best_count = 0
    best_set = []

    def _update_best(idx_list):
        nonlocal best_count, best_set
        if len(idx_list) > best_count:
            best_count = len(idx_list)
            best_set = [int(_IDX_TO_VAL[i]) for i in idx_list]

    # ---------- Fast lower bound (greedy): primes then composites ----------
    primes_idx = [i for i in range(_N) if _is_prime(_IDX_TO_VAL[i])]
    composites_idx = [i for i in range(_N) if not _is_prime(_IDX_TO_VAL[i])]
    composites_idx.sort(key=lambda i: (_pf_count(_IDX_TO_VAL[i]), _IDX_TO_VAL[i]))
    greedy_order = primes_idx + composites_idx

    g_basis = []
    g_idx = []
    for i in greedy_order:
        nb = _independent(_VECTORS[i], g_basis)
        if nb is not None:
            g_basis = nb
            g_idx.append(i)
    _update_best(g_idx)

    # ---------- Exact branch-and-bound ----------
    greedy_set = set(g_idx)
    rest = [i for i in range(_N) if i not in greedy_set]
    rest.sort(key=lambda i: (_pf_count(_IDX_TO_VAL[i]), _IDX_TO_VAL[i]))
    search_order = [i for i in greedy_order if i in greedy_set] + rest

    def _dfs(pos, basis, chosen):
        nonlocal best_count, best_set
        if time.time() > _DEADLINE:
            return
        if pos == _N:
            if len(chosen) > best_count:
                best_count = len(chosen)
                best_set = [int(_IDX_TO_VAL[i]) for i in chosen]
            return
        # Prune: even taking every remaining element cannot exceed best.
        if len(basis) + (_N - pos) <= best_count:
            return
        vec = _VECTORS[search_order[pos]]
        # INCLUDE first (finds large solutions sooner -> tighter pruning).
        nb = _independent(vec, basis)
        if nb is not None:
            chosen.append(search_order[pos])
            _dfs(pos + 1, nb, chosen)
            chosen.pop()
        # EXCLUDE.
        _dfs(pos + 1, basis, chosen)

    _dfs(0, [], [])

    # Safety fallback: never return fewer than the greedy set.
    if len(best_set) < len(g_idx):
        best_set = [int(_IDX_TO_VAL[i]) for i in g_idx]

    return [int(x) for x in best_set]
# EVOLVE-BLOCK-END
