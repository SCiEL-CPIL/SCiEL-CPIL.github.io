"""Rewritten program: search for rank-minimal exact decomposition of 2x3x6 matmul."""
import time
import random
from fractions import Fraction

DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
_START_TIME = time.time()
_DEADLINE = 120.0
_SAFETY = 6.0


def _remaining():
    return _DEADLINE - _SAFETY - (time.time() - _START_TIME)


def _build_T():
    T = {}
    for i in range(2):
        for j in range(3):
            for k in range(2):
                T[(i, j, k)] = Fraction(1)
    return T


def _verify_terms(terms, T=None):
    if T is None:
        T = _build_T()
    acc = {}
    for (U, V, W) in terms:
        for (ij, uval) in U.items():
            if uval == 0:
                continue
            i, j = ij
            for (jk, vval) in V.items():
                if vval == 0:
                    continue
                jj, k = jk
                if jj != j:
                    continue
                for (ki, wval) in W.items():
                    if wval == 0:
                        continue
                    kk, ii = ki
                    if kk != k or ii != i:
                        continue
                    key = (i, j, k)
                    acc[key] = acc.get(key, Fraction(0)) + uval * vval * wval
    for key, val in T.items():
        if acc.get(key, Fraction(0)) != val:
            return False
    for key in acc:
        if key not in T and acc[key] != 0:
            return False
    return True


def _trivial_terms():
    terms = []
    for i in range(2):
        for j in range(3):
            for k in range(2):
                U = {(i, j): Fraction(1)}
                V = {(j, k): Fraction(1)}
                W = {(k, i): Fraction(1)}
                terms.append((U, V, W))
    return terms


_U_KEYS = [(i, j) for i in range(2) for j in range(3)]
_V_KEYS = [(j, k) for j in range(3) for k in range(2)]
_W_KEYS = [(k, i) for k in range(2) for i in range(2)]


def _random_terms(r, rng, entry_set):
    terms = []
    for _t in range(r):
        U = {key: Fraction(rng.choice(entry_set)) for key in _U_KEYS}
        V = {key: Fraction(rng.choice(entry_set)) for key in _V_KEYS}
        W = {key: Fraction(rng.choice(entry_set)) for key in _W_KEYS}
        terms.append((U, V, W))
    return terms


def _local_search(terms, rng, iters, entry_set):
    r = len(terms)
    for it in range(iters):
        if _verify_terms(terms):
            return True
        t = rng.randrange(r)
        U, V, W = terms[t]
        c = rng.random()
        if c < 0.33:
            key = rng.choice(_U_KEYS)
            U[key] = Fraction(rng.choice(entry_set))
        elif c < 0.66:
            key = rng.choice(_V_KEYS)
            V[key] = Fraction(rng.choice(entry_set))
        else:
            key = rng.choice(_W_KEYS)
            W[key] = Fraction(rng.choice(entry_set))
        terms[t] = (U, V, W)
    return _verify_terms(terms)


def _search(rng):
    best = None
    entry_sets = [
        [-1, 0, 1],
        [-2, -1, 0, 1, 2],
    ]
    for r in range(6, 13):
        if _remaining() < 5.0:
            break
        if best is not None and r >= len(best):
            break
        for entry_set in entry_sets:
            if _remaining() < 5.0:
                break
            max_attempts = 40
            iters = 150
            for _a in range(max_attempts):
                if _remaining() < 5.0:
                    break
                terms = _random_terms(r, rng, entry_set)
                if _local_search(terms, rng, iters, entry_set):
                    if best is None or r < len(best):
                        best = terms
                    break
            if best is not None and len(best) == r:
                break
    return best


def _construct_2x3x2():
    rng = random.Random(2024)
    res = _search(rng)
    if res is not None and _verify_terms(res):
        return res
    return _trivial_terms()


def construct():
    n, m, p = DIMENSIONS
    block = _construct_2x3x2()
    rank = len(block)

    U = [[Fraction(0)] * rank for _ in range(n * m)]
    V = [[Fraction(0)] * rank for _ in range(m * p)]
    W = [[Fraction(0)] * rank for _ in range(p * n)]

    for b in range(3):
        for t_idx, (U_t, V_t, W_t) in enumerate(block):
            col = b * rank + t_idx
            for (i, j), val in U_t.items():
                U[i * m + j][col] = val
            for (j, k), val in V_t.items():
                V[j * p + (b * 2 + k)][col] = val
            for (k, i), val in W_t.items():
                W[(b * 2 + k) * n + i][col] = val

    def _conv(mat):
        out = []
        for row in mat:
            newrow = []
            for val in row:
                if val.denominator == 1:
                    newrow.append(val.numerator)
                else:
                    newrow.append(float(val))
            out.append(newrow)
        return out

    return {"U": _conv(U), "V": _conv(V), "W": _conv(W)}
# EVOLVE-BLOCK-END
