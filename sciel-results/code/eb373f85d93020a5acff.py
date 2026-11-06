"""Reciprocal-subsum-free set finder: max independent set via Bron-Kerbosch on conflict graph."""
import time
from fractions import Fraction

# EVOLVE-BLOCK-START
_UNIVERSE = list(range(1, 25))
_INV = {i: Fraction(1, i) for i in _UNIVERSE}
_IDX = {v: i for i, v in enumerate(_UNIVERSE)}
_N = len(_UNIVERSE)

# Conflict graph: _ADJ[u] = bitmask of v that CONFLICT with u
# (1/u == sum of 1/w over some nonempty subset of {v})
_ADJ = [0] * _N
for u in range(_N):
    mask = 0
    target = _INV[_UNIVERSE[u]]
    for v in range(_N):
        if v == u:
            continue
        found = False
        for sub in range(1, 1 << _N):
            if (sub >> v) & 1 == 0:
                continue
            s = Fraction(0)
            for w in range(_N):
                if w != u and ((sub >> w) & 1):
                    s += _INV[_UNIVERSE[w]]
            if s == target:
                found = True
                break
        if found:
            mask |= (1 << v)
    _ADJ[u] = mask

# Complement adjacency (for max clique = max independent set of conflict graph)
_FULL = (1 << _N) - 1
_COMP = [0] * _N
for u in range(_N):
    _COMP[u] = _FULL & ~(1 << u) & ~_ADJ[u]


def _is_valid(sel):
    """sel: iterable of values; True if no reciprocal-subsum violation."""
    idxs = [_IDX[v] for v in sel]
    for a in idxs:
        rest = [b for b in idxs if b != a]
        for sub in range(1, 1 << len(rest)):
            s = Fraction(0)
            for j in range(len(rest)):
                if (sub >> j) & 1:
                    s += _INV[_UNIVERSE[rest[j]]]
            if s == _INV[_UNIVERSE[a]]:
                return False
    return True


def _mask_to_set(mask):
    return frozenset(_UNIVERSE[i] for i in range(_N) if (mask >> i) & 1)


def _bk(R, P, X, deadline, best):
    """Bron-Kerbosch with pivoting on _COMP; best=[size, mask]."""
    if time.time() > deadline:
        return best[0]
    if P == 0:
        if len(R) > best[0]:
            best[0] = len(R)
            best[1] = R
        return best[0]
    if len(R) + bin(P).count("1") <= best[0]:
        return best[0]
    # pivot from P|X maximizing |P & comp[u]|
    union = P | X
    pivot = -1
    maxc = -1
    t = union
    while t:
        lb = t & (-t)
        u = lb.bit_length() - 1
        c = bin(P & _COMP[u]).count("1")
        if c > maxc:
            maxc = c
            pivot = u
        t ^= lb
    candidates = (P & ~_COMP[pivot]) if pivot >= 0 else P
    cand = candidates
    while cand:
        if time.time() > deadline:
            return best[0]
        lb = cand & (-cand)
        u = lb.bit_length() - 1
        cand ^= lb
        _bk(R | lb, P & _COMP[u], X & _COMP[u], deadline, best)
        P &= ~lb
        X |= lb
    return best[0]


def construct():
    start = time.time()
    deadline = start + 80.0  # safety margin under the 90 s wall-clock kill
    # seed fallback (known size-12 valid set)
    seed = frozenset(range(13, 25))
    best = [len(seed), 0]
    # encode seed as mask
    smask = 0
    for v in seed:
        smask |= (1 << _IDX[v])
    best = [len(seed), smask]

    # exact max clique on complement graph
    _bk(0, _FULL, 0, deadline, best)

    result = sorted(_mask_to_set(best[1]))
    if not result or not _is_valid(result):
        result = sorted(seed)
    return result
# EVOLVE-BLOCK-END
