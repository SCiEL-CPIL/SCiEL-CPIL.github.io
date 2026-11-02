"""Rank-minimal exact tensor decomposition for 2x5 by 5x6 matrix multiplication."""
import random
import time
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [2, 5, 6]

# Global deadline guard: the evaluator kills after 150s wall-clock.
T0 = time.time()
DEADLINE = 150.0
SAFETY = 8.0          # seconds to reserve before the hard limit
MAX_RANK_ATTEMPT = 40 # never try to verify a rank this large or larger


def _time_left():
    return DEADLINE - SAFETY - (time.time() - T0)


def _to_halfint(x):
    """Convert a Fraction to a half-integer if possible, else None."""
    if x.denominator == 1:
        return int(x)
    if x.denominator == 2:
        num = x.numerator
        if num % 2 == 0:
            return num // 2
        return num / 2.0
    return None


def _is_halfint(x):
    if isinstance(x, float):
        return abs(x * 2 - round(x * 2)) < 1e-9
    if isinstance(x, int):
        return True
    if isinstance(x, Fraction):
        return x.denominator in (1, 2)
    return False


def _verify(U, V, W):
    """Exact check that sum_c U[:,c] outer V[:,c] outer W[:,c] == T."""
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    rank = len(U[0])
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    S = {}
    for c in range(rank):
        u = U[c]
        v = V[c]
        w = W[c]
        for a in range(N):
            ua = u[a]
            if ua == 0:
                continue
            for b in range(M):
                uv = ua * v[b]
                if uv == 0:
                    continue
                for d in range(P):
                    val = uv * w[d]
                    if val == 0:
                        continue
                    key = (a, b, d)
                    S[key] = S.get(key, 0) + val
    if len(S) != len(T):
        return False
    for key, val in T.items():
        if S.get(key, 0) != val:
            return False
    return True


def _try_rank(r, rng):
    """Attempt to find an exact half-integer decomposition of rank r.
    Returns (U, V, W) as lists of Fractions, or None."""
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    T = {}
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[(i * m + j, j * p + k, k * n + i)] = 1
    K = len(T)

    # Start from the schoolbook (rank n*m*p) rank-1 components.
    comps = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                u = {i * m + j: 1}
                v = {j * p + k: 1}
                w = {k * n + i: 1}
                comps.append((u, v, w))
    base = n * m * p

    # Greedy reduction: try to remove one component at a time by adding a
    # linear combination of the others (a "pivot" that cancels it).
    cur = list(comps)
    for _ in range(base - r):
        if len(cur) <= r:
            break
        if _time_left() <= 0:
            return None
        # Pick a component to remove (random order for diversity).
        order = list(range(len(cur)))
        rng.shuffle(order)
        removed = False
        for cand in order:
            target = cur[cand]
            others = [cur[t] for t in range(len(cur)) if t != cand]
            # Build a random linear combination of the others.
            L = []
            for (u, v, w) in others:
                # coefficient from a small integer set
                coeff = rng.choice([-2, -1, 0, 0, 1, 2])
                if coeff == 0:
                    continue
                L.append((coeff, u, v, w))
            if not L:
                continue
            # Compute the resulting tensor S = T_target + sum coeff * term.
            S = {}
            for key, val in T.items():
                S[key] = val
            for coeff, u, v, w in L:
                for a, ua in u.items():
                    for b, vb in v.items():
                        for d, wd in w.items():
                            val = coeff * ua * vb * wd
                            key = (a, b, d)
                            S[key] = S.get(key, 0) + val
            # If S == T (i.e., the added combination cancels the removed term),
            # then removing `target` and adding the combination preserves T.
            if len(S) == K:
                ok = True
                for key, val in T.items():
                    if S.get(key, 0) != val:
                        ok = False
                        break
                if ok:
                    # Replace cur[cand] with the combination terms.
                    new_comps = [cur[t] for t in range(len(cur)) if t != cand]
                    for coeff, u, v, w in L:
                        nu = {a: coeff * ua for a, ua in u.items()}
                        nv = {b: vb for b, vb in v.items()}
                        nw = {d: wd for d, wd in w.items()}
                        new_comps.append((nu, nv, nw))
                    cur = new_comps
                    removed = True
                    break
        if not removed:
            # No single removal worked; try a 2-component combined removal.
            two = False
            for a, b in combinations(order, 2):
                if a >= len(cur) or b >= len(cur):
                    continue
                ta = cur[a]
                tb = cur[b]
                others = [cur[t] for t in range(len(cur)) if t != a and t != b]
                L = []
                for (u, v, w) in others:
                    coeff = rng.choice([-2, -1, 0, 0, 1, 2])
                    if coeff == 0:
                        continue
                    L.append((coeff, u, v, w))
                if not L:
                    continue
                S = {}
                for key, val in T.items():
                    S[key] = val
                for coeff, u, v, w in L:
                    for aa, ua in u.items():
                        for bb, vb in v.items():
                            for dd, wd in w.items():
                                val = coeff * ua * vb * wd
                                key = (aa, bb, dd)
                                S[key] = S.get(key, 0) + val
                if len(S) == K:
                    ok = True
                    for key, val in T.items():
                        if S.get(key, 0) != val:
                            ok = False
                            break
                    if ok:
                        new_comps = [cur[t] for t in range(len(cur)) if t != a and t != b]
                        for coeff, u, v, w in L:
                            nu = {aa: coeff * ua for aa, ua in u.items()}
                            nv = {bb: vb for bb, vb in v.items()}
                            nw = {dd: wd for dd, wd in w.items()}
                            new_comps.append((nu, nv, nw))
                        cur = new_comps
                        two = True
                        break
                if two:
                    break
            if not two:
                break

    # If we reached the target rank, build the matrices and verify exactly.
    if len(cur) > r:
        return None
    # Pad/truncate to exactly r by dropping zero components.
    comps2 = []
    for (u, v, w) in cur:
        # drop components that are all-zero
        if any(x != 0 for x in u.values()) and any(x != 0 for x in v.values()) and any(x != 0 for x in w.values()):
            comps2.append((u, v, w))
    if len(comps2) < r:
        return None
    comps2 = comps2[:r]
    U = [dict() for _ in range(r)]
    V = [dict() for _ in range(r)]
    W = [dict() for _ in range(r)]
    for c, (u, v, w) in enumerate(comps2):
        U[c] = {a: Fraction(x) for a, x in u.items()}
        V[c] = {b: Fraction(x) for b, x in v.items()}
        W[c] = {d: Fraction(x) for d, x in w.items()}
    # Exact verification.
    S = {}
    for c in range(r):
        for a, ua in U[c].items():
            for b, vb in V[c].items():
                for d, wd in W[c].items():
                    val = ua * vb * wd
                    key = (a, b, d)
                    S[key] = S.get(key, 0) + val
    if len(S) != K:
        return None
    for key, val in T.items():
        if S.get(key, 0) != val:
            return None
    # Convert to half-integers; if any entry is not a half-integer, fail.
    outU = [[Fraction(0)] * r for _ in range(N)]
    outV = [[Fraction(0)] * r for _ in range(M)]
    outW = [[Fraction(0)] * r for _ in range(P)]
    for c in range(r):
        for a, ua in U[c].items():
            outU[a][c] = ua
        for b, vb in V[c].items():
            outV[b][c] = vb
        for d, wd in W[c].items():
            outW[d][c] = wd
    for row in outU:
        for x in row:
            if not _is_halfint(x):
                return None
    for row in outV:
        for x in row:
            if not _is_halfint(x):
                return None
    for row in outW:
        for x in row:
            if not _is_halfint(x):
                return None
    return (outU, outV, outW)


def _finalize(U, V, W):
    n, m, p = DIMENSIONS
    N, M, P = n * m, m * p, p * n
    rank = len(U[0])
    fU = [[_to_halfint(x) for x in row] for row in U]
    fV = [[_to_halfint(x) for x in row] for row in V]
    fW = [[_to_halfint(x) for x in row] for row in W]
    for mat in (fU, fV, fW):
        for row in mat:
            for x in row:
                if x is None:
                    raise ValueError("non half-integer entry")
    return {"U": fU, "V": fV, "W": fW}


def _schoolbook():
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}


def construct():
    rng = random.Random(12345)
    base = 2 * 5 * 6  # 60
    best = None
    # Try ranks from just below the schoolbook rank downward.
    for r in range(base - 1, MAX_RANK_ATTEMPT - 1, -1):
        if _time_left() <= 0:
            break
        res = _try_rank(r, rng)
        if res is not None:
            U, V, W = res
            best = _finalize(U, V, W)
            # Found a valid decomposition; keep searching lower ranks.
            # (A lower rank is strictly better.)
            continue
        # If we failed at this rank, lower ranks are harder; try a few more
        # but stop early if time is short.
        if _time_left() < 15:
            break
    if best is not None:
        return best
    return _schoolbook()
