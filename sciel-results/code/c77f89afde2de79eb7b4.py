"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2_terms():
    """Return the 7 terms of Strassen's 2x2x2 matrix multiplication."""
    return [
        {'U': {(0,0): 1, (1,1): 1}, 'V': {(0,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
        {'U': {(1,0): 1, (1,1): 1}, 'V': {(0,0): 1}, 'W': {(1,0): 1}},
        {'U': {(0,0): 1}, 'V': {(1,0): 1, (1,1): -1}, 'W': {(0,1): 1, (1,1): -1}},
        {'U': {(1,1): 1}, 'V': {(0,1): 1, (0,0): -1}, 'W': {(0,0): -1, (1,0): 1}},
        {'U': {(0,0): 1, (0,1): 1}, 'V': {(1,1): 1}, 'W': {(0,1): 1, (1,1): 1}},
        {'U': {(1,0): 1, (0,0): -1}, 'V': {(0,0): 1, (0,1): 1}, 'W': {(0,0): 1, (1,0): 1}},
        {'U': {(0,1): 1, (1,1): -1}, 'V': {(1,0): 1, (1,1): 1}, 'W': {(0,0): 1, (1,1): 1}},
    ]


def _naive_terms(n, m, p):
    """Return the n*m*p schoolbook terms for an n x m x p multiplication."""
    terms = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                terms.append({'U': {(i, j): 1}, 'V': {(j, k): 1}, 'W': {(k, i): 1}})
    return terms


def _scale_term(term, s):
    return {
        'U': {ij: c * s for ij, c in term['U'].items()},
        'V': {jk: c * s for jk, c in term['V'].items()},
        'W': {ki: c * s for ki, c in term['W'].items()},
    }


def _add_term(t1, t2):
    """Component-wise sum of two terms (same rank slot)."""
    U, V, W = {}, {}, {}
    for d in t1['U']:
        U[d] = U.get(d, 0) + t1['U'][d]
    for d in t2['U']:
        U[d] = U.get(d, 0) + t2['U'][d]
    for d in t1['V']:
        V[d] = V.get(d, 0) + t1['V'][d]
    for d in t2['V']:
        V[d] = V.get(d, 0) + t2['V'][d]
    for d in t1['W']:
        W[d] = W.get(d, 0) + t2['W'][d]
    return {'U': U, 'V': V, 'W': W}


def _strassen_block(n, m, p):
    """Recursive Strassen-style decomposition of an n x m x p multiplication.

    Splits each dimension into a 2-block and a (d-2)-block and composes the
    8 sub-multiplications; uses the 7-term Strassen kernel for 2x2x2 and
    schoolbook for any sub-block touching a 1-dimension.
    """
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2_terms()
    if n == 1 or m == 1 or p == 1:
        return _naive_terms(n, m, p)

    a = 2
    b = n - a
    c = 2
    d = m - c
    e = 2
    f = p - e

    terms = []
    for (x, xx, nx), (y, yy, ny), (z, zz, nz) in (
        ((0, a, 2), (0, c, 2), (0, e, 2)),
        ((0, a, 2), (0, c, 2), (e, f, 2)),
        ((0, a, 2), (c, d, 2), (0, e, 2)),
        ((0, a, 2), (c, d, 2), (e, f, 2)),
        ((a, b, 2), (0, c, 2), (0, e, 2)),
        ((a, b, 2), (0, c, 2), (e, f, 2)),
        ((a, b, 2), (c, d, 2), (0, e, 2)),
        ((a, b, 2), (c, d, 2), (e, f, 2)),
    ):
        for t in _strassen_block(nx, ny, nz):
            terms.append({
                'U': {(x + i, y + j): cc for (i, j), cc in t['U'].items()},
                'V': {(y + j, z + k): cc for (j, k), cc in t['V'].items()},
                'W': {(z + k, x + i): cc for (k, i), cc in t['W'].items()},
            })
    return terms


def _build_block_decomposition(n, m, p, splits):
    """Compose per-block decompositions into full U,V,W matrices."""
    n_off, m_off, p_off = [], [], []
    acc = 0
    for s in splits[0]:
        n_off.append(acc); acc += s
    acc = 0
    for s in splits[1]:
        m_off.append(acc); acc += s
    acc = 0
    for s in splits[2]:
        p_off.append(acc); acc += s

    U = {}
    V = {}
    W = {}
    rank = 0
    for ni, ns in enumerate(splits[0]):
        for mi, ms in enumerate(splits[1]):
            for pi, ps in enumerate(splits[2]):
                no, mo, po = n_off[ni], m_off[mi], p_off[pi]
                for t in _strassen_block(ns, ms, ps):
                    for (i, j), cu in t['U'].items():
                        U.setdefault((no + i, mo + j), []).append((cu, rank))
                    for (j, k), cv in t['V'].items():
                        V.setdefault((mo + j, po + k), []).append((cv, rank))
                    for (k, i), cw in t['W'].items():
                        W.setdefault((po + k, no + i), []).append((cw, rank))
                    rank += 1

    U_mat = [[0] * rank for _ in range(n * m)]
    V_mat = [[0] * rank for _ in range(m * p)]
    W_mat = [[0] * rank for _ in range(p * n)]
    for (i, j), entries in U.items():
        for coeff, r in entries:
            U_mat[i * m + j][r] += coeff
    for (j, k), entries in V.items():
        for coeff, r in entries:
            V_mat[j * p + k][r] += coeff
    for (k, i), entries in W.items():
        for coeff, r in entries:
            W_mat[k * n + i][r] += coeff
    return {"U": U_mat, "V": V_mat, "W": W_mat}


def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p, tol=1e-6):
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0
                for r in range(rank):
                    total += U[i * m + j][r] * V[j * p + k][r] * W[k * n + i][r]
                if abs(total - 1.0) > tol:
                    return False
    return True


def _verify_integer_half_integer(U, V, W, tol=1e-9):
    for M in (U, V, W):
        for row in M:
            for val in row:
                if abs(val * 2 - round(val * 2)) > tol:
                    return False
    return True


def _merge_terms(U, V, W, n, m, p, deadline):
    """Exact rank reduction: merge term r2 into r1 when V(r2)==V(r1) and
    W(r2)==W(r1) (then the combined U column is valid), then drop r2."""
    rank = len(U[0])
    changed = True
    while changed and time.time() < deadline:
        changed = False
        rank = len(U[0])
        for r1 in range(rank):
            for r2 in range(r1 + 1, rank):
                if time.time() >= deadline:
                    return U, V, W
                if any(V[j * p + k][r1] != V[j * p + k][r2]
                       for j in range(m) for k in range(p)):
                    continue
                if any(W[k * n + i][r1] != W[k * n + i][r2]
                       for k in range(p) for i in range(n)):
                    continue
                for i in range(n):
                    for j in range(m):
                        U[i * m + j][r1] += U[i * m + j][r2]
                U.pop(r2)
                V.pop(r2)
                W.pop(r2)
                changed = True
                break
            if changed:
                break
    return U, V, W


def construct():
    """Find a rank-minimal exact tensor decomposition for 4x4 @ 4x5 matrix multiplication."""
    start_time = time.time()
    deadline = start_time + 120.0
    n, m, p = DIMENSIONS

    # Candidate block splits for 4 x 4 x 5.
    candidates = [
        [3, 1], [3, 1], [3, 2],   # recursive Strassen on the 3x3x3 core
        [2, 2], [2, 2], [2, 2, 1], # classic 2x2 Strassen tiling
    ]

    best = None
    for splits in candidates:
        if time.time() >= deadline - 5.0:
            break
        try:
            res = _build_block_decomposition(n, m, p, splits)
            if not _verify_decomposition(res["U"], res["V"], res["W"], n, m, p):
                continue
            if not _verify_integer_half_integer(res["U"], res["V"], res["W"]):
                continue
            U, V, W = _merge_terms(res["U"], res["V"], res["W"], n, m, p, deadline - 5.0)
            if _verify_decomposition(U, V, W, n, m, p) and \
               _verify_integer_half_integer(U, V, W):
                res = {"U": U, "V": V, "W": W}
                if best is None or len(U[0]) < len(best["U"][0]):
                    best = res
        except Exception:
            continue

    if best is not None:
        return best
    return _schoolbook(n, m, p)
# EVOLVE-BLOCK-END
