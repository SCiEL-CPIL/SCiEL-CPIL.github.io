import time
import numpy as np
from fractions import Fraction
from itertools import combinations

DIMENSIONS = [3, 4, 8]


def _build_base(n, m, p):
    """Build the block-Strassen rank-88 base decomposition. Returns U,V,W as lists."""
    rank = 88
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0
    strassen_factors = [
        ({(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}, {(0, 0): 1, (1, 1): 1}),
        ({(1, 0): 1, (1, 1): 1}, {(0, 0): 1}, {(1, 0): 1, (1, 1): -1}),
        ({(0, 0): 1}, {(0, 1): 1, (1, 1): -1}, {(0, 1): 1, (1, 1): 1}),
        ({(1, 1): 1}, {(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (1, 0): 1}),
        ({(0, 0): 1, (0, 1): 1}, {(1, 1): 1}, {(0, 0): -1, (0, 1): 1}),
        ({(1, 0): 1, (0, 0): -1}, {(0, 0): 1, (0, 1): 1}, {(1, 1): 1}),
        ({(0, 1): 1, (1, 1): -1}, {(1, 0): 1, (1, 1): 1}, {(0, 0): 1}),
    ]

    # 2 x M_{2,4,4} blocks for rows i = 0,1
    for c_block in range(2):
        for factor in strassen_factors:
            a_coeffs, b_coeffs, c_coeffs = factor
            for sub_prod_idx in range(4):
                if sub_prod_idx == 0:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 0
                    c_i_off, c_j_off = 0, 0
                elif sub_prod_idx == 1:
                    a_i_off, a_j_off = 0, 0
                    b_i_off, b_j_off = 0, 2
                    c_i_off, c_j_off = 0, 2
                elif sub_prod_idx == 2:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 0
                    c_i_off, c_j_off = 0, 0
                else:
                    a_i_off, a_j_off = 0, 2
                    b_i_off, b_j_off = 2, 2
                    c_i_off, c_j_off = 0, 2

                for (ai, aj), au in a_coeffs.items():
                    U[(ai + a_i_off) * 4 + (aj + a_j_off)][col] += au
                for (bi, bj), bv in b_coeffs.items():
                    V[(bi + b_i_off) * 8 + (bj + b_j_off + 4 * c_block)][col] += bv
                for (ci, cj), cw in c_coeffs.items():
                    W[(cj + c_j_off + 4 * c_block) * 3 + (ci + c_i_off)][col] += cw
                col += 1

    # 2 x M_{1,4,4} blocks for row i = 2
    for c_block in range(2):
        for k in range(4):
            for j in range(4):
                u_idx = 2 * 4 + k
                v_idx = k * 8 + (j + 4 * c_block)
                w_idx = (j + 4 * c_block) * 3 + 2
                U[u_idx][col] = 1
                V[v_idx][col] = 1
                W[w_idx][col] = 1
                col += 1

    return U, V, W


def _tensor_vector(U, V, W, r, n, m, p):
    """Return the flat rank-1 tensor for term r as a list of Fractions."""
    U_r = [Fraction(x) for x in U[:, r]]
    V_r = [Fraction(x) for x in V[:, r]]
    W_r = [Fraction(x) for x in W[:, r]]
    vec = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                vec.append(U_r[i * m + j] * V_r[j * p + k] * W_r[k * n + i])
    return vec


def _verify(U, V, W, n, m, p):
    """Exact verification that U⊗V⊗W equals the matrix-multiplication tensor."""
    R = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(R):
                    val += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if val != 1:
                    return False
    return True


def construct():
    n, m, p = DIMENSIONS
    start = time.time()
    deadline = start + 135.0

    U, V, W = _build_base(n, m, p)
    rank = len(U[0])

    # Precompute tensor vectors (exact) for all terms.
    vecs = []
    for r in range(rank):
        if time.time() > deadline:
            break
        vecs.append(_tensor_vector(U, V, W, r, n, m, p))

    keep = [True] * rank

    # --- Pass 1: proportional terms (T_r = a * T_s) ---
    for r in range(rank):
        if not keep[r]:
            continue
        for s in range(r + 1, rank):
            if not keep[s]:
                continue
            # find a such that vecs[r] = a * vecs[s]
            a = None
            for e in range(len(vecs[r])):
                if vecs[s][e] != 0:
                    a = vecs[r][e] / vecs[s][e]
                    break
            if a is None:
                continue
            ok = True
            for e in range(len(vecs[r])):
                if vecs[r][e] != a * vecs[s][e]:
                    ok = False
                    break
            if ok:
                # drop r, scale s by (1 + a)
                if (1 + a) != 0:
                    W[:, s] = [x * (1 + a) for x in W[:, s]]
                    keep[s] = True
                else:
                    keep[s] = False
                keep[r] = False
                break

    # --- Pass 2: subset-sum dependency (sum over S = a * T_s, s not in S) ---
    kept = [r for r in range(rank) if keep[r]]
    K = len(kept)
    # map kept index -> global index
    gidx = kept

    # precompute tensor vectors for kept terms
    kv = [vecs[r] for r in kept]
    D = len(kv[0])

    # target: for each target t, find a subset S (size >= 2) of the OTHER kept terms
    # such that sum_{r in S} kv[r] = a * kv[t]
    # We search subsets of increasing size.
    # To make it tractable, we only target t that have a "sparse" structure, and we
    # limit subset size.

    found_any = False
    # Try subsets of size 2 and 3
    for size in (2, 3):
        if time.time() > deadline:
            break
        # For each target t, enumerate subsets S of the remaining kept terms.
        # To bound cost, we enumerate all size-`size` subsets of kept terms and
        # test each against every target.
        indices = list(range(K))
        for combo in combinations(indices, size):
            if time.time() > deadline:
                break
            # sum vector
            svec = [Fraction(0)] * D
            for r in combo:
                for e in range(D):
                    svec[e] += kv[r][e]
            # quick: check it's nonzero
            if all(x == 0 for x in svec):
                continue
            # find a target t not in combo such that svec = a * kv[t]
            a = None
            for e in range(D):
                if svec[e] != 0:
                    # need a target with nonzero at e
                    # defer
                    pass
            # Determine a from first nonzero of svec, then require a target with same pattern
            first = None
            for e in range(D):
                if svec[e] != 0:
                    first = e
                    break
            if first is None:
                continue
            for t in indices:
                if t in combo:
                    continue
                if kv[t][first] == 0:
                    continue
                a = svec[first] / kv[t][first]
                ok = True
                for e in range(D):
                    if svec[e] != a * kv[t][e]:
                        ok = False
                        break
                if ok:
                    # absorb: for r in combo, W[r] += a*W[t]; then drop t
                    for r in combo:
                        W[:, r] = [W[r][e] + a * W[t][e] for e in range(len(W[r]))]
                    keep[gidx[t]] = False
                    found_any = True
                    break

    # Rebuild final matrices from kept terms
    final = [r for r in range(rank) if keep[r]]
    Uf = [row[r] for row in U for r in []]  # placeholder
    Uf = np.array(U, dtype=object)[:, final].tolist()
    Vf = np.array(V, dtype=object)[:, final].tolist()
    Wf = np.array(W, dtype=object)[:, final].tolist()

    # Verify exactness; fall back to base if it fails.
    if _verify(Uf, Vf, Wf, n, m, p):
        return {"U": Uf, "V": Vf, "W": Wf}

    # Fallback: return the base (always valid).
    return {"U": U, "V": V, "W": W}


def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct (exact, integer/half-integer safe)."""
    r = len(U[0])
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = Fraction(0)
                for c in range(r):
                    val += Fraction(U[i * m + j][c]) * Fraction(V[j * p + k][c]) * Fraction(W[k * n + i][c])
                if val != 1:
                    return False
    return True
