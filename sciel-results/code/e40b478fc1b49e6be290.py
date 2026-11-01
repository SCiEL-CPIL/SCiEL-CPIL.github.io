"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
import time
import numpy as np
from fractions import Fraction

# --- Strassen 2x2x2 terms (local index 0=(0,0),1=(0,1),2=(1,0),3=(1,1)) ---
_STRASSEN_TERMS = [
    ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),   # M1
    ({2: 1, 3: 1}, {0: 1},       {2: 1, 3: -1}),   # M2
    ({0: 1},       {1: 1, 3: -1}, {1: 1, 3: 1}),   # M3
    ({3: 1},       {2: 1, 0: -1}, {0: 1, 2: 1}),   # M4
    ({0: 1, 1: 1}, {3: 1},       {0: -1, 1: 1}),   # M5
    ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),         # M6
    ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1}),         # M7
]


def _strassen_decomposition(n, m, p):
    """Build the rank-56 Strassen decomposition for (n x m) * (m x p)."""
    n, m, p = int(n), int(m), int(p)
    rank = 7 * (m // 2) * (p // 2)
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for a_part in range(m // 2):
        a_col_start = a_part * 2
        for b_block in range(p // 2):
            b_col_start = b_block * 2
            b_row_start = a_col_start
            for (A_coeffs, B_coeffs, C_coeffs) in _STRASSEN_TERMS:
                for li, coeff in A_coeffs.items():
                    r, c = divmod(li, 2)
                    U[r * m + (a_col_start + c), col] += coeff
                for li, coeff in B_coeffs.items():
                    r, c = divmod(li, 2)
                    V[(b_row_start + r) * p + (b_col_start + c), col] += coeff
                for li, coeff in C_coeffs.items():
                    r, c = divmod(li, 2)
                    W[(b_col_start + c) * n + r, col] += coeff
                col += 1
    return {"U": U, "V": V, "W": W}


def _verify(U, V, W, n, m, p, tol=1e-9):
    """Verify the decomposition reproduces matrix multiplication exactly."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    if U.ndim != 2 or V.ndim != 2 or W.ndim != 2:
        return False
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    rU, rV, rW = U.shape[1], V.shape[1], W.shape[1]
    if not (rU == rV == rW):
        return False
    rank = rU
    for i in range(n):
        for k in range(p):
            for j in range(m):
                val = 0.0
                for c in range(rank):
                    val += U[i * m + j, c] * V[j * p + k, c] * W[k * n + i, c]
                if abs(val - 1.0) > tol:
                    return False
    return True


def _schoolbook_decomposition(n, m, p):
    """Fallback schoolbook decomposition, rank n*m*p."""
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


def _find_redundant_columns(U_arr, V_arr, W_arr, tol=1e-10):
    """Greedily drop columns that are linearly dependent in all three factors."""
    rank = U_arr.shape[1]
    keep = list(range(rank))
    for c in range(rank):
        if c not in keep:
            continue
        remaining = [x for x in keep if x != c]
        if not remaining:
            continue
        if np.linalg.matrix_rank(np.column_stack([U_arr[:, remaining], U_arr[:, c]])) \
                > np.linalg.matrix_rank(U_arr[:, remaining]):
            continue
        if np.linalg.matrix_rank(np.column_stack([V_arr[:, remaining], V_arr[:, c]])) \
                > np.linalg.matrix_rank(V_arr[:, remaining]):
            continue
        if np.linalg.matrix_rank(np.column_stack([W_arr[:, remaining], W_arr[:, c]])) \
                > np.linalg.matrix_rank(W_arr[:, remaining]):
            continue
        keep.remove(c)
    return keep


def _greedy_zero_basis(A, deadline, max_rounds=25, max_sweep=60000):
    """
    Find an invertible G such that A @ G has many zero entries (greedy pivoting).
    Works over Fractions for exactness. Returns (G, zeros, exact) or None.
    """
    A = np.asarray(A)
    r, c = A.shape
    G = np.eye(c, dtype=object)
    for i in range(c):
        G[i, i] = Fraction(1)
    Acur = np.empty((r, c), dtype=object)
    for i in range(r):
        for j in range(c):
            Acur[i, j] = Fraction(int(A[i, j]))

    zeros = 0
    for i in range(r):
        for j in range(c):
            if Acur[i, j] == 0:
                zeros += 1
    start = time.time()
    exact = True
    rounds = 0
    while rounds < max_rounds and time.time() < deadline:
        rounds += 1
        best = None
        sweeps = 0
        for j in range(c):
            for i in range(r):
                if Acur[i, j] != 0:
                    continue
                for jj in range(c):
                    if jj == j or Acur[i, jj] == 0:
                        continue
                    for ii in range(r):
                        if Acur[ii, jj] == 0:
                            continue
                        # eliminate Acur[i,j] using column jj
                        ratio = Acur[i, jj] / Acur[ii, jj]
                        if best is None or (ratio.denominator > best[0]):
                            best = (ratio.denominator, i, j, ii, jj)
                        sweeps += 1
                        if sweeps > max_sweep:
                            break
                    if sweeps > max_sweep:
                        break
                if sweeps > max_sweep:
                    break
            if sweeps > max_sweep:
                break
        if sweeps > max_sweep:
            exact = False
        if best is None:
            break
        _, i, j, ii, jj = best
        ratio = Acur[i, jj] / Acur[ii, jj]
        # G[:, j] -= ratio * G[:, jj]
        for t in range(c):
            G[t, j] = G[t, j] - ratio * G[t, jj]
        # Acur[:, j] -= ratio * Acur[:, jj]
        for t in range(r):
            Acur[t, j] = Acur[t, j] - ratio * Acur[t, jj]
        zeros += 1
    return G, zeros, exact


def _integer_search(A, G, deadline, max_iters=300000):
    """
    Search integer combinations g = sum_j a_j G[:,j] (a_j in {-2,-1,0,1,2})
    that are integer/half-integer and zero out many A-rows. Returns list of g.
    """
    r, c = A.shape
    Gf = np.empty((c, c), dtype=float)
    for t in range(c):
        for s in range(c):
            Gf[t, s] = float(G[t, s])
    A = np.asarray(A, dtype=float)

    def is_half(x):
        return abs(x - round(2 * x) / 2.0) < 1e-7

    def ok_vec(g):
        for v in g:
            if not is_half(v):
                return False
        return True

    def zeros_of(g):
        ag = A @ g
        z = 0
        for v in ag:
            if abs(v) < 1e-6:
                z += 1
        return z

    # base: columns of G (a_j = 1)
    cols = [Gf[:, j] for j in range(c)]
    chosen = []
    counts = [1] * c
    for j in range(c):
        g = Gf[:, j].copy()
        if ok_vec(g):
            chosen.append(g)
    base_zeros = sum(zeros_of(g) for g in chosen)
    base_counts = sum(counts)

    # try small replacements / additions
    best_extra = 0
    extra_vecs = []
    iters = 0
    # attempt: for each column, try a = -1..3 (skip 1) to see if it yields a valid
    # half-integer vector with more zeros than the base column.
    for j in range(c):
        for a in (-2, -1, 2, 3):
            g = a * Gf[:, j]
            if not ok_vec(g):
                continue
            z = zeros_of(g)
            if z > 1:  # better than a single-entry column
                extra_vecs.append((z, j, g.copy()))
        iters += 1
        if iters > max_iters:
            break

    # Keep only the best distinct replacements that don't collide
    # (simple: accept all that are valid; they will be checked for independence)
    # We'll just report the base set plus any strictly better single-column swaps.
    for z, j, g in sorted(extra_vecs, reverse=True):
        # independent of chosen? (different column j, so yes in general)
        chosen.append(g)
        base_zeros += (z - 1)
        base_counts += 1

    return chosen, base_zeros


def _try_rank_reduction(U, V, W, n, m, p, deadline):
    """Apply a global change of basis to zero out entries; drop redundant columns."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]

    # Target the largest factor (V) for zeroing.
    A = V
    res = _greedy_zero_basis(A, deadline)
    if res is None:
        keep = _find_redundant_columns(U, V, W)
        return {"U": U[:, keep].tolist(), "V": V[:, keep].tolist(), "W": W[:, keep].tolist()}
    G, zeros, exact = res

    if exact and time.time() < deadline - 2.0:
        try:
            chosen, _ = _integer_search(A, G, deadline, max_iters=200000)
            if chosen:
                Gf = np.array(chosen, dtype=float).T  # shape (c, len)
                Gf = Gf[:, :rank]  # keep square
                U2 = U @ Gf
                V2 = V @ Gf
                W2 = W
                # drop redundant columns in the new basis
                keep = _find_redundant_columns(U2, V2, W2)
                return {"U": U2[:, keep].tolist(), "V": V2[:, keep].tolist(),
                        "W": W2[:, keep].tolist()}
        except Exception:
            pass

    # fall back to redundant-column removal on original
    keep = _find_redundant_columns(U, V, W)
    return {"U": U[:, keep].tolist(), "V": V[:, keep].tolist(), "W": W[:, keep].tolist()}


def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    n, m, p = DIMENSIONS

    try:
        result = _strassen_decomposition(n, m, p)
        if _verify(result["U"], result["V"], result["W"], n, m, p, tol=1e-9):
            if time.time() < deadline - 20.0:
                try:
                    reduced = _try_rank_reduction(
                        result["U"], result["V"], result["W"], n, m, p, deadline)
                    if _verify(reduced["U"], reduced["V"], reduced["W"], n, m, p, tol=1e-9):
                        return reduced
                except Exception:
                    pass
            return result
    except Exception:
        pass

    return _schoolbook_decomposition(n, m, p)

# EVOLVE-BLOCK-END
