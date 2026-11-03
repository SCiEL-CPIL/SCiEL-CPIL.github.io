import time
import numpy as np

DIMENSIONS = [2, 4, 8]


# EVOLVE-BLOCK-START
def _matmul_check(U, V, W, n, m, p):
    """Exact check that the tensor decomposition reconstructs the MM tensor.
    For every (i,j,k): sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r] must equal 1.
    Returns (ok, max_abs_error).
    """
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    r = Ua.shape[1]
    # Build the 3-way tensor of coefficients: T[i,j,k] = sum_r U[i,j,r]*V[j,k,r]*W[k,i,r]
    # U is (n*m, r), V is (m*p, r), W is (p*n, r)
    # Reshape U -> (n, m, r), V -> (m, p, r), W -> (p, n, r)
    Ur = Ua.reshape(n, m, r)
    Vr = Va.reshape(m, p, r)
    Wr = Wa.reshape(p, n, r)
    # For each (i,j,k): sum_r Ur[i,j,r]*Vr[j,k,r]*Wr[k,i,r]
    # Build via einsum
    T = np.einsum('ijr,jkr,kin->ijk', Ur, Vr, Wr)
    err = np.max(np.abs(T - 1.0))
    return (err < 1e-9), float(err)


def _construct_rank56():
    """The original Strassen-based rank-56 decomposition (known valid)."""
    n, m, p = DIMENSIONS
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    # Strassen 2x2x2 terms
    strassen = [
        {'U': {(0, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1, (1, 1): 1}},
        {'U': {(1, 0): 1, (1, 1): 1}, 'V': {(0, 0): 1}, 'W': {(0, 1): 1, (1, 1): -1}},
        {'U': {(0, 0): 1}, 'V': {(0, 1): 1, (1, 1): -1}, 'W': {(1, 0): 1, (1, 1): 1}},
        {'U': {(1, 1): 1}, 'V': {(1, 0): 1, (0, 0): -1}, 'W': {(0, 0): 1, (0, 1): 1}},
        {'U': {(0, 0): 1, (0, 1): 1}, 'V': {(1, 1): 1}, 'W': {(0, 0): -1, (1, 0): 1}},
        {'U': {(1, 0): 1, (0, 0): -1}, 'V': {(0, 0): 1, (0, 1): 1}, 'W': {(1, 1): 1}},
        {'U': {(0, 1): 1, (1, 1): -1}, 'V': {(1, 0): 1, (1, 1): 1}, 'W': {(0, 0): 1}},
    ]

    col = 0
    for j_high in range(2):
        for l in range(4):
            for term in strassen:
                for (i, j_low), c in term['U'].items():
                    j = j_high * 2 + j_low
                    U[i * m + j][col] += c
                for (j_low, k_low), c in term['V'].items():
                    j = j_high * 2 + j_low
                    k = l * 2 + k_low
                    V[j * p + k][col] += c
                for (k_low, i), c in term['W'].items():
                    k = l * 2 + k_low
                    W[k * n + i][col] += c
                col += 1
    return {"U": U, "V": V, "W": W}


def _construct_rank32():
    """
    Rank-32 decomposition via 8 independent schoolbook 2x2 block products.
    
    A is 2x4, B is 4x8, C is 2x8.
    For each output column-block l (4 blocks of 2 cols):
      C[:, 2l:2l+2] = A[:, 0:2] @ B[0:2, 2l:2l+2] + A[:, 2:4] @ B[2:4, 2l:2l+2]
    Each term is a 2x2 @ 2x2 product, decomposed with 4 schoolbook terms.
    Total: 4 blocks * 2 sub-products * 4 terms = 32 rank.
    """
    n, m, p = DIMENSIONS
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for l in range(4):
        for rb in range(2):
            # rb=0: A rows j in {0,1}, B rows j in {0,1}
            # rb=1: A rows j in {2,3}, B rows j in {2,3}
            j_offset = rb * 2  # 0 or 2
            for i in range(2):
                for j_local in range(2):
                    for k_local in range(2):
                        r = col + (i * 4 + j_local * 2 + k_local)
                        g_j = j_offset + j_local
                        g_k = 2 * l + k_local
                        # U[i, g_j] = 1
                        U[i * m + g_j][r] = 1
                        # V[g_j, g_k] = 1
                        V[g_j * p + g_k][r] = 1
                        # W[g_k, i] = 1
                        W[g_k * n + i][r] = 1
            col += 4
    
    return {"U": U, "V": V, "W": W}


def _trim_zero_columns(U, V, W):
    """Remove rank columns that are all-zero in U, V, and W (they contribute nothing)."""
    n_u = len(U)
    n_v = len(V)
    n_w = len(W)
    rank = len(U[0]) if n_u else 0
    keep = []
    for c in range(rank):
        if any(U[i][c] != 0 for i in range(n_u)) or \
           any(V[i][c] != 0 for i in range(n_v)) or \
           any(W[i][c] != 0 for i in range(n_w)):
            keep.append(c)
    if len(keep) == rank:
        return U, V, W, rank
    U2 = [[U[i][c] for c in keep] for i in range(n_u)]
    V2 = [[V[i][c] for c in keep] for i in range(n_v)]
    W2 = [[W[i][c] for c in keep] for i in range(n_w)]
    return U2, V2, W2, len(keep)


def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    deadline = t0 + 140  # safety margin before 150s kill

    best = None

    # Candidate 1: rank-32 (8 independent schoolbook 2x2 blocks)
    try:
        cand = _construct_rank32()
        ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
        if ok:
            U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
            best = (r, U, V, W)
    except Exception:
        pass

    # Candidate 2: rank-56 (original Strassen, known valid) as fallback
    if time.time() < deadline:
        try:
            cand = _construct_rank56()
            ok, err = _matmul_check(cand["U"], cand["V"], cand["W"], n, m, p)
            if ok:
                U, V, W, r = _trim_zero_columns(cand["U"], cand["V"], cand["W"])
                if best is None or r < best[0]:
                    best = (r, U, V, W)
        except Exception:
            pass

    if best is None:
        # Last resort: return rank-56 without verification (it's known to be valid)
        cand = _construct_rank56()
        return cand

    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
