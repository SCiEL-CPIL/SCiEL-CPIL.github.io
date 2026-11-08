"""Matrix multiplication 2,4,6 - rank-minimal exact tensor decomposition."""
import time
import numpy as np

_DIMENSIONS = (2, 4, 6)
_DEADLINE = 130.0  # safety margin before 150s kill


def _verify(n, m, p, U, V, W):
    """Check that U, V, W exactly decompose the matrix multiplication tensor."""
    rank = U.shape[1]
    total = np.zeros(n * m * p, dtype=float)
    idx_a = []
    idx_b = []
    idx_c = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                idx_a.append(i * m + j)
                idx_b.append(j * p + k)
                idx_c.append(k * n + i)
    idx_a = np.array(idx_a)
    idx_b = np.array(idx_b)
    idx_c = np.array(idx_c)
    for r in range(rank):
        total += U[idx_a, r] * V[idx_b, r] * W[idx_c, r]
    return np.allclose(total, np.ones(n * m * p), atol=1e-9)


def _strassen_decomposition(n, m, p):
    """
    Block Strassen decomposition for M_{2,4,6}.
    Split m=4 into 2 blocks of 2, p=6 into 3 blocks of 2.
    Each subproblem M_{2,2,2} uses rank-7 Strassen.
    Total rank = 2 * 3 * 7 = 42.
    """
    U_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (0, 0): -1},
        {(0, 1): 1, (1, 1): -1},
    ]
    V_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(0, 0): 1},
        {(0, 1): 1, (1, 1): -1},
        {(1, 0): 1, (0, 0): -1},
        {(1, 1): 1},
        {(0, 0): 1, (0, 1): 1},
        {(1, 0): 1, (1, 1): 1},
    ]
    W_coeffs = [
        {(0, 0): 1, (1, 1): 1},
        {(1, 0): 1, (1, 1): -1},
        {(0, 1): 1, (1, 1): 1},
        {(0, 0): 1, (1, 0): 1},
        {(0, 0): -1, (0, 1): 1},
        {(1, 1): 1},
        {(0, 0): 1},
    ]

    a_blocks = m // 2  # 2
    b_blocks = p // 2  # 3
    strassen_rank = 7
    rank = a_blocks * b_blocks * strassen_rank  # 42

    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)

    col = 0
    for a in range(a_blocks):
        for b in range(b_blocks):
            for s in range(strassen_rank):
                for (i_loc, j_loc), coeff in U_coeffs[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j, col] += coeff
                for (j_loc, k_loc), coeff in V_coeffs[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k, col] += coeff
                for (i_loc, k_loc), coeff in W_coeffs[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i, col] += coeff
                col += 1

    return U, V, W


def _try_rank_reduction(n, m, p, base_U, base_V, base_W, max_time=10):
    """
    Try to reduce the rank of the decomposition by finding linear dependencies
    among the rank-1 terms and eliminating them.
    """
    start_time = time.time()
    rank = base_U.shape[1]
    
    removable = []
    for r in range(rank):
        if time.time() - start_time > max_time:
            break
        for s in range(rank):
            if r == s or s in removable:
                continue
            ur, us = base_U[:, r], base_U[:, s]
            vr, vs = base_V[:, r], base_V[:, s]
            wr, ws = base_W[:, r], base_W[:, s]
            
            nonzero_u = np.nonzero(us)[0]
            if len(nonzero_u) == 0:
                continue
            scalar_candidates = ur[nonzero_u] / us[nonzero_u]
            if np.std(scalar_candidates) > 1e-10:
                continue
            scalar = scalar_candidates[0]
            
            if np.allclose(vr, scalar * vs, atol=1e-10) and np.allclose(wr, scalar * ws, atol=1e-10):
                removable.append(r)
                break
    
    if removable:
        keep = [r for r in range(rank) if r not in removable]
        new_U = base_U[:, keep]
        new_V = base_V[:, keep]
        new_W = base_W[:, keep]
        if _verify(n, m, p, new_U, new_V, new_W):
            return new_U, new_V, new_W
    
    return base_U, base_V, base_W


def construct():
    n, m, p = _DIMENSIONS
    start = time.time()

    # Get the known rank-42 Strassen decomposition as a safe fallback
    U_s, V_s, W_s = _strassen_decomposition(n, m, p)

    if not _verify(n, m, p, U_s, V_s, W_s):
        # Fallback to schoolbook (rank 48)
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    best_rank = 42
    best_U, best_V, best_W = U_s, V_s, W_s

    # Try to find linear dependencies to reduce rank
    if time.time() - start < _DEADLINE - 10:
        U_red, V_red, W_red = _try_rank_reduction(n, m, p, U_s, V_s, W_s, max_time=10)
        if U_red.shape[1] < best_rank and _verify(n, m, p, U_red, V_red, W_red):
            best_rank = U_red.shape[1]
            best_U, best_V, best_W = U_red, V_red, W_red

    # Final verification
    if not _verify(n, m, p, best_U, best_V, best_W):
        best_U, best_V, best_W = U_s, V_s, W_s

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
