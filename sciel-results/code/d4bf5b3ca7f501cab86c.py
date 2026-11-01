"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 2, 15]

# EVOLVE-BLOCK-START
import time as _time

_START = _time.time()
_DEADLINE = 115.0  # safety margin before the 150s kill

def _remaining():
    return _DEADLINE - (_time.time() - _START)

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check that sum_k U[:,k] V[:,k] W[:,k] == MM tensor exactly (within tol)."""
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1.0
    tmp = np.einsum('uk,vk->uvk', U, V)
    result = np.einsum('uvk,wk->uvw', tmp, W)
    if np.max(np.abs(result - T)) > tol:
        return False
    return True

def _strassen_222():
    """
    Return rank-7 decomposition of M_{2,2,2} tensor.
    U: shape (4, 7) - indices i*2+j
    V: shape (4, 7) - indices j*2+k
    W: shape (4, 7) - indices k*2+i
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))

    # M1
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    # M2
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    # M3
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    # M4
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    # M5
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    # M6
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    # M7
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1

    return U, V, W

def _rank3_221():
    """
    Return rank-3 decomposition of M_{2,2,1} tensor.
    U: shape (4, 3) - indices i*2+j
    V: shape (2, 3) - indices j (k=0 only)
    W: shape (2, 3) - indices i (k=0 only)
    """
    U = np.zeros((4, 3))
    V = np.zeros((2, 3))
    W = np.zeros((2, 3))

    U[:, 0] = [1, 0, 0, 0]
    V[:, 0] = [1, 0]
    W[:, 0] = [1, 0]

    U[:, 1] = [0, 0, 1, 0]
    V[:, 1] = [0, 1]
    W[:, 1] = [0, 1]

    U[:, 2] = [0, 1, 0, 1]
    V[:, 2] = [1, 1]
    W[:, 2] = [1, 1]

    return U, V, W

def _schoolbook(n, m, p):
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _build_rank3_per_k(n, m, p):
    """
    Rank-3p construction: place an independent rank-3 M_{2,2,1} block at each k.
    Because W-index (k*n+i) and V-index (j*p+k) are disjoint across k, the blocks
    add linearly and the total rank is 3*p. This meets the slice lower bound 3p.
    """
    U_r, V_r, W_r = _rank3_221()
    rank_single = U_r.shape[1]  # 3
    total_rank = rank_single * p

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for k in range(p):
        # U part (shared index space, but each block gets its own rank columns)
        U[:, col:col + rank_single] = U_r

        # V part: local index j -> global j*p + k
        for j in range(m):
            V[j * p + k, col:col + rank_single] = V_r[j]

        # W part: local index i -> global k*n + i
        for i in range(n):
            W[k * n + i, col:col + rank_single] = W_r[i]

        col += rank_single

    return U, V, W

def _build_strassen_plus_single(n, m, p):
    """
    Fallback: 7 Strassen-7 blocks over k-pairs + one rank-3 block for odd k.
    Rank = 7*floor(p/2) + (3 if p odd).
    """
    U_s, V_s, W_s = _strassen_222()
    U_r, V_r, W_r = _rank3_221()

    num_pairs = p // 2
    has_odd = (p % 2 == 1)

    rank_strassen = U_s.shape[1]  # 7
    rank_single = U_r.shape[1]    # 3

    total_rank = num_pairs * rank_strassen + (rank_single if has_odd else 0)

    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))

    col = 0
    for pair_idx in range(num_pairs):
        k0 = pair_idx * 2
        U[:, col:col + rank_strassen] = U_s
        for j_local in range(m):
            for k_local in range(2):
                v_local_idx = j_local * 2 + k_local
                v_global_idx = j_local * p + (k0 + k_local)
                V[v_global_idx, col:col + rank_strassen] = V_s[v_local_idx]
        for k_local in range(2):
            for i in range(n):
                w_local_idx = k_local * 2 + i
                w_global_idx = (k0 + k_local) * n + i
                W[w_global_idx, col:col + rank_strassen] = W_s[w_local_idx]
        col += rank_strassen

    if has_odd:
        k0 = num_pairs * 2
        U[:, col:col + rank_single] = U_r
        for j in range(m):
            V[j * p + k0, col:col + rank_single] = V_r[j]
        for i in range(n):
            W[k0 * n + i, col:col + rank_single] = W_r[i]
        col += rank_single

    return U, V, W

def construct():
    n, m, p = DIMENSIONS

    if _remaining() < 5:
        U, V, W = _schoolbook(n, m, p)
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Primary: rank-3p (45) construction — meets the slice lower bound.
    U, V, W = _build_rank3_per_k(n, m, p)
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Secondary: Strassen-pair + single (52) construction.
    U, V, W = _build_strassen_plus_single(n, m, p)
    if _verify(U, V, W, n, m, p, tol=1e-9):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Final fallback: schoolbook (60).
    U, V, W = _schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
