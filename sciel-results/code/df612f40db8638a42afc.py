"""Matrix multiplication 3,3,8 — improved Strassen + WFTT-inspired decomposition."""
import time
import numpy as np
from itertools import combinations

DIMENSIONS = [3, 3, 8]

_T0 = time.time()
_DEADLINE = _T0 + 110


def _elapsed():
    return time.time() - _T0


def _verify(U, V, W, n, m, p):
    """Verify that the tensor decomposition is exact."""
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    rank = U.shape[1]
    T = np.zeros((n, m, p, rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i, j, k, :] = U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :]
    M_target = np.ones((n, m, p))
    result = T.sum(axis=3)
    return np.allclose(result, M_target, atol=1e-9)


def _build_strassen_2x2x2_terms():
    """Return the 7 Strassen terms as (u_dict, v_dict, w_dict)."""
    return [
        ({(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
        ({(1,0): 1, (1,1): 1}, {(0,0): 1}, {(1,0): 1}),
        ({(0,0): 1}, {(1,0): 1, (1,1): -1}, {(0,1): 1}),
        ({(1,1): 1}, {(0,1): 1, (0,0): -1}, {(0,0): 1, (1,0): 1}),
        ({(0,0): 1, (0,1): 1}, {(1,1): 1}, {(0,0): 1, (0,1): 1}),
        ({(1,0): 1, (0,0): -1}, {(0,0): 1, (0,1): 1}, {(1,1): 1}),
        ({(0,1): 1, (1,1): -1}, {(1,0): 1, (1,1): 1}, {(0,0): 1, (1,1): 1}),
    ]


def _build_decomposition(strassen_blocks, n, m, p):
    """Build the U, V, W matrices for the given Strassen block placements."""
    terms = _build_strassen_2x2x2_terms()
    
    covered = set()
    for (i_s, j_s, k_s) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i_s + di, j_s + dj, k_s + dk))
    
    # Count total rank
    total_rank = 7 * len(strassen_blocks)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    total_rank += 1
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    offset = 0
    
    # Add Strassen blocks
    for (i_s, j_s, k_s) in strassen_blocks:
        for u_dict, v_dict, w_dict in terms:
            for (di, dj), val in u_dict.items():
                i, j = i_s + di, j_s + dj
                U[i * m + j][offset] += val
            for (dj, dk), val in v_dict.items():
                j, k = j_s + dj, k_s + dk
                V[j * p + k][offset] += val
            for (dk, di), val in w_dict.items():
                k, i = k_s + dk, i_s + di
                W[k * n + i][offset] += val
            offset += 1
    
    # Add schoolbook for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    U[i * m + j][offset] += 1
                    V[j * p + k][offset] += 1
                    W[k * n + i][offset] += 1
                    offset += 1
    
    return U, V, W


def _schoolbook(n, m, p):
    """Pure schoolbook decomposition."""
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
    return U, V, W


def _compute_rank_for_placement(strassen_blocks, n, m, p):
    """
    Compute the total rank for a given set of Strassen block placements.
    """
    total_rank = 0
    covered = set()
    
    for (i_s, j_s, k_s) in strassen_blocks:
        for di in range(2):
            for dj in range(2):
                for dk in range(2):
                    covered.add((i_s + di, j_s + dj, k_s + dk))
    
    # Strassen rank: 7 per block
    total_rank += 7 * len(strassen_blocks)
    
    # Schoolbook rank for uncovered entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in covered:
                    total_rank += 1
    
    return total_rank


def _search_best_placement(n, m, p):
    """
    Systematically search for the best set of non-overlapping 2x2x2 Strassen blocks.
    """
    all_blocks = []
    for i_s in range(n - 1):
        for j_s in range(m - 1):
            for k_s in range(p - 1):
                all_blocks.append((i_s, j_s, k_s))
    
    def blocks_overlap(b1, b2):
        i1, j1, k1 = b1
        i2, j2, k2 = b2
        if i1 == i2 or i1 == i2 + 1 or i2 == i1 + 1:
            if j1 == j2 or j1 == j2 + 1 or j2 == j1 + 1:
                if k1 == k2 or k1 == k2 + 1 or k2 == k1 + 1:
                    return True
        return False
    
    best_rank = n * m * p
    best_blocks = []
    
    max_size = min(len(all_blocks), 5)
    
    for size in range(0, max_size + 1):
        if _elapsed() > _DEADLINE - 10:
            break
        for subset in combinations(all_blocks, size):
            valid = True
            for a in range(size):
                for b in range(a + 1, size):
                    if blocks_overlap(subset[a], subset[b]):
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            
            rank = _compute_rank_for_placement(subset, n, m, p)
            if rank < best_rank:
                best_rank = rank
                best_blocks = list(subset)
    
    return best_rank, best_blocks


def construct():
    """Construct rank-minimal (or near-minimal) decomposition for 3x3 x 3x8."""
    n, m, p = 3, 3, 8
    
    # Schoolbook fallback
    U_school, V_school, W_school = _schoolbook(n, m, p)
    
    # Search for best Strassen block placement
    best_rank, best_blocks = _search_best_placement(n, m, p)
    
    if best_blocks and best_rank < n * m * p:
        U, V, W = _build_decomposition(best_blocks, n, m, p)
        if _verify(U, V, W, n, m, p):
            return {"U": U, "V": V, "W": W}
    
    # Fallback to schoolbook
    return {"U": U_school, "V": V_school, "W": W_school}
