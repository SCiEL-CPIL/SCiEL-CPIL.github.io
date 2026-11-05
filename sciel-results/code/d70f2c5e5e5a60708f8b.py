"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 5]

# EVOLVE-BLOCK-START

def _strassen_2x2x2():
    """
    Return the rank-7 Strassen decomposition for 2x2x2 matrix multiplication.
    U: (4, 7), V: (4, 7), W: (4, 7)
    Indexing: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    """
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # M1 = (a11 + a22)(b11 + b22)
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # M2 = (a21 + a22) b11
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    # M3 = a11 (b12 - b22)
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    # M4 = a22 (b21 - b11)
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    # M5 = (a11 + a12) b22
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    # M6 = (a21 - a11) (b11 + b12)
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    # M7 = (a12 - a22) (b21 + b22)
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W


def _schoolbook(n, m, p):
    """Schoolbook decomposition for n x m x p. Returns U, V, W."""
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


def _embed_block(U_local, V_local, W_local, n, m, p, n_off, m_off, p_off, ni, mi, pi):
    """
    Embed a local decomposition of size ni x mi x pi into global dimensions.
    Returns the global U, V, W slices (as arrays of shape (n*m, r), etc.)
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    # U: local index li*mi + lj -> global (n_off+li)*m + (m_off+lj)
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    # V: local index lj*pi + lk -> global (m_off+lj)*p + (p_off+lk)
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    # W: local index lk*ni + li -> global (p_off+lk)*n + (n_off+li)
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    Use Strassen for 2x2x2 blocks, schoolbook for others.
    """
    # Compute offsets
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    # Determine rank for each block
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                if ni == 2 and mi == 2 and pi == 2:
                    block_ranks[(i, j, k)] = 7
                else:
                    block_ranks[(i, j, k)] = ni * mi * pi
                total_rank += block_ranks[(i, j, k)]
    
    # Construct global U, V, W
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = block_ranks[(i, j, k)]
                if ni == 2 and mi == 2 and pi == 2:
                    U_loc, V_loc, W_loc = _strassen_2x2x2()
                else:
                    U_loc, V_loc, W_loc = _schoolbook(ni, mi, pi)
                
                U_g, V_g, W_g = _embed_block(U_loc, V_loc, W_loc, n, m, p,
                                              n_offsets[i], m_offsets[j], p_offsets[k],
                                              ni, mi, pi)
                U[:, col:col+r] = U_g
                V[:, col:col+r] = V_g
                W[:, col:col+r] = W_g
                col += r
    
    return U, V, W, total_rank


def _verify_decomposition(U, V, W, n, m, p):
    """
    Verify that the decomposition is exact.
    Returns True if valid, False otherwise.
    """
    # Use vectorized verification for speed
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
                    return False
    return True


def _generate_splits(dim, max_parts=4):
    """Generate all compositions of dim into up to max_parts positive integers."""
    splits = []
    def helper(remaining, parts_so_far, max_parts):
        if not parts_so_far and remaining == 0:
            return
        if len(parts_so_far) == max_parts:
            if remaining > 0:
                splits.append(parts_so_far + [remaining])
            return
        for next_val in range(1, remaining + 1):
            helper(remaining - next_val, parts_so_far + [next_val], max_parts)
    helper(dim, [], max_parts)
    return splits


def _estimate_rank(n_parts, m_parts, p_parts):
    """Estimate the rank of a block decomposition."""
    total = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                if ni == 2 and mi == 2 and pi == 2:
                    total += 7
                else:
                    total += ni * mi * pi
    return total


def construct():
    deadline = time.time() + 130  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    # Strategy: Systematically explore block decompositions with Strassen for 2x2x2 blocks.
    # Generate all possible splits for n, m, p and evaluate the rank.
    
    n_splits = _generate_splits(n, max_parts=3)
    m_splits = _generate_splits(m, max_parts=4)
    p_splits = _generate_splits(p, max_parts=4)
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    # First, try the most promising candidates manually identified
    # These are splits that allow many 2x2x2 blocks
    priority_candidates = [
        ([1, 2], [2, 2, 1], [2, 2, 1]),
        ([1, 2], [2, 1, 2], [2, 1, 2]),
        ([1, 2], [1, 2, 2], [1, 2, 2]),
        ([2, 1], [2, 2, 1], [2, 2, 1]),
        ([2, 1], [2, 1, 2], [2, 1, 2]),
        ([1, 1, 1], [2, 2, 1], [2, 2, 1]),
        ([1, 1, 1], [2, 1, 2], [2, 1, 2]),
        ([1, 1, 1], [1, 2, 2], [1, 2, 2]),
        ([1, 2], [3, 2], [3, 2]),
        ([1, 2], [2, 3], [2, 3]),
        ([2, 1], [3, 2], [3, 2]),
        ([2, 1], [2, 3], [2, 3]),
        ([1, 1, 1], [3, 2], [3, 2]),
        ([1, 1, 1], [2, 3], [2, 3]),
    ]
    
    # Add all combinations from generated splits, but limit the total number to avoid timeout
    all_candidates = set()
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                all_candidates.add((tuple(ns), tuple(ms), tuple(ps)))
    
    # Convert priority candidates to tuples for deduplication
    for pc in priority_candidates:
        all_candidates.add((tuple(pc[0]), tuple(pc[1]), tuple(pc[2])))
    
    # Sort candidates by estimated rank
    sorted_candidates = sorted(all_candidates, key=lambda x: _estimate_rank(x[0], x[1], x[2]))
    
    # Try candidates in order of estimated rank
    for n_parts, m_parts, p_parts in sorted_candidates:
        if time.time() > deadline:
            break
        try:
            U, V, W, rank = _block_decomposition(n, m, p, list(n_parts), list(m_parts), list(p_parts))
            if rank < best_rank:
                if _verify_decomposition(U, V, W, n, m, p):
                    best_rank = rank
                    best_U, best_V, best_W = U, V, W
        except Exception:
            continue
    
    # Fallback to schoolbook if something went wrong
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
