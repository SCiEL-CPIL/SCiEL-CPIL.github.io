"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np

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
    """
    r = U_local.shape[1]
    U_g = np.zeros((n * m, r))
    V_g = np.zeros((m * p, r))
    W_g = np.zeros((p * n, r))
    
    for li in range(ni):
        for lj in range(mi):
            local_idx = li * mi + lj
            global_idx = (n_off + li) * m + (m_off + lj)
            U_g[global_idx, :] = U_local[local_idx, :]
    
    for lj in range(mi):
        for lk in range(pi):
            local_idx = lj * pi + lk
            global_idx = (m_off + lj) * p + (p_off + lk)
            V_g[global_idx, :] = V_local[local_idx, :]
    
    for lk in range(pi):
        for li in range(ni):
            local_idx = lk * ni + li
            global_idx = (p_off + lk) * n + (n_off + li)
            W_g[global_idx, :] = W_local[local_idx, :]
    
    return U_g, V_g, W_g


def _block_rank(ni, mi, pi):
    """Compute the rank for a block of size ni x mi x pi."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    return ni * mi * pi


def _block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Decompose n x m x p into blocks defined by n_parts, m_parts, p_parts.
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    block_ranks = {}
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = _block_rank(ni, mi, pi)
                block_ranks[(i, j, k)] = r
                total_rank += r
    
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


def _verify_decomposition_fast(U, V, W, n, m, p, tol=1e-9):
    """Fast verification using tensor contraction."""
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True


def _generate_splits(dim, max_parts=5):
    """Generate all compositions of dim into up to max_parts positive integers."""
    splits = []
    def helper(remaining, parts_so_far):
        if remaining == 0:
            if parts_so_far:
                splits.append(list(parts_so_far))
            return
        if len(parts_so_far) >= max_parts:
            return
        for next_val in range(1, remaining + 1):
            helper(remaining - next_val, parts_so_far + [next_val])
    helper(dim, [])
    return splits


def _estimate_rank(n_parts, m_parts, p_parts):
    """Estimate the total rank for a given split."""
    total = 0
    for ni in n_parts:
        for mi in m_parts:
            for pi in p_parts:
                total += _block_rank(ni, mi, pi)
    return total


def _try_reduce_rank(U, V, W, n, m, p, max_reduction, deadline):
    """
    Attempt to reduce the rank of the decomposition by finding linear dependencies.
    Returns potentially reduced U, V, W and the new rank.
    """
    current_rank = U.shape[1]
    if current_rank <= 0:
        return U, V, W, current_rank
    
    U_cur = U.copy()
    V_cur = V.copy()
    W_cur = W.copy()
    
    # Try to eliminate columns one by one by expressing them as combinations
    # of other columns. This is a heuristic approach.
    
    # For each column c, check if it can be expressed as a linear combination
    # of the other columns in a way that preserves the tensor decomposition.
    # This is equivalent to checking if the column is in the span of the others
    # in the tensor product space.
    
    # A simpler approach: try to find pairs of columns that can be merged
    # by finding a linear transformation that eliminates one.
    
    # Given the complexity, let's try a different approach:
    # Use the structure of the block decomposition to identify potential
    # simplifications.
    
    # For now, return the unchanged decomposition as a safe fallback.
    return U_cur, V_cur, W_cur, current_rank


def _search_optimal(deadline):
    """
    Search for an optimal or near-optimal decomposition.
    """
    n, m, p = 3, 5, 5
    
    # Generate all possible splits
    n_splits = _generate_splits(n, max_parts=3)
    m_splits = _generate_splits(m, max_parts=5)
    p_splits = _generate_splits(p, max_parts=5)
    
    all_candidates = []
    for ns in n_splits:
        for ms in m_splits:
            for ps in p_splits:
                rank = _estimate_rank(ns, ms, ps)
                all_candidates.append((rank, ns, ms, ps))
    
    all_candidates.sort(key=lambda x: x[0])
    
    best_U, best_V, best_W = None, None, None
    best_rank = float('inf')
    
    for rank_est, n_parts, m_parts, p_parts in all_candidates:
        if time.time() > deadline:
            break
        if rank_est >= best_rank:
            continue
        
        try:
            U, V, W, actual_rank = _block_decomposition(n, m, p, n_parts, m_parts, p_parts)
            
            # Try to reduce rank further
            U2, V2, W2, reduced_rank = _try_reduce_rank(U, V, W, n, m, p, 
                                                         actual_rank - rank_est, deadline)
            
            if reduced_rank < best_rank:
                if _verify_decomposition_fast(U2, V2, W2, n, m, p):
                    best_rank = reduced_rank
                    best_U, best_V, best_W = U2, V2, W2
        except Exception:
            continue
    
    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W = best_U, best_V, best_W
    
    return U, V, W


def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    n, m, p = DIMENSIONS  # 3, 5, 5
    
    U, V, W = _search_optimal(deadline)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
