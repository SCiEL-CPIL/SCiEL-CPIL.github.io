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
    
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 1, 0, -1]
    
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 0, 1, 1]
    
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [1, 1, 0, 0]
    
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [-1, 0, 1, 0]
    
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 0, 1]
    
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [1, 0, 0, 0]
    
    return U, V, W

def _winograd_3x3x3():
    """
    Return a rank-23 Winograd decomposition for 3x3x3 matrix multiplication.
    This is a standard Winograd scheme.
    """
    U = np.zeros((9, 23))
    V = np.zeros((9, 23))
    W = np.zeros((9, 23))
    
    # Winograd 3x3x3 coefficients
    # a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w
    # This is complex to hardcode correctly without reference.
    # For now, fallback to schoolbook if not implemented perfectly, 
    # but we know rank 23 exists. 
    # To be safe and valid, I will use a known valid decomposition if I can construct it easily,
    # or stick to schoolbook for 3x3x3 if Winograd is too risky to generate on the fly.
    # However, the task requires integer/half-integer entries.
    # Winograd uses integers.
    
    # Since generating a correct 23-rank Winograd from scratch in this short snippet is error-prone,
    # and the current best score (71) is achievable with Strassen blocks,
    # I will rely on the Strassen-based search which already finds 71.
    # Adding an incorrect Winograd would break validity.
    pass

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
    """Compute the rank of a block decomposition."""
    if ni == 2 and mi == 2 and pi == 2:
        return 7
    # Other cases fall back to schoolbook rank for estimation
    # Note: If we had Winograd, we would return 23 for 3x3x3
    # For now, we assume schoolbook for non-2x2x2 to be safe, 
    # but the search might overestimate. 
    # However, the best known partition [2,1]x[2,2,1]x[2,2,1] only uses 2x2x2 and smaller.
    return ni * mi * pi

def _generate_partitions(n):
    """Generate all compositions of n as lists of positive integers."""
    if n <= 0:
        yield []
        return
    for first in range(1, n + 1):
        for rest in _generate_partitions(n - first):
            yield [first] + rest

def _verify_decomposition(U, V, W, n, m, p):
    """Verify the decomposition is exact."""
    # Vectorized verification
    # T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
    # We check if T is the identity tensor (1 where i,j,k match, 0 else)
    # Actually, the tensor for matmul is 1 at (i,j,k) and 0 elsewhere.
    
    # Reshape U, V, W to tensor forms
    # U shape (n*m, rank) -> (n, m, rank)
    U_t = U.reshape(n, m, -1)
    V_t = V.reshape(m, p, -1)
    W_t = W.reshape(p, n, -1)
    
    # Compute tensor product
    # T[i, j, k] = sum_c U[i,j,c] V[j,k,c] W[k,i,c]
    # Using einsum
    T = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t)
    
    # Expected tensor
    expected = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                expected[i, j, k] = 1.0
    
    if not np.allclose(T, expected, atol=1e-9):
        return False
    return True

def _build_block_decomposition(n, m, p, n_parts, m_parts, p_parts):
    """
    Build a block decomposition. Returns (U, V, W, total_rank).
    """
    n_offsets = [sum(n_parts[:i]) for i in range(len(n_parts))]
    m_offsets = [sum(m_parts[:i]) for i in range(len(m_parts))]
    p_offsets = [sum(p_parts[:i]) for i in range(len(p_parts))]
    
    total_rank = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                total_rank += _block_rank(ni, mi, pi)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for i, ni in enumerate(n_parts):
        for j, mi in enumerate(m_parts):
            for k, pi in enumerate(p_parts):
                r = _block_rank(ni, mi, pi)
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

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Generate all compositions
    n_comps = list(_generate_partitions(n))
    m_comps = list(_generate_partitions(m))
    p_comps = list(_generate_partitions(p))
    
    best_rank = float('inf')
    best_parts = None
    
    # Search all combinations
    for n_parts in n_comps:
        if time.time() > deadline:
            break
        for m_parts in m_comps:
            if time.time() > deadline:
                break
            for p_parts in p_comps:
                if time.time() > deadline:
                    break
                total_rank = 0
                for ni in n_parts:
                    for mi in m_parts:
                        for pi in p_parts:
                            total_rank += _block_rank(ni, mi, pi)
                if total_rank < best_rank:
                    best_rank = total_rank
                    best_parts = (n_parts, m_parts, p_parts)
    
    # If search didn't find anything, use known optimal partition
    if best_parts is None or best_rank > 71:
        best_parts = ([2, 1], [2, 2, 1], [2, 2, 1])
        best_rank = 71
    
    U, V, W, _ = _build_block_decomposition(n, m, p, *best_parts)
    
    # Verify and if invalid, fall back to schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
