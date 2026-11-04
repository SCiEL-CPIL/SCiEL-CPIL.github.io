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
    elif ni == 1 and mi == 2 and pi == 2:
        return 4  # 1x2x2: no improvement over schoolbook
    elif ni == 2 and mi == 1 and pi == 2:
        return 4
    elif ni == 2 and mi == 2 and pi == 1:
        return 4
    else:
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
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-9:
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
    deadline = time.time() + 130
    n, m, p = DIMENSIONS
    
    # Generate all compositions
    n_comps = list(_generate_partitions(n))
    m_comps = list(_generate_partitions(m))
    p_comps = list(_generate_partitions(p))
    
    best_rank = float('inf')
    best_parts = None
    
    # Search all combinations - try to find the minimum rank partition
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
    
    if best_parts is None:
        # Fallback to schoolbook
        U, V, W = _schoolbook(n, m, p)
    else:
        U, V, W, _ = _build_block_decomposition(n, m, p, *best_parts)
    
    # Verify and if invalid, fall back to schoolbook
    if not _verify_decomposition(U, V, W, n, m, p):
        U, V, W = _schoolbook(n, m, p)
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
