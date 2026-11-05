import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 5, 7]

def _validate_decomposition(U, V, W, n, m, p, tol=1e-7):
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    if U_arr.shape[0] != n * m or V_arr.shape[0] != m * p or W_arr.shape[0] != p * n:
        return False
    
    rank = U_arr.shape[1]
    if V_arr.shape[1] != rank or W_arr.shape[1] != rank:
        return False
    
    U_reshaped = U_arr.reshape(n, m, rank)
    V_reshaped = V_arr.reshape(m, p, rank)
    W_reshaped = W_arr.reshape(p, n, rank)
    
    try:
        # The tensor decomposition requires:
        # T[i,j,k,l] = sum_r U[i,j,r] * V[j,k,r] * W[k,l,r]
        # For matrix multiplication C = A * B where C[i,l] = sum_j A[i,j]*B[j,l]
        # T[i,j,k,l] = delta(j,k)
        result = np.einsum('ijr,jkr,kir->ijkl', U_reshaped, V_reshaped, W_reshaped)
    except:
        return False
    
    expected = np.zeros((n, m, m, p))
    for a in range(m):
        expected[:, a, a, :] = 1.0
    
    if np.max(np.abs(result - expected)) > tol:
        return False
    return True

def _strassen_2x2x2_terms():
    """
    Returns the 7 Strassen terms for 2x2x2 matrix multiplication.
    Each term is (U_factors, V_factors, W_factors) where each factor list
    contains (row, col, coeff) tuples.
    """
    return [
        # M1 = (A11+A22)(B11+B22)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A21+A22)B11
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A11(B21-B22)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A22(B11-B21)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A11+A12)B22
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A21-A11)(B11+B12)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A12-A22)(B21+B22)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

def _all_partitions(size, max_block=3):
    """Generate all integer partitions of size with parts up to max_block."""
    if size == 0:
        return [[]]
    results = []
    for s in range(1, min(max_block, size) + 1):
        for rest in _all_partitions(size - s, max_block):
            results.append([s] + rest)
    return results

def _compute_rank(n_blocks, m_blocks, p_blocks):
    """Compute the total rank for a given block partition."""
    rank = 0
    for ns in n_blocks:
        for ms in m_blocks:
            for ps in p_blocks:
                if ns == 2 and ms == 2 and ps == 2:
                    rank += 7
                elif ns == 1 and ms == 1 and ps == 1:
                    rank += 1
                else:
                    rank += ns * ms * ps
    return rank

def _build_decomposition(n, m, p, n_blocks, m_blocks, p_blocks):
    """Build the decomposition for the given block partition."""
    strassen_data = _strassen_2x2x2_terms()
    
    n_offsets = [0]
    for s in n_blocks:
        n_offsets.append(n_offsets[-1] + s)
    m_offsets = [0]
    for s in m_blocks:
        m_offsets.append(m_offsets[-1] + s)
    p_offsets = [0]
    for s in p_blocks:
        p_offsets.append(p_offsets[-1] + s)
    
    total_rank = _compute_rank(n_blocks, m_blocks, p_blocks)
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    for ni in range(len(n_blocks)):
        for mi in range(len(m_blocks)):
            for pi in range(len(p_blocks)):
                n0 = n_offsets[ni]
                m0 = m_offsets[mi]
                p0 = p_offsets[pi]
                ns = n_blocks[ni]
                ms = m_blocks[mi]
                ps = p_blocks[pi]
                
                if ns == 2 and ms == 2 and ps == 2:
                    for a_factors, b_factors, c_factors in strassen_data:
                        for (ar, ac, acoeff) in a_factors:
                            U[(n0 + ar) * m + (m0 + ac), col] += acoeff
                        for (br, bc, bcoeff) in b_factors:
                            V[(m0 + br) * p + (p0 + bc), col] += bcoeff
                        for (cr, cc, ccoeff) in c_factors:
                            W[(p0 + cc) * n + (n0 + cr), col] += ccoeff
                        col += 1
                else:
                    for i in range(ns):
                        for j in range(ms):
                            for k in range(ps):
                                U[(n0 + i) * m + (m0 + j), col] = 1
                                V[(m0 + j) * p + (p0 + k), col] = 1
                                W[(p0 + k) * n + (n0 + i), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

def _schoolbook_decomposition(n, m, p):
    """Standard O(nmp) decomposition."""
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    c = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][c] = 1
                V[j * p + k][c] = 1
                W[k * n + i][c] = 1
                c += 1
    
    return {"U": U, "V": V, "W": W}

def construct():
    start_time = time.time()
    deadline = start_time + 120  # 120s limit, keep margin
    
    n, m, p = DIMENSIONS
    
    # Generate all partitions for each dimension
    n_partitions = _all_partitions(n, max_block=3)
    m_partitions = _all_partitions(m, max_block=3)
    p_partitions = _all_partitions(p, max_block=3)
    
    best_rank = n * m * p
    best_partition = (n_partitions[0], m_partitions[0], p_partitions[0])
    
    # Search for the minimum rank partition
    for np_ in n_partitions:
        for mp_ in m_partitions:
            for pp_ in p_partitions:
                r = _compute_rank(np_, mp_, pp_)
                if r < best_rank:
                    best_rank = r
                    best_partition = (np_, mp_, pp_)
    
    if time.time() > deadline - 30:
        return _schoolbook_decomposition(n, m, p)
    
    try:
        result = _build_decomposition(n, m, p, *best_partition)
        if _validate_decomposition(result['U'], result['V'], result['W'], n, m, p):
            return result
        else:
            return _schoolbook_decomposition(n, m, p)
    except Exception:
        return _schoolbook_decomposition(n, m, p)
