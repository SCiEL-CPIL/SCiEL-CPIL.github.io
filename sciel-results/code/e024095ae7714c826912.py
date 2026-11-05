"""Matrix multiplication 3x5x5 rank-minimal tensor decomposition."""
import time
import numpy as np
from functools import lru_cache

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

def _verify_decomposition(U, V, W, n, m, p):
    """Verify that the decomposition is correct."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    rank = U.shape[1]
    
    ii, jj, kk = np.meshgrid(np.arange(n), np.arange(m), np.arange(p), indexing='ij')
    u_idx = ii * m + jj
    v_idx = jj * p + kk
    w_idx = kk * n + ii
    
    U_gathered = U[u_idx, :]
    V_gathered = V[v_idx, :]
    W_gathered = W[w_idx, :]
    
    T = np.sum(U_gathered * V_gathered * W_gathered, axis=-1)
    
    if not np.allclose(T, 1.0, atol=1e-9):
        return False, np.argwhere(np.abs(T - 1.0) > 1e-9)[:5].tolist()
    
    return True, []

@lru_cache(maxsize=None)
def _get_decomposition(n, m, p):
    """
    Recursively find the minimal rank decomposition for n x m x p.
    Returns (U, V, W, rank).
    """
    if n == 1 and m == 1 and p == 1:
        U = np.array([[1.0]])
        V = np.array([[1.0]])
        W = np.array([[1.0]])
        return U, V, W, 1

    if n == 2 and m == 2 and p == 2:
        U, V, W = _strassen_2x2x2()
        return U, V, W, 7

    best_rank = float('inf')
    best_U = None
    best_V = None
    best_W = None

    # Try splitting n
    for n1 in range(1, n):
        n2 = n - n1
        U1, V1, W1, r1 = _get_decomposition(n1, m, p)
        U2, V2, W2, r2 = _get_decomposition(n2, m, p)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            U[:n1*m, :r1] = U1
            U[n1*m:, r1:] = U2
            V[:, :r1] = V1
            V[:, r1:] = V2
            
            for k in range(p):
                for i1 in range(n1):
                    W[k*n + i1, :r1] = W1[k*n1 + i1, :]
                for i2 in range(n2):
                    W[k*n + (n1 + i2), r1:] = W2[k*n2 + i2, :]
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting m
    for m1 in range(1, m):
        m2 = m - m1
        U1, V1, W1, r1 = _get_decomposition(n, m1, p)
        U2, V2, W2, r2 = _get_decomposition(n, m2, p)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            for i in range(n):
                for j1 in range(m1):
                    U[i*m + j1, :r1] = U1[i*m1 + j1, :]
                for j2 in range(m2):
                    U[i*m + (m1 + j2), r1:] = U2[i*m2 + j2, :]
            
            V[:m1*p, :r1] = V1
            V[m1*p:, r1:] = V2
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    # Try splitting p
    for p1 in range(1, p):
        p2 = p - p1
        U1, V1, W1, r1 = _get_decomposition(n, m, p1)
        U2, V2, W2, r2 = _get_decomposition(n, m, p2)
        
        if r1 + r2 < best_rank:
            U = np.zeros((n * m, r1 + r2))
            V = np.zeros((m * p, r1 + r2))
            W = np.zeros((p * n, r1 + r2))
            
            U[:, :r1] = U1
            U[:, r1:] = U2
            
            for j in range(m):
                for k1 in range(p1):
                    V[j*p + k1, :r1] = V1[j*p1 + k1, :]
                for k2 in range(p2):
                    V[j*p + (p1 + k2), r1:] = V2[j*p2 + k2, :]
            
            W[:p1*n, :r1] = W1
            W[p1*n:, r1:] = W2
            
            best_rank = r1 + r2
            best_U, best_V, best_W = U, V, W

    if best_U is None:
        U, V, W = _schoolbook(n, m, p)
        return U, V, W, n * m * p

    return best_U, best_V, best_W, best_rank

def _try_block_partition(n, m, p, n_parts, m_parts, p_parts):
    """
    Try a specific block partition and compute the total rank.
    """
    total_rank = 0
    block_data = []
    i_off = 0
    for n1 in n_parts:
        j_off = 0
        for m1 in m_parts:
            k_off = 0
            for p1 in p_parts:
                U_b, V_b, W_b, r = _get_decomposition(n1, m1, p1)
                block_data.append((U_b, V_b, W_b, r, i_off, j_off, k_off, n1, m1, p1))
                total_rank += r
                k_off += p1
            j_off += m1
        i_off += n1
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col_off = 0
    for U_b, V_b, W_b, r, i_off, j_off, k_off, n1, m1, p1 in block_data:
        for i in range(n1):
            for j in range(m1):
                U[(i_off + i) * m + (j_off + j), col_off:col_off+r] = U_b[i * m1 + j, :]
        
        for j in range(m1):
            for k in range(p1):
                V[(j_off + j) * p + (k_off + k), col_off:col_off+r] = V_b[j * p1 + k, :]
        
        for k in range(p1):
            for i in range(n1):
                W[(k_off + k) * n + (i_off + i), col_off:col_off+r] = W_b[k * n1 + i, :]
        
        col_off += r
    
    return U, V, W, total_rank

def construct():
    deadline = time.time() + 140
    n, m, p = DIMENSIONS
    
    # Get the standard recursive decomposition
    U, V, W, rank = _get_decomposition(n, m, p)
    best_U, best_V, best_W, best_rank = U, V, W, rank
    
    # Try various block partitions
    # Key insight: 5 = 2+2+1 allows using Strassen on 2x2x2 blocks
    n_opts = [[3], [1,2], [2,1], [1,1,1]]
    m_opts = [[5], [2,3], [3,2], [1,4], [4,1], [2,2,1], [2,1,2], [1,2,2], [1,1,3], [1,3,1], [3,1,1]]
    p_opts = [[5], [2,3], [3,2], [1,4], [4,1], [2,2,1], [2,1,2], [1,2,2], [1,1,3], [1,3,1], [3,1,1]]
    
    tried = set()
    
    for np_ in n_opts:
        for mp_ in m_opts:
            for pp_ in p_opts:
                if time.time() > deadline:
                    break
                key = (tuple(np_), tuple(mp_), tuple(pp_))
                if key in tried:
                    continue
                tried.add(key)
                
                try:
                    U_try, V_try, W_try, r_try = _try_block_partition(n, m, p, np_, mp_, pp_)
                    if r_try < best_rank:
                        valid, _ = _verify_decomposition(U_try, V_try, W_try, n, m, p)
                        if valid:
                            best_U, best_V, best_W, best_rank = U_try, V_try, W_try, r_try
                except Exception:
                    pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

# EVOLVE-BLOCK-END
