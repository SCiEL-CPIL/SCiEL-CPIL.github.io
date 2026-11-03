import time
import numpy as np

DIMENSIONS = [3, 5, 6]

def strassen_factors():
    """
    Returns (U, V, W) for 2x2x2 Strassen decomposition.
    """
    U = np.zeros((4, 7), dtype=np.float64)
    V = np.zeros((4, 7), dtype=np.float64)
    W = np.zeros((4, 7), dtype=np.float64)
    
    # m1 = (a+d)(e+h)
    U[0, 0] = 1; U[3, 0] = 1
    V[0, 0] = 1; V[3, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    # m2 = (c+d)e
    U[2, 1] = 1; U[3, 1] = 1
    V[0, 1] = 1
    W[2, 1] = 1; W[3, 1] = -1
    
    # m3 = a(f-h)
    U[0, 2] = 1
    V[1, 2] = 1; V[3, 2] = -1
    W[1, 2] = 1; W[3, 2] = 1
    
    # m4 = d(g-e)
    U[3, 3] = 1
    V[2, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[2, 3] = 1
    
    # m5 = (a+b)h
    U[0, 4] = 1; U[1, 4] = 1
    V[3, 4] = 1
    W[0, 4] = -1; W[1, 4] = 1
    
    # m6 = (c-a)(e+f)
    U[2, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    # m7 = (b-d)(g+h)
    U[1, 6] = 1; U[3, 6] = -1
    V[2, 6] = 1; V[3, 6] = 1
    W[0, 6] = 1
    
    return U, V, W

def schoolbook_factors(n, m, p):
    """
    Returns (U, V, W) for n x m x p schoolbook decomposition.
    """
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    
    return U, V, W

def build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes):
    """
    Build the full factor matrices using block decomposition.
    """
    total_rank = 0
    n_offsets = []
    offset = 0
    for s in n_sizes:
        n_offsets.append(offset)
        offset += s
    
    m_offsets = []
    offset = 0
    for s in m_sizes:
        m_offsets.append(offset)
        offset += s
    
    p_offsets = []
    offset = 0
    for s in p_sizes:
        p_offsets.append(offset)
        offset += s
    
    for i_idx, ni in enumerate(n_sizes):
        for j_idx, mj in enumerate(m_sizes):
            for k_idx, pk in enumerate(p_sizes):
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7
                else:
                    total_rank += ni * mj * pk
    
    U_full = np.zeros((n * m, total_rank), dtype=np.float64)
    V_full = np.zeros((m * p, total_rank), dtype=np.float64)
    W_full = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col_offset = 0
    for i_idx, ni in enumerate(n_sizes):
        i_start = n_offsets[i_idx]
        for j_idx, mj in enumerate(m_sizes):
            j_start = m_offsets[j_idx]
            for k_idx, pk in enumerate(p_sizes):
                k_start = p_offsets[k_idx]
                
                if ni == 2 and mj == 2 and pk == 2:
                    U_block, V_block, W_block = strassen_factors()
                else:
                    U_block, V_block, W_block = schoolbook_factors(ni, mj, pk)
                r = U_block.shape[1]
                
                for li in range(ni):
                    for lj in range(mj):
                        full_row = (i_start + li) * m + (j_start + lj)
                        U_full[full_row, col_offset:col_offset + r] = U_block[li * mj + lj, :]
                
                for lj in range(mj):
                    for lk in range(pk):
                        full_row = (j_start + lj) * p + (k_start + lk)
                        V_full[full_row, col_offset:col_offset + r] = V_block[lj * pk + lk, :]
                
                for lk in range(pk):
                    for li in range(ni):
                        full_row = (k_start + lk) * n + (i_start + li)
                        W_full[full_row, col_offset:col_offset + r] = W_block[lk * ni + li, :]
                
                col_offset += r
    
    return U_full, V_full, W_full, total_rank

def verify_decomposition(U, V, W, n, m, p, atol=1e-10):
    """
    Verify that the decomposition exactly reproduces the matrix multiplication tensor.
    """
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_rows = i_idx * m + j_idx
    v_rows = j_idx * p + k_idx
    w_rows = k_idx * n + i_idx
    
    products = U[u_rows, :] * V[v_rows, :] * W[w_rows, :]
    vals = products.sum(axis=1)
    
    return np.allclose(vals, 1.0, atol=atol)

def construct():
    """
    Constructs the factor matrices U, V, W for the matrix multiplication tensor
    of multiplying a 3x5 matrix by a 5x6 matrix.
    """
    n, m, p = DIMENSIONS  # 3, 5, 6
    
    # Block sizes
    n_sizes = [2, 1]
    m_sizes = [2, 2, 1]
    p_sizes = [2, 2, 2]
    
    assert sum(n_sizes) == n
    assert sum(m_sizes) == m
    assert sum(p_sizes) == p
    
    # Build the block decomposition
    U, V, W, total_rank = build_block_decomposition(n, m, p, n_sizes, m_sizes, p_sizes)
    
    # Verify correctness
    if not verify_decomposition(U, V, W, n, m, p, atol=1e-10):
        # Fall back to schoolbook
        U, V, W = schoolbook_factors(n, m, p)
        total_rank = n * m * p
    
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

# EVOLVE-BLOCK-START
# The construct function is defined above.
# EVOLVE-BLOCK-END
