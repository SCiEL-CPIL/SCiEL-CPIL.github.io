import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def _strassen_block_2x2x2():
    """
    Return the 7 Strassen products as triples of (u_indices, v_indices, w_indices, coeffs).
    Each product contributes to the decomposition.
    """
    # For 2x2x2 blocks, Strassen gives 7 multiplications.
    # A = [[A00, A01], [A10, A11]], B = [[B00, B01], [B10, B11]]
    # C00 = M1 + M4 - M5 - M7
    # C01 = M5 + M6
    # C10 = M2 + M3
    # C11 = M1 - M2 + M3 + M4
    # where:
    # M1 = A00*(B00+B11)
    # M2 = A10*(B00+B01)
    # M3 = A01*B11
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    
    products = []
    
    # M1 = A00 * (B00 + B11)
    # A00 is at (i0,j0), B00 at (j0,k0), B11 at (j1,k1)
    # Result goes to C00 (k0,i0) and C11 (k1,i1)
    products.append({
        'u': [(0, 0, 1)],  # (row_offset, col_offset, coeff)
        'v': [(0, 0, 1), (1, 1, 1)],
        'w': [(0, 0, 1), (1, 1, 1)]
    })
    
    # M2 = A10 * (B00 + B01)
    # A10 at (i1,j0), B00 at (j0,k0), B01 at (j0,k1)
    # Result goes to C10 (k0,i1) and C11 (k1,i1) with -1
    products.append({
        'u': [(1, 0, 1)],
        'v': [(0, 0, 1), (0, 1, 1)],
        'w': [(0, 1, 1), (1, 1, -1)]
    })
    
    # M3 = A01 * B11
    # A01 at (i0,j1), B11 at (j1,k1)
    # Result goes to C01 (k1,i0) and C11 (k1,i1)
    products.append({
        'u': [(0, 1, 1)],
        'v': [(1, 1, 1)],
        'w': [(1, 0, 1), (1, 1, 1)]
    })
    
    # M4 = A11 * (B10 - B00)
    # A11 at (i1,j1), B10 at (j1,k0), B00 at (j0,k0)
    # Result goes to C00 (k0,i0) and C11 (k0,i1)
    products.append({
        'u': [(1, 1, 1)],
        'v': [(1, 0, 1), (0, 0, -1)],
        'w': [(0, 0, 1), (0, 1, 1)]
    })
    
    # M5 = (A00 + A01) * B11
    # A00 at (i0,j0), A01 at (i0,j1), B11 at (j1,k1)
    # Result goes to C00 (k0,i0) with -1 and C01 (k1,i0)
    products.append({
        'u': [(0, 0, 1), (0, 1, 1)],
        'v': [(1, 1, 1)],
        'w': [(0, 0, -1), (1, 0, 1)]
    })
    
    # M6 = (A10 - A00) * (B00 + B01)
    # A10 at (i1,j0), A00 at (i0,j0), B00 at (j0,k0), B01 at (j0,k1)
    # Result goes to C01 (k1,i1)
    products.append({
        'u': [(1, 0, 1), (0, 0, -1)],
        'v': [(0, 0, 1), (0, 1, 1)],
        'w': [(1, 1, 1)]
    })
    
    # M7 = (A01 - A11) * (B10 + B11)
    # A01 at (i0,j1), A11 at (i1,j1), B10 at (j1,k0), B11 at (j1,k1)
    # Result goes to C00 (k0,i0)
    products.append({
        'u': [(0, 1, 1), (1, 1, -1)],
        'v': [(1, 0, 1), (1, 1, 1)],
        'w': [(0, 0, 1)]
    })
    
    return products

def _apply_strassen_to_block(U, V, W, i_start, j_start, k_start, col, n, m, p):
    """Apply Strassen decomposition to a 2x2x2 block."""
    products = _strassen_block_2x2x2()
    for prod in products:
        for (ri, ci, coeff_u) in prod['u']:
            U[(i_start + ri) * m + (j_start + ci)][col] = coeff_u
        for (ri, ci, coeff_v) in prod['v']:
            V[(j_start + ri) * p + (k_start + ci)][col] = coeff_v
        for (ri, ci, coeff_w) in prod['w']:
            W[(k_start + ri) * n + (i_start + ci)][col] = coeff_w
        col += 1
    return col

def _apply_schoolbook_to_block(U, V, W, i_start, j_start, k_start, col, ni, mj, pk, n, m, p):
    """Apply schoolbook decomposition to a block of size ni x mj x pk."""
    for u in range(ni):
        for v in range(mj):
            for w in range(pk):
                U[(i_start + u) * m + (j_start + v)][col] = 1
                V[(j_start + v) * p + (k_start + w)][col] = 1
                W[(k_start + w) * n + (i_start + u)][col] = 1
                col += 1
    return col

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin for 150s limit
    
    n, m, p = DIMENSIONS
    
    # Partition 5 into blocks: [2, 2, 1]
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute start indices
    n_starts = []
    m_starts = []
    p_starts = []
    ns = 0
    ms = 0
    ps = 0
    for sz in n_parts:
        n_starts.append(ns)
        ns += sz
    for sz in m_parts:
        m_starts.append(ms)
        ms += sz
    for sz in p_parts:
        p_starts.append(ps)
        ps += sz
    
    # Calculate total rank
    total_rank = 0
    for i in range(len(n_parts)):
        for j in range(len(m_parts)):
            for k in range(len(p_parts)):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7  # Strassen
                else:
                    total_rank += ni * mj * pk  # Schoolbook
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    num_blocks = len(n_parts) * len(m_parts) * len(p_parts)
    processed = 0
    
    for i in range(len(n_parts)):
        for j in range(len(m_parts)):
            for k in range(len(p_parts)):
                if time.time() > deadline:
                    # If we run out of time, return what we have
                    # But this shouldn't happen for such small dimensions
                    return {"U": U, "V": V, "W": W}
                
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    col = _apply_strassen_to_block(U, V, W, i_start, j_start, k_start, col, n, m, p)
                else:
                    col = _apply_schoolbook_to_block(U, V, W, i_start, j_start, k_start, col, ni, mj, pk, n, m, p)
                
                processed += 1
    
    return {"U": U, "V": V, "W": W}
