import time
import numpy as np

DIMENSIONS = [5, 5, 5]

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin
    
    n, m, p = DIMENSIONS  # 5, 5, 5
    
    # Partitions: 2 + 2 + 1
    n_parts = [2, 2, 1]
    m_parts = [2, 2, 1]
    p_parts = [2, 2, 1]
    
    # Compute start indices
    n_starts = [0]
    m_starts = [0]
    p_starts = [0]
    for i in range(1, len(n_parts)):
        n_starts.append(n_starts[-1] + n_parts[i-1])
    for i in range(1, len(m_parts)):
        m_starts.append(m_starts[-1] + m_parts[i-1])
    for i in range(1, len(p_parts)):
        p_starts.append(p_starts[-1] + p_parts[i-1])
    
    # Calculate total rank
    total_rank = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                ni, mj, pk = n_parts[i], m_parts[j], p_parts[k]
                if ni == 2 and mj == 2 and pk == 2:
                    total_rank += 7  # Strassen for 2x2x2
                else:
                    total_rank += ni * mj * pk  # Schoolbook
    
    # Initialize factor matrices
    U = np.zeros((n * m, total_rank), dtype=np.float64)
    V = np.zeros((m * p, total_rank), dtype=np.float64)
    W = np.zeros((p * n, total_rank), dtype=np.float64)
    
    col = 0
    
    for i in range(3):
        for j in range(3):
            for k in range(3):
                if time.time() > deadline:
                    # Emergency: return what we have (may be incomplete)
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                ni = n_parts[i]
                mj = m_parts[j]
                pk = p_parts[k]
                
                i_start = n_starts[i]
                j_start = m_starts[j]
                k_start = p_starts[k]
                
                if ni == 2 and mj == 2 and pk == 2:
                    # Strassen decomposition for 2x2 block
                    # Indices: i0, i1 for A rows; j0, j1 for B cols; k0, k1 for C rows
                    i0, i1 = i_start, i_start + 1
                    j0, j1 = j_start, j_start + 1
                    k0, k1 = k_start, k_start + 1
                    
                    # M1: (A11 + A22)(B11 + B22)
                    U[i0 * m + j0, col] = 1
                    U[i1 * m + j1, col] = 1
                    V[j0 * p + k0, col] = 1
                    V[j1 * p + k1, col] = 1
                    W[k0 * n + i0, col] = 1
                    W[k1 * n + i1, col] = 1
                    col += 1
                    
                    # M2: (A21 + A22)B11
                    U[i1 * m + j0, col] = 1
                    U[i1 * m + j1, col] = 1
                    V[j0 * p + k0, col] = 1
                    W[k0 * n + i1, col] = 1
                    W[k1 * n + i1, col] = -1
                    col += 1
                    
                    # M3: A11(B21 - B22)
                    U[i0 * m + j0, col] = 1
                    V[j0 * p + k1, col] = 1
                    V[j1 * p + k1, col] = -1
                    W[k1 * n + i0, col] = 1
                    W[k1 * n + i1, col] = 1
                    col += 1
                    
                    # M4: A22(B11 - B12)
                    U[i1 * m + j1, col] = 1
                    V[j1 * p + k0, col] = 1
                    V[j0 * p + k0, col] = -1
                    W[k0 * n + i0, col] = 1
                    W[k0 * n + i1, col] = 1
                    col += 1
                    
                    # M5: (A11 + A12)B22
                    U[i0 * m + j0, col] = 1
                    U[i0 * m + j1, col] = 1
                    V[j1 * p + k1, col] = 1
                    W[k0 * n + i0, col] = -1
                    W[k1 * n + i0, col] = 1
                    col += 1
                    
                    # M6: (A21 - A11)(B11 + B12)
                    U[i1 * m + j0, col] = 1
                    U[i0 * m + j0, col] = -1
                    V[j0 * p + k0, col] = 1
                    V[j0 * p + k1, col] = 1
                    W[k1 * n + i1, col] = 1
                    col += 1
                    
                    # M7: (A12 - A22)(B21 + B22)
                    U[i0 * m + j1, col] = 1
                    U[i1 * m + j1, col] = -1
                    V[j1 * p + k0, col] = 1
                    V[j1 * p + k1, col] = 1
                    W[k0 * n + i0, col] = 1
                    col += 1
                    
                else:
                    # Schoolbook for smaller blocks
                    for u in range(ni):
                        for v in range(mj):
                            for w in range(pk):
                                U[(i_start + u) * m + (j_start + v), col] = 1
                                V[(j_start + v) * p + (k_start + w), col] = 1
                                W[(k_start + w) * n + (i_start + u), col] = 1
                                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
