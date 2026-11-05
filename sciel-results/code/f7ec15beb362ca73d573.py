import numpy as np
import time

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Splits for dimensions
    n_splits = [2, 1]
    m_splits = [2, 2, 1]
    p_splits = [2, 2, 2]
    
    n_starts = [0, 2]
    m_starts = [0, 2, 4]
    p_starts = [0, 2, 4]
    
    # Pre-calculate total rank
    total_rank = 0
    block_ranks = []
    for i_idx in range(len(n_splits)):
        r_i = n_splits[i_idx]
        for k_idx in range(len(m_splits)):
            c_k = m_splits[k_idx]
            for j_idx in range(len(p_splits)):
                c_j = p_splits[j_idx]
                if r_i == 2 and c_k == 2 and c_j == 2:
                    block_ranks.append(7)
                else:
                    block_ranks.append(r_i * c_k * c_j)
                    
    total_rank = sum(block_ranks)
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    
    for i_idx in range(len(n_splits)):
        i_start = n_starts[i_idx]
        r_i = n_splits[i_idx]
        for k_idx in range(len(m_splits)):
            k_start = m_starts[k_idx]
            c_k = m_splits[k_idx]
            for j_idx in range(len(p_splits)):
                j_start = p_starts[j_idx]
                c_j = p_splits[j_idx]
                
                if r_i == 2 and c_k == 2 and c_j == 2:
                    # Strassen decomposition
                    # M1
                    U[(i_start+0)*m + (k_start+0)][col] += 1
                    U[(i_start+1)*m + (k_start+1)][col] += 1
                    V[(k_start+0)*p + (j_start+0)][col] += 1
                    V[(k_start+1)*p + (j_start+1)][col] += 1
                    W[(j_start+0)*n + (i_start+0)][col] += 1
                    W[(j_start+1)*n + (i_start+1)][col] += 1
                    col += 1
                    
                    # M2
                    U[(i_start+1)*m + (k_start+0)][col] += 1
                    U[(i_start+1)*m + (k_start+1)][col] += 1
                    V[(k_start+0)*p + (j_start+0)][col] += 1
                    W[(j_start+0)*n + (i_start+1)][col] += 1
                    col += 1
                    
                    # M3
                    U[(i_start+0)*m + (k_start+0)][col] += 1
                    V[(k_start+1)*p + (j_start+0)][col] += 1
                    V[(k_start+1)*p + (j_start+1)][col] += -1
                    W[(j_start+1)*n + (i_start+0)][col] += 1
                    col += 1
                    
                    # M4
                    U[(i_start+1)*m + (k_start+1)][col] += 1
                    V[(k_start+0)*p + (j_start+1)][col] += 1
                    V[(k_start+0)*p + (j_start+0)][col] += -1
                    W[(j_start+0)*n + (i_start+1)][col] += 1
                    W[(j_start+1)*n + (i_start+1)][col] += 1
                    col += 1
                    
                    # M5
                    U[(i_start+0)*m + (k_start+0)][col] += 1
                    U[(i_start+0)*m + (k_start+1)][col] += 1
                    V[(k_start+1)*p + (j_start+1)][col] += 1
                    W[(j_start+0)*n + (i_start+0)][col] += -1
                    W[(j_start+1)*n + (i_start+0)][col] += 1
                    col += 1
                    
                    # M6
                    U[(i_start+1)*m + (k_start+0)][col] += 1
                    U[(i_start+0)*m + (k_start+0)][col] += -1
                    V[(k_start+0)*p + (j_start+0)][col] += 1
                    V[(k_start+0)*p + (j_start+1)][col] += 1
                    W[(j_start+1)*n + (i_start+1)][col] += 1
                    col += 1
                    
                    # M7
                    U[(i_start+0)*m + (k_start+1)][col] += 1
                    U[(i_start+1)*m + (k_start+1)][col] += -1
                    V[(k_start+1)*p + (j_start+0)][col] += 1
                    V[(k_start+1)*p + (j_start+1)][col] += 1
                    W[(j_start+0)*n + (i_start+0)][col] += 1
                    W[(j_start+1)*n + (i_start+1)][col] += 1
                    col += 1
                    
                else:
                    # Schoolbook decomposition
                    for a in range(r_i):
                        for b in range(c_k):
                            for c in range(c_j):
                                U[(i_start+a)*m + (k_start+b)][col] = 1
                                V[(k_start+b)*p + (j_start+c)][col] = 1
                                W[(j_start+c)*n + (i_start+a)][col] = 1
                                col += 1
                                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
