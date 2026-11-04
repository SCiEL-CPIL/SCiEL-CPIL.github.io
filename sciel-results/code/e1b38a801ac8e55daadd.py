"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 7]

# EVOLVE-BLOCK-START
import time
import numpy as np

def construct():
    n, m, p = DIMENSIONS
    # Splits
    n_splits = [1, 2]
    m_splits = [1, 2, 2]
    p_splits = [1, 2, 2, 2]
    
    n_offsets = np.cumsum([0] + n_splits)[:-1]
    m_offsets = np.cumsum([0] + m_splits)[:-1]
    p_offsets = np.cumsum([0] + p_splits)[:-1]
    
    # Pre-calculate total rank
    total_rank = 0
    blocks = []
    for i_idx, n_sz in enumerate(n_splits):
        for j_idx, m_sz in enumerate(m_splits):
            for k_idx, p_sz in enumerate(p_splits):
                if n_sz == 2 and m_sz == 2 and p_sz == 2:
                    r = 7
                else:
                    r = n_sz * m_sz * p_sz
                blocks.append((i_idx, j_idx, k_idx, n_sz, m_sz, p_sz, r))
                total_rank += r
                
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for i_idx, j_idx, k_idx, n_sz, m_sz, p_sz, r in blocks:
        i_off = n_offsets[i_idx]
        j_off = m_offsets[j_idx]
        k_off = p_offsets[k_idx]
        
        if n_sz == 2 and m_sz == 2 and p_sz == 2:
            # Strassen 2x2x2
            # Terms:
            # 1: (A00+A11)(B00+B11) -> C00+1, C11+1
            # 2: (A10+A11)B00 -> C10+1
            # 3: A00(B01-B11) -> C01+1
            # 4: A11(B10-B00) -> C10+1, C00+1
            # 5: (A00+A01)B11 -> C01+1, C00-1
            # 6: (A10-A00)(B00+B01) -> C11+1
            # 7: (A01-A11)(B10+B11) -> C00+1, C11-1
            
            # Helper to add a term
            def add_term(u_vals, v_vals, w_vals, u_idx, v_idx, w_idx):
                nonlocal col
                # u_idx, v_idx, w_idx are lists of (local_i, local_j), (local_j, local_k), (local_k, local_i)
                for (ui, uj), val in zip(u_idx, u_vals):
                    U[i_off + ui][j_off + uj][col] += val
                for (vj, vk), val in zip(v_idx, v_vals):
                    V[j_off + vj][k_off + vk][col] += val
                for (wk, wi), val in zip(w_idx, w_vals):
                    W[k_off + wk][i_off + wi][col] += val
                col += 1

            # Term 1
            add_term([1, 1], [1, 1], [1, 1], 
                     [(0,0), (1,1)], [(0,0), (1,1)], [(0,0), (1,1)])
            # Term 2
            add_term([1, 1], [1], [1], 
                     [(1,0), (1,1)], [(0,0)], [(0,1)])
            # Term 3
            add_term([1], [1, -1], [1], 
                     [(0,0)], [(0,1), (1,1)], [(1,0)])
            # Term 4
            add_term([1], [1, -1], [1, 1], 
                     [(1,1)], [(1,0), (0,0)], [(0,1), (0,0)])
            # Term 5
            add_term([1, 1], [1], [1, -1], 
                     [(0,0), (0,1)], [(1,1)], [(1,0), (0,0)])
            # Term 6
            add_term([1, -1], [1, 1], [1], 
                     [(1,0), (0,0)], [(0,0), (0,1)], [(1,1)])
            # Term 7
            add_term([1, -1], [1, 1], [1, -1], 
                     [(0,1), (1,1)], [(1,0), (1,1)], [(0,0), (1,1)])

        else:
            # Schoolbook
            for ui in range(n_sz):
                for uj in range(m_sz):
                    for uk in range(p_sz):
                        U[i_off + ui][j_off + uj][col] = 1
                        V[j_off + uj][k_off + uk][col] = 1
                        W[k_off + uk][i_off + ui][col] = 1
                        col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
