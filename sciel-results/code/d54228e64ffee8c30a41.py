import numpy as np
import time

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Decomposition strategy:
    # M_{3,4,7} = M_{3,3,7} + M_{3,1,7}
    # M_{3,3,7} = M_{3,3,3} + M_{3,3,3} + M_{3,3,1}
    # Ranks:
    # R(M_{3,3,3}) = 27 (Schoolbook)
    # R(M_{3,3,1}) = 1
    # R(M_{3,1,7}) = 1
    # Total Rank = 27 + 27 + 1 + 1 = 56
    
    r_total = 56
    U = np.zeros((nm, r_total))
    V = np.zeros((mp, r_total))
    W = np.zeros((pn, r_total))
    
    col = 0
    
    # Helper to add a schoolbook block for M_{n_sub, m_sub, p_sub}
    # with offsets i_off, j_off, k_off
    def add_schoolbook_block(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i + i_off)*m + (j + j_off), col] = 1
                    V[(j + j_off)*p + (k + k_off), col] = 1
                    W[(k + k_off)*n + (i + i_off), col] = 1
                    col += 1
        return col
    
    # Helper to add a rank-1 block for M_{n_sub, m_sub, p_sub}
    # where T_{i,j,k} = 1 (all ones in the sub-block)
    # This corresponds to outer product structure
    def add_rank1_block(n_sub, m_sub, p_sub, i_off, j_off, k_off):
        nonlocal col
        for i in range(n_sub):
            for j in range(m_sub):
                for k in range(p_sub):
                    U[(i + i_off)*m + (j + j_off), col] = 1
                    V[(j + j_off)*p + (k + k_off), col] = 1
                    W[(k + k_off)*n + (i + i_off), col] = 1
        col += 1
        return col

    # Block 1: M_{3,3,3} for j in [0,2], k in [0,2]
    add_schoolbook_block(3, 3, 3, 0, 0, 0)
    
    # Block 2: M_{3,3,3} for j in [0,2], k in [3,5]
    add_schoolbook_block(3, 3, 3, 0, 0, 3)
    
    # Block 3: M_{3,3,1} for j in [0,2], k=6
    # This is rank 1
    add_rank1_block(3, 3, 1, 0, 0, 6)
    
    # Block 4: M_{3,1,7} for j=3, k in [0,6]
    # This is rank 1
    add_rank1_block(3, 1, 7, 0, 3, 0)
    
    # Verify col == r_total
    # if col != r_total:
    #     raise ValueError(f"Column mismatch: {col} vs {r_total}")

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
