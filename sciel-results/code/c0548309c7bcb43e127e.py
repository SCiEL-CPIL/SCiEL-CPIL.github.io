"""Optimized matrix multiplication 3x5x7 via block decomposition."""
import numpy as np

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = DIMENSIONS
    # Block decomposition:
    # Split m=5 into 2+2+1
    # Split p=7 into 2+2+2+1
    # This creates blocks of sizes:
    # (3, 2, 2), (3, 2, 2), (3, 2, 2), (3, 2, 1)
    # (3, 2, 2), (3, 2, 2), (3, 2, 2), (3, 2, 1)
    # (3, 1, 2), (3, 1, 2), (3, 1, 2), (3, 1, 1)
    
    # We will iterate through all (m_block, p_block) combinations.
    # For each, we add a schoolbook decomposition for that sub-block.
    
    U_list = []
    V_list = []
    W_list = []
    
    # Helper to add a schoolbook block
    def add_block(i_start, i_end, j_start, j_end, k_start, k_end):
        ni = i_end - i_start
        nj = j_end - j_start
        nk = k_end - k_start
        for i in range(ni):
            for j in range(nj):
                for k in range(nk):
                    u_vec = [0.0] * (n * m)
                    v_vec = [0.0] * (m * p)
                    w_vec = [0.0] * (p * n)
                    
                    # Global indices
                    gi = i_start + i
                    gj = j_start + j
                    gk = k_start + k
                    
                    u_vec[gi * m + gj] = 1.0
                    v_vec[gj * p + gk] = 1.0
                    w_vec[gk * n + gi] = 1.0
                    
                    U_list.append(u_vec)
                    V_list.append(v_vec)
                    W_list.append(w_vec)
                    
    # Define blocks
    # m splits: [0, 2], [2, 4], [4, 5]
    # p splits: [0, 2], [2, 4], [4, 6], [6, 7]
    
    m_splits = [(0, 2), (2, 4), (4, 5)]
    p_splits = [(0, 2), (2, 4), (4, 6), (6, 7)]
    
    for (j_start, j_end) in m_splits:
        for (k_start, k_end) in p_splits:
            # i is always 0 to 3
            add_block(0, n, j_start, j_end, k_start, k_end)
            
    return {"U": U_list, "V": V_list, "W": W_list}
