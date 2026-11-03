import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    rank = 98
    
    # Deadline guard
    start_time = time.time()
    deadline = 140.0 # 10s margin
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    # Strassen 2x2x2 base case (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    # M2
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1
    # M3
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1
    # M4
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1
    # M5
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1
    # M6
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    # M7
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    steps = [
        {'A': [(1, 0), (1, 3)], 'B': [(1, 0), (1, 3)], 'C': [(1, 0), (1, 3)]},
        {'A': [(1, 2), (1, 3)], 'B': [(1, 0)], 'C': [(1, 2), (-1, 3)]},
        {'A': [(1, 0)], 'B': [(1, 1), (-1, 3)], 'C': [(1, 1), (1, 3)]},
        {'A': [(1, 3)], 'B': [(1, 2), (-1, 0)], 'C': [(1, 0), (1, 2)]},
        {'A': [(1, 0), (1, 1)], 'B': [(1, 3)], 'C': [(-1, 0), (1, 1)]},
        {'A': [(1, 2), (-1, 0)], 'B': [(1, 0), (1, 1)], 'C': [(1, 3)]},
        {'A': [(1, 1), (-1, 3)], 'B': [(1, 2), (1, 3)], 'C': [(1, 0)]}
    ]

    def get_global_indices(block_idx, dim=4):
        r_start = (block_idx // 2) * 2
        c_start = (block_idx % 2) * 2
        indices = []
        for r in range(2):
            for c in range(2):
                indices.append((r_start + r, c_start + c))
        return indices

    for half in range(2):
        for k_idx, step in enumerate(steps):
            for m_s in range(7):
                col = half * 49 + k_idx * 7 + m_s
                
                if time.time() > deadline:
                    break
                
                for coeff_a, block_idx in step['A']:
                    for (r, c) in get_global_indices(block_idx):
                        global_a_idx = r * 4 + c
                        local_r = r % 2
                        local_c = c % 2
                        local_a_idx = local_r * 2 + local_c
                        U[global_a_idx, col] += coeff_a * U_2[local_a_idx, m_s]
                
                for coeff_b, block_idx in step['B']:
                    for (r, c) in get_global_indices(block_idx):
                        global_b_idx = r * 8 + (c + half * 4)
                        local_r = r % 2
                        local_c = c % 2
                        local_b_idx = local_r * 2 + local_c
                        V[global_b_idx, col] += coeff_b * V_2[local_b_idx, m_s]
                
                for coeff_c, block_idx in step['C']:
                    for (r, c) in get_global_indices(block_idx):
                        global_w_idx = (c + half * 4) * 4 + r
                        local_r = r % 2
                        local_c = c % 2
                        local_w_idx = local_r * 2 + local_c
                        W[global_w_idx, col] += coeff_c * W_2[local_w_idx, m_s]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
