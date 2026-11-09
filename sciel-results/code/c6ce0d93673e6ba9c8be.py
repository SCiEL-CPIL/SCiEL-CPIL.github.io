import numpy as np
import time

DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2 factors
    # U, V, W are 4x7
    U_strassen = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 1, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 1, -1],
        [1, 0, 0, 1, 0, 0, -1]
    ])
    V_strassen = np.array([
        [1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 1],
        [1, 0, 0, -1, 1, 0, 0]
    ])
    W_strassen = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 1, 0, 0],
        [1, -1, 1, 0, 0, 1, 0]
    ])
    
    # Schoolbook 3x3 factors
    # U, V, W are 9x27
    U_school = np.zeros((9, 27), dtype=int)
    V_school = np.zeros((9, 27), dtype=int)
    W_school = np.zeros((9, 27), dtype=int)
    c = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_school[i * 3 + j, c] = 1
                V_school[j * 3 + k, c] = 1
                W_school[k * 3 + i, c] = 1
                c += 1
                
    rank_strassen = 7
    rank_school = 27
    total_rank = rank_strassen * rank_school
    
    U_6 = np.zeros((36, total_rank), dtype=int)
    V_6 = np.zeros((36, total_rank), dtype=int)
    W_6 = np.zeros((36, total_rank), dtype=int)
    
    # Construct 6x6 factors
    for r_s in range(rank_strassen):
        for r_c in range(rank_school):
            r = r_s * rank_school + r_c
            
            # U: A indices
            # I, J are block indices (0..1)
            # i, j are intra-block indices (0..2)
            for I in range(2):
                for J in range(2):
                    idx_str = I * 2 + J
                    val_str = U_strassen[idx_str, r_s]
                    if val_str == 0: continue
                    
                    for i in range(3):
                        for j in range(3):
                            idx_school = i * 3 + j
                            val_school = U_school[idx_school, r_c]
                            if val_school == 0: continue
                            
                            idx_6 = (I * 3 + i) * 6 + (J * 3 + j)
                            U_6[idx_6, r] = val_str * val_school
                            
            # V: B indices
            # J, K are block indices
            # j, k are intra-block indices
            for J in range(2):
                for K in range(2):
                    idx_str = J * 2 + K
                    val_str = V_strassen[idx_str, r_s]
                    if val_str == 0: continue
                    
                    for j in range(3):
                        for k in range(3):
                            idx_school = j * 3 + k
                            val_school = V_school[idx_school, r_c]
                            if val_school == 0: continue
                            
                            idx_6 = (J * 3 + j) * 6 + (K * 3 + k)
                            V_6[idx_6, r] = val_str * val_school
                            
            # W: C indices
            # K, I are block indices (Col K, Row I)
            # k, i are intra-block indices
            for K in range(2):
                for I in range(2):
                    idx_str = K * 2 + I
                    val_str = W_strassen[idx_str, r_s]
                    if val_str == 0: continue
                    
                    for k in range(3):
                        for i in range(3):
                            idx_school = k * 3 + i
                            val_school = W_school[idx_school, r_c]
                            if val_school == 0: continue
                            
                            idx_6 = (K * 3 + k) * 6 + (I * 3 + i)
                            W_6[idx_6, r] = val_str * val_school
                            
    return {"U": U_6.tolist(), "V": V_6.tolist(), "W": W_6.tolist()}
