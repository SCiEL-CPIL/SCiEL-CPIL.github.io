import numpy as np
import time

def construct():
    n, m, p = 3, 3, 7
    # Dimensions: U (9, R), V (21, R), W (21, R)
    
    # Part 1: 3x3x3 multiplication (k=0,1,2)
    # Rank 23 decomposition
    # I will use a known rank 23 decomposition for 3x3x3.
    # If I don't have it, I will use rank 27 (schoolbook) which is safe.
    # Let's try to use a rank 24 decomposition if possible, or 23.
    # For now, I will use a rank 27 decomposition to ensure validity, 
    # and then try to improve.
    
    # Schoolbook 3x3x3
    U_333 = np.zeros((9, 27))
    V_333 = np.zeros((9, 27))
    W_333 = np.zeros((9, 27))
    col = 0
    for i in range(3):
        for j in range(3):
            for k in range(3):
                U_333[i*3+j, col] = 1.0
                V_333[j*3+k, col] = 1.0
                W_333[k*3+i, col] = 1.0
                col += 1
                
    # Part 2: 3x3x4 multiplication (k=3,4,5,6)
    # This is 4 copies of 3x3x1
    # Rank 9 for each, total 36
    U_334 = np.zeros((9, 36))
    V_334 = np.zeros((12, 36)) # 3 rows for j, 4 cols for k
    W_334 = np.zeros((12, 36)) # 3 rows for i, 4 cols for k
    
    col = 0
    for k_idx in range(4): # k=3,4,5,6
        for i in range(3):
            for j in range(3):
                U_334[i*3+j, col] = 1.0
                V_334[j*4+k_idx, col] = 1.0
                W_334[k_idx*3+i, col] = 1.0
                col += 1
                
    # Combine
    total_rank = 27 + 36
    U = np.zeros((9, total_rank))
    V = np.zeros((21, total_rank))
    W = np.zeros((21, total_rank))
    
    # Place 3x3x3 part
    U[:, :27] = U_333
    for j in range(3):
        for k in range(3):
            V[j*7+k, :27] = V_333[j*3+k, :]
            W[k*3+i if False else k*3+i, :27] = W_333[k*3+i, :] # Fix index
    # Correct W placement
    for k in range(3):
        for i in range(3):
            W[k*3+i, :27] = W_333[k*3+i, :]
            
    # Place 3x3x4 part
    U[:, 27:] = U_334
    for j in range(3):
        for k_idx in range(4):
            V[j*7 + 3 + k_idx, 27:] = V_334[j*4+k_idx, :]
    for k_idx in range(4):
        for i in range(3):
            W[(3+k_idx)*3+i, 27:] = W_334[k_idx*3+i, :]
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
