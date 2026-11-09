import numpy as np
import time
import random

DIMENSIONS = [4, 5, 6]

def _elapsed(start_time):
    return time.time() - start_time

def _deadline_exceeded(start_time, deadline=130):
    return _elapsed(start_time) > deadline

def _build_optimized():
    n, m, p = DIMENSIONS
    
    # Strassen blocks
    blocks = [
        (0, 2, 0, 2, 0, 2),
        (2, 4, 2, 4, 2, 4)
    ]
    
    # Strassen matrices
    U_strassen = np.zeros((2, 2, 7), dtype=int)
    U_strassen[0, 0, 0] = 1; U_strassen[1, 1, 0] = 1
    U_strassen[1, 0, 1] = 1; U_strassen[1, 1, 1] = 1
    U_strassen[0, 0, 2] = 1
    U_strassen[1, 1, 3] = 1
    U_strassen[0, 0, 4] = 1; U_strassen[0, 1, 4] = 1
    U_strassen[1, 0, 5] = 1; U_strassen[0, 0, 5] = -1
    U_strassen[0, 1, 6] = 1; U_strassen[1, 1, 6] = -1
    
    V_strassen = np.zeros((2, 2, 7), dtype=int)
    V_strassen[0, 0, 0] = 1; V_strassen[1, 1, 0] = 1
    V_strassen[0, 0, 1] = 1
    V_strassen[0, 1, 2] = 1; V_strassen[1, 1, 2] = -1
    V_strassen[1, 0, 3] = 1; V_strassen[0, 0, 3] = -1
    V_strassen[1, 1, 4] = 1
    V_strassen[0, 0, 5] = 1; V_strassen[0, 1, 5] = 1
    V_strassen[1, 0, 6] = 1; V_strassen[1, 1, 6] = 1
    
    W_strassen = np.zeros((2, 2, 7), dtype=int)
    W_strassen[0, 0, 0] = 1; W_strassen[0, 0, 3] = 1; W_strassen[0, 0, 4] = -1; W_strassen[0, 0, 6] = 1
    W_strassen[0, 1, 2] = 1; W_strassen[0, 1, 4] = 1
    W_strassen[1, 0, 1] = 1; W_strassen[1, 0, 3] = 1
    W_strassen[1, 1, 0] = 1; W_strassen[1, 1, 1] = -1; W_strassen[1, 1, 2] = 1; W_strassen[1, 1, 5] = 1

    # Identify indices covered by Strassen
    strassen_indices = set()
    for (i_s, i_e, j_s, j_e, k_s, k_e) in blocks:
        for i in range(i_s, i_e):
            for j in range(j_s, j_e):
                for k in range(k_s, k_e):
                    strassen_indices.add((i, j, k))
    
    # Identify indices for j=4 (full grid)
    j4_indices = set()
    for i in range(n):
        for k in range(p):
            j4_indices.add((i, 4, k))
            
    # Total rank calculation
    # Strassen: 2 * 7 = 14
    # j=4: 1
    # Rest: Schoolbook
    rest_indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) not in strassen_indices and (i, j, k) not in j4_indices:
                    rest_indices.append((i, j, k))
                    
    total_rank = 14 + 1 + len(rest_indices)
    
    U = np.zeros((n * m, total_rank), dtype=int)
    V = np.zeros((m * p, total_rank), dtype=int)
    W = np.zeros((p * n, total_rank), dtype=int)
    
    col = 0
    
    # Add Strassen
    for (i_s, i_e, j_s, j_e, k_s, k_e) in blocks:
        for r in range(7):
            for i_loc in range(2):
                for j_loc in range(2):
                    val = U_strassen[i_loc, j_loc, r]
                    if val != 0:
                        U[(i_s + i_loc) * m + (j_s + j_loc), col] += val
            for j_loc in range(2):
                for k_loc in range(2):
                    val = V_strassen[j_loc, k_loc, r]
                    if val != 0:
                        V[(j_s + j_loc) * p + (k_s + k_loc), col] += val
            for i_loc in range(2):
                for k_loc in range(2):
                    val = W_strassen[i_loc, k_loc, r]
                    if val != 0:
                        W[(k_s + k_loc) * n + (i_s + i_loc), col] += val
            col += 1
            
    # Add j=4 optimization
    # U[i*5+4, col] = 1
    # V[4*6+k, col] = 1
    # W[k*4+i, col] = 1
    for i in range(n):
        U[i * m + 4, col] = 1
    for k in range(p):
        V[4 * p + k, col] = 1
    for i in range(n):
        for k in range(p):
            W[k * n + i, col] = 1
    col += 1
            
    # Add rest schoolbook
    for (i, j, k) in rest_indices:
        U[i * m + j, col] = 1
        V[j * p + k, col] = 1
        W[k * n + i, col] = 1
        col += 1
        
    return U, V, W, total_rank

def construct():
    n, m, p = DIMENSIONS
    U, V, W, rank = _build_optimized()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
