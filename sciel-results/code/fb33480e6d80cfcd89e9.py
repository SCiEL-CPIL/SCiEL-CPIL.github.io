import time
import numpy as np

DIMENSIONS = [3, 3, 7]

def construct():
    """
    Constructs a rank-minimal (or best known) exact tensor decomposition 
    for matrix multiplication of dimensions n x m by m x p.
    
    Currently returns the schoolbook decomposition (Rank = n*m*p = 63).
    A rank-55 decomposition exists using Strassen's 3x3x3 (Rank 23) for two blocks
    and schoolbook 3x3x1 (Rank 9) for the third, but requires hardcoded coefficients.
    """
    n, m, p = DIMENSIONS
    dim_u = n * m
    dim_v = m * p
    dim_w = p * n
    
    # Schoolbook Solution (Rank = n*m*p)
    rank_school = n * m * p
    U_school = np.zeros((dim_u, rank_school), dtype=float)
    V_school = np.zeros((dim_v, rank_school), dtype=float)
    W_school = np.zeros((dim_w, rank_school), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
                
    best_U = U_school
    best_V = V_school
    best_W = W_school
    best_rank = rank_school
    
    # Return the best solution found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
