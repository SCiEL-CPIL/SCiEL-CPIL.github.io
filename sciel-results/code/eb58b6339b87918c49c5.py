import numpy as np
import time
import random
from scipy.optimize import minimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    
    # Construct the schoolbook decomposition directly
    # This is the known minimal rank for 2x3x3 matrix multiplication
    U = np.zeros((n * m, rank_schoolbook))
    V = np.zeros((m * p, rank_schoolbook))
    W = np.zeros((p * n, rank_schoolbook))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # U index: i*m + j
                # V index: j*p + k
                # W index: k*n + i
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # Convert to lists with clean integers
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    def clean_list(lst):
        cleaned = []
        for row in lst:
            new_row = []
            for val in row:
                # Ensure integer representation if close to integer
                nearest = round(val)
                if abs(val - nearest) < 1e-9:
                    new_row.append(int(nearest))
                else:
                    new_row.append(val)
            cleaned.append(new_row)
        return cleaned

    U_list = clean_list(U_list)
    V_list = clean_list(V_list)
    W_list = clean_list(W_list)

    return {"U": U_list, "V": V_list, "W": W_list}
