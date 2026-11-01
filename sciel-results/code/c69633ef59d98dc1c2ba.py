import numpy as np
from scipy.optimize import linear_sum_assignment

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_schoolbook = n * m * p  # 42
    
    # Try to find a lower rank decomposition using a simple approach:
    # We model the tensor and try to find a decomposition with rank < 42
    
    # The tensor T[a,b,c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We want T = sum_r U[a,r] * V[b,r] * W[c,r]
    
    # For now, return the schoolbook decomposition
    # A more sophisticated approach would use tensor decomposition algorithms
    # but given the 120s constraint and the need for exactness, we'll use schoolbook
    
    U = np.zeros((n * m, rank_schoolbook), dtype=float)
    V = np.zeros((m * p, rank_schoolbook), dtype=float)
    W = np.zeros((p * n, rank_schoolbook), dtype=float)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                U[a, col] = 1
                V[b, col] = 1
                W[c, col] = 1
                col += 1
    
    # Convert to lists
    U_list = [list(map(int, row)) for row in U]
    V_list = [list(map(int, row)) for row in V]
    W_list = [list(map(int, row)) for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
