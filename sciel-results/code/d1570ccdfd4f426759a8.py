import numpy as np
from itertools import product

def construct():
    n, m, p = 3, 3, 8
    rank = n * m * p
    
    U_list = []
    V_list = []
    W_list = []
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_col = [0] * (n * m)
                V_col = [0] * (m * p)
                W_col = [0] * (p * n)
                
                U_col[i * m + j] = 1
                V_col[j * p + k] = 1
                W_col[k * n + i] = 1
                
                U_list.append(U_col)
                V_list.append(V_col)
                W_list.append(W_col)
    
    U = list(zip(*U_list)) if U_list else []
    V = list(zip(*V_list)) if V_list else []
    W = list(zip(*W_list)) if W_list else []
    
    return {"U": list(U), "V": list(V), "W": list(W)}
