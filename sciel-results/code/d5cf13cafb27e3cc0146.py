import numpy as np

def construct():
    n, m, p = 2, 4, 6
    rank = n * m  # 8
    
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col_idx = 0
    for k in range(m):
        for i in range(n):
            # U: index (i, k)
            u_row = i * m + k
            U[u_row, col_idx] = 1.0
            
            # V: index (k, j) for all j
            for j in range(p):
                v_row = k * p + j
                V[v_row, col_idx] = 1.0
            
            # W: index (j, i) for all j
            for j in range(p):
                w_row = j * n + i
                W[w_row, col_idx] = 1.0
            
            col_idx += 1
            
    U_list = [[float(x) for x in row] for row in U]
    V_list = [[float(x) for x in row] for row in V]
    W_list = [[float(x) for x in row] for row in W]
    
    return {"U": U_list, "V": V_list, "W": W_list}
