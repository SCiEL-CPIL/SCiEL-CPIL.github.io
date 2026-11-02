import numpy as np

def construct():
    n, m, p = 3, 4, 4
    
    # Shapes
    U_shape = (n * m, n * m * p)
    V_shape = (m * p, n * m * p)
    W_shape = (p * n, n * m * p)
    
    rank = n * m * p
    U = np.zeros(U_shape)
    V = np.zeros(V_shape)
    W = np.zeros(W_shape)
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                # Indices
                u_idx = i * m + j
                v_idx = j * p + k
                w_idx = k * n + i
                
                U[u_idx, col] = 1.0
                V[v_idx, col] = 1.0
                W[w_idx, col] = 1.0
                
                col += 1
                
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
