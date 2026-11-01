import numpy as np

def construct():
    n, m, p = 2, 2, 8
    rank = n * m * p # 32
    
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # We need to map (i, j, k) to a column index
    # and set U[i*m+j, col] = 1
    # V[j*p+k, col] = 1
    # W[k*n+i, col] = 1
    
    cols = []
    u_idxs = []
    v_idxs = []
    w_idxs = []
    
    for k in range(p):
        for i in range(n):
            for j in range(m):
                col = len(cols)
                cols.append(col)
                u_idxs.append(i * m + j)
                v_idxs.append(j * p + k)
                w_idxs.append(k * n + i)
                
    for idx, col in enumerate(cols):
        U[u_idxs[idx], col] = 1
        V[v_idxs[idx], col] = 1
        W[w_idxs[idx], col] = 1
        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
