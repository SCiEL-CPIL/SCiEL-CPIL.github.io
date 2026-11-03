import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Rank-7 Strassen terms for 2x2x2
    # u, v, w are length 4
    strassen_u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 0, 1, -1],
        [0, 1, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_w = [
        [1, 0, 0, 1],
        [0, 0, 1, 0],
        [0, 1, 0, 0],
        [1, 0, 0, -1],
        [-1, 1, 0, 0],
        [0, 0, 1, 1],
        [-1, 0, 0, 1]
    ]
    
    # Generate 49 terms for 4x4x4
    terms_4x4x4 = []
    for r in range(7):
        for s in range(7):
            u = np.kron(strassen_u[r], strassen_u[s])
            v = np.kron(strassen_v[r], strassen_v[s])
            w = np.kron(strassen_w[r], strassen_w[s])
            terms_4x4x4.append((u, v, w))
            
    rank = 49 + 49 + 16
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    # First 4x4x4 block (columns 0-3 of B)
    for u, v, w in terms_4x4x4:
        U[:, col] = u
        V[0:16, col] = v
        W[0:16, col] = w
        col += 1
        
    # Second 4x4x4 block (columns 4-7 of B)
    for u, v, w in terms_4x4x4:
        U[:, col] = u
        V[16:32, col] = v
        W[16:32, col] = w
        col += 1
        
    # 4x4x1 block (column 8 of B)
    for i in range(n):
        for j in range(m):
            for k in range(m):
                U[i*m + j, col] = 1
                V[j*p + 8, col] = 1
                W[k*n + i, col] = 1
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
