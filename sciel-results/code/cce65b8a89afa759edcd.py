import numpy as np
import time

def construct():
    n, m, p = 4, 4, 4
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Strassen's 2x2x2 algorithm
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],  # i=0, j=0
        [0, 0, 0, 0, 1, 0, 1],   # i=0, j=1
        [0, 1, 0, 0, 0, 1, 0],   # i=1, j=0
        [1, 1, 0, 1, 0, 0, -1]   # i=1, j=1
    ], dtype=int)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],  # j=0, k=0
        [0, 0, 1, 0, 0, 1, 0],   # j=0, k=1
        [0, 0, 0, 1, 0, 0, 1],   # j=1, k=0
        [1, 0, -1, 0, 1, 0, 1]   # j=1, k=1
    ], dtype=int)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],  # k=0, i=0
        [0, 1, 0, 1, 0, 0, 0],   # k=0, i=1
        [0, 0, 1, 0, 1, 0, 0],   # k=1, i=0
        [1, -1, 1, 0, 0, 1, 0]   # k=1, i=1
    ], dtype=int)
    
    # Verify 2x2 decomposition
    valid_2x2 = True
    for i in range(2):
        for j in range(2):
            for k in range(2):
                val = np.dot(U2[i*2+j, :], V2[j*2+k, :] * W2[k*2+i, :])
                if val != 1:
                    valid_2x2 = False
                    break
            if not valid_2x2: break
        if not valid_2x2: break
    
    if not valid_2x2:
        return _schoolbook(n, m, p)
        
    # Construct 4x4 via Kronecker Product
    perm_U = np.zeros(16, dtype=int)
    for i in range(4):
        for j in range(4):
            S = i * 4 + j
            i1, i2 = i // 2, i % 2
            j1, j2 = j // 2, j % 2
            r_out = 2 * i1 + j1
            r_in = 2 * i2 + j2
            K = r_out * 4 + r_in
            perm_U[S] = K
            
    perm_V = np.zeros(16, dtype=int)
    for j in range(4):
        for k in range(4):
            S = j * 4 + k
            j1, j2 = j // 2, j % 2
            k1, k2 = k // 2, k % 2
            r_out = 2 * j1 + k1
            r_in = 2 * j2 + k2
            K = r_out * 4 + r_in
            perm_V[S] = K
            
    perm_W = np.zeros(16, dtype=int)
    for k in range(4):
        for i in range(4):
            S = k * 4 + i
            k1, k2 = k // 2, k % 2
            i1, i2 = i // 2, i % 2
            r_out = 2 * k1 + i1
            r_in = 2 * k2 + i2
            K = r_out * 4 + r_in
            perm_W[S] = K
            
    U4_raw = np.kron(U2, U2)
    V4_raw = np.kron(V2, V2)
    W4_raw = np.kron(W2, W2)
    
    U4 = U4_raw[perm_U, :]
    V4 = V4_raw[perm_V, :]
    W4 = W4_raw[perm_W, :]
    
    # Verify 4x4 decomposition
    valid_4x4 = True
    for i in range(4):
        for j in range(4):
            for k in range(4):
                val = np.dot(U4[i*4+j, :], V4[j*4+k, :] * W4[k*4+i, :])
                if val != 1:
                    valid_4x4 = False
                    break
            if not valid_4x4: break
        if not valid_4x4: break
        
    if not valid_4x4:
        return _schoolbook(n, m, p)
        
    # Check deadline
    if time.time() > deadline:
        return _schoolbook(n, m, p)

    U_list = U4.tolist()
    V_list = V4.tolist()
    W_list = W4.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}

def _schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
