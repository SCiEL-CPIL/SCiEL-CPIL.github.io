import numpy as np

def construct():
    n, m, p = 2, 2, 15
    rank = 53
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Strassen vectors for 2x2x2
    # u, v, w are local indices 0..3
    # u: i*2+j
    # v: j*2+k
    # w: k*2+i
    strassen_u = [
        [1, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    strassen_v = [
        [1, 0, 0, 1],
        [0, 1, 1, 0],
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0]
    ]
    strassen_w = [
        [1, 1, 1, 1],
        [0, 1, 0, 1],
        [1, 0, 1, 0],
        [0, 0, 0, 1],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [1, 0, 0, 1]
    ]
    
    # 7 blocks of size 2
    for b in range(7):
        k_start = b * 2
        for s in range(7):
            u_vec = strassen_u[s]
            v_vec = strassen_v[s]
            w_vec = strassen_w[s]
            
            # Map to global U
            for i in range(2):
                for j in range(2):
                    u_idx = i * 2 + j
                    U[u_idx][col] += u_vec[i * 2 + j]
            
            # Map to global V
            for j in range(2):
                for k_loc in range(2):
                    v_idx = j * 15 + (k_start + k_loc)
                    V[v_idx][col] += v_vec[j * 2 + k_loc]
            
            # Map to global W
            for k_loc in range(2):
                for i in range(2):
                    w_idx = (k_start + k_loc) * 2 + i
                    W[w_idx][col] += w_vec[k_loc * 2 + i]
            
            col += 1
            
    # 1 block of size 1 (k=14)
    # Schoolbook for k=14
    # Terms: (i,j,k)
    # i=0, j=0, k=14
    # i=0, j=1, k=14
    # i=1, j=0, k=14
    # i=1, j=1, k=14
    
    # Term 1: A00 B0,14 -> C0,14
    U[0][col] = 1
    V[0 * 15 + 14][col] = 1
    W[14 * 2 + 0][col] = 1
    col += 1
    
    # Term 2: A01 B1,14 -> C0,14
    U[1][col] = 1
    V[1 * 15 + 14][col] = 1
    W[14 * 2 + 0][col] = 1
    col += 1
    
    # Term 3: A10 B0,14 -> C1,14
    U[2][col] = 1
    V[0 * 15 + 14][col] = 1
    W[14 * 2 + 1][col] = 1
    col += 1
    
    # Term 4: A11 B1,14 -> C1,14
    U[3][col] = 1
    V[1 * 15 + 14][col] = 1
    W[14 * 2 + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
