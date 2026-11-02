import numpy as np
import time

DIMENSIONS = [2, 2, 10]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    rank = 35
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # Strassen 7-term decomposition for 2x2x2
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # P1 = (a+d)(e+h), P2 = (c+d)e, P3 = a(f-h), P4 = d(g-e)
    # P5 = (a+b)h, P6 = (c-a)(e+f), P7 = (b-d)(g+h)
    # C00 = P1 + P4 - P5 + P7, C01 = P3 + P5
    # C10 = P2 + P4, C11 = P1 - P2 + P3 + P6
    
    strassen_A = [
        [1, 0, 0, 1],   # P1: a+d
        [0, 0, 1, 1],   # P2: c+d
        [1, 0, 0, 0],   # P3: a
        [0, 0, 0, 1],   # P4: d
        [1, 1, 0, 0],   # P5: a+b
        [-1, 0, 1, 0],  # P6: c-a
        [0, 1, 0, -1]   # P7: b-d
    ]
    
    strassen_B = [
        [1, 0, 0, 1],   # P1: e+h
        [1, 0, 0, 0],   # P2: e
        [0, 1, 0, -1],  # P3: f-h
        [-1, 0, 1, 0],  # P4: g-e
        [0, 0, 0, 1],   # P5: h
        [1, 1, 0, 0],   # P6: e+f
        [0, 0, 1, 1]    # P7: g+h
    ]
    
    # W contributions: [C00, C01, C10, C11]
    w_contrib = [
        [1, 0, 0, 1],    # P1
        [0, 0, 1, -1],   # P2
        [0, 1, 0, 1],    # P3
        [1, 0, 1, 0],    # P4
        [-1, 1, 0, 0],   # P5
        [0, 0, 0, 1],    # P6
        [1, 0, 0, 0]     # P7
    ]
    
    for block in range(5):
        for r in range(7):
            col = block * 7 + r
            
            # U: index i*m + j
            U[0, col] = strassen_A[r][0]  # a
            U[1, col] = strassen_A[r][1]  # b
            U[2, col] = strassen_A[r][2]  # c
            U[3, col] = strassen_A[r][3]  # d
            
            # V: index j*p + k
            k0 = 2 * block
            k1 = 2 * block + 1
            V[0 * p + k0, col] = strassen_B[r][0]  # e
            V[0 * p + k1, col] = strassen_B[r][1]  # f
            V[1 * p + k0, col] = strassen_B[r][2]  # g
            V[1 * p + k1, col] = strassen_B[r][3]  # h
            
            # W: index k*n + i
            W[k0 * n + 0, col] = w_contrib[r][0]  # C00
            W[k1 * n + 0, col] = w_contrib[r][1]  # C01
            W[k0 * n + 1, col] = w_contrib[r][2]  # C10
            W[k1 * n + 1, col] = w_contrib[r][3]  # C11
    
    # Verify with random test
    np.random.seed(123)
    A = np.random.rand(n, m)
    B = np.random.rand(m, p)
    C_true = A @ B
    
    C_decomp = np.zeros((n, p))
    for i in range(n):
        for k in range(p):
            s = 0.0
            for r in range(rank):
                u_val = sum(U[i * m + j, r] * A[i, j] for j in range(m))
                v_val = sum(V[j * p + k, r] * B[j, k] for j in range(m))
                s += u_val * v_val * W[k * n + i, r]
            C_decomp[i, k] = s
    
    if not np.allclose(C_true, C_decomp, atol=1e-10):
        # Fallback: schoolbook rank 40
        rank = 40
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
