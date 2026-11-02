import numpy as np

def construct():
    n, m, p = 2, 2, 8
    nr, mr, pr = n * m, m * p, p * n  # 4, 16, 16
    
    # Local Strassen vectors for 2x2x2
    # U indices: i*2+j
    # V indices: j*2+k
    # W indices: k*2+i
    
    U_local = [
        [1, 0, 0, 1], # P1: A00+A11
        [0, 0, 1, 1], # P2: A10+A11
        [1, 0, 0, 0], # P3: A00
        [0, 0, 0, 1], # P4: A11
        [1, 1, 0, 0], # P5: A00+A01
        [-1, 0, 1, 0], # P6: A10-A00
        [0, 1, 0, -1]  # P7: A01-A11
    ]
    
    V_local = [
        [1, 0, 0, 1], # P1: B00+B11
        [1, 0, 0, 0], # P2: B00
        [0, 1, 0, -1], # P3: B01-B11
        [-1, 0, 1, 0], # P4: B10-B00
        [0, 0, 0, 1], # P5: B11
        [1, 1, 0, 0], # P6: B00+B01
        [0, 0, 1, 1]  # P7: B10+B11
    ]
    
    W_local = [
        [1, 0, 0, 1], # P1: C00+C11
        [0, 1, 0, -1], # P2: C10-C11
        [0, 0, 1, 1], # P3: C01+C11
        [1, 1, 0, 0], # P4: C00+C10
        [-1, 0, 1, 0], # P5: -C00+C01
        [0, 0, 0, 1], # P6: C11
        [1, 0, 0, 0]  # P7: C00
    ]
    
    rank = 28
    U = np.zeros((nr, rank), dtype=float)
    V = np.zeros((mr, rank), dtype=float)
    W = np.zeros((pr, rank), dtype=float)
    
    # 4 blocks of k: {0,1}, {2,3}, {4,5}, {6,7}
    for q in range(4):
        k_base = 2 * q
        for r in range(7):
            t = q * 7 + r
            
            # Fill U
            for i in range(2):
                for j in range(2):
                    idx_u = i * m + j
                    U[idx_u, t] = U_local[r][idx_u]
            
            # Fill V
            for j in range(2):
                for k_local in range(2):
                    idx_v = j * p + (k_base + k_local)
                    idx_v_local = j * 2 + k_local
                    V[idx_v, t] = V_local[r][idx_v_local]
            
            # Fill W
            for k_local in range(2):
                for i in range(2):
                    idx_w = (k_base + k_local) * n + i
                    idx_w_local = k_local * 2 + i
                    W[idx_w, t] = W_local[r][idx_w_local]
                    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
