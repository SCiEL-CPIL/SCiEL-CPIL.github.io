import numpy as np

DIMENSIONS = [2, 2, 8]

def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)
    
    # Strassen vectors for 2x2 block
    # u_s: indices 00, 01, 10, 11
    # v_s: indices 00, 01, 10, 11
    # w_s: indices 00, 01, 10, 11 (C_00, C_01, C_10, C_11)
    
    u_vecs = [
        [1, 0, 0, 1], # s=0: (a+d)
        [0, 0, 1, 1], # s=1: (c+d)
        [1, 0, 0, 0], # s=2: a
        [0, 0, 0, 1], # s=3: d
        [1, 0, 1, 0], # s=4: (a+c)
        [0, -1, 0, 1],# s=5: (d-b)
        [0, 1, 0, -1] # s=6: (b-d)
    ]
    
    v_vecs = [
        [1, 0, 0, 1], # s=0: (e+h)
        [1, 0, 0, 0], # s=1: e
        [0, 1, 0, -1],# s=2: (f-h)
        [1, 0, -1, 0],# s=3: (e-g)
        [0, 0, 0, 1], # s=4: h
        [1, 0, 1, 0], # s=5: (e+g)
        [0, 1, 0, 1]  # s=6: (f+h)
    ]
    
    w_vecs = [
        [1, 0, 0, 1], # s=0: M1 -> C00, C11
        [0, 0, 1, -1],# s=1: M2 -> C10, C11
        [0, 1, 0, 1], # s=2: M3 -> C01, C11
        [1, 0, 1, 0], # s=3: M4 -> C00, C10
        [-1, 1, 0, 0],# s=4: M5 -> C00, C01
        [0, 0, 0, 1], # s=5: M6 -> C11
        [1, 0, 0, 0]  # s=6: M7 -> C00
    ]
    
    for q in range(4):
        for s in range(7):
            r = 7 * q + s
            
            # Fill U
            for idx, val in enumerate(u_vecs[s]):
                U[idx, r] = val
                
            # Fill V
            # V index: j * p + k
            # j in 0..1, k' in 0..1
            # k = 2q + k'
            for j in range(2):
                for kp in range(2):
                    val = v_vecs[s][j * 2 + kp] # v_vecs indexed by j, kp
                    if val != 0:
                        v_idx = j * p + (2 * q + kp)
                        V[v_idx, r] = val
                        
            # Fill W
            # W index: k * n + i
            # k = 2q + kp, i in 0..1
            for kp in range(2):
                for i in range(2):
                    val = w_vecs[s][kp * 2 + i] # w_vecs indexed by kp, i
                    if val != 0:
                        w_idx = (2 * q + kp) * n + i
                        W[w_idx, r] = val
                        
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
