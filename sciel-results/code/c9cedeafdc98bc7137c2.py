import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 120.0
    
    n, m, p = 4, 4, 9
    
    # 2x2x2 Strassen (rank 7)
    U2 = np.array([
        [1, 0, 1, 0, 1, -1, 0],
        [0, 0, 0, 0, 1, 0, 1],
        [0, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, -1]
    ], dtype=float)
    
    V2 = np.array([
        [1, 1, 0, -1, 0, 1, 0],
        [0, 0, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 1],
        [1, 0, -1, 1, 1, 0, 1]
    ], dtype=float)
    
    W2 = np.array([
        [1, 0, 0, 1, -1, 0, 1],
        [1, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
        [1, -1, 1, 0, 0, 1, 0]
    ], dtype=float)
    
    # Build 4x4x4 Strassen (rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)
    
    A_step = [
        [[1, 0], [0, 1]],
        [[0, 0], [1, 1]],
        [[1, 0], [0, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[-1, 0], [1, 0]],
        [[0, 1], [0, -1]],
    ]
    B_step = [
        [[1, 0], [0, 1]],
        [[1, 0], [0, 0]],
        [[0, 1], [0, -1]],
        [[0, -1], [1, 0]],
        [[0, 0], [0, 1]],
        [[1, 1], [0, 0]],
        [[0, 1], [1, 1]],
    ]
    C_contrib = [
        [[1, 1], [1, 1]],
        [[0, 0], [1, -1]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
        [[-1, 1], [0, 0]],
        [[0, 1], [0, 1]],
        [[1, 0], [1, 0]],
    ]
    
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            u2 = U2[:, t]
            v2 = V2[:, t]
            w2 = W2[:, t]
            
            for I in range(2):
                for J in range(2):
                    bc = A_step[s][I][J]
                    if bc == 0: continue
                    for i2 in range(2):
                        for j2 in range(2):
                            gi, gj = 2*I+i2, 2*J+j2
                            sc = u2[i2*2+j2]
                            if sc == 0: continue
                            U4[gi*4+gj, col] += bc * sc
            
            for J in range(2):
                for K in range(2):
                    bc = B_step[s][J][K]
                    if bc == 0: continue
                    for j2 in range(2):
                        for k2 in range(2):
                            gj, gk = 2*J+j2, 2*K+k2
                            sc = v2[j2*2+k2]
                            if sc == 0: continue
                            V4[gj*4+gk, col] += bc * sc
            
            for K in range(2):
                for I in range(2):
                    cc = C_contrib[s][K][I]
                    if cc == 0: continue
                    for k2 in range(2):
                        for i2 in range(2):
                            gk, gi = 2*K+k2, 2*I+i2
                            sc = w2[k2*2+i2]
                            if sc == 0: continue
                            W4[gk*4+gi, col] += cc * sc
    
    # Combine with schoolbook for 4x4x5
    rank_total = 49 + 80
    U = np.zeros((16, rank_total), dtype=float)
    V = np.zeros((16, rank_total), dtype=float)
    W = np.zeros((36, rank_total), dtype=float)
    
    U[:, :49] = U4
    V[:, :49] = V4
    W[:16, :49] = W4
    
    col = 49
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                U[i*4+j, col] = 1
                V[j*4+k, col] = 1
                W[k*4+i, col] = 1
                col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
