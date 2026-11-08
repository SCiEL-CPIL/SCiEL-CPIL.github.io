import numpy as np
import time

def construct():
    t0 = time.time()
    DEADLINE = 110.0
    
    n, m, p = 4, 4, 6
    
    alpha = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    beta = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    recon = [
        [(0, 0, 1), (1, 1, 1)],
        [(1, 0, 1), (1, 1, -1)],
        [(0, 1, 1), (1, 1, 1)],
        [(0, 0, 1), (1, 0, 1)],
        [(0, 0, -1), (0, 1, 1)],
        [(1, 1, 1)],
        [(0, 0, 1)]
    ]
    
    R = 28
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        for i_sub in range(2):
            for k_sub in range(2):
                # U: A-block contribution
                for r in range(4):
                    if alpha[s][r] == 0:
                        continue
                    ar, br = r // 2, r % 2
                    u_idx = (ar * 2 + i_sub) * 4 + (br * 2 + k_sub)
                    U[u_idx, col] += alpha[s][r]
                
                # V: B-block contribution
                for c in range(4):
                    if beta[s][c] == 0:
                        continue
                    ac, bc = c // 2, c % 2
                    for j_sub in range(3):
                        v_idx = (ac * 2 + k_sub) * 6 + (bc * 3 + j_sub)
                        V[v_idx, col] += beta[s][c]
                
                # W: C-block contribution
                for (a, b, gamma) in recon[s]:
                    for j_sub in range(3):
                        w_idx = (b * 3 + j_sub) * 4 + (a * 2 + i_sub)
                        W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
    
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
