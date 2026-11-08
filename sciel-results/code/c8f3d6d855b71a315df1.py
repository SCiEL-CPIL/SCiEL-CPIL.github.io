import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = 4, 4, 6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # Strassen coefficients for 2x2x2
    # alpha[s][a] for a in 0..3 (a = row*2+col of 2x2 block)
    # beta[s][b] for b in 0..3
    # recon[s] = list of (a_block, b_block, coeff) for output reconstruction
    
    alpha = [
        [1, 0, 0, 1],  # s0: A00 + A11
        [0, 0, 1, 1],  # s1: A10 + A11
        [1, 0, 0, 0],  # s2: A00
        [0, 0, 0, 1],  # s3: A11
        [1, 1, 0, 0],  # s4: A00 + A01
        [-1, 0, 1, 0], # s5: A10 - A00
        [0, 1, 0, -1]  # s6: A01 - A11
    ]
    beta = [
        [1, 0, 0, 1],  # s0: B00 + B11
        [1, 0, 0, 0],  # s1: B00
        [0, 1, 0, -1], # s2: B01 - B11
        [-1, 0, 1, 0], # s3: B10 - B00
        [0, 0, 0, 1],  # s4: B11
        [1, 1, 0, 0],  # s5: B00 + B01
        [0, 0, 1, 1]   # s6: B10 + B11
    ]
    # recon[s] = list of (out_row_block, out_col_block, coeff)
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s0
        [(1, 0, 1), (1, 1, -1)],    # s1
        [(0, 1, 1), (1, 1, 1)],     # s2
        [(0, 0, 1), (1, 0, 1)],     # s3
        [(0, 0, -1), (0, 1, 1)],    # s4
        [(1, 1, 1)],                # s5
        [(0, 0, 1)]                 # s6
    ]
    
    # Top-level: 7 Strassen multiplications of (2x2) x (2x3)
    # For each, use: 7 inner Strassen for 2x2x2 part + 4 schoolbook for 2x2x1 part = 11
    R = 7 * 11
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    col = 0
    for s in range(7):
        # X_s is 2x2, Y_s is 2x3
        # Part 1: Inner Strassen for 2x2 x 2x2 (first 2 columns of Y_s)
        for si in range(7):
            # U contributions
            for a in range(4):
                if alpha[si][a] != 0:
                    xr, xc = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + xr) * 4 + (br * 2 + xc)
                            U[u_idx, col] += alpha[si][a] * alpha[s][r]
            
            # V contributions
            for b in range(4):
                if beta[si][b] != 0:
                    yr, yc = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + yr) * 6 + (bc * 3 + yc)
                            V[v_idx, col] += beta[si][b] * beta[s][c]
            
            # W contributions
            for (ar, br, ga) in recon[s]:
                for (yr, yc, gb) in recon[si]:
                    w_idx = (br * 3 + yc) * 4 + (ar * 2 + yr)
                    W[w_idx, col] += ga * gb
            
            col += 1
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        
        # Part 2: Schoolbook for 2x2 x 2x1 (third column of Y_s)
        for u in range(2):
            for v in range(2):
                # U contributions
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V contributions
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W contributions
                for (ar, br, ga) in recon[s]:
                    w_idx = (br * 3 + 2) * 4 + (ar * 2 + u)
                    W[w_idx, col] += ga
                
                col += 1
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
