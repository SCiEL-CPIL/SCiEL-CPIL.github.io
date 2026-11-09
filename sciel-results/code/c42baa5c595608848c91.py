import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    # Top-level Strassen: 7 multiplications of (2x2) x (2x3)
    # For each (2x2)x(2x3): use inner Strassen for 2x2x2 (7 mults) + schoolbook for 2x2x1 (4 mults) = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Top-level Strassen coefficients
    # A blocks: 0=00, 1=01, 2=10, 3=11
    # B blocks: 0=00, 1=01, 2=10, 3=11
    # Using lists for faster access than dicts
    # Index: [block_row, block_col] -> 0,1,2,3
    # 0: 0,0 ; 1: 0,1 ; 2: 1,0 ; 3: 1,1
    
    # alpha[s][a] where a is A-block index
    alpha = [
        [1, 0, 0, 1],  # s=0: A00 + A11
        [0, 0, 1, 1],  # s=1: A10 + A11
        [1, 0, 0, 0],  # s=2: A00
        [0, 0, 0, 1],  # s=3: A11
        [1, 1, 0, 0],  # s=4: A00 + A01
        [-1, 0, 1, 0], # s=5: A10 - A00
        [0, 1, 0, -1]  # s=6: A01 - A11
    ]
    
    # beta[s][b] where b is B-block index
    beta = [
        [1, 0, 0, 1],  # s=0: B00 + B11
        [1, 0, 0, 0],  # s=1: B00
        [0, 1, 0, -1], # s=2: B01 - B11
        [-1, 0, 1, 0], # s=3: B10 - B00
        [0, 0, 0, 1],  # s=4: B11
        [1, 1, 0, 0],  # s=5: B00 + B01
        [0, 0, 1, 1]   # s=6: B10 + B11
    ]
    
    # recon[s] = list of (a_block, b_block, gamma)
    # C_ab += gamma * (X_s * Y_s)
    recon = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    # Inner Strassen for 2x2x2 (same coefficients)
    inner_alpha = alpha
    inner_beta = beta
    inner_recon = recon
    
    col = 0
    for s in range(7):
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix, Y_s is a 2x3 matrix
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            if time.time() > deadline:
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
            # U part
            # U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            # a: inner A-block index (0..3) -> x_row=a//2, x_col=a%2
            # r: top A-block index (0..3) -> ar=r//2, br=r%2
            # u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
            for a in range(4):
                val_a = inner_alpha[si][a]
                if val_a != 0:
                    x_row = a // 2
                    x_col = a % 2
                    for r in range(4):
                        val_r = alpha[s][r]
                        if val_r != 0:
                            ar = r // 2
                            br = r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += val_a * val_r
            
            # V part
            # V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            # b: inner B-block index (0..3) -> y_row=b//2, y_inner_col=b%2
            # c: top B-block index (0..3) -> ac=c//2, bc=c%2
            # v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
            for b in range(4):
                val_b = inner_beta[si][b]
                if val_b != 0:
                    y_row = b // 2
                    y_inner_col = b % 2
                    for c in range(4):
                        val_c = beta[s][c]
                        if val_c != 0:
                            ac = c // 2
                            bc = c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += val_b * val_c
            
            # W part
            # W[w_idx, col] += gamma * zgamma
            # gamma from recon[s] (a, b, gamma)
            # zgamma from inner_recon[si] (za, zb, zgamma)
            # w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # Z[u] = X_s[u,0]*Y_s[0,2] + X_s[u,1]*Y_s[1,2]
        # Terms: X_s[u,v] * Y_s[v,2] for u in 0,1 and v in 0,1
        
        for u in range(2):
            for v in range(2):
                if time.time() > deadline:
                    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                
                # U part
                # U[u_idx, col] += alpha[s][r]
                # u_idx corresponds to A_r[u, v]
                # r: top A-block index -> ar=r//2, br=r%2
                # u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                for r in range(4):
                    val_r = alpha[s][r]
                    if val_r != 0:
                        ar = r // 2
                        br = r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += val_r
                
                # V part
                # V[v_idx, col] += beta[s][c]
                # v_idx corresponds to B_c[v, 2]
                # c: top B-block index -> ac=c//2, bc=c%2
                # v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                for c in range(4):
                    val_c = beta[s][c]
                    if val_c != 0:
                        ac = c // 2
                        bc = c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += val_c
                
                # W part
                # W[w_idx, col] += gamma
                # w_idx corresponds to C_ab[u, 2]
                # a, b from recon[s]
                # w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
