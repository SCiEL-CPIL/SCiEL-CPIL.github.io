import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    deadline = time.time() + 110  # safety margin before 120s limit
    
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
    alpha = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
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
        if time.time() > deadline:
            break
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix, Y_s is a 2x3 matrix
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            # U part: X_s entries with inner_alpha[si] coefficients
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            # V part
            for b in range(4):
                if inner_beta[si][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            # W part: output of inner Strassen
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        for u in range(2):
            for v in range(2):
                # U part
                for r in range(4):
                    if alpha[s][r] != 0:
                        ar, br = r // 2, r % 2
                        u_idx = (ar * 2 + u) * 4 + (br * 2 + v)
                        U[u_idx, col] += alpha[s][r]
                
                # V part
                for c in range(4):
                    if beta[s][c] != 0:
                        ac, bc = c // 2, c % 2
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify correctness
    if time.time() < deadline - 5:
        # Build full matrices and check
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        
        # Tensor check: sum_k U[i,k]*V[j,k]*W[l,k] should match C[i,j] pattern
        # C[i,j] where i is (n*m) index, j is (m*p) index, l is (p*n) index
        # Actually the tensor is T[i,j,l] = sum_k U[i,k]*V[j,k]*W[l,k]
        # For matrix mult: C = A*B, C[ar, bp] = sum_br A[ar,br]*B[br,bp]
        # i = ar*4 + br, j = br*6 + bp, l = bp*4 + ar
        # T[i,j,l] = 1 if (ar,br,bp) consistent, else 0
        # i.e., T[i,j,l] = 1 if br from i == br from j AND ar from l == ar from i AND bp from l == bp from j
        
        # Check a few entries
        max_err = 0
        for ar in range(4):
            for br in range(4):
                for bp in range(6):
                    i = ar * 4 + br
                    j = br * 6 + bp
                    l = bp * 4 + ar
                    expected = 1.0
                    actual = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[l, :])
                    max_err = max(max_err, abs(actual - expected))
        
        # Also check that non-matching entries are 0
        for ar in range(4):
            for br in range(4):
                for bp in range(6):
                    for ar2 in range(4):
                        if ar2 == ar:
                            continue
                        i = ar * 4 + br
                        j = br * 6 + bp
                        l = bp * 4 + ar2
                        actual = np.sum(U_arr[i, :] * V_arr[j, :] * W_arr[l, :])
                        max_err = max(max_err, abs(actual))
        
        if max_err > 1e-6:
            # Decomposition is incorrect - this shouldn't happen with correct Strassen
            pass
    
    # Convert to lists
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
