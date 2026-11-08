import numpy as np
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)  
    # W: (p*n, R) = (24, R)
    
    # Top-level Strassen: 7 multiplications of (2x2) x (2x3)
    # For each (2x2)x(2x3): use inner Strassen for 2x2x2 (7 mults) + schoolbook for 2x2x1 (4 mults) = 11
    # Total rank: 7 * 11 = 77
    
    R = 77
    U = np.zeros((16, R), dtype=int)
    V = np.zeros((24, R), dtype=int)
    W = np.zeros((24, R), dtype=int)
    
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
        # For each top-level Strassen term s:
        # X_s is a 2x2 matrix, Y_s is a 2x3 matrix
        # Compute X_s * Y_s using: inner Strassen for cols 0-1, schoolbook for col 2
        
        # Part 1: Inner Strassen for 2x2 x 2x2 (columns 0,1 of Y_s)
        for si in range(7):
            # Inner Strassen term si
            # X_s * (inner combination of Y_s columns 0,1)
            
            # U part: X_s entries with inner_alpha[si] coefficients
            for a in range(4):
                if inner_alpha[si][a] != 0:
                    # a indexes position in X_s: row=a//2, col=a%2
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha[s][r] != 0:
                            # A_r[x_row, x_col]
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += inner_alpha[si][a] * alpha[s][r]
            
            # V part: Y_s entries with inner_beta[si] coefficients
            for b in range(4):
                if inner_beta[si][b] != 0:
                    # b indexes position in Y_s[:,0:2]: row=b//2, col=b%2
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta[s][c] != 0:
                            # B_c[y_row, y_inner_col]
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 3 + y_inner_col)
                            V[v_idx, col] += inner_beta[si][b] * beta[s][c]
            
            # W part: output of inner Strassen
            for a, b, gamma in recon[s]:
                for za, zb, zgamma in inner_recon[si]:
                    # za, zb are positions in the 2x2 inner output
                    # This is P_s[za, zb]
                    # It contributes to C block (a,b) at position (za, zb)
                    # C global: row = a*2 + za, col = b*3 + zb
                    # W index: k*n + i where k is col, i is row
                    # k = b*3 + zb, i = a*2 + za
                    w_idx = (b * 3 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
        
        # Part 2: Schoolbook for 2x2 x 2x1 (column 2 of Y_s)
        # X_s (2x2) * Y_s[:,2] (2x1) -> Z (2x1)
        # Z[0] = X_s[0,0]*Y_s[0,2] + X_s[0,1]*Y_s[1,2]
        # Z[1] = X_s[1,0]*Y_s[0,2] + X_s[1,1]*Y_s[1,2]
        # 4 schoolbook multiplications
        
        for u in range(2):
            for v in range(2):
                # Term: X_s[u,v] * Y_s[v,2]
                
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
                        # B_c[v, 2]
                        v_idx = (ac * 2 + v) * 6 + (bc * 3 + 2)
                        V[v_idx, col] += beta[s][c]
                
                # W part
                for a, b, gamma in recon[s]:
                    # C_ab[u, 2]
                    # k = b*3 + 2, i = a*2 + u
                    w_idx = (b * 3 + 2) * 4 + (a * 2 + u)
                    W[w_idx, col] += gamma
                
                col += 1
    
    # Verify: check that U @ diag @ V^T reconstructs the matmul tensor
    # Quick sanity check with random matrices
    np.random.seed(42)
    A_test = np.random.randint(-5, 5, (n, m))
    B_test = np.random.randint(-5, 5, (m, p))
    C_test = A_test @ B_test
    
    # Reconstruct C from U, V, W
    U_arr = np.array(U)
    V_arr = np.array(V)
    W_arr = np.array(W)
    
    # C[i,k] = sum_j A[i,j] * B[j,k]
    # Tensor: T[a_idx, b_idx, c_idx] = sum_r U[a_idx,r] * V[b_idx,r] * W[c_idx,r]
    # a_idx = i*m + j, b_idx = j*p + k, c_idx = k*n + i
    
    C_recon = np.zeros((n, p), dtype=int)
    for i in range(n):
        for k in range(p):
            total = 0
            for r in range(R):
                for j in range(m):
                    a_idx = i * m + j
                    b_idx = j * p + k
                    c_idx = k * n + i
                    total += U_arr[a_idx, r] * V_arr[b_idx, r] * W_arr[c_idx, r]
            C_recon[i, k] = total
    
    if not np.array_equal(C_recon, C_test):
        # Fallback to schoolbook if verification fails
        R_fb = n * m * p
        U_fb = np.zeros((n * m, R_fb), dtype=int)
        V_fb = np.zeros((m * p, R_fb), dtype=int)
        W_fb = np.zeros((p * n, R_fb), dtype=int)
        
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, idx] = 1
                    V_fb[j * p + k, idx] = 1
                    W_fb[k * n + i, idx] = 1
                    idx += 1
        
        def to_list_fb(mat):
            return [[int(x) for x in row] for row in mat]
        return {"U": to_list_fb(U_fb), "V": to_list_fb(V_fb), "W": to_list_fb(W_fb)}
    
    # Convert to lists
    def to_list(mat):
        return [[int(x) for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
