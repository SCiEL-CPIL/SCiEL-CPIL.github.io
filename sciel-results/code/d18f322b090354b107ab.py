import numpy as np
import time

DIMENSIONS = [4, 4, 6]
START_TIME = time.time()
TIME_LIMIT = 110.0  # 10s margin

def construct():
    n, m, p = DIMENSIONS  # n=4, m=4, p=6
    # U: (n*m, R) = (16, R)
    # V: (m*p, R) = (24, R)
    # W: (p*n, R) = (24, R)
    
    # Hybrid approach:
    # Part 1: 4x4x4 Strassen (rank 49) for the first 4 columns of B
    # Part 2: 4x4x2 Schoolbook (rank 8) for the last 2 columns of B
    # Total rank: 49 + 8 = 57
    
    R = 57
    U = np.zeros((16, R))
    V = np.zeros((24, R))
    W = np.zeros((24, R))
    
    # Strassen 2x2 factors (for 2x2 x 2x2)
    # alpha[s][a] where a = 0..3 represents A row/col: a//2 = row, a%2 = col
    # beta[s][b] where b = 0..3 represents B row/col: b//2 = row, b%2 = col
    # recon[s] = list of (a_row, a_col, gamma) for output C[a_row*2+a_col]
    
    alpha_2x2 = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: A00 + A11
        {0: 0, 1: 0, 2: 1, 3: 1},  # s=1: A10 + A11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=2: A00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=3: A11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=4: A00 + A01
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=5: A10 - A00
        {0: 0, 1: 1, 2: 0, 3: -1}  # s=6: A01 - A11
    ]
    beta_2x2 = [
        {0: 1, 1: 0, 2: 0, 3: 1},  # s=0: B00 + B11
        {0: 1, 1: 0, 2: 0, 3: 0},  # s=1: B00
        {0: 0, 1: 1, 2: 0, 3: -1}, # s=2: B01 - B11
        {0: -1, 1: 0, 2: 1, 3: 0}, # s=3: B10 - B00
        {0: 0, 1: 0, 2: 0, 3: 1},  # s=4: B11
        {0: 1, 1: 1, 2: 0, 3: 0},  # s=5: B00 + B01
        {0: 0, 1: 0, 2: 1, 3: 1}   # s=6: B10 + B11
    ]
    recon_2x2 = [
        [(0, 0, 1), (1, 1, 1)],     # s=0
        [(1, 0, 1), (1, 1, -1)],    # s=1
        [(0, 1, 1), (1, 1, 1)],     # s=2
        [(0, 0, 1), (1, 0, 1)],     # s=3
        [(0, 0, -1), (0, 1, 1)],    # s=4
        [(1, 1, 1)],                # s=5
        [(0, 0, 1)]                 # s=6
    ]
    
    # Part 1: 4x4x4 Strassen (rank 49)
    # Outer Strassen: 7 multiplications of (2x2) x (2x2)
    # Inner Strassen: each (2x2) x (2x2) uses 7 multiplications
    # Total: 7 * 7 = 49
    
    col = 0
    for s_outer in range(7):
        for s_inner in range(7):
            # U contributions
            for a in range(4):
                if alpha_2x2[s_inner][a] != 0:
                    x_row, x_col = a // 2, a % 2
                    for r in range(4):
                        if alpha_2x2[s_outer][r] != 0:
                            ar, br = r // 2, r % 2
                            u_idx = (ar * 2 + x_row) * 4 + (br * 2 + x_col)
                            U[u_idx, col] += alpha_2x2[s_inner][a] * alpha_2x2[s_outer][r]
            
            # V contributions (for j=0..3)
            for b in range(4):
                if beta_2x2[s_inner][b] != 0:
                    y_row, y_inner_col = b // 2, b % 2
                    for c in range(4):
                        if beta_2x2[s_outer][c] != 0:
                            ac, bc = c // 2, c % 2
                            v_idx = (ac * 2 + y_row) * 6 + (bc * 2 + y_inner_col)
                            V[v_idx, col] += beta_2x2[s_inner][b] * beta_2x2[s_outer][c]
            
            # W contributions (for j=0..3)
            for a, b, gamma in recon_2x2[s_outer]:
                for za, zb, zgamma in recon_2x2[s_inner]:
                    w_idx = (b * 2 + zb) * 4 + (a * 2 + za)
                    W[w_idx, col] += gamma * zgamma
            
            col += 1
    
    # Part 2: 4x4x2 Schoolbook (rank 8)
    # For j=4,5: C[i,j] = sum_k A[i,k] * B[k,j]
    # Each (i,j) pair is a rank-1 term
    # 4 x 2 = 8 terms
    
    for i in range(4):
        for j in range(2):
            j_actual = j + 4  # columns 4, 5 of B
            for k in range(4):
                u_idx = i * 4 + k
                v_idx = k * 6 + j_actual
                w_idx = j_actual * 4 + i
                U[u_idx, col] += 1
                V[v_idx, col] += 1
                W[w_idx, col] += 1
            col += 1
    
    def to_list(mat):
        return [[int(round(x)) if abs(x - round(x)) < 1e-9 else x for x in row] for row in mat]
        
    return {"U": to_list(U), "V": to_list(V), "W": to_list(W)}
