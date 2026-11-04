import numpy as np
import time

def construct():
    start_time = time.time()
    DEADLINE = 140

    n, m, p = 4, 4, 5
    NM, MP, PN = n * m, m * p, p * n  # 16, 20, 20
    
    # Part 1: 4x4x4 part (Rank 49)
    # Part 2: 4x4x1 part (Rank 16)
    rank_444 = 49
    rank_441 = 16
    total_rank = rank_444 + rank_441
    
    U = np.zeros((NM, total_rank))
    V = np.zeros((MP, total_rank))
    W = np.zeros((PN, total_rank))
    
    # --- Part 1: 4x4x4 via Recursive Strassen ---
    
    # Scalar Strassen coefficients
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # M1 = (a+d)(e+h)
    # M2 = (c+d)e
    # M3 = a(g-h)
    # M4 = d(f-e)
    # M5 = (a+b)h
    # M6 = (c-a)(e+f)
    # M7 = (b-d)(g+h)
    
    # Coeffs for A (u)
    # Indices: 0:a, 1:b, 2:c, 3:d
    u_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Coeffs for B (v)
    # Indices: 0:e, 1:f, 2:g, 3:h
    v_coeffs = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 0, 1, -1], # M3
        [0, 1, 0, -1], # M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    
    # Coeffs for C (w)
    # Indices: 0:C00, 1:C01, 2:C10, 3:C11
    w_coeffs = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, -1], # M2
        [0, 1, 0, 1], # M3
        [1, 0, 1, 0], # M4
        [-1, 1, 0, 0], # M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Block Strassen coefficients
    # Blocks: 00, 01, 10, 11
    # L_A[s][b] is coeff of A-block b in X_s
    # L_B[s][b] is coeff of B-block b in Y_s
    # s is 0-indexed (0..6)
    
    L_A = [
        [1, 0, 0, 1], # S1: A00 + A11
        [0, 0, 1, 1], # S2: A10 + A11
        [1, 0, 0, 0], # S3: A00
        [0, 0, 0, 1], # S4: A11
        [1, 1, 0, 0], # S5: A00 + A01
        [-1, 0, 1, 0], # S6: A10 - A00
        [0, 1, 0, -1]  # S7: A01 - A11
    ]
    
    L_B = [
        [1, 0, 0, 1], # S1: B00 + B11
        [1, 0, 0, 0], # S2: B00
        [0, 0, 1, -1], # S3: B10 - B11
        [0, -1, 0, 1], # S4: B01 - B00 (Note: B01 is 01, B00 is 00) -> [0, 1, 0, -1] ? 
                       # S4 = A11 * (B01 - B00). B01 is index 1, B00 is index 0.
                       # So L_B[3] should be [ -1, 1, 0, 0 ]?
                       # Let's re-verify S4.
                       # S4 = A11 (B01 - B00).
                       # B01 is block (0,1). B00 is block (0,0).
                       # Indices: 00=0, 01=1, 10=2, 11=3.
                       # So L_B[3] = [ -1, 1, 0, 0 ].
                       # My previous list had [0, -1, 0, 1] which is wrong.
        [0, 0, 0, 1], # S5: B11
        [1, 1, 0, 0], # S6: B00 + B01
        [0, 0, 1, 1]  # S7: B10 + B11
    ]
    # Correcting L_B[3]
    L_B[3] = [-1, 1, 0, 0]
    
    # K_coeffs[s][I][K] is coeff of S_s in C_IK
    # I, K are 0,1
    K_coeffs = [
        [[1, 0], [0, 1]], # S1: C00=1, C11=1
        [[0, 0], [1, -1]], # S2: C10=1, C11=-1
        [[0, 1], [0, 1]], # S3: C01=1, C11=1
        [[1, 0], [1, 0]], # S4: C00=1, C10=1
        [[-1, 1], [0, 0]], # S5: C00=-1, C01=1
        [[0, 0], [0, 1]], # S6: C11=1
        [[1, 0], [0, 0]]  # S7: C00=1
    ]
    
    # Fill U, V, W for Part 1
    for s in range(7):
        for t in range(7):
            col = s * 7 + t
            
            # U: A part
            # u_coeffs[t] gives coeffs for x_00, x_01, x_10, x_11
            # x_ij = sum_b L_A[s][b] A_b[i,j]
            # So U[A_idx, col] = sum_{b, i, j} ...
            # Actually, for a specific A entry A[i, j], it belongs to block b = (i//2, j//2).
            # Its position in the block is (i%2, j%2).
            # The coefficient of A[i, j] in u_{s,t} is:
            # u_{s,t} = sum_{ij} c_{t,ij} x_{ij}
            # x_{ij} = sum_b L_A[s][b] A_b[i,j]
            # So coeff of A_b[i,j] is c_{t,ij} * L_A[s][b]
            
            for i in range(4):
                for j in range(4):
                    b = (i // 2) * 2 + (j // 2) # 0,1,2,3
                    ii = i % 2
                    jj = j % 2
                    # c_{t,ij} is u_coeffs[t][ii*2 + jj]
                    c_val = u_coeffs[t][ii * 2 + jj]
                    l_val = L_A[s][b]
                    U[i * 4 + j, col] = c_val * l_val
            
            # V: B part
            for j in range(4):
                for k in range(4):
                    b = (j // 2) * 2 + (k // 2)
                    jj = j % 2
                    kk = k % 2
                    d_val = v_coeffs[t][jj * 2 + kk]
                    l_val = L_B[s][b]
                    V[j * 5 + k, col] = d_val * l_val
            
            # W: C part
            # C_{2I+i, 2K+k} = sum_s K[s][I][K] S_s[i,k]
            # S_s[i,k] = sum_t w_coeffs[t][i*2+k] T_{s,t}
            # So W[C_idx, col] = K[s][I][K] * w_coeffs[t][i*2+k]
            
            for i in range(4):
                for k in range(4):
                    I = i // 2
                    K = k // 2
                    ii = i % 2
                    kk = k % 2
                    k_val = K_coeffs[s][I][K]
                    w_val = w_coeffs[t][ii * 2 + kk]
                    W[k * 4 + i, col] = k_val * w_val

    # --- Part 2: 4x4x1 part (Rank 16) ---
    # y_i = sum_j A_ij B_j4
    for i in range(4):
        for j in range(4):
            col = rank_444 + i * 4 + j
            U[i * 4 + j, col] = 1
            V[j * 5 + 4, col] = 1
            W[4 * 4 + i, col] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
