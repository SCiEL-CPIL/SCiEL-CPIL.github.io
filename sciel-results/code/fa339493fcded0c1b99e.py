import numpy as np
import time

DIMENSIONS = [4, 4, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 110

    # Part 1: 4x4x4 part (k=0..3) using Strassen (rank 28)
    # Part 2: 4x4x5 part (k=4..8) using Schoolbook (rank 80)
    
    rank_444 = 28
    rank_445 = 4 * 4 * 5
    total_rank = rank_444 + rank_445
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    # --- Construct 4x4x4 Strassen Decomposition ---
    # Strassen steps 0..6
    # Schoolbook steps within each 2x2 multiplication: 0..3
    # t=0: i'=0, j'=0, k'=0
    # t=1: i'=0, j'=1, k'=1
    # t=2: i'=1, j'=0, k'=0
    # t=3: i'=1, j'=1, k'=1
    
    # Coefficients for A_m[i', j'] = sum alpha[I, J, i', j'] A[2I+i', 2J+j']
    # Coefficients for B_m[j', k'] = sum beta[J, K, j', k'] B[2J+j', 2K+k']
    # Coefficients for C_block[I, K] = sum gamma[I, K] M_m[i', k']
    
    # alpha[s][I][J][i'][j']
    # beta[s][J][K][j'][k']
    # gamma[s][I][K]
    
    alpha = [[[ [ [0]*2 for _ in range(2) ] for _ in range(2) ] for _ in range(2) ] for _ in range(7)]
    beta = [[[ [ [0]*2 for _ in range(2) ] for _ in range(2) ] for _ in range(2) ] for _ in range(7)]
    gamma = [[[0]*2 for _ in range(2)] for _ in range(7)]
    
    # Step 0: M1 = (A11+A22)(B11+B22)
    for i in range(2):
        for j in range(2):
            alpha[0][0][0][i][j] = 1
            alpha[0][1][1][i][j] = 1
            beta[0][0][0][j][i] = 1 # Note: B index is j',k'. Here j'=j, k'=i? No, B is 2x2.
            # B_m[j', k'] = B11[j', k'] + B22[j', k']
            # So beta[0][0][0][j'][k'] = 1, beta[0][1][1][j'][k'] = 1
            # Let's fix the beta indexing: beta[s][J][K][j'][k']
            beta[0][0][0][j][i] = 1 # Wait, k' is the column index of B.
            # In M = A*B, A is 2x2, B is 2x2.
            # A[i', j'] B[j', k'].
            # So for a fixed t (which fixes i', j', k'), we use A[i', j'] and B[j', k'].
            # So beta should be indexed by j', k'.
            # Let's re-initialize beta correctly.
    
    # Re-initialize and fill correctly
    alpha = [[[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    beta = [[[[[0 for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(2)] for _ in range(7)]
    gamma = [[[0 for _ in range(2)] for _ in range(2)] for _ in range(7)]
    
    # Step 0: M1 = (A11+A22)(B11+B22)
    for i in range(2):
        for j in range(2):
            alpha[0][0][0][i][j] = 1
            alpha[0][1][1][i][j] = 1
    for j in range(2):
        for k in range(2):
            beta[0][0][0][j][k] = 1
            beta[0][1][1][j][k] = 1
    gamma[0][0][0] = 1
    gamma[0][1][1] = 1
    
    # Step 1: M2 = (A21+A22)B11
    for i in range(2):
        for j in range(2):
            alpha[1][1][0][i][j] = 1
            alpha[1][1][1][i][j] = 1
    for j in range(2):
        for k in range(2):
            beta[1][0][0][j][k] = 1
    gamma[1][1][0] = 1
    gamma[1][1][1] = -1
    
    # Step 2: M3 = A11(B12-B22)
    for i in range(2):
        for j in range(2):
            alpha[2][0][0][i][j] = 1
    for j in range(2):
        for k in range(2):
            beta[2][0][1][j][k] = 1
            beta[2][1][1][j][k] = -1
    gamma[2][0][1] = 1
    gamma[2][1][0] = 1
    
    # Step 3: M4 = A22(B21-B11)
    for i in range(2):
        for j in range(2):
            alpha[3][1][1][i][j] = 1
    for j in range(2):
        for k in range(2):
            beta[3][1][0][j][k] = 1
            beta[3][0][0][j][k] = -1
    gamma[3][0][0] = -1
    gamma[3][0][1] = 1
    
    # Step 4: M5 = (A11+A12)B22
    for i in range(2):
        for j in range(2):
            alpha[4][0][0][i][j] = 1
            alpha[4][0][1][i][j] = 1
    for j in range(2):
        for k in range(2):
            beta[4][1][1][j][k] = 1
    gamma[4][0][0] = 1
    
    # Step 5: M6 = (A21-A11)(B11+B12)
    for i in range(2):
        for j in range(2):
            alpha[5][1][0][i][j] = 1
            alpha[5][0][0][i][j] = -1
    for j in range(2):
        for k in range(2):
            beta[5][0][0][j][k] = 1
            beta[5][0][1][j][k] = 1
    gamma[5][1][1] = 1
    
    # Step 6: M7 = (A12-A22)(B21+B22)
    for i in range(2):
        for j in range(2):
            alpha[6][0][1][i][j] = 1
            alpha[6][1][1][i][j] = -1
    for j in range(2):
        for k in range(2):
            beta[6][1][0][j][k] = 1
            beta[6][1][1][j][k] = 1
    gamma[6][0][0] = -1
    gamma[6][1][1] = -1
    
    # Fill U, V, W for the 4x4x4 part
    for s in range(7):
        for t in range(4):
            c = s * 4 + t
            # Determine i', j', k' from t
            if t == 0:
                ip, jp, kp = 0, 0, 0
            elif t == 1:
                ip, jp, kp = 0, 1, 1
            elif t == 2:
                ip, jp, kp = 1, 0, 0
            else:
                ip, jp, kp = 1, 1, 1
            
            # U[i*4+j, c]
            for I in range(2):
                for J in range(2):
                    i = 2 * I + ip
                    j = 2 * J + jp
                    U[i * 4 + j][c] = alpha[s][I][J][ip][jp]
            
            # V[j*4+k, c] (Note: V index is j*p+k, here p=4 for this part)
            for J in range(2):
                for K in range(2):
                    j = 2 * J + jp
                    k = 2 * K + kp
                    V[j * 4 + k][c] = beta[s][J][K][jp][kp]
            
            # W[k*4+i, c] (Note: W index is k*n+i, here n=4 for this part)
            for I in range(2):
                for K in range(2):
                    i = 2 * I + ip
                    k = 2 * K + kp
                    W[k * 4 + i][c] = gamma[s][I][K]
    
    # --- Construct 4x4x5 Schoolbook Decomposition ---
    col_offset = rank_444
    for i in range(4):
        for j in range(4):
            for k in range(4, 9):
                c = col_offset + (i * 4 + j) * 5 + (k - 4)
                U[i * 4 + j][c] = 1
                V[j * 9 + k][c] = 1
                W[k * 4 + i][c] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
