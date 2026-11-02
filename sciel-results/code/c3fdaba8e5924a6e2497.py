"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import random
import sys
import numpy as np

DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # 140s safety margin
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    # Schoolbook baseline
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(nm)]
    V = [[0] * rank_school for _ in range(mp)]
    W = [[0] * rank_school for _ in range(pn)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = {'U': U, 'V': V, 'W': W, 'rank': rank_school}
    
    # Verify function
    def verify(U, V, W, rank, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for c in range(rank):
                        s += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                    if abs(s - 1) > 1e-9:
                        return False
        return True
    
    # Construct rank 81 solution: 49 (for 4x4x4) + 32 (for 4x4x2)
    # 4x4x4 uses recursive Strassen (rank 49)
    # 4x4x2 uses schoolbook (rank 32)
    
    # Strassen coefficients for 2x2 matrix multiplication
    A_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                  # M3
        {(1,1): 1},                  # M4
        {(0,0): 1, (0,1): 1},       # M5
        {(1,0): 1, (0,0): -1},      # M6
        {(0,1): 1, (1,1): -1}       # M7
    ]
    B_coeffs = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                  # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                  # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,0): 1, (1,1): 1}        # M7
    ]
    C_coeffs = [
        [
            [1, 0, 0, -1, 1, 0, -1],  # C00
            [0, 0, 1, 0, 1, 0, 0]     # C01
        ],
        [
            [0, 1, 0, 1, 0, 0, 0],    # C10
            [1, -1, 1, 0, 0, 1, 0]    # C11
        ]
    ]
    
    # Block-level coefficients for outer Strassen
    block_A_coeffs = [
        [[1, 0], [0, 1]],    # M1: A00 + A11
        [[0, 0], [1, 1]],    # M2: A10 + A11
        [[1, 0], [0, 0]],    # M3: A00
        [[0, 0], [0, 1]],    # M4: A11
        [[1, 1], [0, 0]],    # M5: A00 + A01
        [[0, 0], [1, -1]],   # M6: A10 - A00
        [[0, 1], [0, -1]]    # M7: A01 - A11
    ]
    block_B_coeffs = [
        [[1, 0], [0, 1]],    # M1: B00 + B11
        [[1, 0], [0, 0]],    # M2: B00
        [[0, 1], [0, -1]],   # M3: B01 - B11
        [[0, -1], [1, 0]],   # M4: B10 - B00
        [[0, 0], [0, 1]],    # M5: B11
        [[1, 1], [0, 0]],    # M6: B00 + B01
        [[0, 1], [1, 1]]     # M7: B10 + B11
    ]
    
    # Build 4x4x4 decomposition using recursive Strassen (rank 49)
    rank_444 = 49
    U_444 = [[0]*rank_444 for _ in range(16)]
    V_444 = [[0]*rank_444 for _ in range(16)]
    W_444 = [[0]*rank_444 for _ in range(16)]
    
    for beta in range(7):
        for alpha in range(7):
            col = beta * 7 + alpha
            
            # Build U_444 for this term
            for i in range(4):
                for j in range(4):
                    block_i = i // 2
                    block_j = j // 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in A_coeffs[alpha]:
                                a_coeff = A_coeffs[alpha][(r, s)]
                                b_coeff = block_A_coeffs[beta][r][s]
                                if r == block_i and s == block_j:
                                    coeff += a_coeff * b_coeff
                    U_444[i*4+j][col] = coeff
            
            # Build V_444 for this term
            for j in range(4):
                for k in range(4):
                    block_j = j // 2
                    block_k = k // 2
                    
                    coeff = 0
                    for r in range(2):
                        for s in range(2):
                            if (r, s) in B_coeffs[alpha]:
                                b_coeff = B_coeffs[alpha][(r, s)]
                                a_coeff = block_B_coeffs[beta][r][s]
                                if r == block_j and s == block_k:
                                    coeff += b_coeff * a_coeff
                    V_444[j*4+k][col] = coeff
            
            # Build W_444 for this term
            for bi in range(2):
                for bk in range(2):
                    for p in range(2):
                        for q in range(2):
                            c_i = 2*bi + p
                            c_k = 2*bk + q
                            coeff = C_coeffs[bi][bk][beta] * C_coeffs[p][q][alpha]
                            W_444[c_k*4+c_i][col] += coeff
    
    # Build 4x4x2 part using schoolbook (rank 32)
    rank_442 = 4 * 4 * 2
    U_442 = [[0]*rank_442 for _ in range(16)]
    V_442 = [[0]*rank_442 for _ in range(8)]
    W_442 = [[0]*rank_442 for _ in range(8)]
    
    col_442 = 0
    for i in range(4):
        for j in range(4):
            for k in range(2):
                U_442[i*4+j][col_442] = 1
                V_442[j*2+k][col_442] = 1
                W_442[k*4+i][col_442] = 1
                col_442 += 1
    
    # Combine: total rank = 49 + 32 = 81
    total_rank = 49 + 32
    U_total = [[0]*total_rank for _ in range(16)]
    V_total = [[0]*total_rank for _ in range(24)]
    W_total = [[0]*total_rank for _ in range(24)]
    
    # Copy 4x4x4 part (columns 0-48)
    for i in range(16):
        for c in range(49):
            U_total[i][c] = U_444[i][c]
            V_total[i][c] = V_444[i][c]
            W_total[i][c] = W_444[i][c]
    
    # Copy 4x4x2 part (columns 49-80)
    for i in range(16):
        for c in range(32):
            U_total[i][49+c] = U_442[i][c]
    for j in range(8):
        for c in range(32):
            V_total[j][49+c] = V_442[j][c]
    for k in range(8):
        for c in range(32):
            W_total[k][49+c] = W_442[k][c]
    
    # Verify the combined solution
    if verify(U_total, V_total, W_total, total_rank, 4, 4, 6):
        best = {'U': U_total, 'V': V_total, 'W': W_total, 'rank': total_rank}
    
    return {'U': best['U'], 'V': best['V'], 'W': best['W']}
# EVOLVE-BLOCK-END
