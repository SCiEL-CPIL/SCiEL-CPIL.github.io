"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
import time

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 6
    
    # Use a rank-23 decomposition for 3x3 matrix multiplication
    # Applied to two 3x3 blocks of the 3x6 matrix
    # Total rank = 2 * 23 = 46
    
    # Define the rank-23 decomposition for 3x3 x 3x3 -> 3x3
    # Using a known explicit construction
    
    # The 23 terms are defined as follows:
    # For each term c (0 to 22), we have U[i,j,c], V[j,k,c], W[k,i,c]
    # where i,j,k are in 0..2
    
    # I'll use a verified construction from the literature.
    # Reference: Strassen's work and subsequent optimizations
    
    # For simplicity and correctness, let me use a practical approach:
    # Define the 23 terms explicitly
    
    rank_3x3 = 23
    U_3x3 = [[0.0] * rank_3x3 for _ in range(9)]
    V_3x3 = [[0.0] * rank_3x3 for _ in range(9)]
    W_3x3 = [[0.0] * rank_3x3 for _ in range(9)]
    
    # Define the 23 terms for 3x3 matrix multiplication
    # This is based on a known rank-23 algorithm
    
    # Term 0: Direct term A[0,0]*B[0,0]
    U_3x3[0*3+0][0] = 1.0
    V_3x3[0*3+0][0] = 1.0
    W_3x3[0*3+0][0] = 1.0
    
    # Term 1: Direct term A[0,0]*B[0,1]
    U_3x3[0*3+0][1] = 1.0
    V_3x3[0*3+1][1] = 1.0
    W_3x3[0*3+0][1] = 1.0
    
    # Term 2: Direct term A[0,0]*B[0,2]
    U_3x3[0*3+0][2] = 1.0
    V_3x3[0*3+2][2] = 1.0
    W_3x3[0*3+0][2] = 1.0
    
    # Term 3: Direct term A[0,1]*B[1,0]
    U_3x3[0*3+1][3] = 1.0
    V_3x3[1*3+0][3] = 1.0
    W_3x3[0*3+0][3] = 1.0
    
    # Term 4: Direct term A[0,1]*B[1,1]
    U_3x3[0*3+1][4] = 1.0
    V_3x3[1*3+1][4] = 1.0
    W_3x3[0*3+0][4] = 1.0
    
    # Term 5: Direct term A[0,1]*B[1,2]
    U_3x3[0*3+1][5] = 1.0
    V_3x3[1*3+2][5] = 1.0
    W_3x3[0*3+0][5] = 1.0
    
    # Term 6: Direct term A[0,2]*B[2,0]
    U_3x3[0*3+2][6] = 1.0
    V_3x3[2*3+0][6] = 1.0
    W_3x3[0*3+0][6] = 1.0
    
    # Term 7: Direct term A[0,2]*B[2,1]
    U_3x3[0*3+2][7] = 1.0
    V_3x3[2*3+1][7] = 1.0
    W_3x3[0*3+0][7] = 1.0
    
    # Term 8: Direct term A[0,2]*B[2,2]
    U_3x3[0*3+2][8] = 1.0
    V_3x3[2*3+2][8] = 1.0
    W_3x3[0*3+0][8] = 1.0
    
    # Term 9: Direct term A[1,0]*B[0,0]
    U_3x3[1*3+0][9] = 1.0
    V_3x3[0*3+0][9] = 1.0
    W_3x3[0*3+1][9] = 1.0
    
    # Term 10: Direct term A[1,0]*B[0,1]
    U_3x3[1*3+0][10] = 1.0
    V_3x3[0*3+1][10] = 1.0
    W_3x3[0*3+1][10] = 1.0
    
    # Term 11: Direct term A[1,0]*B[0,2]
    U_3x3[1*3+0][11] = 1.0
    V_3x3[0*3+2][11] = 1.0
    W_3x3[0*3+1][11] = 1.0
    
    # Term 12: Direct term A[1,1]*B[1,0]
    U_3x3[1*3+1][12] = 1.0
    V_3x3[1*3+0][12] = 1.0
    W_3x3[0*3+1][12] = 1.0
    
    # Term 13: Direct term A[1,1]*B[1,1]
    U_3x3[1*3+1][13] = 1.0
    V_3x3[1*3+1][13] = 1.0
    W_3x3[0*3+1][13] = 1.0
    
    # Term 14: Direct term A[1,1]*B[1,2]
    U_3x3[1*3+1][14] = 1.0
    V_3x3[1*3+2][14] = 1.0
    W_3x3[0*3+1][14] = 1.0
    
    # Term 15: Direct term A[1,2]*B[2,0]
    U_3x3[1*3+2][15] = 1.0
    V_3x3[2*3+0][15] = 1.0
    W_3x3[0*3+1][15] = 1.0
    
    # Term 16: Direct term A[1,2]*B[2,1]
    U_3x3[1*3+2][16] = 1.0
    V_3x3[2*3+1][16] = 1.0
    W_3x3[0*3+1][16] = 1.0
    
    # Term 17: Direct term A[1,2]*B[2,2]
    U_3x3[1*3+2][17] = 1.0
    V_3x3[2*3+2][17] = 1.0
    W_3x3[0*3+1][17] = 1.0
    
    # Term 18: Direct term A[2,0]*B[0,0]
    U_3x3[2*3+0][18] = 1.0
    V_3x3[0*3+0][18] = 1.0
    W_3x3[0*3+2][18] = 1.0
    
    # Term 19: Direct term A[2,0]*B[0,1]
    U_3x3[2*3+0][19] = 1.0
    V_3x3[0*3+1][19] = 1.0
    W_3x3[0*3+2][19] = 1.0
    
    # Term 20: Direct term A[2,0]*B[0,2]
    U_3x3[2*3+0][20] = 1.0
    V_3x3[0*3+2][20] = 1.0
    W_3x3[0*3+2][20] = 1.0
    
    # Term 21: Direct term A[2,1]*B[1,0]
    U_3x3[2*3+1][21] = 1.0
    V_3x3[1*3+0][21] = 1.0
    W_3x3[0*3+2][21] = 1.0
    
    # Term 22: Direct term A[2,1]*B[1,1]
    U_3x3[2*3+1][22] = 1.0
    V_3x3[1*3+1][22] = 1.0
    W_3x3[0*3+2][22] = 1.0
    
    # This is still schoolbook (27 terms would be complete, I have 23 but missing some)
    # Let me just use the full schoolbook for correctness
    
    # For 3x3 x 3x6:
    # Split B = [B1 | B2] where B1, B2 are 3x3
    # C = A * B = [A*B1 | A*B2]
    # Use rank-r decomposition for 3x3 x 3x3, total rank = 2*r
    
    # Use schoolbook rank 27 for each 3x3 block, total rank = 54
    # This is the same as the original, but let me verify it works
    
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
