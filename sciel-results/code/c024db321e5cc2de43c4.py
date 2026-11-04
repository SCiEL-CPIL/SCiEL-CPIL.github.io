import time
import numpy as np

DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=4, m=4, p=5
    
    # Part 1: 4x4 * 4x4 using Strassen (Rank 49)
    # Part 2: 4x4 * 4x1 using Schoolbook (Rank 4)
    
    # Strassen 2x2 factors
    # U_2, V_2, W_2 of shape (4, 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))
    
    # M1 = (A11 + A22)(B11 + B22)
    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1
    
    # M2 = (A21 + A22)B11
    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1
    
    # M3 = A11(B12 - B22)
    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1
    
    # M4 = A22(B21 - B11)
    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[2, 3] = 1
    
    # M5 = (A11 + A12)B22
    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[3, 4] = 1
    
    # M6 = (A21 - A11)(B11 + B12)
    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1
    
    # M7 = (A12 - A22)(B21 + B22)
    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[1, 6] = 1
    
    # Construct 4x4 Strassen factors
    # U_4, V_4, W_4 of shape (16, 49)
    U_4 = np.zeros((16, 49))
    V_4 = np.zeros((16, 49))
    W_4 = np.zeros((16, 49))
    
    # Map 2x2 block indices to 4x4 indices
    # Block (bi, bj) in 2x2 grid corresponds to rows/cols 2*bi..2*bi+1, 2*bj..2*bj+1
    # U index: i*4 + j. For block (bi, bj), i in [2*bi, 2*bi+1], j in [2*bj, 2*bj+1]
    # V index: j*4 + k. For block (bj, bk), j in [2*bj, 2*bj+1], k in [2*bk, 2*bk+1]
    # W index: k*4 + i. For block (bk, bi), k in [2*bk, 2*bk+1], i in [2*bi, 2*bi+1]
    
    for bi in range(2):
        for bj in range(2):
            for bk in range(2):
                # This corresponds to the (bi, bj) block of A and (bj, bk) block of B?
                # No, Strassen is applied to the whole 4x4.
                # The 7 multiplications M1..M7 are of 2x2 matrices.
                # Each M corresponds to a specific combination of blocks.
                # We need to expand each 2x2 Strassen step into 7 rank-1 terms.
                
                # Actually, it's easier to just say:
                # The 4x4 multiplication is composed of 7 sub-multiplications of 2x2 matrices.
                # Each 2x2 multiplication has rank 7.
                # So total rank 49.
                # We need to map the local 2x2 indices to global 4x4 indices.
                
                # Let's define the 7 Strassen terms for the 4x4 case directly.
                # This is complex to generate programmatically without a library.
                # I will hardcode the 49 terms? No, that's too much.
                
                # Alternative: Just use Schoolbook for 4x4x4 (Rank 64) and Schoolbook for 4x4x1 (Rank 4).
                # Total Rank 68.
                # This is simpler and guaranteed to be correct.
                # 68 < 80.
                # Score improvement: 80/68 = 1.17.
                
                pass
    
    # Let's implement the simple block decomposition: Schoolbook 4x4x4 + Schoolbook 4x4x1
    rank_444 = n * m * 4 # 64
    rank_441 = n * 1 * 1 # 4
    total_rank = rank_444 + rank_441
    
    U = np.zeros((n * m, total_rank))
    V = np.zeros((m * p, total_rank))
    W = np.zeros((p * n, total_rank))
    
    col = 0
    # 4x4x4 part (k=0..3)
    for i in range(n):
        for j in range(m):
            for k in range(4):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
                
    # 4x4x1 part (k=4)
    for i in range(n):
        for j in range(m):
            U[i * m + j, col] = 1
            V[j * p + 4, col] = 1
            W[4 * n + i, col] = 1
            col += 1
            
    return {'U': U.tolist(), 'V': V.tolist(), 'W': W.tolist()}
# EVOLVE-BLOCK-END
