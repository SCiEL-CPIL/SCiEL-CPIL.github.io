import numpy as np
import time

DIMENSIONS = [3, 4, 5]

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 120
    
    # Schoolbook rank is n*m*p = 60
    # We use a recursive decomposition to reduce rank.
    # Split n=3 into 2 and 1.
    # C = A1 * B + A2 * B
    # A1 is 2x4, A2 is 1x4.
    # Rank of 2x4x5:
    #   Split m=4 into 2 and 2.
    #   A1 = [A11, A12]
    #   C1 = A11 * B1 + A12 * B2
    #   B1, B2 are 2x5.
    #   Rank of 2x2x5:
    #     Schoolbook is 20.
    #     Can we do better?
    #     For 2x2x1, rank 2.
    #     For 2x2x2, rank 7.
    #     For 2x2x5, we can use 5 * rank(2x2x1)? No.
    #     We can use 5 * 2 = 10?
    #     Or 2 * rank(2x2x5/2)?
    #     Let's use schoolbook for 2x2x5, rank 20.
    #     So 2x4x5 rank is 20 + 20 = 40?
    #     Wait, 2x2x5 schoolbook is 2*2*5 = 20.
    #     So 2x4x5 is 40.
    #     1x4x5 rank is 4 (as derived before? No, schoolbook is 20).
    #     If 1x4x5 is 20, total is 60.
    #     
    #     Let's check rank of 1x4x5 again.
    #     C_c = sum_b A_b B_{b,c}.
    #     This is a sum of 4 terms for each c.
    #     Total 20 terms.
    #     Can we reduce?
    #     No, it's a linear map from A to C for fixed B.
    #     The rank is 4?
    #     If rank is 4, then 1x4x5 is 4.
    #     Then total is 40 + 4 = 44.
    
    # Let's assume 1xm p rank is m.
    # And 2x2xp rank is 2p?
    # If so, 2x4x5 is 2*2*5 + 2*2*5 = 40.
    # Total 44.
    
    # This is better than 60.
    
    # Constructing the 44-rank decomposition.
    rank = 44
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    col = 0
    
    # Part 1: A1 (2x4) * B (4x5)
    # Split B into B1 (2x5) and B2 (2x5)
    # A1 = [A11 (2x2), A12 (2x2)]
    
    # A11 * B1 (2x2x5)
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i*4 + j][col] = 1
                V[j*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # A12 * B2 (2x2x5)
    for i in range(2):
        for j in range(2):
            for k in range(5):
                U[i*4 + (j+2)][col] = 1
                V[(j+2)*5 + k][col] = 1
                W[k*3 + i][col] = 1
                col += 1
                
    # Part 2: A2 (1x4) * B (4x5)
    # Rank 4?
    # If rank is 4, we need 4 terms.
    # C_k = sum_b A_b B_{b,c}
    # We can use 4 multiplications: z_b = A_b * 1?
    # No, we need to multiply A_b by B_{b,c}.
    # This is 20 multiplications in schoolbook.
    # If rank is 4, how?
    # Maybe it's not 4.
    # Let's use schoolbook for 1x4x5, rank 20.
    # Total rank 40 + 20 = 60.
    
    # So we are back to 60.
    
    # Let's try to use Strassen for 2x2x5.
    # Rank of 2x2x2 is 7.
    # For 2x2x5, we can do 5 * 7 = 35?
    # No, Strassen works on 2x2 blocks.
    # 2x2x5 is 5 independent 2x2x1 multiplications?
    # No, it's one 2x2 matrix times a 2x5 matrix.
    # We can use Strassen to compute the 2x2 product, but the output is 2x5.
    # This is not standard.
    
    # Given the complexity, I will return the schoolbook decomposition.
    # It is safe and correct.
    
    U = np.zeros((n*m, n*m*p))
    V = np.zeros((m*p, n*m*p))
    W = np.zeros((p*n, n*m*p))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
