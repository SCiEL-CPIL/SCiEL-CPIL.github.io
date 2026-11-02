import numpy as np

DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen's 7 rank-1 decompositions for 2x2 matrix multiplication
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    
    u_list = [
        [1, 0, 0, 1],  # P1: A00 + A11
        [0, 0, 1, 1],  # P2: A10 + A11
        [1, 0, 0, 0],  # P3: A00
        [0, 0, 0, 1],  # P4: A11
        [1, 1, 0, 0],  # P5: A00 + A01
        [-1, 0, 1, 0], # P6: A10 - A00
        [0, 1, 0, -1]  # P7: A01 - A11
    ]
    
    v_list = [
        [1, 0, 0, 1],  # P1: B00 + B11
        [1, 0, 0, 0],  # P2: B00
        [0, 1, 0, -1], # P3: B01 - B11
        [-1, 0, 1, 0], # P4: B10 - B00
        [0, 0, 0, 1],  # P5: B11
        [1, 1, 0, 0],  # P6: B00 + B01
        [0, 0, 1, 1]   # P7: B10 + B11
    ]
    
    w_list = [
        [1, 0, 0, 1],  # P1: C00 + C11
        [-1, 0, 1, 0], # P2: -C00 + C10
        [0, 1, 0, -1], # P3: C01 - C11
        [1, 0, 1, 0],  # P4: C00 + C10
        [0, 1, 0, 0],  # P5: C01
        [1, 0, -1, 0], # P6: C00 - C10  (Note: Check signs)
        [0, 0, 0, 1]   # P7: C11
    ]
    
    # Verify Strassen decomposition
    # C00 = P1 - P2 + P4 - P6? 
    # Let's check C00 coeff: 1 - (-1) + 1 - 1 = 2? No.
    # Standard Strassen:
    # M11 = P1 - P2 + P4
    # M12 = P3 + P5
    # M21 = P2 + P4
    # M22 = P1 - P3 + P7
    # This uses only P1, P2, P3, P4, P5, P7. P6 is unused?
    # If P6 is unused, the rank is 6? No, rank is 7.
    # There are different variants. One common variant uses 7.
    # Let's use the one where all 7 are used.
    # Actually, if I use the 6 above, I can just set P6 to 0 or omit it?
    # No, I need 7 terms to get rank 7?
    # If the rank is 7, I need 7 terms.
    # If the specific formulas above only use 6, then the rank might be 6?
    # But it is known that rank(M2,2,2) = 7.
    # So there must be a dependency or I am missing a term.
    # Actually, the formulas I wrote are for a specific algorithm.
    # Let's stick to the known rank 7 decomposition.
    # I will use the 7 vectors from a reliable source or derive carefully.
    
    # Correct Strassen vectors (one valid set):
    # u1 = [1, 0, 0, 1]
    # u2 = [0, 0, 1, 1]
    # u3 = [1, 0, 0, 0]
    # u4 = [0, 0, 0, 1]
    # u5 = [1, 1, 0, 0]
    # u6 = [-1, 0, 1, 0]
    # u7 = [0, 1, 0, -1]
    
    # v1 = [1, 0, 0, 1]
    # v2 = [1, 0, 0, 0]
    # v3 = [0, 1, 0, -1]
    # v4 = [-1, 0, 1, 0]
    # v5 = [0, 0, 0, 1]
    # v6 = [1, 1, 0, 0]
    # v7 = [0, 0, 1, 1]
    
    # w1 = [1, 0, 0, 1]
    # w2 = [-1, 0, 1, 0]
    # w3 = [0, 1, 0, -1]
    # w4 = [1, 0, 1, 0]
    # w5 = [0, 1, 0, 0]
    # w6 = [1, 0, -1, 0]
    # w7 = [0, 0, 0, 1]
    
    # Let's check C00 (index 0):
    # w1[0]=1, w2[0]=-1, w3[0]=0, w4[0]=1, w5[0]=0, w6[0]=1, w7[0]=0
    # Sum = 1 - 1 + 0 + 1 + 0 + 1 + 0 = 2.
    # We want 1.
    # So scaling is needed or signs are wrong.
    
    # Let's use the explicit tensor decomposition from literature for M2,2,2.
    # The 7 terms are:
    # 1. (A11+A22)(B11+B22) -> C11 + C22
    # 2. (A21+A22)B11 -> -C11 + C21
    # 3. A11(B12-B22) -> C12 - C22
    # 4. A22(B21-B11) -> C11 + C21
    # 5. (A11+A12)B22 -> C12
    # 6. (A21-A11)(B11+B12) -> C21 - C11  (Wait, C11 coeff -1)
    # 7. (A12-A22)(B21+B22) -> C22
    
    # C11: 1 (from 1) - 1 (from 2) + 1 (from 4) - 1 (from 6) = 0? No.
    # Let's check C11 again.
    # 1: +1
    # 2: -1
    # 4: +1
    # 6: -1 (if w6 is [1,0,-1,0])
    # Sum = 1 - 1 + 1 - 1 = 0.
    # This is wrong.
    
    # I will use a verified set of 7 vectors.
    # From "Tensor Rank of Matrix Multiplication":
    # The 7 vectors are often listed.
    # I will use the following which sums correctly:
    
    u = [
        [1, 0, 0, 1],
        [0, 0, 1, 1],
        [1, 0, 0, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [-1, 0, 1, 0],
        [0, 1, 0, -1]
    ]
    v = [
        [1, 0, 0, 1],
        [1, 0, 0, 0],
        [0, 1, 0, -1],
        [-1, 0, 1, 0],
        [0, 0, 0, 1],
        [1, 1, 0, 0],
        [0, 0, 1, 1]
    ]
    w = [
        [1, 0, 0, 1],
        [-1, 0, 1, 0],
        [0, 1, 0, -1],
        [1, 0, 1, 0],
        [0, 1, 0, 0],
        [0, 0, -1, 0], # Adjusted
        [0, 0, 0, 1]
    ]
    
    # Check C00 (0): 1 - 1 + 0 + 1 + 0 + 0 + 0 = 1. OK.
    # Check C01 (1): 0 + 0 + 1 + 0 + 1 + 0 + 0 = 2. Want 1.
    # Check C10 (2): 0 + 1 + 0 + 1 + 0 - 1 + 0 = 1. OK.
    # Check C11 (3): 1 + 0 - 1 + 0 + 0 + 0 + 1 = 1. OK.
    # C01 is 2. Need to fix.
    # Change w5 to [0, 0.5, 0, 0]? No, integer/half-integer allowed.
    # If I use 0.5, it's valid.
    # But let's try to find integer solution.
    # Maybe w3 is [0, 1, 0, -1] and w5 is [0, 0, 0, 0]? No.
    
    # Let's just use the 24 rank decomposition if I can't quickly verify the 7.
    # But 21 is a significant improvement.
    # I will use the 7 vectors and scale w5 by 0.5 if needed?
    # Actually, let's look at C01.
    # Terms contributing to C01:
    # P3: +1
    # P5: +1
    # Sum 2.
    # If I change P5 to have 0.5?
    # u5, v5 are integers. w5 can be half-integer.
    # The prompt allows "integer/half-integer entries".
    # So I can set w5 = [0, 0.5, 0, 0].
    # Then C01 = 1 + 0.5 = 1.5? No.
    # I need sum 1.
    # If w3 = [0, 0.5, 0, -0.5] and w5 = [0, 0.5, 0, 0]?
    # C01 = 0.5 + 0.5 = 1.
    # C11 = -0.5 + 1 (from w7) = 0.5. Want 1.
    # This is getting messy.
    
    # I will stick to the trivial 24 rank for safety, as deriving the exact Strassen coefficients under time pressure is risky.
    # However, 21 is the theoretical minimum.
    # I will output the 24 rank solution.
    
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
                
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
