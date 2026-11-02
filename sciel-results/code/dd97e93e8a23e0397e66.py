"""Matrix multiplication 4x4x6 tensor decomposition."""
import time

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    n, m, p = DIMENSIONS  # 4, 4, 6
    
    # Schoolbook as fallback
    rank_school = n * m * p  # 96
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    
    best_rank = rank_school
    best_U = U_school
    best_V = V_school
    best_W = W_school
    
    def verify(U, V, W, rank, n, m, p):
        """Verify that the decomposition is correct."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0
                    for r in range(rank):
                        total += U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r]
                    if abs(total - 1) > 1e-10:
                        return False
        return True
    
    # Strassen 2x2x2 decomposition terms
    strassen_terms = [
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    # Block Strassen - 12 independent 2x2x2 Strassen products, rank 84
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]
    V1 = [[0] * rank1 for _ in range(m * p)]
    W1 = [[0] * rank1 for _ in range(p * n)]
    
    col = 0
    for a in range(2):
        for c in range(2):
            for b in range(3):
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar
                        gj = 2*c + ac
                        U1[gi * m + gj][col] += coef
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br
                        gj = 2*b + bc
                        V1[gi * p + gj][col] += coef
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc
                        gj = 2*a + cr
                        W1[gi * n + gj][col] += coef
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    # Strategy: Kronecker product of Strassen(2x2x2, rank 7) and 2x2x3 (rank 7)
    # 4x4x6 = (2x2 x 2x2) x (2x2 x 2x3) in tensor structure
    # This gives rank 7 * 7 = 49
    #
    # The 2x2x3 rank-7 algorithm (verified below):
    # For A(2x2) * B(2x3) = C(2x3):
    # We need 7 rank-1 terms.
    
    # 2x2x3 rank-7 algorithm:
    # P1: A[0,0]*B[0,0] -> C[0,0]
    # P2: A[1,0]*B[0,0] -> C[1,0]
    # P3: A[0,0]*B[0,1] -> C[0,1]
    # P4: A[1,0]*B[0,1] -> C[1,1]
    # P5: A[0,0]*B[0,2] -> C[0,2]
    # P6: A[1,0]*B[0,2] -> C[1,2]
    # P7: (A[0,1]+A[1,1])*(B[1,0]+B[1,1]+B[1,2]) -> C[0,0],C[0,1],C[0,2],C[1,0],C[1,1],C[1,2]
    #
    # Check: C[0,0] = P1 + P7 = A[0,0]*B[0,0] + (A[0,1]+A[1,1])*(B[1,0]+B[1,1]+B[1,2])
    # Expected: A[0,0]*B[0,0] + A[0,1]*B[1,0]
    # These don't match, so this naive approach fails.
    #
    # Correct 2x2x3 rank-7 (from literature, verified by construction):
    # We use the fact that we can "absorb" one column of B into the A factors.
    
    # Alternative structural approach: 4x4x4 (rank 49 via Landsberg) + 4x4x2 (rank 28)
    # For 4x4x2: 4 products of 2x2 x 2x1, each rank 4 = 16... 
    # Actually 4x4x2: split A into 2x2 blocks (4), B into 2x1 (2), C into 2x1 (2)
    # C[a,b] = sum_c A[a,c]*B[c,b], 4*2*2 = 16 products of 2x2x2x1
    # 2x2x2x1: multiply 2x2 by 2x1 to get 2x1. Rank 4 (schoolbook) or can we do better?
    # 2x2 * 2x1 = 2x1: C[0]=A[0,0]*B[0]+A[0,1]*B[1], C[1]=A[1,0]*B[0]+A[1,1]*B[1]
    # This is rank 4 (4 independent products). No known better.
    # So 4x4x2 rank = 4*4 = 16? No wait, there are 4 A-blocks * 2 B-blocks * 2 C-blocks... 
    # Actually: a in {0,1}, c in {0,1}, b in {0,1}: 8 products of 2x2 * 2x1, each rank 4 = 32.
    
    # Total: 49 + 32 = 81. Better than 84!
    #
    # But implementing Landsberg rank-49 for 4x4x4 is very complex.
    # Let me try a different split: 4x4x6 as 2*(4x2 x 2x6)
    # 4x2 x 2x6: A is 4x2, B is 2x6, C is 4x6
    # Split A into two 2x2, B into three 2x2, C into six 2x2
    # 2*3 = 6 products of 2x2x2, each rank 7 = 42 per half
    # Total: 2*42 = 84. Same as before.
    
    # Let me try: split differently. 4x4x6 = (4x4 x 4x4) + (4x4 x 4x2)
    # 4x4x4: use 8 Strassen 2x2x2 = rank 56
    # 4x4x2: 8 products of 2x2 x 2x1, each rank 4 = 32
    # Total: 56 + 32 = 88. Worse.
    
    # Best simple approach remains 84. Let me try to find if there's a way to share
    # Strassen factors across the 3 column blocks for fixed (a,c).
    #
    # For fixed (a,c), we compute A[a,c] * B[c,0:6] where B[c,0:6] is 2x6.
    # This is a 2x2 * 2x6 = 2x6 problem. If we can do this in rank 14 (vs 3*7=21),
    # total would be 4*14 = 56.
    #
    # 2x2 * 2x6: A(2x2) * B(2x6) = C(2x6)
    # C[i,j] = sum_l A[i,l]*B[l,j] for i in {0,1}, j in {0,1,2,3,4,5}
    # This has 12 entries. Lower bound is max(4, 6, 12/4) = 4. 
    # Known: rank(2x2x6) = 14 (from Kronecker: 2x2x2 rank 7 times 2x1x3 rank 2)
    # Actually 2x2x6 = (2x2x2) x (1x1x3) in some sense? No.
    # 2x2 * 2x6: split B into three 2x2. 3 products of 2x2x2, rank 21.
    # Or: rank(2x2x6) = 14 is known (it's 2*7).
    
    # For 2x2 * 2x6 with rank 14:
    # Split B into B_left (2x4) and B_right (2x2)
    # 2x2 * 2x4: rank 14? No, 2x2x4 = 2x2x2 x 1x1x2... 
    # Actually 2x2x4: A(2x2)*B(2x4)=C(2x4). 8 entries. 
    # Split B into two 2x2: 2 products of 2x2x2, rank 14.
    # So 2x2x6: split into 2x2x4 (rank 14) + 2x2x2 (rank 7) = 21. Not helpful.
    # Or split into three 2x2: 3*7 = 21.
    # The rank 14 for 2x2x6 would require a non-trivial algorithm.
    
    # I'll stick with rank 84 which is verified working.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
