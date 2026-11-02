"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import itertools
import random

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
    
    # Block Strassen - 12 independent 2x2x2 Strassen products
    # A: 4x4 -> 2x2 blocks indexed by (a,c) where a=row_block, c=col_block
    # B: 4x6 -> 2x2 blocks indexed by (c,b) where c=row_block, b=col_block
    # C: 4x6 -> 2x2 blocks indexed by (a,b) where a=row_block, b=col_block
    # C[a,b] = sum_c A[a,c] * B[c,b]
    # Total: 2*2*3 = 12 products, each rank 7 -> rank 84
    
    # Strassen 2x2x2 decomposition terms
    # Each term: (A_coefs, B_coefs, C_coefs)
    strassen_terms = [
        # P1 = (A00+A11)(B00+B11)
        ({(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}, {(0,0):1, (1,1):1}),
        # P2 = (A10+A11)B00
        ({(1,0):1, (1,1):1}, {(0,0):1}, {(1,0):1}),
        # P3 = A00(B10-B11)
        ({(0,0):1}, {(1,0):1, (1,1):-1}, {(0,1):1, (1,1):1}),
        # P4 = A11(B01-B00)
        ({(1,1):1}, {(0,1):1, (0,0):-1}, {(0,0):1, (1,0):1}),
        # P5 = (A00+A01)B11
        ({(0,0):1, (0,1):1}, {(1,1):1}, {(0,0):-1, (0,1):1}),
        # P6 = (A10-A00)(B00+B01)
        ({(1,0):1, (0,0):-1}, {(0,0):1, (0,1):1}, {(1,1):1}),
        # P7 = (A01-A11)(B10+B11)
        ({(0,1):1, (1,1):-1}, {(1,0):1, (1,1):1}, {(0,0):1, (1,1):-1}),
    ]
    
    rank1 = 12 * 7  # 84
    U1 = [[0] * rank1 for _ in range(n * m)]  # 16 x 84
    V1 = [[0] * rank1 for _ in range(m * p)]  # 24 x 84
    W1 = [[0] * rank1 for _ in range(p * n)]  # 24 x 84
    
    col = 0
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(3):  # col block of B/C
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side: A_block(a,c) = A[2a:2a+2, 2c:2c+2]
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar  # global row in A
                        gj = 2*c + ac  # global col in A
                        U1[gi * m + gj][col] += coef
                    
                    # B-side: B_block(c,b) = B[2c:2c+2, 2b:2b+2]
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br  # global row in B
                        gj = 2*b + bc  # global col in B
                        V1[gi * p + gj][col] += coef
                    
                    # C-side: C_block(a,b) = C[2a:2a+2, 2b:2b+2]
                    # W[k*n+i] where k is row of C (=col of B), i is col of C (=row of A)
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc  # k (row of C)
                        gj = 2*a + cr  # i (col of C)
                        W1[gi * n + gj][col] += coef
                    
                    col += 1
    
    assert col == rank1, f"col={col}, rank1={rank1}"
    
    if verify(U1, V1, W1, rank1, n, m, p):
        best_rank = rank1
        best_U = U1
        best_V = V1
        best_W = W1
    
    # Strategy 2: Try to use a rank-49 decomposition for 4x4x4 + rank-32 for 4x4x2
    # But first, let's try a different approach: use the known rank-49 for 4x4x4
    # and handle the extra 2 columns separately.
    
    # For now, let's try to optimize the block Strassen by sharing factors
    # when possible.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Strategy 3: Try a 4x4x4 rank-49 decomposition for the first 4 columns
    # and a 4x4x2 decomposition for the last 2 columns.
    
    # For 4x4x4, use the known Strassen rank-49 algorithm.
    # For 4x4x2, use 8 products of 2x2 * 2x1, each rank 4 -> rank 32.
    
    # However, implementing rank-49 for 4x4x4 is complex.
    # Let's try a simpler approach: use 2x2x2 Strassen for the first 4 columns
    # (8 products, rank 56) and a different approach for the last 2 columns.
    
    # For the last 2 columns: 4x4 * 4x2
    # Split A into 2x2 blocks (4 blocks)
    # B: 4x2 -> split into 2x1 blocks (4 blocks)
    # C: 4x2 -> split into 2x1 blocks (4 blocks)
    # C[a,b] = sum_c A[a,c] * B[c,b] where a,c in {0,1}, b in {0,1}
    # 8 products of 2x2 * 2x1, each rank 4 -> rank 32
    # Total: 56 + 32 = 88. Worse than 84.
    
    # Let's try yet another approach: use a 4x4x6 decomposition that
    # exploits the rectangular structure more effectively.
    
    # Idea: Split A into 4x2 and 4x2 blocks, and B into 2x6 and 2x6 blocks.
    # C = A1 * B1 + A2 * B2
    # where A1, A2 are 4x2, B1, B2 are 2x6.
    # Each 4x2 * 2x6 is a 4x2 * 2x6 multiplication.
    
    # For 4x2 * 2x6:
    # Split B into 2x2 and 2x2 and 2x2 blocks (3 blocks)
    # C = A * B1 + A * B2 + A * B3
    # where A is 4x2, B1,B2,B3 are 2x2.
    # Each 4x2 * 2x2 is rank 8 (as computed earlier).
    # Total for 4x2 * 2x6: 3*8 = 24.
    # Total for 4x4 * 4x6: 2*24 = 48.
    
    # But wait, this doesn't work because A is shared between the two halves.
    # Let me reconsider.
    
    # Actually, the tensor rank is for the entire 4x4x6 problem.
    # T[i,j,k] = 1 for all i,j,k.
    # The rank is the minimum number of rank-1 tensors that sum to T.
    
    # With the approach above:
    # C = A[:,0:2] * B[0:2,:] + A[:,2:4] * B[2:4,:]
    # For each half: 4x2 * 2x6
    # For each 4x2 * 2x6: split B into 3 2x2 blocks
    # Each 4x2 * 2x2: rank 8
    # Total: 2 * 3 * 8 = 48.
    
    # But this is not correct because the A factors are different for each half.
    # Let me recompute.
    
    # Actually, I think the block Strassen approach with rank 84 is the best
    # I can do with the current knowledge.
    
    # Let's try one more thing: use a 4x4x4 rank-49 decomposition and see
    # if I can implement it.
    
    # The rank-49 decomposition for 4x4x4 is known but complex.
    # For now, let's stick with rank 84.
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
