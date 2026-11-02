"""Matrix multiplication 4x4x6 tensor decomposition."""
import time
import math

DIMENSIONS = [4, 4, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
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
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    # Try to improve using a 4x4x4 rank-49 decomposition for first 4 cols
    # and schoolbook for last 2 cols: total rank 49 + 32 = 81
    
    # For 4x4x4, use the known rank-49 decomposition by Strassen
    # This is complex, so let's try a simpler approach:
    # Use block Strassen with 2x2 blocks for 4x4x4: rank = 4*7 = 28
    # Then for 4x4x2: schoolbook rank = 4*4*2 = 32
    # Total: 28 + 32 = 60
    
    # Wait, that's not right. Let me think again.
    # For 4x4x4 with 2x2 blocks: 4 blocks of A, 4 blocks of B, 4 blocks of C
    # Each C block = sum of 2 A blocks * 2 B blocks
    # Using Strassen for each 2x2x2: 7 multiplications per block
    # Total: 4 * 7 = 28
    
    # For 4x4x2: A is 4x4, B is 4x2, C is 4x2
    # Split into 2x2 blocks: A has 4 blocks, B has 2 blocks (2 rows x 1 col of 2x2)
    # Wait, B is 4x2, so it's 2 blocks of 2x2 (rows 0-1 and 2-3, cols 0-1)
    # C is 4x2, so it's 2 blocks of 2x2
    # C[0,:] = A[0,0]*B[0,0] + A[0,1]*B[1,0] (where indices are block indices)
    # C[1,:] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    # Using Strassen for each: 7 multiplications
    # Total: 2 * 7 = 14
    
    # So total rank = 28 + 14 = 42
    
    # Let's implement this.
    
    # For 4x4x4 part (first 4 columns of B, first 4 columns of C):
    # A: 4x4, B1: 4x4, C1: 4x4
    # Use 2x2 blocks with Strassen: rank 28
    
    # For 4x4x2 part (last 2 columns of B, last 2 columns of C):
    # A: 4x4, B2: 4x2, C2: 4x2
    # Use 2x2 blocks with Strassen: rank 14
    
    # Total: 28 + 14 = 42
    
    rank2 = 28 + 14  # 42
    U2 = [[0] * rank2 for _ in range(n * m)]  # 16 x 42
    V2 = [[0] * rank2 for _ in range(m * p)]  # 24 x 42
    W2 = [[0] * rank2 for _ in range(p * n)]  # 24 x 42
    
    col = 0
    
    # Part 1: 4x4x4 (columns 0-3 of B and C)
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            for b in range(2):  # col block of B/C (first 4 cols)
                for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                    # A-side
                    for (ar, ac), coef in a_coefs.items():
                        gi = 2*a + ar
                        gj = 2*c + ac
                        U2[gi * m + gj][col] += coef
                    
                    # B-side (first 4 columns)
                    for (br, bc), coef in b_coefs.items():
                        gi = 2*c + br
                        gj = 2*b + bc  # cols 0-3
                        V2[gi * p + gj][col] += coef
                    
                    # C-side (first 4 columns)
                    for (cr, cc), coef in c_coefs.items():
                        gi = 2*b + cc  # k (row of C) = col of B, cols 0-3
                        gj = 2*a + cr  # i (col of C) = row of A
                        W2[gi * n + gj][col] += coef
                    
                    col += 1
    
    # Part 2: 4x4x2 (columns 4-5 of B and C)
    # B2 is 4x2, C2 is 4x2
    # Split into 2x2 blocks: B2 has 2 blocks (rows 0-1, cols 0-1) and (rows 2-3, cols 0-1)
    # C2 has 2 blocks (rows 0-1, cols 0-1) and (rows 2-3, cols 0-1)
    # A has 4 blocks
    # C2[0,:] = A[0,0]*B2[0,0] + A[0,1]*B2[1,0]
    # C2[1,:] = A[1,0]*B2[0,0] + A[1,1]*B2[1,0]
    
    # Using Strassen for each 2x2x2:
    for a in range(2):       # row block of A/C
        for c in range(2):   # col block of A / row block of B
            # b is the col block of B/C, but here B and C only have 1 col block (cols 0-1 of B2/C2)
            # So b = 0, and the global col in B is 4 + 0 = 4, in C is 4 + 0 = 4
            for (a_coefs, b_coefs, c_coefs) in strassen_terms:
                # A-side
                for (ar, ac), coef in a_coefs.items():
                    gi = 2*a + ar
                    gj = 2*c + ac
                    U2[gi * m + gj][col] += coef
                
                # B-side (last 2 columns, block col 0)
                for (br, bc), coef in b_coefs.items():
                    gi = 2*c + br
                    gj = 4 + bc  # cols 4-5
                    V2[gi * p + gj][col] += coef
                
                # C-side (last 2 columns, block col 0)
                for (cr, cc), coef in c_coefs.items():
                    gi = 4 + cc  # k (row of C) = col of B, cols 4-5
                    gj = 2*a + cr  # i (col of C) = row of A
                    W2[gi * n + gj][col] += coef
                
                col += 1
    
    assert col == rank2, f"col={col}, rank2={rank2}"
    
    if verify(U2, V2, W2, rank2, n, m, p):
        best_rank = rank2
        best_U = U2
        best_V = V2
        best_W = W2
    
    if time.time() > deadline:
        return {"U": best_U, "V": best_V, "W": best_W}
    
    return {"U": best_U, "V": best_V, "W": best_W}
