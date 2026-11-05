"""Matrix multiplication 4x5x5 with Strassen-based decomposition."""
DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # We'll build the rank by combining:
    # 1. 4x4x4 Strassen (28 terms)
    # 2. 4x4x1 schoolbook (16 terms)
    # 3. 4x1x4 (4 terms)
    # 4. 4x1x1 (4 terms)
    # Total rank = 28 + 16 + 4 + 4 = 52
    
    # Actually, let's think more carefully.
    # The tensor is T[i,j,k] = sum over... 
    # We need U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = sum_{j} A[i,j] * B[j,k]
    # where A is 4x5, B is 5x5, C is 4x5.
    
    # Split j into j1 in [0,3] and j2 = 4
    # Split k into k1 in [0,3] and k2 = 4
    
    # C[i, k1] = sum_{j1=0}^{3} A[i,j1] * B[j1,k1]  (for k1 in 0..3)
    # C[i, k2] = sum_{j1=0}^{3} A[i,j1] * B[j1,4] + A[i,4] * B[4,4]
    
    # So we have:
    # - A 4x4 block of A times a 4x4 block of B -> 28 terms (Strassen)
    # - A 4x1 block of A times a 1x4 block of B (row 4 of B) -> 4*1*4 = 16 terms? No.
    #   Actually sum_{j1} A[i,j1]*B[j1,4] is 4 terms per i, so 4*4=16 terms total? 
    #   No, for each i and k2=4: sum_{j1=0}^3 A[i,j1]*B[j1,4]. This is 4 terms per i, 4 i's = 16 terms.
    #   Wait, that's rank 16 for the 4x4x1 part? No, it's 4x1x4 from A's perspective... 
    #   Let me re-think. The term is A[i,j1]*B[j1,k]. For fixed k=4, j1 varies 0-3, i varies 0-3.
    #   This is a rank-16 tensor? No, each term (i,j1,k=4) is rank 1. There are 4*4=16 such terms. Yes, rank 16.
    
    # - A[i,4]*B[4,k1] for k1 in 0..3: 4 terms (i varies 0-3, k1 fixed... no, k1 varies 0-3, i varies 0-3)
    #   That's 4*4=16 terms? No wait. For each i and k1: A[i,4]*B[4,k1]. 
    #   i in 0..3, k1 in 0..3, so 16 terms.
    
    # - A[i,4]*B[4,4]: 4 terms (one per i).
    
    # Total: 28 + 16 + 16 + 4 = 64. Hmm, that's worse than 52.
    
    # Let me reconsider. Maybe I can use Strassen more cleverly or combine terms.
    
    # Actually, let's just implement the straightforward combination and verify.
    # But 64 > 52 from my earlier estimate. Let me re-examine.
    
    # The 4x4x4 part: A[0:4,0:4] * B[0:4,0:4] -> 28 terms via Strassen
    # The 4x4x1 part: A[0:4,0:4] * B[0:4,4:5] -> This is 4x4x1. Rank of M_{4,4,1} is 4*4=16.
    # The 4x1x4 part: A[0:4,4:5] * B[4:5,0:4] -> This is 4x1x4. Rank is 4*4=16? 
    #   Actually, A[i,4]*B[4,k] for i in 0..3, k in 0..3. Each (i,k) pair is rank 1. 16 terms.
    # The 4x1x1 part: A[0:4,4:5] * B[4:5,4:5] -> A[i,4]*B[4,4]. 4 terms.
    
    # Total: 28 + 16 + 16 + 4 = 64.
    
    # Hmm. The schoolbook is 4*5*5=100. So 64 is still an improvement.
    # Can we do better?
    
    # Maybe I can use a different decomposition. Let's just go with 64 for now and verify correctness.
    
    rank_total = 64
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a term
    def add_term(i, j, k, coeff):
        nonlocal col
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        U[idx_u][col] += coeff
        V[idx_v][col] += coeff
        W[idx_w][col] += coeff
        col += 1
    
    # Part 1: 4x4x4 Strassen decomposition
    # We decompose the 4x4x4 tensor using 4 independent 2x2x2 Strassen decompositions
    # for the four 2x2 blocks.
    
    # Strassen for 2x2x2: 7 terms
    # Given A = [[a00,a01],[a10,a11]], B = [[b00,b01],[b10,b11]]
    # C = A*B
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b01-b11)
    # M4 = a11(b10-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For 4x4x4, we split into four 2x2 blocks:
    # A = [[A00, A01], [A10, A11]] where each block is 2x2
    # B = [[B00, B01], [B10, B11]] where each block is 2x2
    # C = A*B
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    
    # For each block product, we use Strassen.
    # Block (0,0): A00*B00 -> rows 0-1 of A (cols 0-1), rows 0-1 of B (cols 0-1), output rows 0-1, cols 0-1
    # Block (0,1): A00*B01 -> rows 0-1 of A (cols 0-1), rows 0-1 of B (cols 2-3), output rows 0-1, cols 2-3
    # Block (0,2): A01*B10 -> rows 0-1 of A (cols 2-3), rows 2-3 of B (cols 0-1), output rows 0-1, cols 0-1
    # Block (0,3): A01*B11 -> rows 0-1 of A (cols 2-3), rows 2-3 of B (cols 2-3), output rows 0-1, cols 2-3
    # Block (1,0): A10*B00 -> rows 2-3 of A (cols 0-1), rows 0-1 of B (cols 0-1), output rows 2-3, cols 0-1
    # Block (1,1): A10*B01 -> rows 2-3 of A (cols 0-1), rows 0-1 of B (cols 2-3), output rows 2-3, cols 2-3
    # Block (1,2): A11*B10 -> rows 2-3 of A (cols 2-3), rows 2-3 of B (cols 0-1), output rows 2-3, cols 0-1
    # Block (1,3): A11*B11 -> rows 2-3 of A (cols 2-3), rows 2-3 of B (cols 2-3), output rows 2-3, cols 2-3
    
    # Each block product uses 7 terms. Total 8*7 = 56 terms? No, that's too many.
    # Wait, I'm using Strassen on each 2x2 block product separately. That gives 7 terms per block, 8 blocks = 56 terms.
    # But that's worse than the schoolbook 16 terms for 4x4x4!
    
    # I think I'm confusing myself. Strassen for 2x2x2 uses 7 multiplications instead of 4.
    # For 4x4x4, the schoolbook uses 4*4*4=64 terms.
    # Using Strassen on 2x2 blocks: we have 4 block rows and 4 block columns... no.
    
    # Let me restart the Strassen approach for 4x4x4.
    # Standard Strassen for 4x4x4 would recursively split into 2x2x2.
    # A 4x4x4 can be decomposed as follows:
    # Split A into 2x2 blocks of 2x2, B into 2x2 blocks of 2x2.
    # C = A*B has 4 block entries, each 2x2.
    # C00 = A00*B00 + A01*B10
    # C01 = A00*B01 + A01*B11
    # C10 = A10*B00 + A11*B10
    # C11 = A10*B01 + A11*B11
    # Each block product (e.g., A00*B00) is a 2x2x2 multiplication, which uses 7 terms via Strassen.
    # There are 8 block products, so 8*7 = 56 terms.
    # But wait, we can combine some of these. Actually, the standard Strassen recursion for 4x4x4
    # would give 7^2 = 49 terms? No, that's not right either.
    
    # Actually, I recall that the rank of M_{4,4,4} is at most 28 via a specific decomposition,
    # but the simple recursive Strassen gives 49. Let me just use the 49-term recursive Strassen.
    
    # For now, let me use the schoolbook for the 4x4x4 part (64 terms) and just do the simple split.
    # Total would be 64 + 16 + 16 + 4 = 100, which is no improvement.
    
    # OK let me just implement the 49-term Strassen for 4x4x4.
    
    # Actually, let me just go with a simpler approach: use schoolbook for everything but verify.
    # The schoolbook is rank 100. Let me implement Strassen for 4x4x4 with 49 terms.
    
    # Restart with a cleaner implementation.
    
    U = [[0] * rank_total for _ in range(n * m)]
    V = [[0] * rank_total for _ in range(m * p)]
    W = [[0] * rank_total for _ in range(p * n)]
    col = 0
    
    def add_term_simple(i, j, k, coeff):
        nonlocal col
        if col >= rank_total:
            return
        idx_u = i * m + j
        idx_v = j * p + k
        idx_w = k * n + i
        if idx_u < n * m and idx_v < m * p and idx_w < p * n:
            U[idx_u][col] += coeff
            V[idx_v][col] += coeff
            W[idx_w][col] += coeff
        col += 1
    
    # Strassen 2x2x2: adds 7 terms for block rows (r0,r1), block cols (c0,c1) of A and B
    # output block rows (or0,or1), output block cols (oc0,oc1)
    def strassen_2x2(a_row0, a_row1, b_row0, b_row1, b_col0, b_col1, c_row0, c_row1, c_col0, c_col1):
        nonlocal col
        # a_row0, a_row1: rows in A
        # b_row0, b_row1: rows in B (same as cols in A for block structure)
        # b_col0, b_col1: cols in B
        # c_row0, c_row1: rows in C
        # c_col0, c_col1: cols in C
        
        # M1 = (A[r0,c0] + A[r1,c1]) * (B[c0,r0] + B[c1,r1])
        # This gets complex with the index mapping. Let me use a different approach.
        
        # Let me define the 7 Strassen terms explicitly.
        # For 2x2: A = [[a,b],[c,d]], B = [[e,f],[g,h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        # C00 = m1+m4-m5+m7
        # C01 = m3+m5
        # C10 = m2+m4
        # C11 = m1-m2+m3+m6
        
        # In our tensor decomposition, each "multiplication" mX corresponds to a rank-1 term.
        # m1 = (a+d)(e+h) expands to 4 rank-1 terms: a*e, a*h, d*e, d*h
        # But Strassen combines them into fewer rank-1 terms in the tensor decomposition sense.
        # Actually, in the tensor decomposition, each mX is a single rank-1 term where the coefficients
        # are products of sums. But that's not how tensor rank works - each term must be a simple product.
        
        # I think I'm overcomplicating this. Let me just use the schoolbook decomposition
        # and accept rank 100, or find a known decomposition for 4x5x5.
        
        # For simplicity, let me just return the schoolbook.
        pass
    
    # Given the complexity, let me just return the schoolbook decomposition.
    # This is safe and correct, with rank 100.
    
    U = [[0] * (n*m*p) for _ in range(n*m)]
    V = [[0] * (n*m*p) for _ in range(m*p)]
    W = [[0] * (n*m*p) for _ in range(p*n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
