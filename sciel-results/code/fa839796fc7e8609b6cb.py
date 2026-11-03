"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [5, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    # Hybrid decomposition: Strassen 4x4 (rank 49) + explicit 5th-index interactions
    n, m, p = DIMENSIONS
    # We'll build U (25 x rank), V (25 x rank), W (25 x rank)
    # Index convention: U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
    # For 5x5x5: i in 0..4 (row of A), j in 0..4 (col of A / row of B), k in 0..4 (col of B)
    # U index: i*m + j = i*5 + j
    # V index: j*p + k = j*5 + k  
    # W index: k*n + i = k*5 + i
    
    # Start with Strassen 4x4 decomposition for indices 0..3
    # Strassen 2x2 has 7 terms. 4x4 = tensor product of two 2x2 Strassen = 49 terms.
    
    # Strassen 2x2 terms (for blocks indexed by 0,1):
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # For 4x4, we decompose into 2x2 blocks where each entry is a 2x2 matrix
    # Block indices: (a,b) where a,b in {0,1} for block position, and (c,d) where c,d in {0,1} for within-block
    # So matrix indices: 2*a+c, 2*b+d
    
    # The 49 terms are indexed by (a_strassen, b_strassen) where each is in 0..6
    # For term (s1, s2), we multiply the s1-th Strassen term of block-row with s2-th of block-col
    
    # Let's build this systematically
    # Strassen coefficients for 2x2:
    # Each term has coefficients for A entries and B entries
    # A entries: (0,0), (0,1), (1,0), (1,1) -> indices in 2x2
    # B entries: same
    
    # Strassen term definitions:
    # term 0 (M1): A: [(0,0,1),(1,1,1)], B: [(0,0,1),(1,1,1)]
    # term 1 (M2): A: [(1,0,1),(1,1,1)], B: [(0,0,1)]
    # term 2 (M3): A: [(0,0,1)], B: [(0,1,1),(1,1,-1)]
    # term 3 (M4): A: [(1,1,1)], B: [(1,0,1),(0,0,-1)]
    # term 4 (M5): A: [(0,0,1),(0,1,1)], B: [(1,1,1)]
    # term 5 (M6): A: [(1,0,1),(0,0,-1)], B: [(0,0,1),(0,1,1)]
    # term 6 (M7): A: [(0,1,1),(1,1,-1)], B: [(1,0,1),(1,1,1)]
    
    strassen_A = [
        [(0,0,1),(1,1,1)],
        [(1,0,1),(1,1,1)],
        [(0,0,1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(0,0,-1)],
        [(0,1,1),(1,1,-1)]
    ]
    strassen_B = [
        [(0,0,1),(1,1,1)],
        [(0,0,1)],
        [(0,1,1),(1,1,-1)],
        [(1,0,1),(0,0,-1)],
        [(1,1,1)],
        [(0,0,1),(0,1,1)],
        [(1,0,1),(1,1,1)]
    ]
    
    # For 4x4, the Strassen decomposition operates on 2x2 blocks
    # Block (a,b) contains elements at positions (2*a+c, 2*b+d) for c,d in {0,1}
    # So for a 4x4 matrix, entry (i,j) = block(a=i//2, b=j//2), within(c=i%2, d=j%2)
    
    # The 49 terms: for each pair (s1, s2) in 0..6 x 0..6
    # Term (s1, s2) computes: (sum over strassen_A[s1] entries in block-row) * (sum over strassen_B[s2] entries in block-col)
    # This gives a 2x2 block result, which we need to map to C entries
    
    # Actually, let me think about this differently. The standard Kronecker product approach:
    # 4x4 multiplication tensor = kron(2x2 mult tensor, 2x2 mult tensor)
    # Rank = 7 * 7 = 49
    
    # Let's build it directly using the Kronecker structure
    # For term index t = s1 * 7 + s2 where s1, s2 in 0..6
    # U part: for each (a,c) where a is block row index, c is within-block row
    #   A entry at (2*a+c, j) contributes based on strassen_A[s1]
    # V part: for each (b,d) where b is block col index, d is within-block col
    #   B entry at (i, 2*b+d) contributes based on strassen_B[s2]
    # W part: for each (a', c') output block position
    #   C entry at (2*a'+c', k) where k is... 
    
    # This is getting complex. Let me just build the schoolbook for the 5th index parts
    # and use a known 49-term 4x4 decomposition.
    
    # Alternative: use the fact that we can decompose 5x5 as:
    # - 4x4 block (top-left): rank 49 via Strassen
    # - Row 4 (bottom row, indices 4,0 to 4,4): needs 4*5 = 20 schoolbook terms for A[4,:]*B
    #   But we can share some. Actually A[4,j]*B[j,k] for j=0..4, k=0..4
    # - Col 4 (right col, indices 0,4 to 3,4): A[i,4]*B[4,k] for i=0..3, k=0..4
    # - Corner (4,4): A[4,4]*B[4,4]
    
    # The 4x4 Strassen handles A[0:4,0:4]*B[0:4,0:4] -> C[0:4,0:4]
    # Remaining:
    # C[4,k] = sum_{j=0}^{4} A[4,j]*B[j,k] for k=0..4
    # C[i,4] = sum_{j=0}^{4} A[i,j]*B[j,4] for i=0..3
    # C[4,4] is included in both above
    
    # For C[4,k]: we need 5 terms (one per j) for each k, so 5*5=25 terms if schoolbook
    # For C[i,4]: we need 5 terms (one per j) for each i, so 4*5=20 terms if schoolbook
    # But C[4,4] = sum_j A[4,j]*B[j,4] is counted in both. Let's be careful.
    
    # Actually, let's just do:
    # 49 terms for 4x4 core
    # For the remaining entries, use schoolbook:
    # C[4,k] for k=0..4: each needs sum over j of A[4,j]*B[j,k] -> 5 terms each, 25 total
    # C[i,4] for i=0..3: each needs sum over j of A[i,j]*B[j,4] -> 5 terms each, 20 total
    # But wait, this double-counts C[4,4]. Let me handle it properly.
    
    # Total = 49 + 25 + 20 = 94? That's still better than 125.
    # But we might be able to do better by sharing terms.
    
    # Let me just implement the 49-term 4x4 Strassen plus schoolbook for the rest.
    
    rank = 49 + 25 + 20  # 94
    U = [[0] * rank for _ in range(25)]
    V = [[0] * rank for _ in range(25)]
    W = [[0] * rank for _ in range(25)]
    
    col = 0
    
    # Part 1: 4x4 Strassen decomposition (indices 0..3)
    # Using Kronecker product of two 2x2 Strassen
    # Term (s1, s2) for s1,s2 in 0..6
    # A index: (2*a + c, 2*b + d) where strassen_A[s1] has entry (a,c,coeff_a)
    #          and the within-block position maps to actual index
    # Wait, I need to be more careful.
    
    # In the 2x2 Strassen, we have 4 A-entries: positions (0,0), (0,1), (1,0), (1,1)
    # In the 4x4, these map to 2x2 blocks. Block (a,b) has entries at (2a+c, 2b+d).
    # The Strassen decomposition on blocks: for block-matrix M with blocks M_{ab},
    # we compute block-products using Strassen.
    
    # For the Kronecker product construction:
    # The 49 terms are indexed by (r, s) where r,s in 0..6
    # Term (r,s):
    #   U part: for each (a, c) in strassen_A[r] with coefficient ua,
    #           for each d in {0,1}: U[2*a + c, 2*b + d, col] += ua * strassen_B_within[s][b, d]
    #   Hmm, this is getting tangled.
    
    # Let me use a simpler approach: just enumerate all 49 terms explicitly
    # by computing the Kronecker product of the two 2x2 Strassen factorizations.
    
    # 2x2 Strassen factors:
    # U2[entry, term], V2[entry, term], W2[entry, term] for 7 terms
    # A entry (i,j) -> U2[i*2+j, t]
    # B entry (j,k) -> V2[j*2+k, t]  
    # C entry (i,k) -> W2[i*2+k, t]
    
    # 2x2 Strassen:
    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    
    # C00 = M0 + M3 - M4 + M6
    # C01 = M2 + M4
    # C10 = M1 + M3
    # C11 = M0 - M1 + M2 + M5
    
    U2 = [[0]*7 for _ in range(4)]
    V2 = [[0]*7 for _ in range(4)]
    W2 = [[0]*7 for _ in range(4)]
    
    # Term 0: (A00+A11)(B00+B11)
    U2[0][0] = 1; U2[3][0] = 1
    V2[0][0] = 1; V2[3][0] = 1
    W2[0][0] = 1; W2[1][0] = 0; W2[2][0] = 0; W2[3][0] = 1
    
    # Term 1: (A10+A11)B00
    U2[2][1] = 1; U2[3][1] = 1
    V2[0][1] = 1
    W2[2][1] = 1; W2[3][1] = -1
    
    # Term 2: A00(B01-B11)
    U2[0][2] = 1
    V2[1][2] = 1; V2[3][2] = -1
    W2[1][2] = 1; W2[3][2] = 1
    
    # Term 3: A11(B10-B00)
    U2[3][3] = 1
    V2[2][3] = 1; V2[0][3] = -1
    W2[0][3] = 1; W2[2][3] = 1
    
    # Term 4: (A00+A01)B11
    U2[0][4] = 1; U2[1][4] = 1
    V2[3][4] = 1
    W2[0][4] = -1; W2[1][4] = 1
    
    # Term 5: (A10-A00)(B00+B01)
    U2[2][5] = 1; U2[0][5] = -1
    V2[0][5] = 1; V2[1][5] = 1
    W2[3][5] = 1
    
    # Term 6: (A01-A11)(B10+B11)
    U2[1][6] = 1; U2[3][6] = -1
    V2[2][6] = 1; V2[3][6] = 1
    W2[0][6] = 1
    
    # Now 4x4 = kron(2x2, 2x2)
    # 4x4 index: (i,j) = (2*i2 + i1, 2*j2 + j1) where i2,j2 in {0,1} block, i1,j1 in {0,1} within
    # Term (t2, t1) for t2,t1 in 0..6:
    #   U4[2*i2+i1, 2*j2+j1, col] += U2[i2*2+i1, t2] * U2[j2*2+j1, t1]
    #   V4[2*j2+j1, 2*k2+k1, col] += V2[j2*2+j1, t2] * V2[k2*2+k1, t1]
    #   W4[2*k2+k1, 2*i2+i1, col] += W2[k2*2+k1, t2] * W2[i2*2+i1, t1]
    
    for t2 in range(7):
        for t1 in range(7):
            for i2 in range(2):
                for i1 in range(2):
                    for j2 in range(2):
                        for j1 in range(2):
                            for k2 in range(2):
                                for k1 in range(2):
                                    u_val = U2[i2*2+i1][t2] * U2[j2*2+j1][t1]
                                    v_val = V2[j2*2+j1][t2] * V2[k2*2+k1][t1]
                                    w_val = W2[k2*2+k1][t2] * W2[i2*2+i1][t1]
                                    if u_val != 0:
                                        U[2*i2+i1, col] += u_val
                                    if v_val != 0:
                                        V[2*j2+j1, col] += v_val
                                    if w_val != 0:
                                        W[2*k2+k1, col] += w_val
            col += 1
    
    # Part 2: C[4,k] = sum_{j=0}^{4} A[4,j]*B[j,k] for k=0..4
    # Schoolbook: 5 terms per k, 25 total
    for j in range(5):
        for k in range(5):
            U[4*5+j, col] = 1   # A[4,j]
            V[j*5+k, col] = 1   # B[j,k]
            W[k*5+4, col] = 1   # C[4,k]
            col += 1
    
    # Part 3: C[i,4] = sum_{j=0}^{4} A[i,j]*B[j,4] for i=0..3
    # Schoolbook: 5 terms per i, 20 total
    # But note: C[4,4] is already handled in Part 2 (when k=4, j=4 gives A[4,4]*B[4,4])
    # For i=0..3, j=0..4: A[i,j]*B[j,4] -> C[i,4]
    for i in range(4):
        for j in range(5):
            U[i*5+j, col] = 1   # A[i,j]
            V[j*5+4, col] = 1   # B[j,4]
            W[4*5+i, col] = 1   # C[i,4]
            col += 1
    
    # Trim to actual rank used
    actual_rank = col
    U = [row[:actual_rank] for row in U]
    V = [row[:actual_rank] for row in V]
    W = [row[:actual_rank] for row in W]
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
