import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140

    n, m, p = 2, 5, 6
    
    # We'll build a rank-50 decomposition
    # Strategy: Use Strassen-like sharing more aggressively
    # 
    # The key idea: for the 2x5x6 problem, we can group the 5 j-indices
    # and 6 k-indices into pairs and use Strassen on each 2x2x2 sub-block,
    # but share some terms between adjacent blocks.
    #
    # Alternative: use a "tensor train" style decomposition
    #
    # Let me try: rank 50
    # Part A: j in {0,1}, k in {0,1,2,3,4,5} -> 3 Strassen blocks = 21 terms
    # Part B: j in {2,3}, k in {0,1,2,3,4,5} -> 3 Strassen blocks = 21 terms  
    # Part C: j=4, k in {0,1,2,3,4,5} -> 12 terms
    # Total: 54
    #
    # To get to 50, I need to save 4 terms. Let me try overlapping Strassen
    # blocks or using a different algebraic identity.
    #
    # Actually, let me try a different grouping:
    # Group j into {0,1,2} and {3,4}, k into pairs
    # For j in {0,1,2}: this is 2x3x6, can be decomposed as...
    # 
    # Let me just carefully construct a valid rank-50 decomposition.
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]  # (10, 50)
    V = [[0] * rank for _ in range(m * p)]  # (30, 50)
    W = [[0] * rank for _ in range(p * n)]  # (12, 50)
    
    col = 0
    
    # Define Strassen 7-term decomposition for 2x2x2:
    # Given A = [[a,b],[c,d]], B = [[e,f],[g,h]], C = [[C00,C01],[C10,C11]]
    # P1 = (a+d)(e+h) -> C00 += P1, C11 += P1
    # P2 = (c+d)e    -> C10 += P2, C11 -= P2
    # P3 = a(f-h)    -> C01 += P3, C11 += P3
    # P4 = d(g-e)    -> C00 += P4, C10 += P4
    # P5 = (a+b)h    -> C00 -= P5, C01 += P5
    # P6 = (c-a)(e+f) -> C10 += P6
    # P7 = (b-d)(g+h) -> C01 += P7
    
    def strassen_block(a_row0, a_row1, j0, j1, k0, k1, col_start):
        """Apply Strassen 2x2x2 decomposition.
        a_row0, a_row1: row indices in A (i=0,1)
        j0, j1: column indices in A / row indices in B
        k0, k1: column indices in B
        """
        c = col_start
        
        # P1 = (A[0,j0]+A[1,j1])(B[j0,k0]+B[j1,k1])
        U[0*m + j0][c] += 1
        U[1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # P2 = (A[1,j0]+A[1,j1]) * B[j0,k0]
        U[1*m + j0][c] += 1
        U[1*m + j1][c] += 1
        V[j0*p + k0][c] += 1
        W[k0*n + 1][c] += 1
        W[k1*n + 1][c] -= 1
        c += 1
        
        # P3 = A[0,j0] * (B[j0,k1]-B[j1,k1])
        U[0*m + j0][c] += 1
        V[j0*p + k1][c] += 1
        V[j1*p + k1][c] -= 1
        W[k1*n + 0][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # P4 = A[1,j1] * (B[j1,k0]-B[j0,k0])
        U[1*m + j1][c] += 1
        V[j1*p + k0][c] += 1
        V[j0*p + k0][c] -= 1
        W[k0*n + 0][c] += 1
        W[k0*n + 1][c] += 1
        c += 1
        
        # P5 = (A[0,j0]+A[0,j1]) * B[j1,k1]
        U[0*m + j0][c] += 1
        U[0*m + j1][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] -= 1
        W[k1*n + 0][c] += 1
        c += 1
        
        # P6 = (A[1,j0]-A[0,j0]) * (B[j0,k0]+B[j0,k1])
        U[1*m + j0][c] += 1
        U[0*m + j0][c] -= 1
        V[j0*p + k0][c] += 1
        V[j0*p + k1][c] += 1
        W[k1*n + 1][c] += 1
        c += 1
        
        # P7 = (A[0,j1]-A[1,j1]) * (B[j1,k0]+B[j1,k1])
        U[0*m + j1][c] += 1
        U[1*m + j1][c] -= 1
        V[j1*p + k0][c] += 1
        V[j1*p + k1][c] += 1
        W[k0*n + 0][c] += 1
        c += 1
        
        return c
    
    # Apply Strassen to 6 blocks of 2x2x2:
    # j-pairs: (0,1), (2,3)
    # k-pairs: (0,1), (2,3), (4,5)
    j_pairs = [(0, 1), (2, 3)]
    k_pairs = [(0, 1), (2, 3), (4, 5)]
    
    for j0, j1 in j_pairs:
        for k0, k1 in k_pairs:
            col = strassen_block(0, 1, j0, j1, k0, k1, col)
    
    # col should be 42 now
    # Remaining: j=4, all 6 k values: 12 terms needed, but we only have 50-42=8 slots
    # That's not enough! We need 12 terms for j=4.
    # Total would be 42+12=54, which is what we had before.
    
    # To get rank 50, I need a different approach.
    # Let me reset and try a smarter decomposition.
    
    # Reset
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    # New strategy: use overlapping Strassen blocks
    # Instead of disjoint j-pairs, use overlapping ones:
    # j-pairs: (0,1), (1,2), (2,3), (3,4) with appropriate k-pairs
    # But this gets complicated. Let me try a different approach.
    
    # Actually, let me try: for each pair of j's and each pair of k's,
    # use a 7-term Strassen, but allow the same (j,k) pair to appear
    # in multiple blocks with different coefficients.
    
    # Let me try a rank-50 construction by hand:
    # Terms 1-7: Strassen for (j=0,1; k=0,1)
    # Terms 8-14: Strassen for (j=0,1; k=2,3)
    # Terms 15-21: Strassen for (j=0,1; k=4,5)
    # Terms 22-28: Strassen for (j=2,3; k=0,1)
    # Terms 29-35: Strassen for (j=2,3; k=2,3)
    # Terms 36-42: Strassen for (j=2,3; k=4,5)
    # Terms 43-50: 8 terms for j=4 (instead of 12)
    
    # For j=4, we need: C[i,k] += A[i,4]*B[4,k] for i=0,1; k=0..5
    # That's 12 scalar products. Can we do it in 8?
    # C[0,k] = A[0,4]*B[4,k] and C[1,k] = A[1,4]*B[4,k]
    # These are independent for each (i,k) pair, so naive is 12.
    # But we can combine: for each k, C[0,k] and C[1,k] share B[4,k].
    # Use: P_k = (A[0,4]+A[1,4])*B[4,k] gives C[0,k]+=P_k, C[1,k]+=P_k
    # Then we need to subtract A[1,4]*B[4,k] from C[0,k] and A[0,4]*B[4,k] from C[1,k]
    # That's 3 terms per k, still 18 total. Worse.
    
    # Alternative: group k's. For k=(0,1):
    # C[0,0]=A[0,4]B[4,0], C[0,1]=A[0,4]B[4,1], C[1,0]=A[1,4]B[4,0], C[1,1]=A[1,4]B[4,1]
    # This is a 2x1x2 problem (or 1x2x2 transposed). Rank of 2x1x2 is 2 (naive).
    # So 3 pairs of k -> 3*2 = 6 terms? No, 2x1x2 has rank 2, so 3*2=6 terms for 6 outputs.
    # But we have 4 outputs per pair, so 2 terms per pair, 3 pairs = 6 terms.
    # Total: 42+6=48! That's better than 50!
    
    # Let me verify: for j=4, k-pair (k0,k1):
    # We need C[0,k0]=A[0,4]B[4,k0], C[0,k1]=A[0,4]B[4,k1], 
    #           C[1,k0]=A[1,4]B[4,k0], C[1,k1]=A[1,4]B[4,k1]
    # Term 1: A[0,4]*B[4,k0] -> C[0,k0]
    # Term 2: A[0,4]*B[4,k1] -> C[0,k1]
    # Term 3: A[1,4]*B[4,k0] -> C[1,k0]
    # Term 4: A[1,4]*B[4,k1] -> C[1,k1]
    # That's 4 terms per k-pair, 3 pairs = 12 terms. Back to 54.
    
    # The 2x1x2 tensor: T[i,k] = A[i]*B[k] for i in {0,1}, k in {0,1}
    # This is just an outer product of a 2-vector and a 2-vector, which has rank 2.
    # But we need to produce 4 separate outputs, each being a single product.
    # The tensor is sum_{i,k} A[i]*B[k]*E_{i,k} which has rank 4 (each E_{i,k} is distinct).
    # So naive is indeed 4 terms per k-pair.
    
    # I'm stuck at 54. Let me try a completely different approach.
    # What if I don't use Strassen at all, but find a more global decomposition?
    
    # Let me try: for each j, express A[i,j]*B[j,k] using shared terms.
    # The total tensor is sum_{i,j,k} A[i,j]*B[j,k]*E_{i,k}
    # = sum_j (sum_i A[i,j]*E_{i,:}) (sum_k B[j,k]*E_{:,k})
    # This is a sum of 5 rank-? terms, but each inner product is not rank-1.
    
    # I think 54 might be close to optimal for this approach. Let me try to verify
    # the current 54-term decomposition is correct and see if I can shave off a few.
    
    # Let me try one more idea: use a 3x3 Strassen-like decomposition for some blocks.
    # Or use the fact that 2x3x2 has known rank 5 (not 12).
    
    # For 2x3x2: A is 2x3, B is 3x2, C is 2x2
    # C[0,0]=A[0,0]B[0,0]+A[0,1]B[1,0]+A[0,2]B[2,0]
    # C[0,1]=A[0,0]B[0,1]+A[0,1]B[1,1]+A[0,2]B[2,1]
    # C[1,0]=A[1,0]B[0,0]+A[1,1]B[1,0]+A[1,2]B[2,0]
    # C[1,1]=A[1,0]B[0,1]+A[1,1]B[1,1]+A[1,2]B[2,1]
    
    # Known result: rank of 2x3x2 matrix multiplication is 5.
    # Let me use that!
    
    # Decompose 2x5x6 as:
    # j in {0,1,2}, k in {0,1,2,3,4,5}: 2x3x6 = 3 * (2x3x2) = 3*5 = 15 terms
    # j in {3,4}, k in {0,1,2,3,4,5}: 2x2x6 = 3 * (2x2x2) = 3*7 = 21 terms
    # Total: 15+21 = 36 terms!
    
    # Let me implement the 2x3x2 rank-5 decomposition.
    # Reference: the 5-term decomposition for 2x3x2 matrix multiplication.
    # 
    # After research, a known 5-term decomposition for 2x3x2:
    # Let A = [[a,b,c],[d,e,f]], B = [[g,h],[i,j],[k,l]]
    # C = [[ag+bh+ck, ah+bj+cl], [dg+ei+fk, dh+ej+fl]]
    #
    # One known decomposition (from literature):
    # P1 = (a+d+e)(g+i+k)  -> contributes to C[0,0] and C[1,1] with adjustments
    # This is getting complex. Let me look up a specific 5-term scheme.
    
    # Actually, I recall that the border rank of 2x3x2 is 5, but the exact rank might be 6.
    # Let me be safe and use rank 6 for 2x3x2, giving 3*6+21 = 39 terms.
    
    # Hmm, but I'm not 100% sure of the exact decomposition. Let me use a safe approach.
    
    # Let me just implement the 54-term Strassen decomposition correctly and verify it.
    # Then try to optimize if time permits.
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            col = strassen_block(0, 1, j0, j1, k0, k1, col)
    
    # j=4: 12 terms
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    # col should be 42+12=54, but rank=50, so we have overflow!
    # Let me fix: use rank=54
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    
    for j0, j1 in [(0, 1), (2, 3)]:
        for k0, k1 in [(0, 1), (2, 3), (4, 5)]:
            col = strassen_block(0, 1, j0, j1, k0, k1, col)
    
    for k in range(p):
        U[0*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 0][col] += 1
        col += 1
        
        U[1*m + 4][col] += 1
        V[4*p + k][col] += 1
        W[k*n + 1][col] += 1
        col += 1
    
    assert col == rank, f"col={col}, rank={rank}"
    
    return {"U": U, "V": V, "W": W}
