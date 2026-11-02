import time

DIMENSIONS = [2, 2, 13]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 13
    deadline = time.time() + 140  # 10s safety margin
    
    # First, build the target tensor T[i,j,k] = 1 if i*m+j and j*p+k produce output k*n+i
    # T is a 3D tensor of shape (n*m, m*p, p*n) = (4, 26, 26)
    # T[i*m+j, j*p+k, k*n+i] = 1
    
    # Build the full multiplication tensor as a list of non-zero entries
    # Each entry is (a_idx, b_idx, c_idx) where:
    # a_idx = i*m+j, b_idx = j*p+k, c_idx = k*n+i
    tensor_entries = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a_idx = i * m + j
                b_idx = j * p + k
                c_idx = k * n + i
                tensor_entries.append((a_idx, b_idx, c_idx))
    
    # Build Strassen-based initial decomposition
    # For 2x2 x 2x2: Strassen uses 7 multiplications
    # We'll tile the 13 columns: 6 blocks of 2 + 1 block of 1
    
    # Strassen 2x2 x 2x2 decomposition:
    # A = [[a00, a01], [a10, a11]], B = [[b00, b01], [b10, b11]]
    # C = A*B
    # m1 = (a00+a11)*(b00+b11)
    # m2 = (a10+a11)*b00
    # m3 = a00*(b01-b11)
    # m4 = a11*(b10-b00)
    # m5 = (a00+a01)*b11
    # m6 = (a10-a00)*(b00+b10)
    # m7 = (a01-a11)*(b01+b11)
    # C00 = m1+m4-m5+m7
    # C01 = m3+m5
    # C10 = m2+m4
    # C11 = m1-m2+m3+m6
    
    # For our case: A is 2x2, B is 2x13
    # We can group B into blocks: columns 0-1, 2-3, ..., 10-11, 12
    
    def strassen_2x2_block(a_indices, b_indices, c_indices, rank_offset):
        """
        Returns list of (U_row, V_col, W_row, coeff) for one 2x2x2 Strassen block.
        a_indices: [a00, a01, a10, a11] - indices into U
        b_indices: [b00, b01, b10, b11] - indices into V (for 2 columns)
        c_indices: [c00, c01, c10, c11] - indices into W (for 2 columns)
        """
        a00, a01, a10, a11 = a_indices
        b00, b01, b10, b11 = b_indices
        c00, c01, c10, c11 = c_indices
        
        terms = []
        # m1 = (a00+a11)*(b00+b11)
        terms.append((a00, b00, c00, 1))
        terms.append((a11, b00, c00, 1))
        terms.append((a00, b11, c00, 1))
        terms.append((a11, b11, c00, 1))
        terms.append((a00, b00, c11, 1))
        terms.append((a11, b00, c11, 1))
        terms.append((a00, b11, c11, 1))
        terms.append((a11, b11, c11, 1))
        terms.append((a00, b00, c00, -1))  # Wait, this isn't right for rank decomposition
        
        # Actually, for rank decomposition, each "multiplication" is one rank-1 term
        # m1 corresponds to one rank-1 term with U vector having 1s at a00,a11 and V having 1s at b00,b11
        # But we need to express the sum as separate rank-1 terms
        
        # Let me reconsider. In Strassen, each mi is a product of two linear combinations.
        # For rank decomposition, each mi contributes one rank-1 term where:
        # U has the coefficients of the A-linear-combination
        # V has the coefficients of the B-linear-combination  
        # W has the coefficients showing which C elements get this term
        
        # m1 = (a00+a11)*(b00+b11) contributes to C00 (+1) and C11 (+1)
        # So: U has 1 at a00, 1 at a11; V has 1 at b00, 1 at b11; W has 1 at c00, 1 at c11
        # But this is NOT a valid rank-1 term in the tensor sense.
        
        # Actually, in tensor rank decomposition:
        # T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
        # Each term r is a rank-1 tensor.
        
        # For m1: the term is (a00+a11)*(b00+b11) which adds to C00 and C11
        # This means: U has 1 at a00 and 1 at a11, V has 1 at b00 and 1 at b11,
        # W has 1 at c00 and 1 at c11.
        # But wait, that would create cross terms: a00*b00*c11, a00*b11*c00, etc.
        # That's not correct.
        
        # I need to think more carefully. The Strassen algorithm computes:
        # C00 = m1 + m4 - m5 + m7
        # where m1 = (a00+a11)*(b00+b11)
        # The contribution of m1 to C00 is (a00+a11)*(b00+b11)
        # This is a bilinear form in A and B that produces a scalar added to C00.
        
        # In tensor decomposition, each rank-1 term is:
        # u * v * w where u is in R^(n*m), v in R^(m*p), w in R^(p*n)
        # The tensor entry T[a,b,c] = sum_r U[a,r]*V[b,r]*W[c,r]
        
        # For the naive decomposition, each term is like:
        # U[a00, r] = 1, V[b00, r] = 1, W[c00, r] = 1
        # This gives T[a00, b00, c00] += 1
        
        # For Strassen, m1 contributes to both C00 and C11:
        # (a00+a11)*(b00+b11) adds to C00 AND C11
        # This is NOT a single rank-1 term. It's actually 4 rank-1 terms:
        # a00*b00 -> C00, a00*b11 -> C00, a11*b00 -> C00, a11*b11 -> C00
        # a00*b00 -> C11, a00*b11 -> C11, a11*b00 -> C11, a11*b11 -> C11
        # But that's 8 terms, not 1.
        
        # I think I'm confusing myself. Let me restart with a cleaner approach.
        
        return None  # This function is broken, let me redo
    
    # Let me use a simpler approach: start with the naive rank-52 decomposition
    # and try to reduce it via randomization.
    
    # Naive decomposition (rank 52)
    rank = 4 * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for k in range(p):
        base = 4 * k
        U[0][base] = 1
        V[k][base] = 1
        W[2*k][base] = 1
        U[1][base + 1] = 1
        V[13 + k][base + 1] = 1
        W[2*k][base + 1] = 1
        U[2][base + 2] = 1
        V[k][base + 2] = 1
        W[2*k + 1][base + 2] = 1
        U[3][base + 3] = 1
        V[13 + k][base + 3] = 1
        W[2*k + 1][base + 3] = 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to reduce rank by merging terms
    # Two rank-1 terms can be merged if they share structure
    # For now, let's try a simple greedy approach
    
    # Check if decomposition is valid
    def is_valid(U, V, W):
        """Check if U, V, W correctly decompose the tensor."""
        rank = len(U[0])
        # Build the tensor from decomposition
        tensor = {}
        for r in range(rank):
            u = U[:, r] if isinstance(U[0], list) else [row[r] for row in U]
            v = V[:, r] if isinstance(V[0], list) else [row[r] for row in V]
            w = W[:, r] if isinstance(W[0], list) else [row[r] for row in W]
            for i in range(n * m):
                if u[i] != 0:
                    for j in range(m * p):
                        if v[j] != 0:
                            for k in range(p * n):
                                if w[k] != 0:
                                    key = (i, j, k)
                                    tensor[key] = tensor.get(key, 0) + u[i] * v[j] * w[k]
        
        # Check against expected
        for (a_idx, b_idx, c_idx) in tensor_entries:
            if tensor.get((a_idx, b_idx, c_idx), 0) != 1:
                return False
        # Check no spurious entries
        for (a_idx, b_idx, c_idx), val in tensor.items():
            if val != 1 and val != 0:
                return False
        return True
    
    # Verify initial
    assert is_valid(U, V, W), "Initial decomposition is invalid"
    
    # Try local search: perturb and see if we can reduce rank
    import random
    random.seed(42)
    
    # Strategy: try to find linear dependencies among the rank-1 terms
    # If we can express one term as a linear combination of others, we can remove it
    
    # For now, return the naive decomposition if no time for search
    if time.time() > deadline:
        return {"U": U, "V": V, "W": W}
    
    # Try a simple optimization: look for patterns where we can share terms
    # For example, if two terms have the same U and V patterns, we might merge
    
    # Actually, let me try a different structural approach:
    # Use the fact that 13 = 2*6 + 1, and apply a mixed decomposition
    
    # For the first 12 columns (6 blocks of 2), use Strassen-like decomposition
    # For the last column, use naive (4 terms)
    
    # But Strassen for 2x2x2x2x2 (i.e., 2x2 times 2x2 producing 2x2) has rank 7
    # However, in our tensor decomposition framework, each "multiplication" in Strassen
    # corresponds to a rank-1 term in the tensor.
    
    # Let me carefully construct the Strassen decomposition for 2x2 x 2x2:
    # Input: A (2x2), B (2x2)
    # Output: C = A*B (2x2)
    # 
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # B indices: 0=b00, 1=b01, 2=b10, 3=b11
    # C indices: 0=c00, 1=c01, 2=c10, 3=c11 (using k*n+i convention for 2x2: c[0]=c00, c[1]=c01, c[2]=c10, c[3]=c11)
    # Wait, the convention is W[k*n+i, r]. For 2x2 output, k is column, i is row.
    # So C[0] = C[0,0], C[1] = C[0,1], C[2] = C[1,0], C[3] = C[1,1]
    # That means: row-major with columns as the outer index? No:
    # W[k*n+i, r] where n=2, so index = k*2+i
    # k=0,i=0: index 0 -> C[0,0]
    # k=0,i=1: index 1 -> C[1,0] (row 1, col 0)
    # k=1,i=0: index 2 -> C[0,1] (row 0, col 1)
    # k=1,i=1: index 3 -> C[1,1]
    
    # So the output indexing is: index = k*n + i, where i is row, k is column
    # This is column-major!
    
    # Let me re-verify with the original code:
    # W[2*k][base] = 1 for term producing C[0,k]
    # W[2*k+1][base+2] = 1 for term producing C[1,k]
    # So W index 2*k = C[0,k] (row 0, col k), W index 2*k+1 = C[1,k] (row 1, col k)
    # This means: index = k*n + i where i is row, k is column. Yes, column-major.
    
    # For 2x2 output (n=2, p=2):
    # C[0,0]: index 0*2+0 = 0
    # C[1,0]: index 0*2+1 = 1
    # C[0,1]: index 1*2+0 = 2
    # C[1,1]: index 1*2+1 = 3
    
    # Now Strassen for 2x2 x 2x2:
    # m1 = (a00+a11)*(b00+b11) -> contributes to C[0,0] and C[1,1]
    # m2 = (a10+a11)*b00 -> contributes to C[1,0] and C[1,1]
    # m3 = a00*(b01-b11) -> contributes to C[0,1] and C[1,1]
    # m4 = a11*(b10-b00) -> contributes to C[1,0] and C[0,0]
    # m5 = (a00+a01)*b11 -> contributes to C[0,0] and C[0,1]
    # m6 = (a10-a00)*(b00+b10) -> contributes to C[1,0]
    # m7 = (a01-a11)*(b01+b11) -> contributes to C[0,1]
    
    # Each mi is a product of a linear combination of A entries and a linear combination of B entries.
    # In tensor rank decomposition, each mi corresponds to ONE rank-1 term where:
    # U has the coefficients of the A-linear-combination
    # V has the coefficients of the B-linear-combination
    # W has 1s at the C positions this term contributes to
    
    # BUT: this only works if the contribution to each C element is the same scalar multiple.
    # For m1: contributes (a00+a11)*(b00+b11) to BOTH C[0,0] and C[1,1].
    # So U = [1, 0, 0, 1] (coefficients for a00, a01, a10, a11)
    # V = [1, 0, 0, 1] (coefficients for b00, b01, b10, b11)
    # W = [1, 0, 0, 1] (coefficients for C[0,0], C[1,0], C[0,1], C[1,1])
    # This gives: T[a,b,c] += U[a]*V[b]*W[c]
    # For a=0,b=0,c=0: 1*1*1 = 1 (contributes to C[0,0])
    # For a=0,b=0,c=3: 1*1*1 = 1 (contributes to C[1,1])
    # For a=3,b=3,c=0: 1*1*1 = 1 (contributes to C[0,0])
    # etc.
    # This is correct! The cross terms (like a00*b11*c00) are also generated, but they
    # are part of the bilinear form (a00+a11)*(b00+b11) which expands to
    # a00*b00 + a00*b11 + a11*b00 + a11*b11, all contributing to C[0,0] and C[1,1].
    
    # So the Strassen decomposition for 2x2 x 2x2 has rank 7!
    
    # Let me verify: for the full 2x2 x 2x2 multiplication tensor, the rank is 7 (Strassen).
    # Our problem is 2x2 x 2x13. We can tile B into 6 blocks of 2 columns + 1 block of 1 column.
    # For each 2-column block, we use 7 rank-1 terms (Strassen).
    # For the 1-column block, we use 4 rank-1 terms (naive).
    # Total: 6*7 + 4 = 42 + 4 = 46
    
    # Let's construct this!
    
    rank_strassen = 6 * 7 + 4  # 46
    U_s = [[0] * rank_strassen for _ in range(n * m)]  # 4 x 46
    V_s = [[0] * rank_strassen for _ in range(m * p)]  # 26 x 46
    W_s = [[0] * rank_strassen for _ in range(p * n)]  # 26 x 46
    
    # A indices: 0=a00, 1=a01, 2=a10, 3=a11
    # For each block of 2 columns (k, k+1):
    # B indices: 13*0+k, 13*0+(k+1), 13*1+k, 13*1+(k+1)
    # C indices for output: 
    #   C[0,k]: 2*k+0, C[1,k]: 2*k+1, C[0,k+1]: 2*(k+1)+0, C[1,k+1]: 2*(k+1)+1
    
    def add_strassen_block(U, V, W, k, rank_offset):
        """Add Strassen terms for columns k and k+1."""
        # B indices for this block
        b00 = 0 * p + k      # B[0,k]
        b01 = 0 * p + (k+1)  # B[0,k+1]
        b10 = 1 * p + k      # B[1,k]
        b11 = 1 * p + (k+1)  # B[1,k+1]
        
        # C indices for this block
        c00 = k * n + 0      # C[0,k]
        c10 = k * n + 1      # C[1,k]
        c01 = (k+1) * n + 0  # C[0,k+1]
        c11 = (k+1) * n + 1  # C[1,k+1]
        
        # A indices
        a00 = 0  # a00
        a01 = 1  # a01
        a10 = 2  # a10
        a11 = 3  # a11
        
        # m1 = (a00+a11)*(b00+b11) -> C[0,0] and C[1,1]
        r = rank_offset + 0
        U[a00][r] += 1
        U[a11][r] += 1
        V[b00][r] += 1
        V[b11][r] += 1
        W[c00][r] += 1
        W[c11][r] += 1
        
        # m2 = (a10+a11)*b00 -> C[1,0] and C[1,1]
        r = rank_offset + 1
        U[a10][r] += 1
        U[a11][r] += 1
        V[b00][r] += 1
        W[c10][r] += 1
        W[c11][r] += 1
        
        # m3 = a00*(b01-b11) -> C[0,1] and C[1,1]
        r = rank_offset + 2
        U[a00][r] += 1
        V[b01][r] += 1
        V[b11][r] += -1
        W[c01][r] += 1
        W[c11][r] += 1
        
        # m4 = a11*(b10-b00) -> C[1,0] and C[0,0]
        r = rank_offset + 3
        U[a11][r] += 1
        V[b10][r] += 1
        V[b00][r] += -1
        W[c10][r] += 1
        W[c00][r] += 1
        
        # m5 = (a00+a01)*b11 -> C[0,0] and C[0,1]
        r = rank_offset + 4
        U[a00][r] += 1
        U[a01][r] += 1
        V[b11][r] += 1
        W[c00][r] += 1
        W[c01][r] += 1
        
        # m6 = (a10-a00)*(b00+b10) -> C[1,0]
        r = rank_offset + 5
        U[a10][r] += 1
        U[a00][r] += -1
        V[b00][r] += 1
        V[b10][r] += 1
        W[c10][r] += 1
        
        # m7 = (a01-a11)*(b01+b11) -> C[0,1]
        r = rank_offset + 6
        U[a01][r] += 1
        U[a11][r] += -1
        V[b01][r] += 1
        V[b11][r] += 1
        W[c01][r] += 1
    
    # Add 6 Strassen blocks for columns (0,1), (2,3), (4,5), (6,7), (8,9), (10,11)
    for block in range(6):
        k = 2 * block
        add_strassen_block(U_s, V_s, W_s, k, block * 7)
    
    # Add naive decomposition for column 12
    base = 6 * 7  # 42
    # C[0,12] = A[0,0]*B[0,12] + A[0,1]*B[1,12]
    U_s[0][base] = 1
    V_s[0 * p + 12][base] = 1
    W_s[12 * n + 0][base] = 1
    
    U_s[1][base + 1] = 1
    V_s[1 * p + 12][base + 1] = 1
    W_s[12 * n + 0][base + 1] = 1
    
    # C[1,12] = A[1,0]*B[0,12] + A[1,1]*B[1,12]
    U_s[2][base + 2] = 1
    V_s[0 * p + 12][base + 2] = 1
    W_s[12 * n + 1][base + 2] = 1
    
    U_s[3][base + 3] = 1
    V_s[1 * p + 12][base + 3] = 1
    W_s[12 * n + 1][base + 3] = 1
    
    # Verify Strassen decomposition
    if is_valid(U_s, V_s, W_s):
        best_rank = rank_strassen
        best_U, best_V, best_W = U_s, V_s, W_s
    else:
        # Fall back to naive
        best_rank = rank
        best_U, best_V, best_W = U, V, W
    
    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
