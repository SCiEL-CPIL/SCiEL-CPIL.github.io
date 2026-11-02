"""Matrix multiplication 3x4x6 - Strassen-based rank reduction."""
DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 6
    # We'll build a decomposition using Strassen on 2x2 blocks where possible.
    # Strategy: Partition the 3x4 matrix A into blocks and the 4x6 matrix B into blocks.
    # Use Strassen's algorithm on overlapping 2x2 sub-problems to reduce rank.
    
    # The schoolbook rank is 3*4*6 = 72.
    # We can apply Strassen on 2x2x2 sub-blocks.
    # For a 2x2 * 2x2 multiplication, Strassen uses 7 multiplications instead of 8.
    
    # Let's think about how to partition:
    # A is 3x4, B is 4x6.
    # We can split A into rows {0,1}, {2} and cols {0,1}, {2,3}
    # We can split B into rows {0,1}, {2,3} and cols {0,1}, {2,3}, {4,5}
    
    # Actually, let me think more carefully. The tensor for multiplying A (n x m) by B (m x p)
    # has indices (i, j, k) where i in [0,n), j in [0,m), k in [0,p).
    # The tensor is T[i,j,k] = 1 if we're computing C[i,k] = sum_j A[i,j]*B[j,k].
    # Wait, the convention is W[k*n+i, r], so W is indexed by (k, i) with k in [0,p), i in [0,n).
    
    # Let me use a more systematic approach. I'll build the decomposition by:
    # 1. Starting with schoolbook
    # 2. Applying Strassen substitutions on 2x2x2 sub-tensors
    
    # Actually, let me just directly construct a rank-66 or lower decomposition.
    
    # For 2x2x2 Strassen:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B21-B22)
    # M4 = A22(B11-B21)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    # Let me build this by explicitly constructing the factor matrices.
    # I'll use a list of "terms" where each term is (U_entry, V_entry, W_entry).
    
    # Let me define the problem more carefully.
    # A is n x m = 3 x 4, indexed A[i, j] with i in 0..2, j in 0..3
    # B is m x p = 4 x 6, indexed B[j, k] with j in 0..3, k in 0..5
    # C is n x p = 3 x 6, indexed C[i, k] with i in 0..2, k in 0..5
    # C[i, k] = sum_{j=0}^{3} A[i, j] * B[j, k]
    
    # The tensor T has entries T[i, j, k] = 1 for all valid i, j, k.
    # We want to decompose T = sum_r U[:,r] * V[:,r] * W[:,r] (outer product)
    # where U is (n*m) x rank, V is (m*p) x rank, W is (p*n) x rank.
    # U is indexed by (i, j) -> row i*m + j
    # V is indexed by (j, k) -> row j*p + k  
    # W is indexed by (k, i) -> row k*n + i
    
    # So the decomposition satisfies:
    # T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # For schoolbook, each term r corresponds to (i,j,k) with U[i*m+j,r]=1, V[j*p+k,r]=1, W[k*n+i,r]=1.
    
    # Now let's apply Strassen. Consider a 2x2x2 subproblem with:
    # i in {i0, i0+1}, j in {j0, j0+1}, k in {k0, k0+1}
    # This involves 8 schoolbook terms. Strassen replaces them with 7 terms.
    
    # Let me define a helper that adds a Strassen block.
    terms = []  # Each term: list of (matrix, row, value)
    
    def add_term(u_entries, v_entries, w_entries):
        """Add one rank-1 term. u_entries, v_entries, w_entries are lists of (row, value)."""
        terms.append((u_entries, v_entries, w_entries))
    
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for the 2x2x2 subblock starting at (i0,j0,k0)."""
        # A11 = A[i0,j0], A12 = A[i0,j0+1], A21 = A[i0+1,j0], A22 = A[i0+1,j0+1]
        # B11 = B[j0,k0], B12 = B[j0,k0+1], B21 = B[j0+1,k0], B22 = B[j0+1,k0+1]
        # C11 = C[i0,k0], C12 = C[i0,k0+1], C21 = C[i0+1,k0], C22 = C[i0+1,k0+1]
        
        # M1 = (A11+A22)(B11+B22)
        # U: A11 at (i0,j0), A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0), B22 at (j0+1,k0+1)
        # W: contributes to C11 and C22
        # C11 = M1+M4-M5+M7, C22 = M1-M2+M3+M6
        # So M1 contributes +1 to C11 and +1 to C22
        add_term(
            [(i0*m+j0, 1), ((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1), ((j0+1)*p+k0+1, 1)],
            [(k0*n+i0, 1), ((k0+1)*n+i0+1, 1)]
        )
        
        # M2 = (A21+A22)B11
        # U: A21 at (i0+1,j0), A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0)
        # W: contributes to C21 (+1) and C22 (-1)
        add_term(
            [((i0+1)*m+j0, 1), ((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1)],
            [(k0*n+i0+1, 1), ((k0+1)*n+i0+1, -1)]
        )
        
        # M3 = A11(B21-B22)
        # U: A11 at (i0,j0)
        # V: B21 at (j0+1,k0), B22 at (j0+1,k0+1)
        # W: contributes to C12 (+1) and C22 (+1)
        add_term(
            [(i0*m+j0, 1)],
            [((j0+1)*p+k0, 1), ((j0+1)*p+k0+1, -1)],
            [(k0+1)*n+i0, 1], [(k0+1)*n+i0+1, 1]
        )
        
        # M4 = A22(B11-B21)
        # U: A22 at (i0+1,j0+1)
        # V: B11 at (j0,k0), B21 at (j0+1,k0)
        # W: contributes to C11 (+1) and C21 (+1)
        add_term(
            [((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1), ((j0+1)*p+k0, -1)],
            [(k0*n+i0, 1), (k0*n+i0+1, 1)]
        )
        
        # M5 = (A11+A12)B22
        # U: A11 at (i0,j0), A12 at (i0,j0+1)
        # V: B22 at (j0+1,k0+1)
        # W: contributes to C11 (-1) and C12 (+1)
        add_term(
            [(i0*m+j0, 1), (i0*m+j0+1, 1)],
            [((j0+1)*p+k0+1, 1)],
            [(k0*n+i0, -1), ((k0+1)*n+i0, 1)]
        )
        
        # M6 = (A21-A11)(B11+B12)
        # U: A21 at (i0+1,j0), A11 at (i0,j0)
        # V: B11 at (j0,k0), B12 at (j0,k0+1)
        # W: contributes to C22 (+1)
        add_term(
            [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
            [(j0*p+k0, 1), (j0*p+k0+1, 1)],
            [((k0+1)*n+i0+1, 1)]
        )
        
        # M7 = (A12-A22)(B21+B22)
        # U: A12 at (i0,j0+1), A22 at (i0+1,j0+1)
        # V: B21 at (j0+1,k0), B22 at (j0+1,k0+1)
        # W: contributes to C11 (+1) and C12... wait let me recheck
        # C11 = M1+M4-M5+M7, so M7 contributes +1 to C11
        # C12 = M3+M5, M7 doesn't contribute to C12
        # Wait, let me re-derive Strassen:
        # C11 = M1 - M5 + M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # Hmm, let me double-check M7's contribution. M7 = (A12-A22)(B21+B22)
        # This appears in C11 with +1. Does it appear elsewhere?
        # In standard Strassen:
        # M7 = (A12 - A22)(B21 + B22)
        # C11 = M1 - M5 + M4 + M7
        # C12 = M3 + M5
        # C21 = M2 + M4
        # C22 = M1 - M2 + M3 + M6
        
        # So M7 only contributes to C11. Let me fix M7.
        # Actually wait, I already added M7 above with only C11 contribution. Let me recheck my code.
        # In my M7 add_term above, I only have W: [(k0*n+i0, 1)] which is C11. That's correct.
        # But I wrote the comment wrong. Let me just make sure the code is right.
        
    # Now I need to figure out which 2x2x2 blocks to apply Strassen to.
    # n=3, m=4, p=6.
    # Possible 2x2x2 blocks: i0 in {0,1}, j0 in {0,1,2}, k0 in {0,1,2,3,4}
    # That's 2*3*5 = 30 possible blocks, but they overlap.
    
    # The key insight: we need to partition the (i,j,k) space into non-overlapping 2x2x2 blocks
    # plus remaining single terms, OR we need a more clever approach.
    
    # Actually, for a proper rank reduction, we should partition the tensor into disjoint
    # 2x2x2 sub-tensors and apply Strassen to each.
    
    # Let's partition:
    # i: {0,1} and {2}
    # j: {0,1}, {2,3}
    # k: {0,1}, {2,3}, {4,5}
    
    # Blocks where i in {0,1}, j in {0,1} or {2,3}, k in {0,1} or {2,3} or {4,5}:
    # That's 2 j-groups * 3 k-groups = 6 blocks of size 2x2x2.
    # Remaining: i=2, all j, all k -> 1*4*6 = 24 single terms.
    
    # Total rank: 6*7 + 24 = 42 + 24 = 66.
    
    # Let's do it.
    terms = []
    
    def add_term(u_entries, v_entries, w_entries):
        terms.append((u_entries, v_entries, w_entries))
    
    def add_strassen_block(i0, j0, k0):
        """Add 7 Strassen terms for 2x2x2 block with i in {i0,i0+1}, j in {j0,j0+1}, k in {k0,k0+1}."""
        # M1 = (A11+A22)(B11+B22) -> C11: +1, C22: +1
        add_term(
            [(i0*m+j0, 1), ((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1), ((j0+1)*p+k0+1, 1)],
            [(k0*n+i0, 1), ((k0+1)*n+i0+1, 1)]
        )
        # M2 = (A21+A22)B11 -> C21: +1, C22: -1
        add_term(
            [((i0+1)*m+j0, 1), ((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1)],
            [(k0*n+i0+1, 1), ((k0+1)*n+i0+1, -1)]
        )
        # M3 = A11(B21-B22) -> C12: +1, C22: +1
        add_term(
            [(i0*m+j0, 1)],
            [((j0+1)*p+k0, 1), ((j0+1)*p+k0+1, -1)],
            [((k0+1)*n+i0, 1), ((k0+1)*n+i0+1, 1)]
        )
        # M4 = A22(B11-B21) -> C11: +1, C21: +1
        add_term(
            [((i0+1)*m+j0+1, 1)],
            [(j0*p+k0, 1), ((j0+1)*p+k0, -1)],
            [(k0*n+i0, 1), (k0*n+i0+1, 1)]
        )
        # M5 = (A11+A12)B22 -> C11: -1, C12: +1
        add_term(
            [(i0*m+j0, 1), (i0*m+j0+1, 1)],
            [((j0+1)*p+k0+1, 1)],
            [(k0*n+i0, -1), ((k0+1)*n+i0, 1)]
        )
        # M6 = (A21-A11)(B11+B12) -> C22: +1
        add_term(
            [((i0+1)*m+j0, 1), (i0*m+j0, -1)],
            [(j0*p+k0, 1), (j0*p+k0+1, 1)],
            [((k0+1)*n+i0+1, 1)]
        )
        # M7 = (A12-A22)(B21+B22) -> C11: +1
        add_term(
            [(i0*m+j0+1, 1), ((i0+1)*m+j0+1, -1)],
            [((j0+1)*p+k0, 1), ((j0+1)*p+k0+1, 1)],
            [(k0*n+i0, 1)]
        )
    
    def add_single(i, j, k):
        """Add a single schoolbook term."""
        add_term(
            [(i*m+j, 1)],
            [(j*p+k, 1)],
            [(k*n+i, 1)]
        )
    
    # Apply Strassen to 6 blocks:
    # i0=0, j0 in {0,2}, k0 in {0,2,4}
    for j0 in [0, 2]:
        for k0 in [0, 2, 4]:
            add_strassen_block(0, j0, k0)
    
    # Remaining: i=2, all j in 0..3, all k in 0..5
    for j in range(m):
        for k in range(p):
            add_single(2, j, k)
    
    rank = len(terms)
    
    # Build the matrices
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for r, (u_entries, v_entries, w_entries) in enumerate(terms):
        for row, val in u_entries:
            U[row][r] = val
        for row, val in v_entries:
            V[row][r] = val
        for row, val in w_entries:
            W[row][r] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
