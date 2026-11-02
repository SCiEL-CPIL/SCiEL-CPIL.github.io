"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
import numpy as np

def _strassen_2x2():
    """Return U, V, W for 2x2 * 2x2 with Strassen rank 7.
    Indexing: U[i*m+j, c], V[j*p+k, c], W[k*n+i, c]
    For 2x2*2x2: n=m=p=2, so U is 4x7, V is 4x7, W is 4x7.
    W index: k*n+i where k in {0,1}, i in {0,1}
    U index: i*m+j where i in {0,1}, j in {0,1}
    V index: j*p+k where j in {0,1}, k in {0,1}
    """
    n = m = p = 2
    rank = 7
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    
    # Strassen's 7 products for C = A*B where A, B are 2x2
    # A = [[a11, a12], [a21, a22]], B = [[b11, b12], [b21, b22]]
    # C = [[c11, c12], [c21, c22]]
    
    # P1 = (a11+a22)(b11+b22)  -> contributes to c11, c22
    # P2 = (a21+a22)b11       -> contributes to c21, c22
    # P3 = a11(b12-b22)       -> contributes to c12, c22... wait let me be careful
    # P4 = a22(b21-b11)       -> 
    # P5 = (a11+a12)b22       -> 
    # P6 = (a21-a11)(b11+b12) -> 
    # P7 = (a12-a22)(b21+b22) ->
    
    # Standard Strassen:
    # C11 = P1 - P4 + P5 - P7
    # C12 = P3 + P7
    # C21 = P2 + P6
    # C22 = P1 - P3 + P4 + P6
    
    # So:
    # P1: (a11+a22) * (b11+b22)
    # P2: (a21+a22) * b11
    # P3: a11 * (b12-b22)
    # P4: a22 * (b21-b11)
    # P5: (a11+a12) * b22
    # P6: (a21-a11) * (b11+b12)
    # P7: (a12-a22) * (b21+b22)
    
    # For each product c, we need U row indices (a indices), V row indices (b indices), W row indices (c indices)
    # U[i*m+j, c] = coefficient of a_{ij} in the left factor
    # V[j*p+k, c] = coefficient of b_{jk} in the right factor
    # W[k*n+i, c] = coefficient of c_{ki} in the result
    
    # Product 1: (a11+a22)(b11+b22) -> c11 += 1, c22 += 1
    # a indices: (0,0) and (1,1) -> U[0*2+0]=U[0], U[1*2+1]=U[3]
    # b indices: (0,0) and (0,1), (1,0) and (1,1) -> V[0*2+0]=V[0], V[0*2+1]=V[1], V[1*2+0]=V[2], V[1*2+1]=V[3]
    # Wait, b11 is b[0][0], b12 is b[0][1], b21 is b[1][0], b22 is b[1][1]
    # b index j*p+k: j=0,k=0 -> 0; j=0,k=1 -> 1; j=1,k=0 -> 2; j=1,k=1 -> 3
    # For P1: left = a11+a22, right = b11+b22
    # But this is a product of sums, so in tensor decomposition we need to handle it as:
    # Actually in tensor decomposition, each rank-1 term is a single product of linear forms.
    # So P1 contributes to rank-1 terms:
    # a11*b11, a11*b22, a22*b11, a22*b22
    # Each contributes to c11 and c22.
    
    # This is getting complex. Let me use the standard Strassen decomposition directly.
    
    # Let me define the 7 rank-1 terms explicitly.
    # Term 1: a11*b11 -> c11
    # But Strassen groups them differently. Let me just enumerate all 7 terms with their contributions.
    
    # Actually, let me use a cleaner approach. The Strassen decomposition gives:
    # c11 = P1 - P4 + P5 - P7
    # c12 = P3 + P7
    # c21 = P2 + P6
    # c22 = P1 - P3 + P4 + P6
    # where:
    # P1 = (a11+a22)(b11+b22)
    # P2 = (a21+a22)b11
    # P3 = a11(b12-b22)
    # P4 = a22(b21-b11)
    # P5 = (a11+a12)b22
    # P6 = (a21-a11)(b11+b12)
    # P7 = (a12-a22)(b21+b22)
    
    # In tensor rank decomposition, each Pi is a rank-1 term (product of two linear forms).
    # The coefficient of each a_ij*b_kl in each Pi determines the entries.
    
    # Let me expand each Pi:
    # P1 = a11*b11 + a11*b22 + a22*b11 + a22*b22
    #     contributes: c11 += 1, c22 += 1 for each of these 4 sub-terms
    # But in the tensor, each rank-1 term is a single a_ij * b_kl -> c_ki contribution.
    
    # I think the cleanest way is to directly construct the 7 vectors.
    
    # Let me index: a_00, a_01, a_10, a_11 (i,j)
    # b_00, b_01, b_10, b_11 (j,k)
    # c_00, c_01, c_10, c_11 (k,i) -> c index = k*n+i = k*2+i
    
    # For each of the 7 Strassen products, I'll write out the left vector (over a's), right vector (over b's), and result vector (over c's).
    
    # P1: (a00+a11)*(b00+b11) ... wait, b11 means b[1][1]
    # Actually let me use 1-based for clarity: a11=a[0][0], a12=a[0][1], a21=a[1][0], a22=a[1][1]
    # b11=b[0][0], b12=b[0][1], b21=b[1][0], b22=b[1][1]
    # c11=c[0][0], c12=c[0][1], c21=c[1][0], c22=c[1][1]
    
    # P1 = (a11+a22)(b11+b22)
    # = a11*b11 + a11*b22 + a22*b11 + a22*b22
    # Each term contributes to c11 and c22.
    # So in the tensor:
    #   a11*b11 -> c11: 1, c22: 1
    #   a11*b22 -> c11: 1, c22: 1
    #   a22*b11 -> c11: 1, c22: 1
    #   a22*b22 -> c11: 1, c22: 1
    # But these are 4 separate rank-1 terms if we don't combine. However, Strassen combines them into one "product" but in tensor rank, we need to count the actual rank.
    
    # I think I'm overcomplicating this. Let me just use the known fact that Strassen has rank 7 and construct the matrices directly.
    
    # U rows: a11=0, a12=1, a21=2, a22=3  (i*m+j: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    # V rows: b11=0, b12=1, b21=2, b22=3  (j*p+k: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    # W rows: c11=0, c12=1, c21=2, c22=3  (k*n+i: 0*2+0=0, 0*2+1=1, 1*2+0=2, 1*2+1=3)
    
    # The 7 terms of Strassen:
    # T1: (a11+a22)*(b11+b22) -> this is not a single rank-1 term.
    
    # OK I need to think about this differently. The Strassen algorithm computes 7 products, each of which is a product of two linear combinations. In the tensor decomposition sense, each such product corresponds to a rank-1 term where the left factor is a linear form in the a's and the right factor is a linear form in the b's, and the result is a linear form in the c's.
    
    # So for P1 = (a11+a22)(b11+b22):
    # Left linear form: a11 + a22 -> U vector: [1, 0, 0, 1]
    # Right linear form: b11 + b22 -> V vector: [1, 0, 0, 1]
    # Result: this product contributes to c11 and c22.
    # But in the final result, c11 = P1 - P4 + P5 - P7 and c22 = P1 - P3 + P4 + P6
    # So P1 contributes +1 to both c11 and c22.
    # W vector for P1: [1, 0, 0, 1] (c11=1, c12=0, c21=0, c22=1)
    
    # P2 = (a21+a22)*b11
    # Left: a21+a22 -> U: [0, 0, 1, 1]
    # Right: b11 -> V: [1, 0, 0, 0]
    # Contributes to c21 and c22: c21 += 1, c22 += 1
    # W: [0, 0, 1, 1]
    
    # P3 = a11*(b12-b22)
    # Left: a11 -> U: [1, 0, 0, 0]
    # Right: b12-b22 -> V: [0, 1, 0, -1]
    # Contributes to c12 and c22: c12 += 1, c22 -= 1
    # W: [0, 1, 0, -1]
    
    # P4 = a22*(b21-b11)
    # Left: a22 -> U: [0, 0, 0, 1]
    # Right: b21-b11 -> V: [-1, 0, 1, 0]
    # Contributes to c11 and c22: c11 -= 1, c22 += 1
    # W: [-1, 0, 0, 1]
    
    # P5 = (a11+a12)*b22
    # Left: a11+a12 -> U: [1, 1, 0, 0]
    # Right: b22 -> V: [0, 0, 0, 1]
    # Contributes to c11 and c22: c11 += 1, c22 += 1
    # W: [1, 0, 0, 1]
    
    # P6 = (a21-a11)*(b11+b12)
    # Left: a21-a11 -> U: [-1, 0, 1, 0]
    # Right: b11+b12 -> V: [1, 1, 0, 0]
    # Contributes to c21 and c22: c21 += 1, c22 += 1
    # W: [0, 0, 1, 1]
    
    # P7 = (a12-a22)*(b21+b22)
    # Left: a12-a22 -> U: [0, 1, 0, -1]
    # Right: b21+b22 -> V: [0, 0, 1, 1]
    # Contributes to c12 and c22: c12 += 1, c22 -= 1
    # W: [0, 1, 0, -1]
    
    U = np.zeros((4, 7))
    V = np.zeros((4, 7))
    W = np.zeros((4, 7))
    
    # Column 0: P1
    U[:, 0] = [1, 0, 0, 1]
    V[:, 0] = [1, 0, 0, 1]
    W[:, 0] = [1, 0, 0, 1]
    
    # Column 1: P2
    U[:, 1] = [0, 0, 1, 1]
    V[:, 1] = [1, 0, 0, 0]
    W[:, 1] = [0, 0, 1, 1]
    
    # Column 2: P3
    U[:, 2] = [1, 0, 0, 0]
    V[:, 2] = [0, 1, 0, -1]
    W[:, 2] = [0, 1, 0, -1]
    
    # Column 3: P4
    U[:, 3] = [0, 0, 0, 1]
    V[:, 3] = [-1, 0, 1, 0]
    W[:, 3] = [-1, 0, 0, 1]
    
    # Column 4: P5
    U[:, 4] = [1, 1, 0, 0]
    V[:, 4] = [0, 0, 0, 1]
    W[:, 4] = [1, 0, 0, 1]
    
    # Column 5: P6
    U[:, 5] = [-1, 0, 1, 0]
    V[:, 5] = [1, 1, 0, 0]
    W[:, 5] = [0, 0, 1, 1]
    
    # Column 6: P7
    U[:, 6] = [0, 1, 0, -1]
    V[:, 6] = [0, 0, 1, 1]
    W[:, 6] = [0, 1, 0, -1]
    
    return U, V, W

def _identity_block(n, m, p):
    """Return trivial rank n*m*p decomposition for n x m * m x p."""
    rank = n * m * p
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m + j, col] = 1
                V[j*p + k, col] = 1
                W[k*n + i, col] = 1
                col += 1
    return U, V, W

def _combine_decompositions(U1, V1, W1, U2, V2, W2):
    """Combine two block-diagonal decompositions.
    Block 1 is for dimensions (n1, m1, p1), Block 2 for (n2, m2, p2).
    The combined matrix multiplication is for (n1+n2, m1+m2, p2) ... 
    Actually this is for when we split the intermediate dimension or the output.
    
    Let me think about this more carefully.
    
    If we have A (n1 x m1) * B1 (m1 x p) and A (n2 x m2) * B2 (m2 x p), 
    this isn't quite right for a single matrix multiplication.
    
    Actually, for the Kronecker product approach:
    If we can decompose n x m * m x p as a block structure...
    
    For 3x5*5x6:
    We can think of 3 = 1+2, 5 = 1+4 or 2+3, 6 = 2+4 or 3+3, etc.
    
    The Kronecker product approach: if we have decompositions for (n1,m1,p1) with rank r1 and (n2,m2,p2) with rank r2,
    then we can get a decomposition for (n1*n2, m1*m2, p1*p2) with rank r1*r2.
    
    But 3x5*5x6 doesn't factorize nicely as a Kronecker product of smaller standard sizes.
    
    3 = 3, 5 = 5, 6 = 6. No obvious Kronecker factorization.
    
    Alternative: use the fact that we can do block matrix multiplication.
    If we partition the matrices into blocks, we can use known decompositions for the block sizes.
    
    For 3x5 * 5x6:
    Partition A into rows: 1+2, so A = [A1; A2] where A1 is 1x5, A2 is 2x5.
    Partition B into columns: 2+4, so B = [B1 B2] where B1 is 5x2, B2 is 5x4.
    
    Then C = A*B = [A1*B1, A1*B2; A2*B1, A2*B2]
    
    C11 = A1*B1: 1x5 * 5x2 -> rank 10 (schoolbook) or can we do better?
    C12 = A1*B2: 1x5 * 5x4 -> rank 20
    C21 = A2*B1: 2x5 * 5x2 -> rank 20
    C22 = A2*B2: 2x5 * 5x4 -> rank 40
    
    Total schoolbook: 10+20+20+40 = 90 = 3*5*6. No improvement.
    
    But if we use Strassen for some blocks:
    C21 = A2*B1: 2x5 * 5x2. We can view this as 2x(2+3) * (2+3)x2.
    This doesn't directly help.
    
    Actually, let me try a different partition.
    3 = 1+2, 5 = 2+3, 6 = 2+4
    
    A (3x5) = [A11 A12; A21 A22] where A11 is 1x2, A12 is 1x3, A21 is 2x2, A22 is 2x3
    B (5x6) = [B11 B12; B21 B22] where B11 is 2x2, B12 is 2x4, B21 is 3x2, B22 is 3x4
    
    C11 = A11*B11 + A12*B21: 1x2*2x2 + 1x3*3x2 -> 1x2
    C12 = A11*B12 + A12*B22: 1x2*2x4 + 1x3*3x4 -> 1x4
    C21 = A21*B11 + A22*B21: 2x2*2x2 + 2x3*3x2 -> 2x2
    C22 = A21*B12 + A22*B22: 2x2*2x4 + 2x3*3x4 -> 2x4
    
    For C21: 2x2*2x2 + 2x3*3x2
    The 2x2*2x2 part can use Strassen (rank 7)
    The 2x3*3x2 part: rank 12 (schoolbook) or maybe better?
    2x3*3x2: we can partition 3 = 1+2
    2x(1+2) * (1+2)x2
    = 2x1*1x2 + 2x2*2x2
    2x1*1x2: rank 4
    2x2*2x2: rank 7 (Strassen)
    Total for 2x3*3x2: 4+7 = 11
    
    So C21: 7 + 11 = 18
    
    For C22: 2x2*2x4 + 2x3*3x4
    2x2*2x4: partition 4 = 2+2, so 2x2*(2+2) = 2x2*2x2 + 2x2*2x2 = 7+7 = 14
    2x3*3x4: partition 3=1+2, 4=2+2
    = 2x1*1x4 + 2x2*2x4
    2x1*1x4: rank 8
    2x2*2x4: 2x2*(2+2) = 7+7 = 14
    Total for 2x3*3x4: 8+14 = 22
    So C22: 14 + 22 = 36
    
    For C11: 1x2*2x2 + 1x3*3x2
    1x2*2x2: rank 4
    1x3*3x2: partition 3=1+2
    = 1x1*1x2 + 1x2*2x2 = 2 + 4 = 6
    Total for C11: 4+6 = 10
    
    For C12: 1x2*2x4 + 1x3*3x4
    1x2*2x4: 1x2*(2+2) = 1x2*2x2 + 1x2*2x2 = 4+4 = 8
    1x3*3x4: 1x(1+2)*(1+2)x4
    = 1x1*1x4 + 1x2*2x4
    1x1*1x4: 4
    1x2*2x4: 1x2*(2+2) = 4+4 = 8
    Total for 1x3*3x4: 4+8 = 12
    So C12: 8+12 = 20
    
    Total: C11 + C12 + C21 + C22 = 10 + 20 + 18 + 36 = 84
    
    That's better than 90! Let me verify this logic is sound.
    
    The key insight is that when we partition the matrices into blocks, the block matrix multiplication can be decomposed into sub-problems, and we can use optimal decompositions for each sub-problem.
    
    Let me implement this carefully.
    
    Actually, implementing this correctly with proper index mappings is quite involved. Let me try a different approach.
    
    Let me just use the schoolbook decomposition but with the partitioning to get 84, and construct the matrices explicitly.
    
    Actually, let me reconsider. The block decomposition approach requires careful index management. Let me implement it step by step.
    """
    pass

def _decompose_nm_p(n, m, p):
    """Decompose n x m * m x p matrix multiplication.
    Uses recursive block decomposition with Strassen for 2x2 blocks.
    Returns U, V, W as numpy arrays.
    """
    # Base case: if any dimension is 0, return empty
    if n == 0 or m == 0 or p == 0:
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        return U, V, W
    
    # Base case: 2x2*2x2 -> Strassen
    if n == 2 and m == 2 and p == 2:
        return _strassen_2x2()
    
    # Base case: if m == 1, it's just outer product scaled
    if m == 1:
        rank = n * p
        U = np.zeros((n*1, rank))
        V = np.zeros((1*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for k in range(p):
                U[i, col] = 1
                V[0, col] = 1
                W[k*n + i, col] = 1
                col += 1
        return U, V, W
    
    # If n == 1
    if n == 1:
        rank = m * p
        U = np.zeros((1*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*1, rank))
        col = 0
        for j in range(m):
            for k in range(p):
                U[j, col] = 1
                V[j*p + k, col] = 1
                W[k, col] = 1
                col += 1
        return U, V, W
    
    # If p == 1
    if p == 1:
        rank = n * m
        U = np.zeros((n*m, rank))
        V = np.zeros((m*1, rank))
        W = np.zeros((1*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                U[i*m + j, col] = 1
                V[j, col] = 1
                W[i, col] = 1
                col += 1
        return U, V, W
    
    # Try to split m into m1 + m2 where m1, m2 > 0
    # Choose split that minimizes total rank
    best_rank = n * m * p
    best_U = None
    best_V = None
    best_W = None
    
    for m1 in range(1, m):
        m2 = m - m1
        if m2 <= 0:
            continue
        # Get decompositions for (n, m1, p) and (n, m2, p)
        U1, V1, W1 = _decompose_nm_p(n, m1, p)
        U2, V2, W2 = _decompose_nm_p(n, m2, p)
        r1 = U1.shape[1]
        r2 = U2.shape[1]
        total_rank = r1 + r2
        if total_rank < best_rank:
            best_rank = total_rank
            # Combine: U1 and U2 have different row indices because m1 != m2
            # U1 has n*m1 rows, U2 has n*m2 rows
            # We need to interleave them properly.
            # Actually, the rows of U correspond to (i, j) where i in [0,n), j in [0,m)
            # For m1: j in [0, m1)
            # For m2: j in [m1, m)
            # So U is block diagonal with U1 (n*m1 rows) and U2 (n*m2 rows)
            U = np.zeros((n*m, total_rank))
            V = np.zeros((m*p, total_rank))
            W = np.zeros((p*n, total_rank))
            
            # U1 goes in rows 0..n*m1-1, columns 0..r1-1
            # But we need to map the j indices correctly.
            # U1[i*m1 + j, c] corresponds to A[i, j] where j in [0, m1)
            # In the full U, A[i, j] is at row i*m + j
            # So for j in [0, m1): full_row = i*m + j
            # U1 row index: i*m1 + j
            # So full_row = i*m + j = (i*m1 + j) + i*(m - m1) = U1_row + i*(m-m1)
            # This is not a simple block diagonal; we need to scatter the rows.
            
            for i in range(n):
                for j in range(m1):
                    u1_row = i * m1 + j
                    full_row = i * m + j
                    U[full_row, :r1] = U1[u1_row, :]
            
            for i in range(n):
                for j in range(m2):
                    u2_row = i * m2 + j
                    full_row = i * m + (m1 + j)
                    U[full_row, r1:] = U2[u2_row, :]
            
            # V1: rows are (j, k) where j in [0, m1), k in [0, p)
            # V1[j*p + k, c]
            # Full V: (j, k) where j in [0, m), k in [0, p)
            # Full row: j*p + k
            # V1 row: j*p + k for j in [0, m1)
            # So V1 rows 0..m1*p-1 map directly to full V rows 0..m1*p-1
            V[:m1*p, :r1] = V1
            # V2: j in [0, m2), full j' = m1 + j
            # Full row: (m1+j)*p + k = m1*p + j*p + k
            # V2 row: j*p + k
            V[m1*p:, r1:] = V2
            
            # W is the same for both since the output dimensions are the same
            W[:, :r1] = W1
            W[:, r1:] = W2
            
            best_U = U
            best_V = V
            best_W = W
    
    if best_U is not None:
        return best_U, best_V, best_W
    
    # Fallback: schoolbook
    return _identity_block(n, m, p)

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _decompose_nm_p(n, m, p)
    rank = U.shape[1]
    
    # Convert to list of lists with integer/half-integer entries
    U_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in U.tolist()]
    V_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in V.tolist()]
    W_list = [[int(round(x, 4)) if abs(x - round(x, 4)) < 1e-9 else float(x) for x in row] for row in W.tolist()]
    
    return {"U": U_list, "V": V_list, "W": W_list}
# EVOLVE-BLOCK-END
