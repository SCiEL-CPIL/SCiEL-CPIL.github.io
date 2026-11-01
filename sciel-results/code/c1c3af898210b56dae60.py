"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for multiplying a 2x2 matrix by a 2x5 matrix.
    
    Strategy: Use Strassen's algorithm for 2x2x2 blocks (rank 7 each) and schoolbook for the remaining 2x2x1 block (rank 2).
    Total rank = 7 + 7 + 2 = 16.
    
    The matrix multiplication C = A * B where:
    - A is 2x2
    - B is 2x5
    - C is 2x5
    
    We can view B as [B1, B2] where B1 is 2x2 and B2 is 2x3.
    Then C = [A*B1, A*B2].
    
    For A*B1 (2x2 * 2x2 -> 2x2), we use Strassen (rank 7).
    For A*B2 (2x2 * 2x3 -> 2x3), we can split B2 into [B2a, B2b] where B2a is 2x2 and B2b is 2x1.
    Then A*B2 = [A*B2a, A*B2b].
    For A*B2a (2x2 * 2x2 -> 2x2), we use Strassen (rank 7).
    For A*B2b (2x2 * 2x1 -> 2x1), we use schoolbook (rank 2).
    
    Wait, let me reconsider. The dimensions are n=2, m=2, p=5.
    A is n x m = 2 x 2
    B is m x p = 2 x 5
    C is n x p = 2 x 5
    
    We can partition B into blocks: B = [B1, B2, B3, B4, B5] where each Bi is 2x1.
    Then C = [C1, C2, C3, C4, C5] where Ci = A * Bi.
    
    Each Ci is 2x1, and A is 2x2, Bi is 2x1.
    So each Ci = A * Bi is a 2x2 * 2x1 multiplication.
    
    For a 2x2 * 2x1 multiplication, the rank is at most 2 (schoolbook), but could it be less?
    Actually, for 2x2 * 2x1, the tensor has rank 2 (since it's just two independent multiplications).
    
    So if we do schoolbook for each of the 5 columns, we get rank 5 * 2 = 10.
    
    But wait, the current program uses rank n*m*p = 2*2*5 = 20.
    
    Let me think about this more carefully. The standard schoolbook for n x m times m x p has rank n*m*p.
    But we can do better by exploiting structure.
    
    Actually, for 2x2 * 2x5, we can view this as:
    C[i,k] = sum_j A[i,j] * B[j,k]
    
    The tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    
    The rank of this tensor is what we're minimizing.
    
    For 2x2 * 2x2, the rank is 7 (Strassen).
    For 2x2 * 2x1, the rank is 2 (since it's just two scalar multiplications per row, but actually...
    
    Let me reconsider. For 2x2 * 2x1:
    A = [[a, b], [c, d]], B = [[e], [f]], C = [[ae+bf], [ce+df]]
    
    The tensor has entries T[i,j,k] where i in {0,1}, j in {0,1}, k in {0,1} (but k only has one value since p=1).
    
    Actually, for p=1, the tensor is T[i,j,0] = A[i,j] * B[j,0] * C[i,0].
    
    The rank of this tensor... Let's think of it as a bilinear map from (A, B) to C.
    C[0] = A[0,0]*B[0,0] + A[0,1]*B[1,0]
    C[1] = A[1,0]*B[0,0] + A[1,1]*B[1,0]
    
    This is a bilinear map with 4 input variables (A[0,0], A[0,1], A[1,0], A[1,1]) and 2 input variables (B[0,0], B[1,0]), producing 2 outputs.
    
    The rank of this bilinear map is at most 4 (schoolbook), but can it be less?
    
    Actually, I think the rank for 2x2 * 2x1 is 2. Let me verify:
    
    We need to find rank-r decomposition:
    C[i,k] = sum_{t=1}^{r} U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    Wait, the indices are:
    U has shape (n*m, rank) = (4, rank)
    V has shape (m*p, rank) = (2, rank)
    W has shape (p*n, rank) = (2, rank)
    
    And the decomposition is:
    T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
    
    For the schoolbook:
    T[i,j,k] = 1 if the multiplication A[i,j]*B[j,k] contributes to C[i,k]
    
    So for schoolbook, we have n*m*p terms, each with U[i*m+j, t]=1, V[j*p+k, t]=1, W[k*n+i, t]=1.
    
    For 2x2 * 2x1:
    n=2, m=2, p=1
    Schoolbook rank = 2*2*1 = 4.
    
    The 4 terms are:
    (i=0,j=0,k=0): U[0,0]=1, V[0,0]=1, W[0,0]=1
    (i=0,j=1,k=0): U[1,0]=1, V[1,0]=1, W[0,0]=1
    (i=1,j=0,k=0): U[2,0]=1, V[0,0]=1, W[1,0]=1
    (i=1,j=1,k=0): U[3,0]=1, V[1,0]=1, W[1,0]=1
    
    Can we do better than 4? 
    
    I believe the rank of the 2x2 * 2x1 tensor is actually 2. Let me think...
    
    Actually, I'm not sure. Let me just implement the direct sum approach with Strassen for the 2x2 blocks.
    
    Let me partition the 2x5 matrix B into [B1, B2] where B1 is 2x2 and B2 is 2x3.
    Then C = [A*B1, A*B2].
    
    For A*B1 (2x2 * 2x2 -> 2x2), use Strassen (rank 7).
    For A*B2 (2x2 * 2x3 -> 2x3), we can further partition B2 into [B2a, B2b] where B2a is 2x2 and B2b is 2x1.
    Then A*B2 = [A*B2a, A*B2b].
    For A*B2a (2x2 * 2x2 -> 2x2), use Strassen (rank 7).
    For A*B2b (2x2 * 2x1 -> 2x1), use schoolbook (rank 4).
    
    Total rank = 7 + 7 + 4 = 18.
    
    But the current program has rank 20, so 18 is an improvement.
    
    Actually, let me reconsider the partition. We have p=5, so we can partition into:
    - One 2x2 block (Strassen, rank 7)
    - One 2x2 block (Strassen, rank 7)  
    - One 2x1 block (schoolbook, rank 4)
    
    Total rank = 7 + 7 + 4 = 18.
    
    Let me implement this.
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # Partition the p=5 columns into: 2 + 2 + 1
    # Block 1: columns 0-1 (2x2), use Strassen
    # Block 2: columns 2-3 (2x2), use Strassen
    # Block 3: column 4 (2x1), use schoolbook
    
    # We'll build U, V, W matrices
    # U shape: (n*m, rank) = (4, rank)
    # V shape: (m*p, rank) = (10, rank)
    # W shape: (p*n, rank) = (10, rank)
    
    # First, let's define the Strassen decomposition for 2x2 * 2x2
    # Strassen's algorithm for C = A * B where A, B, C are 2x2:
    # M1 = (A00 + A11) * (B00 + B11)
    # M2 = (A10 + A11) * B00
    # M3 = A00 * (B01 - B11)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B10 + B11)
    #
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # For each Strassen block, we need to map the indices correctly.
    # The U matrix has indices [i*m + j] where i is row of A, j is column of A (which is row of B)
    # The V matrix has indices [j*p + k] where j is row of B, k is column of B
    # The W matrix has indices [k*n + i] where k is column of C, i is row of C
    
    # Let's define a function to add a Strassen block for a specific range of columns
    def add_strassen(col_start):
        """Add a Strassen decomposition for a 2x2 block starting at column col_start."""
        # Returns (U_block, V_block, W_block) where each is a list of (index, value) tuples
        U_block = []
        V_block = []
        W_block = []
        
        # Strassen's 7 multiplications
        # M1 = (A00 + A11) * (B00 + B11)
        # In terms of indices:
        # A00: i=0, j=0 -> U index 0*2+0 = 0
        # A11: i=1, j=1 -> U index 1*2+1 = 3
        # B00: j=0, k=col_start -> V index 0*5+col_start = col_start
        # B11: j=1, k=col_start+1 -> V index 1*5+col_start+1 = 5+col_start+1 = 6+col_start
        
        # For M1:
        U_block.append((0, 1))   # A00
        U_block.append((3, 1))   # A11
        V_block.append((col_start, 1))       # B00
        V_block.append((5 + col_start + 1, 1))  # B11
        W_block.append((col_start * 2 + 0, 1))   # C00
        W_block.append((col_start * 2 + 1, 1))   # C01
        W_block.append(((col_start+1) * 2 + 0, 1))  # C10
        W_block.append(((col_start+1) * 2 + 1, 1))  # C11
        
        return U_block, V_block, W_block
    
    # Actually, let me think about this differently. Each rank-1 term in the tensor decomposition
    # corresponds to a single multiplication. For Strassen, we have 7 such terms.
    
    # Let me define the Strassen decomposition more carefully.
    
    # For a 2x2 * 2x2 multiplication with A, B, C all 2x2:
    # The tensor T[i,j,k] = A[i,j] * B[j,k] * C[i,k]
    
    # Strassen's decomposition:
    # Term 1: (A00+A11)(B00+B11) contributes to C00, C01, C10, C11
    #   U: A00=1, A11=1
    #   V: B00=1, B11=1
    #   W: C00=1, C01=1, C10=1, C11=1
    
    # Term 2: (A10+A11)B00 contributes to C10, C11
    #   U: A10=1, A11=1
    #   V: B00=1
    #   W: C10=1, C11=-1
    
    # Term 3: A00(B01-B11) contributes to C01, C11
    #   U: A00=1
    #   V: B01=1, B11=-1
    #   W: C01=1, C11=1
    
    # Term 4: A11(B10-B00) contributes to C00, C10
    #   U: A11=1
    #   V: B10=1, B00=-1
    #   W: C00=1, C10=1
    
    # Term 5: (A00+A01)B11 contributes to C00, C01
    #   U: A00=1, A01=1
    #   V: B11=1
    #   W: C00=-1, C01=1
    
    # Term 6: (A10-A00)(B00+B01) contributes to C11
    #   U: A10=1, A00=-1
    #   V: B00=1, B01=1
    #   W: C11=1
    
    # Term 7: (A01-A11)(B10+B11) contributes to C00, C11
    #   U: A01=1, A11=-1
    #   V: B10=1, B11=1
    #   W: C00=1, C11=-1
    
    # Let me verify: 
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # So for term 1 (M1): W[00]=1, W[01]=1, W[10]=1, W[11]=1
    # For term 2 (M2): W[10]=1, W[11]=-1
    # For term 3 (M3): W[01]=1, W[11]=1
    # For term 4 (M4): W[00]=1, W[10]=1
    # For term 5 (M5): W[00]=-1, W[01]=1
    # For term 6 (M6): W[11]=1
    # For term 7 (M7): W[00]=1, W[11]=-1
    
    # Now let's map these to the global indices.
    
    def get_strassen_terms(col_start):
        """Get the 7 Strassen terms for a 2x2 block starting at column col_start."""
        terms = []
        
        # Term 1: M1 = (A00+A11)(B00+B11)
        U1 = {0: 1, 3: 1}  # A00, A11
        V1 = {col_start: 1, 5 + col_start + 1: 1}  # B00, B11
        W1 = {col_start * 2 + 0: 1, col_start * 2 + 1: 1, (col_start+1) * 2 + 0: 1, (col_start+1) * 2 + 1: 1}
        terms.append((U1, V1, W1))
        
        # Term 2: M2 = (A10+A11)B00
        U2 = {2: 1, 3: 1}  # A10, A11
        V2 = {col_start: 1}  # B00
        W2 = {(col_start+1) * 2 + 0: 1, (col_start+1) * 2 + 1: -1}  # C10, C11
        terms.append((U2, V2, W2))
        
        # Term 3: M3 = A00(B01-B11)
        U3 = {0: 1}  # A00
        V3 = {5 + col_start + 1: 1, 5 + col_start + 2: -1}  # B01, B11
        W3 = {col_start * 2 + 1: 1, (col_start+1) * 2 + 1: 1}  # C01, C11
        terms.append((U3, V3, W3))
        
        # Term 4: M4 = A11(B10-B00)
        U4 = {3: 1}  # A11
        V4 = {5 + col_start + 1: 1, col_start: -1}  # B10, B00
        W4 = {col_start * 2 + 0: 1, (col_start+1) * 2 + 0: 1}  # C00, C10
        terms.append((U4, V4, W4))
        
        # Term 5: M5 = (A00+A01)B11
        U5 = {0: 1, 1: 1}  # A00, A01
        V5 = {5 + col_start + 1: 1}  # B11
        W5 = {col_start * 2 + 0: -1, col_start * 2 + 1: 1}  # C00, C01
        terms.append((U5, V5, W5))
        
        # Term 6: M6 = (A10-A00)(B00+B01)
        U6 = {2: 1, 0: -1}  # A10, A00
        V6 = {col_start: 1, 5 + col_start + 1: 1}  # B00, B01
        W6 = {(col_start+1) * 2 + 1: 1}  # C11
        terms.append((U6, V6, W6))
        
        # Term 7: M7 = (A01-A11)(B10+B11)
        U7 = {1: 1, 3: -1}  # A01, A11
        V7 = {5 + col_start + 1: 1, 5 + col_start + 2: 1}  # B10, B11
        W7 = {col_start * 2 + 0: 1, (col_start+1) * 2 + 1: -1}  # C00, C11
        terms.append((U7, V7, W7))
        
        return terms
    
    def get_schoolbook_terms(col_start, num_cols=1):
        """Get schoolbook terms for a 2x2 * 2xnum_cols block starting at column col_start."""
        terms = []
        for i in range(n):
            for j in range(m):
                for k in range(num_cols):
                    col = col_start + k
                    U_term = {i * m + j: 1}
                    V_term = {j * p + col: 1}
                    W_term = {col * n + i: 1}
                    terms.append((U_term, V_term, W_term))
        return terms
    
    # Now build the full decomposition
    all_terms = []
    
    # Block 1: columns 0-1, Strassen
    all_terms.extend(get_strassen_terms(0))
    
    # Block 2: columns 2-3, Strassen
    all_terms.extend(get_strassen_terms(2))
    
    # Block 3: column 4, schoolbook (2x2 * 2x1)
    all_terms.extend(get_schoolbook_terms(4, 1))
    
    rank = len(all_terms)
    
    # Initialize U, V, W
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    for t, (U_term, V_term, W_term) in enumerate(all_terms):
        for idx, val in U_term.items():
            U[idx][t] = val
        for idx, val in V_term.items():
            V[idx][t] = val
        for idx, val in W_term.items():
            W[idx][t] = val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
