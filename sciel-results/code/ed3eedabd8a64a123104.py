"""Matrix multiplication 2,2,5 with rank-minimal exact decomposition."""

DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 2x2 by 2x5 matrix multiplication.
    
    We use a block decomposition approach:
    - Split the 5 columns into two groups of 2 and one group of 1.
    - For each pair of columns, use Strassen's algorithm for 2x2x2 (rank 7).
    - For the single column, use schoolbook 2x2x1 (rank 4).
    - Total rank: 7 + 7 + 4 = 18.
    
    The output factor W follows the official trace convention W[k*n+i, r].
    """
    n, m, p = DIMENSIONS  # n=2, m=2, p=5
    
    # We'll build U, V, W as lists of lists.
    # U: (n*m, rank) = (4, rank)
    # V: (m*p, rank) = (10, rank)
    # W: (p*n, rank) = (10, rank)
    
    # Block 1: columns 0,1 -> 2x2x2 Strassen, rank 7
    # Block 2: columns 2,3 -> 2x2x2 Strassen, rank 7
    # Block 3: column 4 -> 2x2x1 schoolbook, rank 4
    
    total_rank = 18
    
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col_offset = 0
    
    # Helper: add a 2x2x2 Strassen block for A(2x2) * B2(2x2) -> C2(2x2)
    # B2 columns correspond to global columns c0, c1
    # Strassen products:
    # P1 = (a00+a11)*(b00+b11)
    # P2 = (a10+a11)*b00
    # P3 = a00*(b11-b00)
    # P4 = a11*(b10-b00)
    # P5 = (a00+a01)*b11
    # P6 = (a10-a00)*(b00+b01)
    # P7 = (a01-a11)*(b10+b11)
    #
    # C00 = P1 - P4 + P7
    # C01 = P3 + P1
    # C10 = P2 + P4
    # C11 = P1 - P3 + P5
    
    def add_strassen_block(c0, c1, offset):
        """Add Strassen block for columns c0, c1 starting at column offset."""
        # P1 = (a00+a11)*(b00+b11)
        U[0][offset] = 1  # a00
        U[3][offset] = 1  # a11
        V[0*p + c0][offset] = 1  # b00
        V[1*p + c1][offset] = 1  # b11
        W[c0*n + 0][offset] = 1  # C00
        W[c1*n + 0][offset] = 1  # C01
        W[c0*n + 1][offset] = 1  # C10
        W[c1*n + 1][offset] = 1  # C11
        offset += 1
        
        # P2 = (a10+a11)*b00
        U[2][offset] = 1  # a10
        U[3][offset] = 1  # a11
        V[0*p + c0][offset] = 1  # b00
        W[c0*n + 1][offset] = 1  # C10
        offset += 1
        
        # P3 = a00*(b11-b00)
        U[0][offset] = 1  # a00
        V[1*p + c1][offset] = 1  # b11
        V[0*p + c0][offset] = -1  # -b00
        W[c1*n + 0][offset] = 1  # C01
        offset += 1
        
        # P4 = a11*(b10-b00)
        U[3][offset] = 1  # a11
        V[1*p + c0][offset] = 1  # b10
        V[0*p + c0][offset] = -1  # -b00
        W[c0*n + 0][offset] = -1  # -C00
        W[c0*n + 1][offset] = 1  # C10
        offset += 1
        
        # P5 = (a00+a01)*b11
        U[0][offset] = 1  # a00
        U[1][offset] = 1  # a01
        V[1*p + c1][offset] = 1  # b11
        W[c1*n + 1][offset] = 1  # C11
        offset += 1
        
        # P6 = (a10-a00)*(b00+b01)
        U[2][offset] = 1  # a10
        U[0][offset] = -1  # -a00
        V[0*p + c0][offset] = 1  # b00
        V[0*p + c1][offset] = 1  # b01
        W[c0*n + 1][offset] = 1  # C10
        W[c1*n + 1][offset] = 1  # C11
        offset += 1
        
        # P7 = (a01-a11)*(b10+b11)
        U[1][offset] = 1  # a01
        U[3][offset] = -1  # -a11
        V[1*p + c0][offset] = 1  # b10
        V[1*p + c1][offset] = 1  # b11
        W[c0*n + 0][offset] = 1  # C00
        W[c1*n + 1][offset] = 1  # C11
        offset += 1
        
        return offset
    
    # Helper: add a 2x2x1 schoolbook block for column c
    def add_schoolbook_column(c, offset):
        """Add schoolbook block for single column c starting at column offset."""
        # C0c = a00*b0c + a01*b1c
        # C1c = a10*b0c + a11*b1c
        
        # Term 1: a00 * b0c -> C0c
        U[0][offset] = 1  # a00
        V[0*p + c][offset] = 1  # b0c
        W[c*n + 0][offset] = 1  # C0c
        offset += 1
        
        # Term 2: a01 * b1c -> C0c
        U[1][offset] = 1  # a01
        V[1*p + c][offset] = 1  # b1c
        W[c*n + 0][offset] = 1  # C0c
        offset += 1
        
        # Term 3: a10 * b0c -> C1c
        U[2][offset] = 1  # a10
        V[0*p + c][offset] = 1  # b0c
        W[c*n + 1][offset] = 1  # C1c
        offset += 1
        
        # Term 4: a11 * b1c -> C1c
        U[3][offset] = 1  # a11
        V[1*p + c][offset] = 1  # b1c
        W[c*n + 1][offset] = 1  # C1c
        offset += 1
        
        return offset
    
    # Block 1: columns 0, 1 using Strassen
    col_offset = add_strassen_block(0, 1, col_offset)
    
    # Block 2: columns 2, 3 using Strassen
    col_offset = add_strassen_block(2, 3, col_offset)
    
    # Block 3: column 4 using schoolbook
    col_offset = add_schoolbook_column(4, col_offset)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
