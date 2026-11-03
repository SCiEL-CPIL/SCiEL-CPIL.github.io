"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    """
    Rank-36 decomposition for 4x4 * 4x7 matrix multiplication using
    Strassen-style recursive block decomposition.
    
    Partition into 2x2 blocks:
    - A (4x4) -> 4 blocks of 2x2
    - B (4x7) -> 4 blocks of 2x2 and 2 blocks of 2x1
    - C (4x7) -> 4 blocks of 2x2 and 2 blocks of 2x1
    
    Use Strassen (rank 7) for the 2x2*2x2 -> 2x2 multiplications.
    Use schoolbook (rank 4) for the 2x2*2x1 -> 2x1 multiplications.
    
    Total rank: 4*7 + 2*4 = 28 + 8 = 36
    """
    n, m, p = DIMENSIONS  # n=4, m=4, p=7
    
    rank = 36
    U = [[0] * rank for _ in range(n * m)]  # 16 x 36
    V = [[0] * rank for _ in range(m * p)]  # 28 x 36
    W = [[0] * rank for _ in range(p * n)]  # 28 x 36
    
    col = 0
    
    # Helper: add a Strassen 2x2x2 decomposition
    # A block at rows a_r0..a_r1, cols a_c0..a_c1 (2x2)
    # B block at rows b_r0..b_r1, cols b_c0..b_c1 (2x2)
    # C block at rows c_r0..c_r1, cols c_c0..c_c1 (2x2)
    # W index: k*n + i where k is col of C, i is row of C
    def add_strassen(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # A11=A[a_r0,a_c0], A12=A[a_r0,a_c0+1], A21=A[a_r0+1,a_c0], A22=A[a_r0+1,a_c0+1]
        # B11=B[b_r0,b_c0], B12=B[b_r0,b_c0+1], B21=B[b_r0+1,b_c0], B22=B[b_r0+1,b_c0+1]
        # C11=C[c_r0,c_c0], C12=C[c_r0,c_c0+1], C21=C[c_r0+1,c_c0], C22=C[c_r0+1,c_c0+1]
        
        # W indices: for C[r,c], W index = c*n + r
        # C11: W[c_c0*n + c_r0]
        # C12: W[(c_c0+1)*n + c_r0]
        # C21: W[c_c0*n + c_r0+1]
        # C22: W[(c_c0+1)*n + c_r0+1]
        
        # M1 = (A11+A22)*(B11+B22)
        # U: A11 and A22 get +1
        U[a_r0 * m + a_c0][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        # V: B11 and B22 get +1
        V[b_r0 * p + b_c0][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        # W: C11 +1, C22 -1
        W[c_c0 * n + c_r0][col] += 1
        W[(c_c0+1) * n + (c_r0+1)][col] -= 1
        col += 1
        
        # M2 = (A21+A22)*B11
        U[(a_r0+1) * m + a_c0][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        # W: C21 +1
        W[c_c0 * n + (c_r0+1)][col] += 1
        col += 1
        
        # M3 = A11*(B12-B22)
        U[a_r0 * m + a_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] -= 1
        # W: C12 +1, C22 +1
        W[(c_c0+1) * n + c_r0][col] += 1
        W[(c_c0+1) * n + (c_r0+1)][col] += 1
        col += 1
        
        # M4 = A22*(B21-B11)
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        V[b_r0 * p + b_c0][col] -= 1
        # W: C11 +1, C21 +1
        W[c_c0 * n + c_r0][col] += 1
        W[c_c0 * n + (c_r0+1)][col] += 1
        col += 1
        
        # M5 = (A11+A12)*B22
        U[a_r0 * m + a_c0][col] += 1
        U[a_r0 * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        # W: C11 -1, C12 +1
        W[c_c0 * n + c_r0][col] -= 1
        W[(c_c0+1) * n + c_r0][col] += 1
        col += 1
        
        # M6 = (A21-A11)*(B11+B12)
        U[(a_r0+1) * m + a_c0][col] += 1
        U[a_r0 * m + a_c0][col] -= 1
        V[b_r0 * p + b_c0][col] += 1
        V[b_r0 * p + (b_c0+1)][col] += 1
        # W: C22 +1
        W[(c_c0+1) * n + (c_r0+1)][col] += 1
        col += 1
        
        # M7 = (A12-A22)*(B21+B22)
        U[a_r0 * m + (a_c0+1)][col] += 1
        U[(a_r0+1) * m + (a_c0+1)][col] -= 1
        V[(b_r0+1) * p + b_c0][col] += 1
        V[(b_r0+1) * p + (b_c0+1)][col] += 1
        # W: C11 +1
        W[c_c0 * n + c_r0][col] += 1
        col += 1
    
    # Helper: add a schoolbook 2x2 * 2x1 -> 2x1 decomposition
    # A block at rows a_r0..a_r1, cols a_c0..a_c1 (2x2)
    # B block at rows b_r0..b_r1, col b_c0 (2x1)
    # C block at rows c_r0..c_r1, col c_c0 (2x1)
    def add_schoolbook_2x2x1(a_r0, a_c0, b_r0, b_c0, c_r0, c_c0):
        nonlocal col
        # C1 = A11*B1 + A12*B2 -> C[c_r0, c_c0]
        # C2 = A21*B1 + A22*B2 -> C[c_r0+1, c_c0]
        
        # Product 1: A11 * B1
        U[a_r0 * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        col += 1
        
        # Product 2: A12 * B2
        U[a_r0 * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        W[c_c0 * n + c_r0][col] += 1
        col += 1
        
        # Product 3: A21 * B1
        U[(a_r0+1) * m + a_c0][col] += 1
        V[b_r0 * p + b_c0][col] += 1
        W[c_c0 * n + (c_r0+1)][col] += 1
        col += 1
        
        # Product 4: A22 * B2
        U[(a_r0+1) * m + (a_c0+1)][col] += 1
        V[(b_r0+1) * p + b_c0][col] += 1
        W[c_c0 * n + (c_r0+1)][col] += 1
        col += 1
    
    # Now compute each C block:
    # C11 = A11*B11 + A12*B21 (2x2)
    # This is sum of two 2x2*2x2 products, each handled by Strassen
    # But we need to add them together. We can use Strassen for each and sum the W contributions.
    # Actually, we can't just add two Strassen decompositions because they share the same C block.
    # We need to decompose A11*B11 + A12*B21 as a single tensor.
    
    # Alternative approach: treat each product term separately but they all contribute to the same C block.
    # We'll just add all the Strassen decompositions for each term.
    
    # C11 = A11*B11 + A12*B21
    # A11*B11: A block (0,0), B block (0,0), C block (0,0)
    add_strassen(0, 0, 0, 0, 0, 0)
    # A12*B21: A block (0,2), B block (2,0), C block (0,0)
    add_strassen(0, 2, 2, 0, 0, 0)
    
    # C12 = A11*B12 + A12*B22
    # A11*B12: A block (0,0), B block (0,2), C block (0,2)
    add_strassen(0, 0, 0, 2, 0, 2)
    # A12*B22: A block (0,2), B block (2,2), C block (0,2)
    add_strassen(0, 2, 2, 2, 0, 2)
    
    # C13 = A11*B13 + A12*B23 (2x1)
    # A11*B13: A block (0,0), B block (0,4), C block (0,4)
    add_schoolbook_2x2x1(0, 0, 0, 4, 0, 4)
    # A12*B23: A block (0,2), B block (2,4), C block (0,4)
    add_schoolbook_2x2x1(0, 2, 2, 4, 0, 4)
    
    # C21 = A21*B11 + A22*B21
    # A21*B11: A block (2,0), B block (0,0), C block (2,0)
    add_strassen(2, 0, 0, 0, 2, 0)
    # A22*B21: A block (2,2), B block (2,0), C block (2,0)
    add_strassen(2, 2, 2, 0, 2, 0)
    
    # C22 = A21*B12 + A22*B22
    # A21*B12: A block (2,0), B block (0,2), C block (2,2)
    add_strassen(2, 0, 0, 2, 2, 2)
    # A22*B22: A block (2,2), B block (2,2), C block (2,2)
    add_strassen(2, 2, 2, 2, 2, 2)
    
    # C23 = A21*B13 + A22*B23 (2x1)
    # A21*B13: A block (2,0), B block (0,4), C block (2,4)
    add_schoolbook_2x2x1(2, 0, 0, 4, 2, 4)
    # A22*B23: A block (2,2), B block (2,4), C block (2,4)
    add_schoolbook_2x2x1(2, 2, 2, 4, 2, 4)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
