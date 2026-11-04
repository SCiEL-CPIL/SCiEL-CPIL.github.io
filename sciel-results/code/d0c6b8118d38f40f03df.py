"""Tensor decomposition for 2x3x7 matrix multiplication using Strassen-style recursion."""
DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START

def construct():
    """
    Construct a tensor decomposition for 2x3 x 3x7 matrix multiplication.
    
    Strategy:
    1. Split A (2x3) into A_left (2x2) and A_right (2x1).
    2. Split B (3x7) into B_top (2x7) and B_bottom (1x7).
    3. C = A_left * B_top + A_right * B_bottom.
    
    Part 1: A_right * B_bottom (2x1 * 1x7)
    Rank = 2 * 1 * 7 = 14.
    Terms: A_right[i,0] * B_bottom[0,j] for i in 0..1, j in 0..6.
    
    Part 2: A_left * B_top (2x2 * 2x7)
    Split B_top columns into pairs and one single.
    Pairs: (0,1), (2,3), (4,5). Single: (6).
    For each pair of columns (col_k, col_k+1), use Strassen's 7 products.
    For the single column, use 4 products.
    Total rank for Part 2 = 3 * 7 + 4 = 25.
    
    Total Rank = 14 + 25 = 39.
    """
    n, m, p = DIMENSIONS  # n=2, m=3, p=7
    
    rank = 39
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A_right (col 2) * B_bottom (row 2)
    # A_right is A[:, 2] (2x1)
    # B_bottom is B[2, :] (1x7)
    # Product C[i, j] += A[i, 2] * B[2, j]
    for i in range(n):
        for j in range(p):
            U[i * m + 2][col] = 1
            V[2 * p + j][col] = 1
            W[j * n + i][col] = 1
            col += 1
            
    # Part 2: A_left (cols 0,1) * B_top (rows 0,1)
    # A_left is A[:, 0:2] (2x2)
    # B_top is B[0:2, :] (2x7)
    
    # Helper to add Strassen decomposition for 2x2 * 2x2 block (cols c1, c2 of B)
    def add_strassen_block(c1, c2):
        nonlocal col
        # A_left indices: 
        # A00 -> U[0], A01 -> U[1]
        # A10 -> U[2], A11 -> U[3]
        # B_top cols c1, c2:
        # B0_c1 -> V[0*p + c1] = V[c1]
        # B1_c1 -> V[1*p + c1] = V[p + c1]
        # B0_c2 -> V[c2]
        # B1_c2 -> V[p + c2]
        
        # Strassen products:
        # m1 = (A00 + A11) * (B0_c1 + B1_c2)
        # m2 = (A10 + A11) * B0_c1
        # m3 = A00 * (B1_c2 - B0_c1)  # Wait, standard Strassen uses B11-B22 etc.
        # Let's map standard Strassen for A=[[a,b],[c,d]], B=[[e,f],[g,h]]
        # m1 = (a+d)(e+h)
        # m2 = (c+d)e
        # m3 = a(f-h)
        # m4 = d(g-e)
        # m5 = (a+b)h
        # m6 = (c-a)(e+f)
        # m7 = (b-d)(g+h)
        
        # Here A = [[A00, A01], [A10, A11]]
        # B block = [[B0_c1, B0_c2], [B1_c1, B1_c2]]
        
        # m1
        U[0][col] = 1; U[3][col] = 1
        V[c1][col] = 1; V[p + c2][col] = 1
        W[c1 * n + 0][col] = 1; W[c1 * n + 1][col] = 1
        W[c2 * n + 0][col] = 1; W[c2 * n + 1][col] = 1
        # W[k*n+i] gets +1 for both i=0,1?
        # C[0, c1] += m1, C[1, c1] += m1
        # C[0, c2] += m1, C[1, c2] += m1
        # So W[c1*n+0] += 1, W[c1*n+1] += 1
        # W[c2*n+0] += 1, W[c2*n+1] += 1
        col += 1
        
        # m2 = (A10 + A11) * B0_c1
        U[2][col] = 1; U[3][col] = 1
        V[c1][col] = 1
        W[c1 * n + 1][col] = 1
        col += 1
        
        # m3 = A00 * (B1_c2 - B0_c1)
        U[0][col] = 1
        V[p + c2][col] = 1; V[c1][col] = -1
        W[c2 * n + 0][col] = 1; W[c1 * n + 0][col] = -1
        col += 1
        
        # m4 = A11 * (B1_c1 - B0_c1)
        U[3][col] = 1
        V[p + c1][col] = 1; V[c1][col] = -1
        W[c1 * n + 1][col] = 1; W[c1 * n + 0][col] = -1
        col += 1
        
        # m5 = (A00 + A01) * B1_c2
        U[0][col] = 1; U[1][col] = 1
        V[p + c2][col] = 1
        W[c2 * n + 0][col] = 1; W[c2 * n + 1][col] = 1
        col += 1
        
        # m6 = (A10 - A00) * (B0_c1 + B0_c2)
        U[2][col] = 1; U[0][col] = -1
        V[c1][col] = 1; V[c2][col] = 1
        W[c1 * n + 1][col] = 1; W[c2 * n + 1][col] = 1
        col += 1
        
        # m7 = (A01 - A11) * (B1_c1 + B1_c2)
        U[1][col] = 1; U[3][col] = -1
        V[p + c1][col] = 1; V[p + c2][col] = 1
        W[c1 * n + 0][col] = 1; W[c2 * n + 0][col] = 1
        col += 1

    # Add Strassen blocks for column pairs
    add_strassen_block(0, 1)
    add_strassen_block(2, 3)
    add_strassen_block(4, 5)
    
    # Add schoolbook for single column 6
    # A_left * B_top[:, 6]
    # C[0, 6] = A00*B0_6 + A01*B1_6
    # C[1, 6] = A10*B0_6 + A11*B1_6
    
    # Term 1: A00 * B0_6
    U[0][col] = 1
    V[6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 * B1_6
    U[1][col] = 1
    V[p + 6][col] = 1
    W[6 * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 * B0_6
    U[2][col] = 1
    V[6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 * B1_6
    U[3][col] = 1
    V[p + 6][col] = 1
    W[6 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}

# EVOLVE-BLOCK-END
