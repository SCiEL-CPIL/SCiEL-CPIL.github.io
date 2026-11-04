import numpy as np

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # We split the multiplication A(2x3) * B(3x7) into:
    # A = [A1 (2x2), A2 (2x1)]
    # B = [B1 (2x7); B2 (1x7)]
    # C = A1*B1 + A2*B2
    
    # Part 1: A1 (2x2) * B1 (2x7)
    # We decompose B1 into 3 blocks of width 2 and 1 block of width 1.
    # B1 = [B1a (2x2), B1b (2x2), B1c (2x2), B1d (2x1)]
    # C1a = A1 * B1a (2x2 * 2x2) -> Rank 7 (Strassen)
    # C1b = A1 * B1b (2x2 * 2x2) -> Rank 7
    # C1c = A1 * B1c (2x2 * 2x2) -> Rank 7
    # C1d = A1 * B1d (2x2 * 2x1) -> Rank 4 (Schoolbook)
    # Total Rank Part 1 = 7 + 7 + 7 + 4 = 25
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # C2 = A2 * B2 (2x1 * 1x7) -> Rank 14 (Schoolbook, outer product)
    # Total Rank Part 2 = 14
    
    # Total Rank = 25 + 14 = 39
    
    rank = 39
    U = [[0] * rank for _ in range(n * m)] # 6 x 39
    V = [[0] * rank for _ in range(m * p)] # 21 x 39
    W = [[0] * rank for _ in range(p * n)] # 14 x 39
    
    # Indices for A: A[0]=a11, A[1]=a12, A[2]=a21, A[3]=a22, A[4]=a13, A[5]=a23
    # A1 = [[A[0], A[1]], [A[2], A[3]]]
    # A2 = [A[4], A[5]]
    
    # Indices for B: B is 3x7.
    # B[0..6] = Row 0 (b11..b17)
    # B[7..13] = Row 1 (b21..b27)
    # B[14..20] = Row 2 (b31..b37)
    # B1 = Rows 0,1 of B.
    # B2 = Row 2 of B.
    
    # Indices for C: C is 2x7.
    # W[k*n + i] -> W[0..6] = Row 0 (c11..c17), W[7..13] = Row 1 (c21..c27)
    # C1 = A1*B1 (2x7)
    # C2 = A2*B2 (2x7)
    # C = C1 + C2
    
    col = 0
    
    # Helper to add Strassen 2x2x2 decomposition
    # A1 (2x2), B1 (2x2), C1 (2x2)
    # Strassen factors:
    # M1 = (A11+A22)(B11+B22)
    # M2 = (A21+A22)B11
    # M3 = A11(B12-B22)
    # M4 = A22(B21-B11)
    # M5 = (A11+A12)B22
    # M6 = (A21-A11)(B11+B12)
    # M7 = (A12-A22)(B21+B22)
    # C11 = M1+M4-M5+M7
    # C12 = M3+M5
    # C21 = M2+M4
    # C22 = M1-M2+M3+M6
    
    def add_strassen(offset_u, offset_v, offset_w):
        nonlocal col
        # offset_u: index in U for A1 (0,1,2,3)
        # offset_v: index in V for B1 (0,1,2,3)
        # offset_w: index in W for C1 (0,1,2,3)
        
        # M1
        U[offset_u + 0][col] = 1
        U[offset_u + 3][col] = 1
        V[offset_v + 0][col] = 1
        V[offset_v + 3][col] = 1
        W[offset_w + 0][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M2
        U[offset_u + 2][col] = 1
        U[offset_u + 3][col] = 1
        V[offset_v + 0][col] = 1
        W[offset_w + 2][col] = 1
        W[offset_w + 3][col] = -1
        col += 1
        
        # M3
        U[offset_u + 0][col] = 1
        V[offset_v + 1][col] = 1
        V[offset_v + 3][col] = -1
        W[offset_w + 1][col] = 1
        W[offset_w + 2][col] = 1
        col += 1
        
        # M4
        U[offset_u + 3][col] = 1
        V[offset_v + 2][col] = 1
        V[offset_v + 0][col] = -1
        W[offset_w + 0][col] = -1
        W[offset_w + 2][col] = 1
        col += 1
        
        # M5
        U[offset_u + 0][col] = 1
        U[offset_u + 1][col] = 1
        V[offset_v + 3][col] = 1
        W[offset_w + 0][col] = -1
        W[offset_w + 1][col] = 1
        col += 1
        
        # M6
        U[offset_u + 2][col] = 1
        U[offset_u + 0][col] = -1
        V[offset_v + 0][col] = 1
        V[offset_v + 1][col] = 1
        W[offset_w + 3][col] = 1
        col += 1
        
        # M7
        U[offset_u + 1][col] = 1
        U[offset_u + 3][col] = -1
        V[offset_v + 2][col] = 1
        V[offset_v + 3][col] = 1
        W[offset_w + 0][col] = 1
        col += 1

    # Helper to add Schoolbook 2x2x1 decomposition
    def add_schoolbook_2x2_1(offset_u, offset_v, offset_w):
        nonlocal col
        # A1 (2x2), B1 (2x1), C1 (2x1)
        # C11 = A11*B11 + A12*B12
        # C21 = A21*B11 + A22*B12
        
        # A11*B11
        U[offset_u + 0][col] = 1
        V[offset_v + 0][col] = 1
        W[offset_w + 0][col] = 1
        col += 1
        
        # A12*B12
        U[offset_u + 1][col] = 1
        V[offset_v + 1][col] = 1
        W[offset_w + 0][col] = 1
        col += 1
        
        # A21*B11
        U[offset_u + 2][col] = 1
        V[offset_v + 0][col] = 1
        W[offset_w + 1][col] = 1
        col += 1
        
        # A22*B12
        U[offset_u + 3][col] = 1
        V[offset_v + 1][col] = 1
        W[offset_w + 1][col] = 1
        col += 1

    # Helper to add Schoolbook 2x1x1 decomposition (Outer product part)
    def add_outer_product_2x1_7(offset_u, offset_v, offset_w):
        nonlocal col
        # A2 (2x1), B2 (1x7), C2 (2x7)
        # C2[i][j] = A2[i] * B2[j]
        
        for i in range(2):
            for j in range(7):
                U[offset_u + i][col] = 1
                V[offset_v + j][col] = 1
                W[offset_w + i * 7 + j][col] = 1
                col += 1

    # Part 1: A1 (2x2) * B1 (2x7)
    # A1 is U[0:4]
    # B1 is V[0:14] (Rows 0,1 of B)
    # C1 is W[0:14] (Rows 0,1 of C)
    
    # B1 columns 0,1 -> B1a
    add_strassen(0, 0, 0)
    # B1 columns 2,3 -> B1b
    add_strassen(0, 2, 2)
    # B1 columns 4,5 -> B1c
    add_strassen(0, 4, 4)
    # B1 column 6 -> B1d (2x2 * 2x1)
    add_schoolbook_2x2_1(0, 6, 6)
    
    # Part 2: A2 (2x1) * B2 (1x7)
    # A2 is U[4:6]
    # B2 is V[14:21] (Row 2 of B)
    # C2 is W[0:14] (Added to C1)
    
    add_outer_product_2x1_7(4, 14, 0)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
