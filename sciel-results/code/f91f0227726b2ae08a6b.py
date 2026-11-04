"""
Rank 26 decomposition for 2x3 by 3x6 matrix multiplication.
Splits the 3x6 matrix into two 3x3 blocks.
Uses a rank 13 decomposition for each 2x3x3 block (Strassen for 2x2x2 + Outer Product for 2x1x3).
"""

DIMENSIONS = [2, 3, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 26
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_decomp_233(offset_u, offset_v, offset_w, start_r):
        """
        Adds a rank 13 decomposition for 2x3x3 multiplication.
        offset_u: index offset in U (should be 0)
        offset_v: index offset in V
        offset_w: index offset in W
        start_r: starting rank index
        """
        # Local indices for 2x3 A, 3x3 B, 2x3 C
        # U: 6 elements (2 rows, 3 cols)
        # V: 9 elements (3 rows, 3 cols)
        # W: 6 elements (3 cols, 2 rows) -> W[col*2 + row]
        
        # Part 1: Strassen for A[0:2, 0:2] * B[0:2, 0:2] -> C[0:2, 0:2]
        # U indices: 0,1 (row 0, cols 0,1), 3,4 (row 1, cols 0,1)
        # V indices: 0,1 (row 0, cols 0,1), 3,4 (row 1, cols 0,1)
        # W indices: 0,1 (col 0, rows 0,1), 2,3 (col 1, rows 0,1)
        
        # M1
        r = start_r
        U[offset_u + 0][r] = 1; U[offset_u + 4][r] = 1
        V[offset_v + 0][r] = 1; V[offset_v + 4][r] = 1
        W[offset_w + 0][r] = 1; W[offset_w + 3][r] = 1
        
        # M2
        r = start_r + 1
        U[offset_u + 3][r] = 1; U[offset_u + 4][r] = 1
        V[offset_v + 0][r] = 1
        W[offset_w + 1][r] = 1; W[offset_w + 3][r] = -1
        
        # M3
        r = start_r + 2
        U[offset_u + 0][r] = 1
        V[offset_v + 1][r] = 1; V[offset_v + 4][r] = -1
        W[offset_w + 2][r] = 1; W[offset_w + 3][r] = 1
        
        # M4
        r = start_r + 3
        U[offset_u + 4][r] = 1
        V[offset_v + 3][r] = 1; V[offset_v + 0][r] = -1
        W[offset_w + 0][r] = 1; W[offset_w + 1][r] = 1
        
        # M5
        r = start_r + 4
        U[offset_u + 0][r] = 1; U[offset_u + 1][r] = 1
        V[offset_v + 4][r] = 1
        W[offset_w + 0][r] = -1; W[offset_w + 2][r] = 1
        
        # M6
        r = start_r + 5
        U[offset_u + 3][r] = 1; U[offset_u + 0][r] = -1
        V[offset_v + 0][r] = 1; V[offset_v + 1][r] = 1
        W[offset_w + 3][r] = 1
        
        # M7
        r = start_r + 6
        U[offset_u + 1][r] = 1; U[offset_u + 4][r] = -1
        V[offset_v + 3][r] = 1; V[offset_v + 4][r] = 1
        W[offset_w + 0][r] = 1

        # Part 2: Outer product for A[0:2, 2] * B[2, 0:3] -> C[0:2, 0:3]
        # U indices: 2 (row 0, col 2), 5 (row 1, col 2)
        # V indices: 6,7,8 (row 2, cols 0,1,2)
        # W indices: 0..5 (all)
        
        for i in range(2):
            for j in range(3):
                r = start_r + 7 + i * 3 + j
                U[offset_u + i * 3 + 2][r] = 1
                V[offset_v + 6 + j][r] = 1
                W[offset_w + j * 2 + i][r] = 1

    # Block 1: B cols 0,1,2 -> C cols 0,1,2
    # V offset 0, W offset 0
    add_decomp_233(0, 0, 0, 0)
    
    # Block 2: B cols 3,4,5 -> C cols 3,4,5
    # V offset 9, W offset 6
    add_decomp_233(0, 9, 6, 13)

    return {"U": U, "V": V, "W": W}
