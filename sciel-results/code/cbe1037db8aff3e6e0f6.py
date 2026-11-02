"""Matrix multiplication 4x5x6 with Strassen-inspired decomposition."""
DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 4, 5, 6
    
    def strassen_2x2x2():
        """Return U,V,W for 2x2*2x2->2x2 with rank 7.
        U: (4,7) for A(2x2), V: (4,7) for B(2x2), W: (4,7) for C(2x2)
        Indexing: A[i,j] -> U[i*2+j], B[i,j] -> V[i*2+j], C[i,j] -> W[i*2+j]
        """
        rank = 7
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(4)]
        W = [[0]*rank for _ in range(4)]
        
        # M1 = (A00+A11)*(B00+B11)
        U[0][0] = 1; U[3][0] = 1
        V[0][0] = 1; V[3][0] = 1
        W[0][0] = 1; W[1][0] = 1; W[2][0] = 1; W[3][0] = 1
        
        # M2 = (A10+A11)*B00
        U[2][1] = 1; U[3][1] = 1
        V[0][1] = 1
        W[2][1] = 1; W[3][1] = 1
        
        # M3 = A00*(B01-B11)
        U[0][2] = 1
        V[1][2] = 1; V[3][2] = -1
        W[0][2] = 1; W[1][2] = 1
        
        # M4 = A11*(B10-B00)
        U[3][3] = 1
        V[2][3] = 1; V[0][3] = -1
        W[2][3] = 1; W[3][3] = 1
        
        # M5 = (A00+A10)*B11
        U[0][4] = 1; U[2][4] = 1
        V[3][4] = 1
        W[0][4] = -1; W[1][4] = 1
        
        # M6 = (A11-A00)*(B00+B10)
        U[3][5] = 1; U[0][5] = -1
        V[0][5] = 1; V[2][5] = 1
        W[0][5] = 1; W[3][5] = 1
        
        # M7 = (A01-A11)*(B10+B11)
        U[1][6] = 1; U[3][6] = -1
        V[2][6] = 1; V[3][6] = 1
        W[0][6] = 1; W[3][6] = -1
        
        return U, V, W
    
    # Full decomposition:
    # A (4x5) indexed as A[i,j] -> U[i*5+j]
    # B (5x6) indexed as B[i,j] -> V[i*6+j]  
    # C (4x6) indexed as C[i,j] -> W[i*6+j]
    
    # Split: A = [A1(4x2) A2(4x2) A3(4x1)]
    #         B = [B1(2x6); B2(2x6); B3(1x6)]
    # C = A1*B1 + A2*B2 + A3*B3
    
    # A1*B1: 4x2 * 2x6 -> 4x6
    # Split into 2x2 * 2x2 blocks: 2 row-blocks x 3 col-blocks = 6 blocks, each rank 7
    # Total: 42
    
    # A2*B2: same structure, 42
    
    # A3*B3: 4x1 * 1x6 -> 4x6, schoolbook rank 24
    
    total_rank = 42 + 42 + 24
    U = [[0]*total_rank for _ in range(20)]
    V = [[0]*total_rank for _ in range(30)]
    W = [[0]*total_rank for _ in range(24)]
    
    col = 0
    # A1*B1: A rows 0-3, cols 0-1; B rows 0-1, cols 0-5
    for rb in range(2):  # row block: 0 or 1 (each 2 rows)
        for cb in range(3):  # col block: 0,1,2 (each 2 cols)
            U7, V7, W7 = strassen_2x2x2()
            for i in range(4):
                for c in range(7):
                    U[rb*2 + i//2][col + c] = U7[i][c]
            for i in range(4):
                for c in range(7):
                    V[cb*2 + i//2][col + c] = V7[i][c]
            for i in range(4):
                for c in range(7):
                    W[rb*2 + i//2][col + c] = W7[i][c]
            col += 7
    
    # A2*B2: A rows 0-3, cols 2-3; B rows 2-3, cols 0-5
    for rb in range(2):
        for cb in range(3):
            U7, V7, W7 = strassen_2x2x2()
            for i in range(4):
                for c in range(7):
                    U[rb*2 + i//2][col + c] = U7[i][c]
            for i in range(4):
                for c in range(7):
                    V[2 + cb*2 + i//2][col + c] = V7[i][c]
            for i in range(4):
                for c in range(7):
                    W[rb*2 + i//2][col + c] = W7[i][c]
            col += 7
    
    # A3*B3: A col 4, B row 4
    for i in range(4):
        for k in range(6):
            U[i*5 + 4][col] = 1
            V[4*6 + k][col] = 1
            W[i*6 + k][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
