"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 18 construction:
    # 2 Strassen blocks for cols (0,1) and (2,3)
    # 1 Schoolbook block for col 4
    rank = 18
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen block for B columns k0, k1
    def add_strassen(k0, k1, start_col):
        # U indices: 0: A00, 1: A01, 2: A10, 3: A11
        # V indices: 
        # j=0, k=k0 -> 0*p + k0
        # j=0, k=k1 -> 0*p + k1
        # j=1, k=k0 -> 1*p + k0
        # j=1, k=k1 -> 1*p + k1
        v00 = 0 * p + k0
        v01 = 0 * p + k1
        v10 = 1 * p + k0
        v11 = 1 * p + k1
        
        # W indices:
        # C00 -> k0*n + 0
        # C10 -> k0*n + 1
        # C01 -> k1*n + 0
        # C11 -> k1*n + 1
        w00 = k0 * n + 0
        w10 = k0 * n + 1
        w01 = k1 * n + 0
        w11 = k1 * n + 1
        
        c = start_col
        # M1 = (A00+A11)(B00+B11)
        U[0][c] = 1; U[3][c] = 1
        V[v00][c] = 1; V[v11][c] = 1
        W[w00][c] = 1; W[w11][c] = 1
        c += 1
        
        # M2 = (A10+A11)B00
        U[2][c] = 1; U[3][c] = 1
        V[v00][c] = 1
        W[w10][c] = 1
        c += 1
        
        # M3 = A00(B01-B11)
        U[0][c] = 1
        V[v01][c] = 1; V[v11][c] = -1
        W[w01][c] = 1
        c += 1
        
        # M4 = A11(B10-B00)
        U[3][c] = 1
        V[v10][c] = 1; V[v00][c] = -1
        W[w00][c] = 1; W[w10][c] = 1
        c += 1
        
        # M5 = (A00+A01)B11
        U[0][c] = 1; U[1][c] = 1
        V[v11][c] = 1
        W[w00][c] = -1; W[w01][c] = 1
        c += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[2][c] = 1; U[0][c] = -1
        V[v00][c] = 1; V[v01][c] = 1
        W[w11][c] = 1
        c += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[1][c] = 1; U[3][c] = -1
        V[v10][c] = 1; V[v11][c] = 1
        W[w00][c] = 1; W[w11][c] = -1
        c += 1
        
        return c

    # Block 1: Cols 0, 1
    col = add_strassen(0, 1, col)
    
    # Block 2: Cols 2, 3
    col = add_strassen(2, 3, col)
    
    # Block 3: Col 4 (Schoolbook)
    # C04 = A00 B04 + A01 B14
    # C14 = A10 B04 + A11 B14
    
    # Term 1: A00 B04
    U[0][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 2: A01 B14
    U[1][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 0][col] = 1
    col += 1
    
    # Term 3: A10 B04
    U[2][col] = 1
    V[0 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    # Term 4: A11 B14
    U[3][col] = 1
    V[1 * p + 4][col] = 1
    W[4 * n + 1][col] = 1
    col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
