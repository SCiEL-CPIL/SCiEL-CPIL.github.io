"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = 2, 3, 5  # A is n×m, B is m×p, C is n×p
    
    rank = 28
    U = [[0] * rank for _ in range(n * m)]  # 6 x 28
    V = [[0] * rank for _ in range(m * p)]  # 15 x 28
    W = [[0] * rank for _ in range(p * n)]  # 10 x 28
    
    col = 0
    
    # Helper: add Strassen block for A[i0:i1, j0:j1] @ B[j0:j1, p0:p1]
    # where i1-i0=2, j1-j0=2, p1-p0=2
    def add_strassen_block(i0, j0, p0):
        nonlocal col
        # A indices
        A00 = i0 * m + j0
        A01 = i0 * m + (j0 + 1)
        A10 = (i0 + 1) * m + j0
        A11 = (i0 + 1) * m + (j0 + 1)
        # B indices
        B00 = j0 * p + p0
        B01 = j0 * p + (p0 + 1)
        B10 = (j0 + 1) * p + p0
        B11 = (j0 + 1) * p + (p0 + 1)
        # C indices (W[k*n+i] where k=p-index, i=n-index)
        C00 = p0 * n + i0
        C01 = (p0 + 1) * n + i0
        C10 = p0 * n + (i0 + 1)
        C11 = (p0 + 1) * n + (i0 + 1)
        
        # M1 = (A00+A11)(B00+B11)
        U[A00][col] = 1; U[A11][col] = 1
        V[B00][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C01][col] = 1; W[C10][col] = -1; W[C11][col] = 1
        col += 1
        
        # M2 = (A10+A11)B00
        U[A10][col] = 1; U[A11][col] = 1
        V[B00][col] = 1
        W[C10][col] = 1; W[C11][col] = -1
        col += 1
        
        # M3 = A00(B01-B11)
        U[A00][col] = 1
        V[B01][col] = 1; V[B11][col] = -1
        W[C01][col] = 1; W[C11][col] = 1
        col += 1
        
        # M4 = A11(B10-B00)
        U[A11][col] = 1
        V[B10][col] = 1; V[B00][col] = -1
        W[C00][col] = 1; W[C10][col] = 1
        col += 1
        
        # M5 = (A00+A01)B11
        U[A00][col] = 1; U[A01][col] = 1
        V[B11][col] = 1
        W[C00][col] = -1; W[C01][col] = 1
        col += 1
        
        # M6 = (A10-A00)(B00+B01)
        U[A10][col] = 1; U[A00][col] = -1
        V[B00][col] = 1; V[B01][col] = 1
        W[C11][col] = 1
        col += 1
        
        # M7 = (A01-A11)(B10+B11)
        U[A01][col] = 1; U[A11][col] = -1
        V[B10][col] = 1; V[B11][col] = 1
        W[C00][col] = 1; W[C01][col] = 1
        col += 1
    
    # First Strassen block: i0=0, j0=0, p0=0
    add_strassen_block(0, 0, 0)
    
    # Second Strassen block: i0=0, j0=0, p0=2
    add_strassen_block(0, 0, 2)
    
    # Remaining schoolbook terms
    for i in range(n):
        for j in range(m):
            for p_idx in range(p):
                if j < 2 and p_idx < 4:
                    continue
                U[i * m + j][col] = 1
                V[j * p + p_idx][col] = 1
                W[p_idx * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
