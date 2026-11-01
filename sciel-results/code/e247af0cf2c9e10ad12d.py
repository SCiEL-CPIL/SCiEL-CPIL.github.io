"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 11]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Use Strassen for pairs of columns: 5 pairs (cols 0-1, 2-3, 4-5, 6-7, 8-9) + 1 single (col 10)
    # 5*7 + 4 = 39 rank
    rank = 39
    U = [[0] * rank for _ in range(n * m)]  # 4 x 39
    V = [[0] * rank for _ in range(m * p)]  # 22 x 39
    W = [[0] * rank for _ in range(p * n)]  # 22 x 39
    
    col_idx = 0
    
    # Strassen terms for a 2x2 * 2x2 multiplication
    # For B_sub = [[b00, b01], [b10, b11]] (rows of B for two columns)
    # M1 = (a00+a11)(b00+b11)
    # M2 = (a10+a11)b00
    # M3 = a00(b10-b11)
    # M4 = a11(b01-b00)
    # M5 = (a00+a01)b11
    # M6 = (a10-a00)(b00+b01)
    # M7 = (a01-a11)(b10+b11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    def add_strassen_pair(k):
        """Add 7 rank-1 tensors for multiplying A (2x2) by B[:, 2k:2k+2] (2x2)."""
        nonlocal col_idx
        b00_col = 0 * p + 2 * k      # B[0, 2k]
        b01_col = 0 * p + (2 * k + 1)  # B[0, 2k+1]
        b10_col = 1 * p + 2 * k      # B[1, 2k]
        b11_col = 1 * p + (2 * k + 1)  # B[1, 2k+1]
        
        c00_col = (2 * k) * n + 0    # C[0, 2k]
        c01_col = (2 * k + 1) * n + 0  # C[0, 2k+1]
        c10_col = (2 * k) * n + 1    # C[1, 2k]
        c11_col = (2 * k + 1) * n + 1  # C[1, 2k+1]
        
        # M1 = (a00+a11)(b00+b11)
        U[0][col_idx] += 1; U[3][col_idx] += 1
        V[b00_col][col_idx] += 1; V[b11_col][col_idx] += 1
        W[c00_col][col_idx] += 1; W[c11_col][col_idx] += 1
        col_idx += 1
        
        # M2 = (a10+a11)b00
        U[2][col_idx] += 1; U[3][col_idx] += 1
        V[b00_col][col_idx] += 1
        W[c10_col][col_idx] += 1
        col_idx += 1
        
        # M3 = a00(b10-b11)
        U[0][col_idx] += 1
        V[b10_col][col_idx] += 1; V[b11_col][col_idx] += -1
        W[c01_col][col_idx] += 1
        col_idx += 1
        
        # M4 = a11(b01-b00)
        U[3][col_idx] += 1
        V[b01_col][col_idx] += 1; V[b00_col][col_idx] += -1
        W[c00_col][col_idx] += 1; W[c10_col][col_idx] += 1
        col_idx += 1
        
        # M5 = (a00+a01)b11
        U[0][col_idx] += 1; U[1][col_idx] += 1
        V[b11_col][col_idx] += 1
        W[c00_col][col_idx] += -1; W[c01_col][col_idx] += 1
        col_idx += 1
        
        # M6 = (a10-a00)(b00+b01)
        U[2][col_idx] += 1; U[0][col_idx] += -1
        V[b00_col][col_idx] += 1; V[b01_col][col_idx] += 1
        W[c11_col][col_idx] += 1
        col_idx += 1
        
        # M7 = (a01-a11)(b10+b11)
        U[1][col_idx] += 1; U[3][col_idx] += -1
        V[b10_col][col_idx] += 1; V[b11_col][col_idx] += 1
        W[c00_col][col_idx] += 1
        col_idx += 1
    
    def add_single_column(k):
        """Add 4 rank-1 tensors for multiplying A (2x2) by B[:, k] (2x1)."""
        nonlocal col_idx
        b0_col = 0 * p + k  # B[0, k]
        b1_col = 1 * p + k  # B[1, k]
        
        c0_col = k * n + 0  # C[0, k]
        c1_col = k * n + 1  # C[1, k]
        
        # C[0,k] = a00*b0 + a01*b1
        U[0][col_idx] += 1
        V[b0_col][col_idx] += 1
        W[c0_col][col_idx] += 1
        col_idx += 1
        
        U[1][col_idx] += 1
        V[b1_col][col_idx] += 1
        W[c0_col][col_idx] += 1
        col_idx += 1
        
        # C[1,k] = a10*b0 + a11*b1
        U[2][col_idx] += 1
        V[b0_col][col_idx] += 1
        W[c1_col][col_idx] += 1
        col_idx += 1
        
        U[3][col_idx] += 1
        V[b1_col][col_idx] += 1
        W[c1_col][col_idx] += 1
        col_idx += 1
    
    # Process 5 pairs: columns (0,1), (2,3), (4,5), (6,7), (8,9)
    for k in range(5):
        add_strassen_pair(k)
    
    # Process last single column: column 10
    add_single_column(10)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
