"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    
    # Rank-50 decomposition using Strassen on 2x2x2 blocks
    # A (2x4) = [A_left A_right], B (4x7) = [B_top; B_bottom]
    # C = A_left @ B_top + A_right @ B_bottom
    # Each 2x2 @ 2x7 split as 3*(2x2@2x2) + (2x2@2x1)
    # Rank per 2x2@2x7: 3*7 + 4 = 25
    # Total: 2*25 = 50
    
    rank = 50
    U = [[0] * rank for _ in range(n * m)]   # 8 x 50
    V = [[0] * rank for _ in range(m * p)]   # 28 x 50
    W = [[0] * rank for _ in range(p * n)]   # 14 x 50
    
    col = 0
    
    def add_strassen222(a_row, b_row, c_col):
        """Strassen 2x2x2. A[a_row:a_row+2, b_row:b_row+2] @ B[b_row:b_row+2, c_col:c_col+2] -> C[a_row:a_row+2, c_col:c_col+2]"""
        nonlocal col
        # A00=a, A01=b, A10=c, A11=d
        # B00=e, B01=f, B10=g, B11=h
        # C00=i, C01=j, C10=k, C11=l
        
        a_idx0 = a_row * m + b_row
        a_idx1 = a_row * m + (b_row + 1)
        a_idx2 = (a_row + 1) * m + b_row
        a_idx3 = (a_row + 1) * m + (b_row + 1)
        
        b_idx0 = b_row * p + c_col
        b_idx1 = b_row * p + (c_col + 1)
        b_idx2 = (b_row + 1) * p + c_col
        b_idx3 = (b_row + 1) * p + (c_col + 1)
        
        c_idx0 = c_col * n + a_row
        c_idx1 = (c_col + 1) * n + a_row
        c_idx2 = c_col * n + (a_row + 1)
        c_idx3 = (c_col + 1) * n + (a_row + 1)
        
        # P1 = (a+d)(e+h)
        U[a_idx0][col] += 1
        U[a_idx3][col] += 1
        V[b_idx0][col] += 1
        V[b_idx3][col] += 1
        W[c_idx0][col] += 1
        W[c_idx3][col] += 1
        col += 1
        
        # P2 = (c+d)*e
        U[a_idx2][col] += 1
        U[a_idx3][col] += 1
        V[b_idx0][col] += 1
        W[c_idx2][col] += 1
        col += 1
        
        # P3 = a*(f-h)
        U[a_idx0][col] += 1
        V[b_idx1][col] += 1
        V[b_idx3][col] += -1
        W[c_idx1][col] += 1
        col += 1
        
        # P4 = d*(g-e)
        U[a_idx3][col] += 1
        V[b_idx2][col] += 1
        V[b_idx0][col] += -1
        W[c_idx2][col] += 1
        col += 1
        
        # P5 = (a+b)*h
        U[a_idx0][col] += 1
        U[a_idx1][col] += 1
        V[b_idx3][col] += 1
        W[c_idx1][col] += 1
        W[c_idx2][col] += -1
        col += 1
        
        # P6 = (c-a)*(e+f)
        U[a_idx2][col] += 1
        U[a_idx0][col] += -1
        V[b_idx0][col] += 1
        V[b_idx1][col] += 1
        W[c_idx2][col] += 1
        W[c_idx3][col] += 1
        col += 1
        
        # P7 = (b-d)*(g+h)
        U[a_idx1][col] += 1
        U[a_idx3][col] += -1
        V[b_idx2][col] += 1
        V[b_idx3][col] += 1
        W[c_idx0][col] += 1
        W[c_idx1][col] += 1
        col += 1
        
        return 7
    
    def add_schoolbook221(a_row, b_row, c_col):
        """Schoolbook 2x2 @ 2x1 -> 2x1. Rank 4."""
        nonlocal col
        a_idx0 = a_row * m + b_row
        a_idx1 = a_row * m + (b_row + 1)
        a_idx2 = (a_row + 1) * m + b_row
        a_idx3 = (a_row + 1) * m + (b_row + 1)
        
        b_idx0 = b_row * p + c_col
        b_idx1 = (b_row + 1) * p + c_col
        
        c_idx0 = c_col * n + a_row
        c_idx1 = c_col * n + (a_row + 1)
        
        # C[a_row, c_col] += A[a_row, b_row] * B[b_row, c_col]
        U[a_idx0][col] += 1
        V[b_idx0][col] += 1
        W[c_idx0][col] += 1
        col += 1
        
        # C[a_row, c_col] += A[a_row, b_row+1] * B[b_row+1, c_col]
        U[a_idx1][col] += 1
        V[b_idx1][col] += 1
        W[c_idx0][col] += 1
        col += 1
        
        # C[a_row+1, c_col] += A[a_row+1, b_row] * B[b_row, c_col]
        U[a_idx2][col] += 1
        V[b_idx0][col] += 1
        W[c_idx1][col] += 1
        col += 1
        
        # C[a_row+1, c_col] += A[a_row+1, b_row+1] * B[b_row+1, c_col]
        U[a_idx3][col] += 1
        V[b_idx1][col] += 1
        W[c_idx1][col] += 1
        col += 1
        
        return 4
    
    # Build decomposition
    for group in range(2):
        b_row = group * 2  # rows 0-1 or 2-3 of B
        # Three Strassen 2x2x2 blocks for C columns 0-1, 2-3, 4-5
        for cc in range(3):
            c_col = cc * 2
            add_strassen222(0, b_row, c_col)
        # One schoolbook 2x2x1 block for C column 6
        add_schoolbook221(0, b_row, 6)
    
    assert col == rank, f"Expected rank {rank}, got {col}"
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
