"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 4, 4, 6
    rank = 84
    U = [[0] * rank for _ in range(n * m)]  # 16 x 84
    V = [[0] * rank for _ in range(m * p)]  # 24 x 84
    W = [[0] * rank for _ in range(p * n)]  # 24 x 84
    
    col = 0
    
    def add_strassen(x_row, x_col, y_row, y_col, out_row, out_col):
        nonlocal col
        x11 = x_row * 2 + x_col * 2
        x12 = x_row * 2 + x_col * 2 + 1
        x21 = x_row * 2 + 1 + x_col * 2
        x22 = x_row * 2 + 1 + x_col * 2 + 1
        
        y11 = y_row * 2 + y_col * 2
        y12 = y_row * 2 + y_col * 2 + 1
        y21 = y_row * 2 + 1 + y_col * 2
        y22 = y_row * 2 + 1 + y_col * 2 + 1
        
        z11 = out_row * n + out_col * 2
        z12 = out_row * n + out_col * 2 + 1
        z21 = (out_row + 1) * n + out_col * 2
        z22 = (out_row + 1) * n + out_col * 2 + 1
        
        U[x11][col] += 1; U[x22][col] += 1; V[y11][col] += 1; V[y22][col] += 1; W[z11][col] += 1; W[z22][col] += 1; col += 1
        U[x21][col] += 1; U[x22][col] += 1; V[y11][col] += 1; W[z21][col] += 1; W[z22][col] += 1; col += 1
        U[x11][col] += 1; V[y12][col] += 1; V[y22][col] -= 1; W[z12][col] += 1; W[z22][col] -= 1; col += 1
        U[x22][col] += 1; V[y21][col] += 1; V[y11][col] -= 1; W[z21][col] += 1; W[z22][col] -= 1; col += 1
        U[x11][col] += 1; U[x12][col] += 1; V[y22][col] += 1; W[z11][col] += 1; W[z12][col] += 1; col += 1
        U[x21][col] += 1; U[x11][col] -= 1; V[y11][col] += 1; V[y12][col] += 1; W[z21][col] += 1; W[z22][col] += 1; col += 1
        U[x12][col] += 1; U[x22][col] -= 1; V[y21][col] += 1; V[y22][col] += 1; W[z11][col] -= 1; W[z12][col] += 1; col += 1
    
    add_strassen(0, 0, 0, 0, 0, 0); add_strassen(0, 1, 1, 0, 0, 0)
    add_strassen(0, 0, 0, 1, 0, 1); add_strassen(0, 1, 1, 1, 0, 1)
    add_strassen(0, 0, 0, 2, 0, 2); add_strassen(0, 1, 1, 2, 0, 2)
    add_strassen(1, 0, 0, 0, 1, 0); add_strassen(1, 1, 1, 0, 1, 0)
    add_strassen(1, 0, 0, 1, 1, 1); add_strassen(1, 1, 1, 1, 1, 1)
    add_strassen(1, 0, 0, 2, 1, 2); add_strassen(1, 1, 1, 2, 1, 2)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
