"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 28
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen(col, i_off, j_off, k_off):
        # X is 2x2, Y is 2x2, Z is 2x2
        # X indices: (i_off + r, j_off + c)
        # Y indices: (j_off + r, k_off + c)
        # Z indices: (k_off + c, i_off + r) -> W[k*n + i]
        
        # Term 1: (X00 + X11)(Y00 + Y11) -> Z00 + 1, Z11 + 1
        U[i_off * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        
        # Term 2: (X10 + X11)Y00 -> Z10 + 1, Z11 - 1
        U[(i_off + 1) * m + j_off][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + k_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] -= 1
        
        # Term 3: X00(Y10 - Y11) -> Z10 + 1, Z11 + 1
        U[i_off * m + j_off][col] += 1
        V[(j_off + 1) * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] -= 1
        W[k_off * n + (i_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        
        # Term 4: X11(Y01 - Y00) -> Z00 + 1, Z10 + 1
        U[(i_off + 1) * m + (j_off + 1)][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        V[j_off * p + k_off][col] -= 1
        W[k_off * n + i_off][col] += 1
        W[k_off * n + (i_off + 1)][col] += 1
        
        # Term 5: (X00 + X01)Y11 -> Z00 - 1, Z01 + 1
        U[i_off * m + j_off][col] += 1
        U[i_off * m + (j_off + 1)][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] -= 1
        W[(k_off + 1) * n + i_off][col] += 1
        
        # Term 6: (X10 - X00)(Y00 + Y01) -> Z11 + 1
        U[(i_off + 1) * m + j_off][col] += 1
        U[i_off * m + j_off][col] -= 1
        V[j_off * p + k_off][col] += 1
        V[j_off * p + (k_off + 1)][col] += 1
        W[(k_off + 1) * n + (i_off + 1)][col] += 1
        
        # Term 7: (X01 - X11)(Y10 + Y11) -> Z00 + 1, Z01 + 1
        U[i_off * m + (j_off + 1)][col] += 1
        U[(i_off + 1) * m + (j_off + 1)][col] -= 1
        V[(j_off + 1) * p + k_off][col] += 1
        V[(j_off + 1) * p + (k_off + 1)][col] += 1
        W[k_off * n + i_off][col] += 1
        W[(k_off + 1) * n + i_off][col] += 1

    # 4 sub-multiplications
    add_strassen(0, 0, 0, 0)   # A1 B11
    add_strassen(7, 0, 2, 0)   # A2 B21
    add_strassen(14, 0, 0, 2)  # A1 B12
    add_strassen(21, 0, 2, 2)  # A2 B22

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
