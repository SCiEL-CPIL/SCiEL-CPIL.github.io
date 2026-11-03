"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 18
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    def add_strassen_block(j_start, k_start, col_offset):
        # Strassen factors for 2x2x2
        u_00 = 0 * m + j_start
        u_01 = 0 * m + j_start + 1
        u_10 = 1 * m + j_start
        u_11 = 1 * m + j_start + 1
        
        v_00 = j_start * p + k_start
        v_01 = j_start * p + k_start + 1
        v_10 = (j_start + 1) * p + k_start
        v_11 = (j_start + 1) * p + k_start + 1
        
        w_00 = k_start * n + 0
        w_10 = k_start * n + 1
        w_01 = (k_start + 1) * n + 0
        w_11 = (k_start + 1) * n + 1
        
        # M1 = (A00 + A11)(B00 + B11)
        U[u_00][col_offset] += 1
        U[u_11][col_offset] += 1
        V[v_00][col_offset] += 1
        V[v_11][col_offset] += 1
        W[w_00][col_offset] += 1
        W[w_11][col_offset] += 1
        
        # M2 = (A10 + A11)B00
        U[u_10][col_offset+1] += 1
        U[u_11][col_offset+1] += 1
        V[v_00][col_offset+1] += 1
        W[w_10][col_offset+1] += 1
        W[w_11][col_offset+1] -= 1
        
        # M3 = A00(B10 - B11)
        U[u_00][col_offset+2] += 1
        V[v_10][col_offset+2] += 1
        V[v_11][col_offset+2] -= 1
        W[w_01][col_offset+2] += 1
        W[w_11][col_offset+2] += 1
        
        # M4 = A11(B01 - B00)
        U[u_11][col_offset+3] += 1
        V[v_01][col_offset+3] += 1
        V[v_00][col_offset+3] -= 1
        W[w_00][col_offset+3] += 1
        W[w_10][col_offset+3] += 1
        
        # M5 = (A00 + A01)B11
        U[u_00][col_offset+4] += 1
        U[u_01][col_offset+4] += 1
        V[v_11][col_offset+4] += 1
        W[w_00][col_offset+4] -= 1
        W[w_01][col_offset+4] += 1
        
        # M6 = (A10 - A00)(B00 + B01)
        U[u_10][col_offset+5] += 1
        U[u_00][col_offset+5] -= 1
        V[v_00][col_offset+5] += 1
        V[v_01][col_offset+5] += 1
        W[w_11][col_offset+5] += 1
        
        # M7 = (A01 - A11)(B10 + B11)
        U[u_01][col_offset+6] += 1
        U[u_11][col_offset+6] -= 1
        V[v_10][col_offset+6] += 1
        V[v_11][col_offset+6] += 1
        W[w_00][col_offset+6] += 1

    add_strassen_block(0, 0, 0)
    add_strassen_block(2, 2, 7)
    
    # Block 3: j=4, k=4,5
    U[4][14] = 1
    V[28][14] = 1
    W[8][14] = 1
    
    U[4][15] = 1
    V[29][15] = 1
    W[10][15] = 1
    
    U[9][16] = 1
    V[28][16] = 1
    W[9][16] = 1
    
    U[9][17] = 1
    V[29][17] = 1
    W[11][17] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
