"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 56 decomposition using Strassen on 2x2x2 blocks
    rank = 56
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add a Strassen decomposition for a 2x2x2 block
    def add_strassen(i1, i2, j1, j2, k1, k2, start_col):
        # i1, i2 are row indices in A (0, 1)
        # j1, j2 are col indices in A / row indices in B
        # k1, k2 are col indices in B
        
        # U indices
        u_00 = i1 * m + j1
        u_11 = i2 * m + j2
        u_10 = i2 * m + j1
        u_01 = i1 * m + j2
        
        # V indices
        v_00 = j1 * p + k1
        v_11 = j2 * p + k2
        v_01 = j1 * p + k2
        v_10 = j2 * p + k1
        
        # W indices (k*n + i)
        w_00 = k1 * n + i1
        w_01 = k2 * n + i1
        w_10 = k1 * n + i2
        w_11 = k2 * n + i2
        
        # M1 = (X00 + X11)(Y00 + Y11)
        U[u_00][start_col] += 1
        U[u_11][start_col] += 1
        V[v_00][start_col] += 1
        V[v_11][start_col] += 1
        W[w_00][start_col] += 1
        W[w_11][start_col] += 1
        col_idx = start_col + 1
        
        # M2 = (X10 + X11)Y00
        U[u_10][col_idx] += 1
        U[u_11][col_idx] += 1
        V[v_00][col_idx] += 1
        W[w_10][col_idx] += 1
        W[w_11][col_idx] -= 1
        col_idx += 1
        
        # M3 = X00(Y01 - Y11)
        U[u_00][col_idx] += 1
        V[v_01][col_idx] += 1
        V[v_11][col_idx] -= 1
        W[w_01][col_idx] += 1
        W[w_11][col_idx] += 1
        col_idx += 1
        
        # M4 = X11(Y10 - Y00)
        U[u_11][col_idx] += 1
        V[v_10][col_idx] += 1
        V[v_00][col_idx] -= 1
        W[w_00][col_idx] += 1
        W[w_10][col_idx] += 1
        col_idx += 1
        
        # M5 = (X00 + X01)Y11
        U[u_00][col_idx] += 1
        U[u_01][col_idx] += 1
        V[v_11][col_idx] += 1
        W[w_00][col_idx] -= 1
        W[w_01][col_idx] += 1
        col_idx += 1
        
        # M6 = (X10 - X00)(Y00 + Y01)
        U[u_10][col_idx] += 1
        U[u_00][col_idx] -= 1
        V[v_00][col_idx] += 1
        V[v_01][col_idx] += 1
        W[w_11][col_idx] += 1
        col_idx += 1
        
        # M7 = (X01 - X11)(Y10 + Y11)
        U[u_01][col_idx] += 1
        U[u_11][col_idx] -= 1
        V[v_10][col_idx] += 1
        V[v_11][col_idx] += 1
        W[w_00][col_idx] -= 1
        W[w_11][col_idx] -= 1
        
        return start_col + 7

    # We need to cover the whole 2x4x8 product.
    # Split into 8 blocks of 2x2x2?
    # A is 2x4. B is 4x8.
    # We can view this as 8 independent 2x2x2 problems if we partition carefully?
    # No, the previous analysis showed:
    # Split B into B1 (cols 0-3) and B2 (cols 4-7).
    # For B1:
    #   Split A into AL (cols 0-1) and AR (cols 2-3).
    #   Split B1 into BT (rows 0-1) and BB (rows 2-3).
    #   C1 = AL*BT + AR*BB.
    #   AL*BT is 2x2 * 2x4. Split BT into BT1 (cols 0-1) and BT2 (cols 2-3).
    #   AL*BT1 is 2x2x2.
    #   AL*BT2 is 2x2x2.
    #   Similarly for AR*BB.
    
    # Block 1: i=0,1, j=0,1, k=0,1 (AL * BT1)
    col = add_strassen(0, 1, 0, 1, 0, 1, col)
    
    # Block 2: i=0,1, j=0,1, k=2,3 (AL * BT2)
    col = add_strassen(0, 1, 0, 1, 2, 3, col)
    
    # Block 3: i=0,1, j=2,3, k=0,1 (AR * BB1)
    col = add_strassen(0, 1, 2, 3, 0, 1, col)
    
    # Block 4: i=0,1, j=2,3, k=2,3 (AR * BB2)
    col = add_strassen(0, 1, 2, 3, 2, 3, col)
    
    # Now for B2 (cols 4-7)
    # Block 5: i=0,1, j=0,1, k=4,5 (AL * BT3)
    col = add_strassen(0, 1, 0, 1, 4, 5, col)
    
    # Block 6: i=0,1, j=0,1, k=6,7 (AL * BT4)
    col = add_strassen(0, 1, 0, 1, 6, 7, col)
    
    # Block 7: i=0,1, j=2,3, k=4,5 (AR * BB3)
    col = add_strassen(0, 1, 2, 3, 4, 5, col)
    
    # Block 8: i=0,1, j=2,3, k=6,7 (AR * BB4)
    col = add_strassen(0, 1, 2, 3, 6, 7, col)
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
