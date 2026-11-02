"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 4]

# EVOLVE-BLOCK-START
def construct():
    # Rank 14 decomposition using Strassen's algorithm for each 2x2 block
    n, m, p = DIMENSIONS
    rank = 14
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B01-B11)
    # M4 = A11(B10-B00)
    # M5 = (A00+A01)B11
    # M6 = (A10-A00)(B00+B01)
    # M7 = (A01-A11)(B10+B11)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Block 1: B cols 0,1. C cols 0,1.
    # V indices for B1: j=0 -> 0,1; j=1 -> 4,5
    # W indices for C1: c00->0, c10->1, c01->2, c11->3
    
    # Block 2: B cols 2,3. C cols 2,3.
    # V indices for B2: j=0 -> 2,3; j=1 -> 6,7
    # W indices for C2: c00->4, c10->5, c01->6, c11->7
    
    for block in range(2):
        base_col = block * 7
        # V offsets for B
        # B00 is at j=0, k=block*2
        # B01 is at j=0, k=block*2+1
        # B10 is at j=1, k=block*2
        # B11 is at j=1, k=block*2+1
        v_b00 = 0 * p + (block * 2)
        v_b01 = 0 * p + (block * 2 + 1)
        v_b10 = 1 * p + (block * 2)
        v_b11 = 1 * p + (block * 2 + 1)
        
        # W offsets for C
        # C00 is at k=block*2, i=0 -> idx = block*2 * n + 0
        # C01 is at k=block*2+1, i=0 -> idx = (block*2+1) * n + 0
        # C10 is at k=block*2, i=1 -> idx = block*2 * n + 1
        # C11 is at k=block*2+1, i=1 -> idx = (block*2+1) * n + 1
        w_c00 = (block * 2) * n + 0
        w_c01 = (block * 2 + 1) * n + 0
        w_c10 = (block * 2) * n + 1
        w_c11 = (block * 2 + 1) * n + 1
        
        # U indices for A
        # A00: 0
        # A01: 1
        # A10: 2
        # A11: 3
        
        # M1
        col = base_col + 0
        U[0][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        W[w_c11][col] += 1
        
        # M2
        col = base_col + 1
        U[2][col] += 1
        U[3][col] += 1
        V[v_b00][col] += 1
        W[w_c10][col] += 1
        W[w_c11][col] += -1
        
        # M3
        col = base_col + 2
        U[0][col] += 1
        V[v_b01][col] += 1
        V[v_b11][col] += -1
        W[w_c01][col] += 1
        W[w_c11][col] += 1
        
        # M4
        col = base_col + 3
        U[3][col] += 1
        V[v_b10][col] += 1
        V[v_b00][col] += -1
        W[w_c00][col] += 1
        W[w_c10][col] += 1
        
        # M5
        col = base_col + 4
        U[0][col] += 1
        U[1][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += -1
        W[w_c01][col] += 1
        
        # M6
        col = base_col + 5
        U[2][col] += 1
        U[0][col] += -1
        V[v_b00][col] += 1
        V[v_b01][col] += 1
        W[w_c11][col] += 1
        
        # M7
        col = base_col + 6
        U[1][col] += 1
        U[3][col] += -1
        V[v_b10][col] += 1
        V[v_b11][col] += 1
        W[w_c00][col] += 1
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
