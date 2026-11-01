"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 9]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 32 decomposition using Strassen for 2x2 blocks and schoolbook for 2x1 block
    rank = 32
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A part)
    # M1..M7
    U_strassen = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Define the blocks of columns
    # Blocks: (0,1), (2,3), (4,5), (6,7)
    blocks = [(0, 1), (2, 3), (4, 5), (6, 7)]
    
    col_idx = 0
    for k1, k2 in blocks:
        # For each Strassen multiplication M1..M7
        for q in range(7):
            # Fill U
            for i in range(n):
                for j in range(m):
                    U[i * m + j][col_idx] = U_strassen[q][i * m + j]
            
            # Fill V
            # B indices:
            # B00 -> k1 (row 0)
            # B01 -> k2 (row 0)
            # B10 -> 9+k1 (row 1)
            # B11 -> 9+k2 (row 1)
            
            if q == 0: # M1: B00 + B11
                V[k1][col_idx] = 1
                V[9 + k2][col_idx] = 1
            elif q == 1: # M2: B00
                V[k1][col_idx] = 1
            elif q == 2: # M3: B01 - B11
                V[k2][col_idx] = 1
                V[9 + k2][col_idx] = -1
            elif q == 3: # M4: B10 - B00
                V[9 + k1][col_idx] = 1
                V[k1][col_idx] = -1
            elif q == 4: # M5: B11
                V[9 + k2][col_idx] = 1
            elif q == 5: # M6: B00 + B01
                V[k1][col_idx] = 1
                V[k2][col_idx] = 1
            elif q == 6: # M7: B10 + B11
                V[9 + k1][col_idx] = 1
                V[9 + k2][col_idx] = 1
            
            # Fill W
            # C indices:
            # C0k1 -> 2*k1
            # C0k2 -> 2*k2
            # C1k1 -> 2*k1 + 1
            # C1k2 -> 2*k2 + 1
            
            if q == 0: # M1: C00 + C11
                W[2 * k1][col_idx] = 1
                W[2 * k2 + 1][col_idx] = 1
            elif q == 1: # M2: C10 - C11
                W[2 * k1 + 1][col_idx] = 1
                W[2 * k2 + 1][col_idx] = -1
            elif q == 2: # M3: C01 + C10
                W[2 * k2][col_idx] = 1
                W[2 * k1 + 1][col_idx] = 1
            elif q == 3: # M4: -C00 + C01
                W[2 * k1][col_idx] = -1
                W[2 * k2][col_idx] = 1
            elif q == 4: # M5: C00
                W[2 * k1][col_idx] = 1
            elif q == 5: # M6: C11
                W[2 * k2 + 1][col_idx] = 1
            elif q == 6: # M7: C00 + C11
                W[2 * k1][col_idx] = 1
                W[2 * k2 + 1][col_idx] = 1
                
            col_idx += 1
            
    # Handle the last column k=8 with schoolbook (rank 4)
    k8 = 8
    # M1: A00 * B08
    U[0][col_idx] = 1
    V[k8][col_idx] = 1
    W[2 * k8][col_idx] = 1
    col_idx += 1
    
    # M2: A01 * B18
    U[1][col_idx] = 1
    V[9 + k8][col_idx] = 1
    W[2 * k8][col_idx] = 1
    col_idx += 1
    
    # M3: A10 * B08
    U[2][col_idx] = 1
    V[k8][col_idx] = 1
    W[2 * k8 + 1][col_idx] = 1
    col_idx += 1
    
    # M4: A11 * B18
    U[3][col_idx] = 1
    V[9 + k8][col_idx] = 1
    W[2 * k8 + 1][col_idx] = 1
    col_idx += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
