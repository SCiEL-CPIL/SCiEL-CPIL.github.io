"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Using block Strassen: 4 blocks of 2x2x2, rank 7 each. Total rank 28.
    R = 28
    U = [[0] * R for _ in range(n * m)]
    V = [[0] * R for _ in range(m * p)]
    W = [[0] * R for _ in range(p * n)]
    
    # Strassen coefficients for A (U)
    # U indices: i*m + j
    # M1: a11+a22 -> U[0, r]=1, U[3, r]=1
    # M2: a21+a22 -> U[2, r]=1, U[3, r]=1
    # M3: a11     -> U[0, r]=1
    # M4: a22     -> U[3, r]=1
    # M5: a11+a12 -> U[0, r]=1, U[1, r]=1
    # M6: a21-a11 -> U[2, r]=1, U[0, r]=-1
    # M7: a12-a22 -> U[1, r]=1, U[3, r]=-1
    
    U_forms = [
        {0: 1, 3: 1},
        {2: 1, 3: 1},
        {0: 1},
        {3: 1},
        {0: 1, 1: 1},
        {2: 1, 0: -1},
        {1: 1, 3: -1}
    ]
    
    # Strassen coefficients for B (V)
    # V indices: j*p + k
    # For block b, k in {2b, 2b+1}
    # M1: b11+b22 -> V[2b, r]=1, V[8+2b+1, r]=1
    # M2: b11     -> V[2b, r]=1
    # M3: b12-b22 -> V[2b+1, r]=1, V[8+2b+1, r]=-1
    # M4: b11-b21 -> V[2b, r]=1, V[8+2b, r]=-1
    # M5: b22     -> V[8+2b+1, r]=1
    # M6: b11+b12 -> V[2b, r]=1, V[2b+1, r]=1
    # M7: b21+b22 -> V[8+2b, r]=1, V[8+2b+1, r]=1
    
    # Strassen coefficients for C (W)
    # W indices: k*n + i
    # For block b, k in {2b, 2b+1}, i in {0, 1}
    # W[4b, r]   (k=2b, i=0)   C11: M1+M4-M5+M7
    # W[4b+1, r] (k=2b, i=1)   C21: M2+M4
    # W[4b+2, r] (k=2b+1, i=0) C12: M3+M5
    # W[4b+3, r] (k=2b+1, i=1) C22: M1-M2+M3+M6
    
    W_forms = [
        {0: 1, 2: -1, 3: 1, 4: 0, 5: 0, 6: 0, 7: 1}, # Placeholder, will fill in loop
    ]
    
    # Actually, let's just fill them directly in the loop for clarity.
    
    for b in range(4):
        col_base = b * 7
        for s in range(7):
            col = col_base + s
            
            # Fill U
            for idx, val in U_forms[s].items():
                U[idx][col] = val
            
            # Fill V
            # V indices: j*p + k
            # j=0, k=2b -> 2b
            # j=0, k=2b+1 -> 2b+1
            # j=1, k=2b -> 8+2b
            # j=1, k=2b+1 -> 8+2b+1
            
            if s == 0: # M1
                V[2*b][col] = 1
                V[8+2*b+1][col] = 1
            elif s == 1: # M2
                V[2*b][col] = 1
            elif s == 2: # M3
                V[2*b+1][col] = 1
                V[8+2*b+1][col] = -1
            elif s == 3: # M4
                V[2*b][col] = 1
                V[8+2*b][col] = -1
            elif s == 4: # M5
                V[8+2*b+1][col] = 1
            elif s == 5: # M6
                V[2*b][col] = 1
                V[2*b+1][col] = 1
            elif s == 6: # M7
                V[8+2*b][col] = 1
                V[8+2*b+1][col] = 1
            
            # Fill W
            # W indices: k*n + i
            # k=2b, i=0 -> 4b
            # k=2b, i=1 -> 4b+1
            # k=2b+1, i=0 -> 4b+2
            # k=2b+1, i=1 -> 4b+3
            
            if s == 0: # M1
                W[4*b][col] = 1
                W[4*b+3][col] = 1
            elif s == 1: # M2
                W[4*b+1][col] = 1
                W[4*b+3][col] = -1
            elif s == 2: # M3
                W[4*b+2][col] = 1
                W[4*b+3][col] = 1
            elif s == 3: # M4
                W[4*b][col] = 1
                W[4*b+1][col] = 1
            elif s == 4: # M5
                W[4*b][col] = -1
                W[4*b+2][col] = 1
            elif s == 5: # M6
                W[4*b+3][col] = 1
            elif s == 6: # M7
                W[4*b][col] = 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
