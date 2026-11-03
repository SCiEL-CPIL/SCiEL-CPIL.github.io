"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen for 2x2 blocks and schoolbook for 3x3 blocks
    rank = 189
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2 matrix multiplication
    # Each entry is a list of (index, coeff) for the local 2x2 block
    # Local indices: (0,0)=0, (0,1)=1, (1,0)=2, (1,1)=3
    # A indices: 0, 1, 2, 3
    # B indices: 0, 1, 2, 3
    # C indices: 0, 1, 2, 3
    
    strassen_A = [
        [(0, 1), (3, 1)],       # M1: A00 + A11
        [(2, 1), (3, 1)],       # M2: A10 + A11
        [(0, 1)],               # M3: A00
        [(3, 1)],               # M4: A11
        [(0, 1), (1, 1)],       # M5: A00 + A01
        [(2, 1), (0, -1)],      # M6: A10 - A00
        [(1, 1), (3, -1)]       # M7: A01 - A11
    ]
    
    strassen_B = [
        [(0, 1), (3, 1)],       # M1: B00 + B11
        [(0, 1)],               # M2: B00
        [(1, 1), (3, -1)],      # M3: B01 - B11
        [(2, 1), (0, -1)],      # M4: B10 - B00
        [(3, 1)],               # M5: B11
        [(0, 1), (1, 1)],       # M6: B00 + B01
        [(2, 1), (3, 1)]        # M7: B10 + B11
    ]
    
    strassen_C = [
        [(0, 1), (3, 1)],       # M1: C00 + C11
        [(2, 1), (3, -1)],      # M2: C10 - C11
        [(1, 1), (3, 1)],       # M3: C01 + C11
        [(0, 1), (2, 1)],       # M4: C00 + C10
        [(0, -1), (1, 1)],      # M5: -C00 + C01
        [(3, 1)],               # M6: C11
        [(0, 1)]                # M7: C00
    ]
    
    col = 0
    # Iterate over 3x3 blocks
    for i in range(3):
        for j in range(3):
            for k in range(3):
                for s in range(7):
                    # Process A contributions
                    for idx, coeff in strassen_A[s]:
                        r, c = divmod(idx, 2)
                        # Global index in U (n*m = 36)
                        # Row 2i+r, Col 2k+c
                        u_row = (2 * i + r) * 6 + (2 * k + c)
                        U[u_row][col] = coeff
                        
                    # Process B contributions
                    for idx, coeff in strassen_B[s]:
                        r, c = divmod(idx, 2)
                        # Global index in V (m*p = 36)
                        # Row 2k+r, Col 2j+c
                        v_row = (2 * k + r) * 6 + (2 * j + c)
                        V[v_row][col] = coeff
                        
                    # Process C contributions
                    for idx, coeff in strassen_C[s]:
                        r, c = divmod(idx, 2)
                        # Global index in W (p*n = 36)
                        # Row 2i+r, Col 2j+c
                        w_row = (2 * i + r) * 6 + (2 * j + c)
                        W[w_row][col] = coeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
