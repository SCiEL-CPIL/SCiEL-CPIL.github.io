"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 14]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=2, p=14
    # Rank of 2x2x2 multiplication is 7 (Strassen)
    # We have p/m = 14/2 = 7 blocks.
    # Total rank = 7 * 7 = 49
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix)
    # Order: a00, a01, a10, a11 -> rows 0, 1, 2, 3
    U_coeffs = [
        [1, 0, 0, 1],  # M1
        [0, 0, 1, 1],  # M2
        [1, 0, 0, 0],  # M3
        [0, 0, 0, 1],  # M4
        [1, 1, 0, 0],  # M5
        [-1, 0, 1, 0], # M6
        [0, 1, 0, -1]  # M7
    ]
    
    # Strassen coefficients for V (B matrix block)
    # Local indices: b00, b01, b10, b11
    V_coeffs = [
        [1, 0, 0, 1],  # M1: b00 + b11
        [1, 0, 0, 0],  # M2: b00
        [0, 1, 0, -1], # M3: b01 - b11
        [-1, 0, 1, 0], # M4: b10 - b00
        [0, 0, 0, 1],  # M5: b11
        [1, 1, 0, 0],  # M6: b00 + b01
        [0, 0, 1, 1]   # M7: b10 + b11
    ]
    
    # Strassen coefficients for W (C matrix block)
    # Local indices: c00, c01, c10, c11
    W_coeffs = [
        [1, 0, 0, 1],  # M1: c00 + c11
        [0, 0, 1, -1], # M2: c10 - c11
        [0, 1, 0, 1],  # M3: c01 + c11
        [1, 0, 1, 0],  # M4: c00 + c10
        [-1, 1, 0, 0], # M5: -c00 + c01
        [0, 0, 0, 1],  # M6: c11
        [1, 0, 0, 0]   # M7: c00
    ]
    
    num_blocks = p // m # 14 // 2 = 7
    
    for t in range(num_blocks):
        for s in range(7):
            col = t * 7 + s
            
            # Fill U
            # U row index: i*m + j
            for idx in range(4):
                U[idx][col] = U_coeffs[s][idx]
                
            # Fill V
            # V row index: j*p + k
            # j=0, k=2t -> 14*0 + 2t = 2t
            # j=0, k=2t+1 -> 14*0 + 2t+1 = 2t+1
            # j=1, k=2t -> 14*1 + 2t = 14+2t
            # j=1, k=2t+1 -> 14*1 + 2t+1 = 14+2t+1
            
            # Local idx 0: b00 -> row 2t
            # Local idx 1: b01 -> row 2t+1
            # Local idx 2: b10 -> row 14+2t
            # Local idx 3: b11 -> row 14+2t+1
            
            v_rows = [2*t, 2*t+1, 14+2*t, 14+2*t+1]
            for idx in range(4):
                V[v_rows[idx]][col] = V_coeffs[s][idx]
                
            # Fill W
            # W row index: k*n + i
            # k=2t, i=0 -> 2t*2 + 0 = 4t
            # k=2t, i=1 -> 2t*2 + 1 = 4t+1
            # k=2t+1, i=0 -> (2t+1)*2 + 0 = 4t+2
            # k=2t+1, i=1 -> (2t+1)*2 + 1 = 4t+3
            
            # Local idx 0: c00 -> row 4t
            # Local idx 1: c01 -> row 4t+2
            # Local idx 2: c10 -> row 4t+1
            # Local idx 3: c11 -> row 4t+3
            
            w_rows = [4*t, 4*t+2, 4*t+1, 4*t+3]
            for idx in range(4):
                W[w_rows[idx]][col] = W_coeffs[s][idx]
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
