"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 28 decomposition using block Strassen
    # Split p=8 into 4 blocks of size 2
    num_blocks = p // 2
    rank = num_blocks * 7
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for U (A matrix, 2x2)
    # Rows of U correspond to a00, a01, a10, a11
    # Columns correspond to the 7 Strassen multiplications
    U_coeffs = [
        [1, 0, 1, 0, 1, -1, 0],  # a00
        [0, 0, 0, 0, 1, 0, 1],   # a01
        [0, 1, 0, 0, 0, 1, 0],   # a10
        [1, 1, 0, 1, 0, 0, -1]   # a11
    ]
    
    # For each block
    for t in range(num_blocks):
        # Column offset for this block
        col_offset = t * 7
        
        # V coefficients for this block's B submatrix (2x2)
        # B block columns are 2t and 2t+1
        # V rows correspond to b00...b17 (row-major: b_{j,k} at j*p+k)
        # b_{0, 2t} is at index 0*p + 2t = 2t
        # b_{0, 2t+1} is at index 0*p + 2t+1 = 2t+1
        # b_{1, 2t} is at index 1*p + 2t = p + 2t
        # b_{1, 2t+1} is at index 1*p + 2t+1 = p + 2t+1
        
        # Strassen V vectors for the 7 terms
        # M1: b00 + b11
        # M2: b01
        # M3: b10 - b11
        # M4: b00 - b10
        # M5: b11
        # M6: b00 + b10
        # M7: b01 + b11
        
        # We iterate through the 7 terms
        for s in range(7):
            col = col_offset + s
            
            # Fill U
            for i in range(n * m):
                U[i][col] = U_coeffs[i][s]
                
            # Fill V
            # Initialize V column to 0 (already done by init, but we set specific values)
            # We need to reset V column to 0 first if we were reusing, but here it's fresh.
            # However, we only set non-zeros.
            
            # Helper to set V
            def set_v(row, val):
                V[row][col] = val
                
            if s == 0: # M1
                set_v(2*t, 1)
                set_v(p + 2*t + 1, 1)
            elif s == 1: # M2
                set_v(2*t + 1, 1)
            elif s == 2: # M3
                set_v(p + 2*t, 1)
                set_v(p + 2*t + 1, -1)
            elif s == 3: # M4
                set_v(2*t, 1)
                set_v(p + 2*t, -1)
            elif s == 4: # M5
                set_v(p + 2*t + 1, 1)
            elif s == 5: # M6
                set_v(2*t, 1)
                set_v(p + 2*t, 1)
            elif s == 6: # M7
                set_v(2*t + 1, 1)
                set_v(p + 2*t + 1, 1)
            
            # Fill W
            # W rows correspond to c_{i,k} at index k*n + i
            # Block t corresponds to C columns 2t and 2t+1
            # c_{0, 2t} at index 2t*n + 0 = 4t
            # c_{0, 2t+1} at index (2t+1)*n + 0 = 4t+2
            # c_{1, 2t} at index 2t*n + 1 = 4t+1
            # c_{1, 2t+1} at index (2t+1)*n + 1 = 4t+3
            
            def set_w(row, val):
                W[row][col] = val
                
            if s == 0: # M1
                set_w(4*t, 1)       # c00
                set_w(4*t + 3, 1)   # c11
            elif s == 1: # M2
                set_w(4*t + 2, 1)   # c01
                set_w(4*t + 3, -1)  # c11
            elif s == 2: # M3
                set_w(4*t + 1, 1)   # c10
                set_w(4*t + 3, 1)   # c11
            elif s == 3: # M4
                set_w(4*t, -1)      # c00
                set_w(4*t + 1, 1)   # c10
            elif s == 4: # M5
                set_w(4*t, 1)       # c00
            elif s == 5: # M6
                set_w(4*t + 3, 1)   # c11
            elif s == 6: # M7
                set_w(4*t, -1)      # c00
                set_w(4*t + 2, 1)   # c01
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
