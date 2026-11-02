"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 2, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 21
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen coefficients for 2x2x2
    # M1: (A00+A11)(B00+B11)
    # M2: (A10+A11)B00
    # M3: A00(B01-B11)
    # M4: A11(B10-B00)
    # M5: (A00+A10)B11
    # M6: (A11-A00)(B00+B01)
    # M7: (A00-A11)(B11-B10)
    
    # U vectors (i, j)
    # V vectors (j, k)
    # W vectors (k, i)
    
    # Note: U index is i*m+j
    # V index is j*p+k
    # W index is k*n+i
    
    # Define the 7 patterns for one 2x2 block
    # Each pattern is a dict of non-zero entries for U, V, W
    # But W depends on the output position (k, i)
    
    # Let's define U and V for the 7 products
    # U[r] is a vector of length n*m (4)
    # V[r] is a vector of length m*p (12)
    # W[r] is a vector of length p*n (12)
    
    # We will fill 7 columns per group g (0, 1, 2)
    
    # U patterns (i, j)
    u_patterns = [
        {(0,0): 1, (1,1): 1},       # M1
        {(1,0): 1, (1,1): 1},       # M2
        {(0,0): 1},                 # M3
        {(1,1): 1},                 # M4
        {(0,0): 1, (1,0): 1},       # M5
        {(1,1): 1, (0,0): -1},      # M6
        {(0,0): 1, (1,1): -1}       # M7
    ]
    
    # V patterns (j, k_local) where k_local is 0 or 1 within the block
    # We need to map k_local to global k = 2*g + k_local
    v_patterns_local = [
        {(0,0): 1, (1,1): 1},       # M1
        {(0,0): 1},                 # M2
        {(0,1): 1, (1,1): -1},      # M3
        {(1,0): 1, (0,0): -1},      # M4
        {(1,1): 1},                 # M5
        {(0,0): 1, (0,1): 1},       # M6
        {(1,1): 1, (1,0): -1}       # M7
    ]
    
    # W patterns (k_local, i) -> coeff
    # C00 (i=0, k=0): M1+M4-M5+M7
    # C01 (i=0, k=1): M3+M5
    # C10 (i=1, k=0): M2+M4
    # C11 (i=1, k=1): M1-M2+M3+M6
    
    w_coeffs = [
        # M1
        {(0,0): 1, (1,1): 1},
        # M2
        {(0,1): 1, (1,1): -1},
        # M3
        {(1,0): 1, (1,1): 1},
        # M4
        {(0,0): 1, (0,1): 1},
        # M5
        {(0,0): -1, (1,0): 1},
        # M6
        {(1,1): 1},
        # M7
        {(0,0): 1}
    ]
    
    # Construct the matrices
    for g in range(3):
        base_col = g * 7
        for t in range(7):
            col = base_col + t
            
            # Fill U
            for (i, j), val in u_patterns[t].items():
                U[i * m + j][col] = val
            
            # Fill V
            for (j, k_local), val in v_patterns_local[t].items():
                k_global = 2 * g + k_local
                V[j * p + k_global][col] = val
            
            # Fill W
            for (k_local, i), val in w_coeffs[t].items():
                k_global = 2 * g + k_local
                W[k_global * n + i][col] = val
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
