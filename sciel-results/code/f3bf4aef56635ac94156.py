import time

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Deadline guard
    start_time = time.time()
    timeout = 120.0
    
    # Strassen-based decomposition: partition p=12 columns into 6 pairs
    # Each pair uses 7 rank-1 terms (Strassen for 2x2x2 matrix mult)
    # Total rank = 6 * 7 = 42
    
    num_blocks = p // 2  # 6 blocks
    rank = num_blocks * 7  # 42
    
    # Initialize U, V, W as lists of lists
    # U: (n*m, rank) -> (4, 42)
    # V: (m*p, rank) -> (24, 42)
    # W: (p*n, rank) -> (24, 42)
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen 7-product decomposition for 2x2x2
    # Indices for A (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3
    # Indices for B (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    # Indices for C (2x2): 00, 01, 10, 11 -> 0, 1, 2, 3 (local)
    
    # Strassen terms (u, v, w) coefficients
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B01
    # M3 = (A01 + A11) * B10
    # M4 = A11 * (B00 - B10)
    # M5 = (A00 + A01) * (B10 - B11)
    # M6 = (A10 - A00) * (B00 + B01)
    # M7 = (A01 - A11) * (B11 + B10)
    
    # C00 = M1 + M4 - M5 + M7
    # C01 = M2 + M3 - M6 + M7
    # C10 = M2 + M4 - M5 + M6
    # C11 = M1 - M3 + M5 + M6
    
    strassen_terms = [
        # M1
        {0: 1}, {0: 1, 3: 1}, {0: 1, 2: -1, 3: 1}, # Wait, standard Strassen W coefficients
        # Let's use the explicit list from the current program which was verified to work (score 1.1429)
    ]
    
    # Re-using the verified coefficients from the current program
    strassen_terms = [
        {0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1},       # M1 (U, V, W) - Wait, the current program has 3 dicts per term?
        # Current program structure:
        # strassen_terms[s*3] -> U coeffs
        # strassen_terms[s*3+1] -> V coeffs
        # strassen_terms[s*3+2] -> W coeffs
        
        # Let's copy the exact structure from the current program to ensure correctness
    ]
    
    strassen_terms = [
        # M1
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}, 
        {0: 1, 3: 1}, 
        # M2
        {2: 1, 3: 1}, 
        {0: 1},        
        {2: 1, 3: -1}, 
        # M3
        {0: 1},       
        {1: 1, 3: -1}, 
        {1: 1, 3: 1},  
        # M4
        {3: 1},       
        {2: 1, 0: -1}, 
        {0: 1, 2: 1},  
        # M5
        {0: 1, 1: 1}, 
        {3: 1},        
        {0: -1, 1: 1}, 
        # M6
        {0: -1, 2: 1}, 
        {0: 1, 1: 1},  
        {3: 1},        
        # M7
        {1: 1, 3: -1}, 
        {2: 1, 3: 1},  
        {0: 1},        
    ]
    
    for t in range(num_blocks):
        if time.time() - start_time > timeout - 5:
            break
        for s in range(7):
            col = t * 7 + s
            u_coeffs = strassen_terms[s * 3]
            v_coeffs = strassen_terms[s * 3 + 1]
            w_coeffs = strassen_terms[s * 3 + 2]
            
            # Set U
            for a_idx, val in u_coeffs.items():
                U[a_idx][col] = val
            
            # Set V
            for local_b, val in v_coeffs.items():
                if local_b == 0:
                    global_b = 2 * t
                elif local_b == 1:
                    global_b = 2 * t + 1
                elif local_b == 2:
                    global_b = m * p // 2 + 2 * t
                else:
                    global_b = m * p // 2 + 2 * t + 1
                V[global_b][col] = val
            
            # Set W
            for local_c, val in w_coeffs.items():
                if local_c == 0:
                    global_w = (2 * t) * n + 0
                elif local_c == 1:
                    global_w = (2 * t + 1) * n + 0
                elif local_c == 2:
                    global_w = (2 * t) * n + 1
                else:
                    global_w = (2 * t + 1) * n + 1
                W[global_w][col] += val
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
