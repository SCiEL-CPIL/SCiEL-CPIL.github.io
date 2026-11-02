"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Inner rank-7 decomposition for 2x2 matrix multiplication
    # Indices for 2x2: U (2x2=4), V (2x2=4), W (2x2=4)
    # W index: k*2 + i
    inner_U = [
        [1, 0, 0, 1], # M1
        [0, 0, 1, 1], # M2
        [1, 0, 0, 0], # M3
        [0, 0, 0, 1], # M4
        [1, 1, 0, 0], # M5
        [-1, 0, 1, 0],# M6
        [0, 1, 0, -1] # M7
    ]
    inner_V = [
        [1, 0, 0, 1], # M1
        [1, 0, 0, 0], # M2
        [0, 1, 0, -1],# M3
        [0, -1, 1, 0],# M4
        [0, 0, 0, 1], # M5
        [1, 1, 0, 0], # M6
        [0, 0, 1, 1]  # M7
    ]
    inner_W = [
        [1, 0, 0, 1], # M1
        [0, 1, 0, 0], # M2
        [0, 0, 1, 1], # M3
        [1, 1, 0, 0], # M4
        [-1, 0, 1, 0],# M5
        [0, 0, 0, 1], # M6
        [1, 0, 0, 0]  # M7
    ]
    
    # Outer Strassen coefficients for 4x4 (blocks 00, 01, 10, 11)
    # alpha: A blocks, beta: C blocks
    # Format: list of 7 dicts or tuples (alpha, beta)
    # alpha is dict {block_idx: coeff}, block_idx 0..3 (0=00, 1=01, 2=10, 3=11)
    # beta is dict {block_idx: coeff}
    
    outer_terms = [
        # 1
        ({0: 1, 3: 1}, {0: 1, 3: 1}),
        # 2
        ({2: 1, 3: 1}, {2: 1}),
        # 3
        ({0: 1}, {1: 1, 3: 1}),
        # 4
        ({3: 1}, {0: 1, 2: 1}),
        # 5
        ({0: 1, 1: 1}, {0: -1, 1: 1}),
        # 6
        ({2: 1, 0: -1}, {3: 1}),
        # 7
        ({1: 1, 3: -1}, {0: 1})
    ]
    
    rank = 98
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    # Generate 49 terms for first half of B (k=0..3)
    for k_out in range(7):
        alpha, beta = outer_terms[k_out]
        for r_in in range(7):
            u_in = inner_U[r_in]
            v_in = inner_V[r_in]
            w_in = inner_W[r_in]
            
            # Fill U
            # U index: i*m + j. i,j in 0..3.
            # Block I,J in 0..1. i = I*2 + i', j = J*2 + j'.
            # u_in index: i'*2 + j'.
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha:
                        coeff = alpha[block_idx]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = u_in[i_prime * 2 + j_prime]
                                if u_val != 0:
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    U[i * m + j][col] += coeff * u_val
                                    
            # Fill V
            # V index: j*p + k. j in 0..3, k in 0..7.
            # First half: k in 0..3.
            # Block I,J for B? B is 4x4.
            # B blocks: B_00, B_01, B_10, B_11.
            # j = I*2 + j', k = J*2 + k'.
            # v_in index: j'*2 + k'.
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha: # Same alpha for B? 
                       # Wait, Strassen terms use specific linear combos for A and B.
                       # M1: (A11+A22)(B11+B22). alpha_A = {00:1, 22:1}, alpha_B = {00:1, 22:1}.
                       # My outer_terms tuple was (alpha, beta). I need alpha_B too.
                       pass
            
            # Re-evaluating outer terms structure
            # I need alpha_A and alpha_B separately.
            
            # Let's redefine outer_terms properly.
            
            # 1: A: 00,11 ; B: 00,11 ; C: 00,11
            # 2: A: 10,11 ; B: 00 ; C: 10
            # 3: A: 00 ; B: 01,-11 ; C: 01,11
            # 4: A: 11 ; B: 10,-01 ; C: 00,10
            # 5: A: 00,01 ; B: 11 ; C: 00,-01 (Wait, C11 gets -M5, C12 gets +M5)
            #    C blocks: 00 coeff -1, 01 coeff 1.
            # 6: A: 10,-00 ; B: 00,01 ; C: 11
            # 7: A: 01,-11 ; B: 10,11 ; C: 00
            
            # I will hardcode the alpha_A and alpha_B in the loop below to avoid confusion.
            break
        break
        
    # Redefining with explicit alpha_A, alpha_B
    outer_defs = [
        # (alpha_A, alpha_B, beta_C)
        ({0: 1, 3: 1}, {0: 1, 3: 1}, {0: 1, 3: 1}),
        ({2: 1, 3: 1}, {0: 1}, {2: 1}),
        ({0: 1}, {1: 1, 3: -1}, {1: 1, 3: 1}),
        ({3: 1}, {2: 1, 1: -1}, {0: 1, 2: 1}),
        ({0: 1, 1: 1}, {3: 1}, {0: -1, 1: 1}),
        ({2: 1, 0: -1}, {0: 1, 1: 1}, {3: 1}),
        ({1: 1, 3: -1}, {2: 1, 3: 1}, {0: 1})
    ]

    col = 0
    for k_out in range(7):
        alpha_A, alpha_B, beta_C = outer_defs[k_out]
        for r_in in range(7):
            u_in = inner_U[r_in]
            v_in = inner_V[r_in]
            w_in = inner_W[r_in]
            
            # Fill U
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha_A:
                        coeff = alpha_A[block_idx]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = u_in[i_prime * 2 + j_prime]
                                if u_val != 0:
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    U[i * m + j][col] += coeff * u_val
                                    
            # Fill V (First half: k=0..3)
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha_B:
                        coeff = alpha_B[block_idx]
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_val = v_in[j_prime * 2 + k_prime]
                                if v_val != 0:
                                    j = I * 2 + j_prime
                                    k = J * 2 + k_prime
                                    V[j * p + k][col] += coeff * v_val
                                    
            # Fill W (First half: k=0..3)
            # W index: k*n + i
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in beta_C:
                        coeff = beta_C[block_idx]
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_val = w_in[k_prime * 2 + i_prime]
                                if w_val != 0:
                                    k = J * 2 + k_prime
                                    i = I * 2 + i_prime
                                    W[k * n + i][col] += coeff * w_val
                                    
            col += 1
            
    # Duplicate for second half (k=4..7)
    for k_out in range(7):
        alpha_A, alpha_B, beta_C = outer_defs[k_out]
        for r_in in range(7):
            u_in = inner_U[r_in]
            v_in = inner_V[r_in]
            w_in = inner_W[r_in]
            
            # Fill U (Same)
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha_A:
                        coeff = alpha_A[block_idx]
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_val = u_in[i_prime * 2 + j_prime]
                                if u_val != 0:
                                    i = I * 2 + i_prime
                                    j = J * 2 + j_prime
                                    U[i * m + j][col] += coeff * u_val
                                    
            # Fill V (Second half: k=4..7)
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in alpha_B:
                        coeff = alpha_B[block_idx]
                        for j_prime in range(2):
                            for k_prime in range(2):
                                v_val = v_in[j_prime * 2 + k_prime]
                                if v_val != 0:
                                    j = I * 2 + j_prime
                                    k = 4 + J * 2 + k_prime
                                    V[j * p + k][col] += coeff * v_val
                                    
            # Fill W (Second half: k=4..7)
            for I in range(2):
                for J in range(2):
                    block_idx = I * 2 + J
                    if block_idx in beta_C:
                        coeff = beta_C[block_idx]
                        for k_prime in range(2):
                            for i_prime in range(2):
                                w_val = w_in[k_prime * 2 + i_prime]
                                if w_val != 0:
                                    k = 4 + J * 2 + k_prime
                                    i = I * 2 + i_prime
                                    W[k * n + i][col] += coeff * w_val
                                    
            col += 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
