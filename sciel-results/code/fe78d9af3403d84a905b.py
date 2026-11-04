"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen blocks for 2x2x2
    # Local indices i, j, k in 0..1
    # M1: A11 * (B11 + B22)
    # M2: (A11 + A22) * B22
    # M3: (A12 - A22) * (B21 + B22)
    # M4: A11 * B21
    # M5: (A12 + A22) * B11
    # M6: (A12 - A11) * (B11 + B21)
    # M7: (A21 + A22) * (B22 - B21)
    
    # Coefficients for C11, C12, C21, C22
    # C11 = M1 - M4 + M5 - M6 + M7
    # C12 = M3 + M5
    # C21 = M2 + M4
    # C22 = M1 - M2 + M3 + M7
    
    # We represent each term r by U_r, V_r, W_r
    # U_r is a 2x2 matrix (flattened to 4)
    # V_r is a 2x2 matrix (flattened to 4)
    # W_r is a 2x2 matrix (flattened to 4)
    
    strassen_terms = []
    
    # Helper to add a term
    def add_term(u, v, w):
        strassen_terms.append((u, v, w))
        
    # M1
    u = [1, 0, 0, 0] # A11
    v = [1, 1, 0, 0] # B11 + B22 (B11 at 0, B22 at 1? No, B is 2x2. B11=0, B12=1, B21=2, B22=3)
    # V indices: j, k. B_jk.
    # B11 is j=0, k=0. B22 is j=1, k=1.
    v = [1, 0, 0, 1] # B11 + B22
    w = [1, 0, 0, 0] # C11 coeff 1
    add_term(u, v, w)
    
    # M2
    u = [1, 0, 0, 1] # A11 + A22
    v = [0, 0, 0, 1] # B22
    w = [0, 0, 1, 1] # C21 coeff 1, C22 coeff 1
    add_term(u, v, w)
    
    # M3
    u = [0, 1, 0, -1] # A12 - A22
    v = [0, 0, 1, 1] # B21 + B22
    w = [0, 1, 0, 1] # C12 coeff 1, C22 coeff 1
    add_term(u, v, w)
    
    # M4
    u = [1, 0, 0, 0] # A11
    v = [0, 0, 1, 0] # B21
    w = [-1, 0, 1, 0] # C11 coeff -1, C21 coeff 1
    add_term(u, v, w)
    
    # M5
    u = [0, 1, 0, 1] # A12 + A22
    v = [1, 0, 0, 0] # B11
    w = [1, 1, 0, 0] # C11 coeff 1, C12 coeff 1
    add_term(u, v, w)
    
    # M6
    u = [0, 1, -1, 0] # A12 - A11
    v = [1, 0, 1, 0] # B11 + B21
    w = [-1, 0, 0, 0] # C11 coeff -1
    add_term(u, v, w)
    
    # M7
    u = [0, 0, 1, 1] # A21 + A22
    v = [0, 0, -1, 1] # B22 - B21
    w = [1, 0, 0, 1] # C11 coeff 1, C22 coeff 1
    add_term(u, v, w)
    
    # Initialize global matrices
    rank = 6 * 7 + 1
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    j_blocks = [(0, 1), (2, 3)] # 0-indexed: {0,1}, {2,3}
    k_blocks = [(0, 1), (2, 3), (4, 5)]
    
    for j1, j2 in j_blocks:
        for k1, k2 in k_blocks:
            for u_loc, v_loc, w_loc in strassen_terms:
                # u_loc is 2x2 flattened (i, j_loc)
                # v_loc is 2x2 flattened (j_loc, k_loc)
                # w_loc is 2x2 flattened (k_loc, i)
                
                for i in range(2):
                    for j_l in range(2):
                        j_global = j1 + j_l
                        U[i * m + j_global][col] += u_loc[i * 2 + j_l]
                        
                for j_l in range(2):
                    for k_l in range(2):
                        j_global = j1 + j_l
                        k_global = k1 + k_l
                        V[j_global * p + k_global][col] += v_loc[j_l * 2 + k_l]
                        
                for k_l in range(2):
                    for i in range(2):
                        k_global = k1 + k_l
                        W[k_global * n + i][col] += w_loc[k_l * 2 + i]
                        
                col += 1
                
    # Add term for j=4 (0-indexed)
    # U[i, 4] = 1
    # V[4, k] = 1
    # W[k, i] = 1
    for i in range(n):
        U[i * m + 4][col] = 1
    for k in range(p):
        V[4 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
