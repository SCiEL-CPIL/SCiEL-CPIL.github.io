DIMENSIONS = [2, 2, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 25
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Helper to add Strassen block for 2 columns starting at k_start
    def add_strassen(k_start):
        nonlocal col
        # Local A indices 0..3
        # Local B indices 0..3 (j*2 + k_local)
        # Local C indices 0..3 (k_local*n + i)
        
        # Term 1: M1 = (a+d)(x+w)
        U[0][col] += 1; U[3][col] += 1
        V[0][col] += 1; V[3][col] += 1
        W[0][col] += 1; W[3][col] += 1 # C00, C11
        # Map to global
        # U is global A, no shift
        # V: j*p + k. k = k_start + k_local
        # W: k*n + i. k = k_start + k_local
        
        # We need to map local V/W indices to global.
        # It's easier to just use a general add_term function.
        pass
        
    # Let's just hardcode the 25 terms or generate them properly.
    # Generating is safer.
    
    def add_term(u_coeffs, v_local_coeffs, w_local_coeffs, k_start, v_width, w_width):
        nonlocal col
        # u_coeffs: list of 4
        # v_local_coeffs: list of v_width
        # w_local_coeffs: list of w_width
        
        for i in range(4):
            if u_coeffs[i] != 0:
                U[i][col] += u_coeffs[i]
                
        for j in range(2):
            for k_loc in range(v_width // 2):
                idx = j * (v_width // 2) + k_loc
                if v_local_coeffs[idx] != 0:
                    global_k = k_start + k_loc
                    V[j * p + global_k][col] += v_local_coeffs[idx]
                    
        for k_loc in range(w_width // 2):
            for i_loc in range(2):
                idx = k_loc * 2 + i_loc
                if w_local_coeffs[idx] != 0:
                    global_k = k_start + k_loc
                    W[global_k * n + i_loc][col] += w_local_coeffs[idx]
                    
        col += 1

    # Strassen coefficients for M_{2,2,2}
    # u, v, w local
    strassen_terms = [
        # M1
        ([1, 0, 0, 1], [1, 0, 0, 1], [1, 0, 0, 1]),
        # M2
        ([0, 0, 1, 1], [1, 0, 0, 0], [0, 0, 1, -1]),
        # M3
        ([1, 0, 0, 0], [0, 1, 0, -1], [0, 1, 0, 1]),
        # M4
        ([0, 0, 0, 1], [-1, 0, 1, 0], [1, 0, 1, 0]),
        # M5
        ([1, 1, 0, 0], [0, 0, 0, 1], [-1, 1, 0, 0]),
        # M6
        ([-1, 0, 1, 0], [1, 0, 1, 0], [0, 0, 0, 1]),
        # M7
        ([0, 1, 0, -1], [0, 1, 0, 1], [1, 0, 0, 0]),
    ]
    
    # Trivial terms for M_{2,2,1}
    trivial_terms = [
        ([1, 0, 0, 0], [1, 0], [1, 0]), # a*x
        ([0, 1, 0, 0], [0, 1], [1, 0]), # b*z
        ([0, 0, 1, 0], [1, 0], [0, 1]), # c*x
        ([0, 0, 0, 1], [0, 1], [0, 1]), # d*z
    ]
    
    # Block 1: k=0,1 (Strassen)
    for t in strassen_terms:
        add_term(t[0], t[1], t[2], 0, 4, 4)
        
    # Block 2: k=2 (Trivial)
    for t in trivial_terms:
        add_term(t[0], t[1], t[2], 2, 2, 2)
        
    # Block 3: k=3,4 (Strassen)
    for t in strassen_terms:
        add_term(t[0], t[1], t[2], 3, 4, 4)
        
    # Block 4: k=5,6 (Strassen)
    for t in strassen_terms:
        add_term(t[0], t[1], t[2], 5, 4, 4)
        
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
