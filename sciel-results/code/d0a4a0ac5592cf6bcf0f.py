"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=5
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    def add_strassen(j_s, k_s):
        nonlocal col
        # Local A indices: 00, 01, 10, 11
        # Global U indices
        u_00 = 0 * m + j_s
        u_01 = 0 * m + (j_s + 1)
        u_10 = 1 * m + j_s
        u_11 = 1 * m + (j_s + 1)
        
        # Local B indices: 00, 01, 10, 11
        # Global V indices
        v_00 = j_s * p + k_s
        v_01 = j_s * p + (k_s + 1)
        v_10 = (j_s + 1) * p + k_s
        v_11 = (j_s + 1) * p + (k_s + 1)
        
        # Local C indices: 00, 01, 10, 11
        # Global W indices
        w_00 = k_s * n + 0
        w_01 = k_s * n + 1
        w_10 = k_s * n + 2
        w_11 = k_s * n + 3
        
        # Strassen M1-M7 definitions
        # U_vec, V_vec, W_vec (coeffs for C00, C01, C10, C11)
        strassen_blocks = [
            # M1
            ([(u_00, 1), (u_11, 1)], [(v_00, 1), (v_11, 1)], [(w_00, 1), (w_11, 1)]),
            # M2
            ([(u_10, 1), (u_11, 1)], [(v_00, 1)], [(w_10, 1), (w_11, -1)]),
            # M3
            ([(u_00, 1)], [(v_01, 1), (v_11, -1)], [(w_01, 1), (w_11, 1)]),
            # M4
            ([(u_11, 1)], [(v_10, 1), (v_00, -1)], [(w_00, 1), (w_10, 1)]),
            # M5
            ([(u_00, 1), (u_01, 1)], [(v_11, 1)], [(w_00, -1), (w_01, 1)]),
            # M6
            ([(u_10, 1), (u_00, -1)], [(v_00, 1), (v_01, 1)], [(w_11, 1)]),
            # M7
            ([(u_01, 1), (u_11, -1)], [(v_10, 1), (v_11, 1)], [(w_00, 1)])
        ]
        
        for u_terms, v_terms, w_terms in strassen_blocks:
            for idx, val in u_terms:
                U[idx][col] += val
            for idx, val in v_terms:
                V[idx][col] += val
            for idx, val in w_terms:
                W[idx][col] += val
            col += 1

    def add_schoolbook_2x2x1(j_s, k_s):
        nonlocal col
        # Local A indices: 00, 01, 10, 11
        u_00 = 0 * m + j_s
        u_01 = 0 * m + (j_s + 1)
        u_10 = 1 * m + j_s
        u_11 = 1 * m + (j_s + 1)
        
        # Local B indices: 00, 10 (c is fixed 0)
        v_00 = j_s * p + k_s
        v_10 = (j_s + 1) * p + k_s
        
        # Local C indices: 00, 10 (c is fixed 0)
        w_00 = k_s * n + 0
        w_10 = k_s * n + 2
        
        # Terms:
        # 1. A00 B00 -> C00
        U[u_00][col] = 1
        V[v_00][col] = 1
        W[w_00][col] = 1
        col += 1
        
        # 2. A01 B10 -> C00
        U[u_01][col] = 1
        V[v_10][col] = 1
        W[w_00][col] = 1
        col += 1
        
        # 3. A10 B00 -> C10
        U[u_10][col] = 1
        V[v_00][col] = 1
        W[w_10][col] = 1
        col += 1
        
        # 4. A11 B10 -> C10
        U[u_11][col] = 1
        V[v_10][col] = 1
        W[w_10][col] = 1
        col += 1

    # Add 4 Strassen blocks (2x2x2)
    add_strassen(0, 0) # j in 0,1; k in 0,1
    add_strassen(0, 2) # j in 0,1; k in 2,3
    add_strassen(2, 0) # j in 2,3; k in 0,1
    add_strassen(2, 2) # j in 2,3; k in 2,3
    
    # Add 2 Schoolbook blocks (2x2x1)
    add_schoolbook_2x2x1(0, 4) # j in 0,1; k in 4
    add_schoolbook_2x2x1(2, 4) # j in 2,3; k in 4

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
