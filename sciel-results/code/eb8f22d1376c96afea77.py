"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # n=2, m=4, p=6
    # Rank 42 decomposition using 3 * 2 * 7 = 42 terms
    # 3 blocks for p (size 2 each), 2 blocks for m (size 2 each)
    # Each block is a 2x2x2 multiplication using Strassen's algorithm (rank 7)
    
    rank = 42
    U = [[0] * rank for _ in range(n * m)] # 8 x 42
    V = [[0] * rank for _ in range(m * p)] # 24 x 42
    W = [[0] * rank for _ in range(p * n)] # 12 x 42
    
    # Strassen coefficients for 2x2x2
    # M1 = (a11+a22)(b11+b22)
    # M2 = (a21+a22)b11
    # M3 = a11(b12-b22)
    # M4 = a22(b21-b11)
    # M5 = (a11+a12)b22
    # M6 = (a21-a11)(b11+b12)
    # M7 = (a12-a22)(b21+b22)
    
    # U coefficients for a11, a12, a21, a22
    # V coefficients for b11, b12, b21, b22
    # W coefficients for c11, c12, c21, c22
    
    # Indices in local 2x2 matrix:
    # a11: 0, a12: 1, a21: 2, a22: 3
    # b11: 0, b12: 1, b21: 2, b22: 3
    
    # Strassen steps: (U_coeffs, V_coeffs)
    # U_coeffs is dict {local_a_idx: coeff}
    # V_coeffs is dict {local_b_idx: coeff}
    
    strassen_steps = [
        ({0: 1, 3: 1}, {0: 1, 3: 1}), # M1
        ({2: 1, 3: 1}, {0: 1}),       # M2
        ({0: 1}, {1: 1, 3: -1}),      # M3
        ({3: 1}, {2: 1, 0: -1}),      # M4
        ({0: 1, 1: 1}, {3: 1}),       # M5
        ({2: 1, 0: -1}, {0: 1, 1: 1}),# M6
        ({1: 1, 3: -1}, {2: 1, 3: 1}) # M7
    ]
    
    # W coefficients: for each step m (0-6), contribution to c11, c12, c21, c22
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # W_coeffs[m] is dict {local_c_idx: coeff}
    # local_c_idx: 0=c11, 1=c12, 2=c21, 3=c22
    
    W_coeffs = [
        {0: 1},                # M1 -> c11
        {2: 1},                # M2 -> c21
        {1: 1, 3: 1},          # M3 -> c12, c22
        {0: 1, 2: 1},          # M4 -> c11, c21
        {0: -1, 1: 1},         # M5 -> c11, c12
        {3: 1},                # M6 -> c22
        {0: 1}                 # M7 -> c11
    ]
    # Wait, let's re-verify c22
    # c22 = M1 - M2 + M3 + M6
    # M1: c22 += 1? No, standard Strassen:
    # c11 = M1 + M4 - M5 + M7
    # c12 = M3 + M5
    # c21 = M2 + M4
    # c22 = M1 - M2 + M3 + M6
    
    # Let's re-list W_coeffs carefully:
    # m=0 (M1): c11+=1, c22+=1
    # m=1 (M2): c21+=1, c22-=-1 -> c22+=-1
    # m=2 (M3): c12+=1, c22+=1
    # m=3 (M4): c11+=1, c21+=1
    # m=4 (M5): c11-=-1 -> c11+=-1, c12+=1
    # m=5 (M6): c22+=1
    # m=6 (M7): c11+=1
    
    W_coeffs = [
        {0: 1, 3: 1},       # M1
        {2: 1, 3: -1},      # M2
        {1: 1, 3: 1},       # M3
        {0: 1, 2: 1},       # M4
        {0: -1, 1: 1},      # M5
        {3: 1},             # M6
        {0: 1}              # M7
    ]

    col = 0
    for q in range(3): # 3 blocks of width 2 in p
        for s in range(2): # 2 blocks of height 2 in m
            for m_idx in range(7):
                # Map local indices to global indices
                # A is 2x4. Local a_ij corresponds to A[i, 2s+j]
                # B is 4x6. Local b_ij corresponds to B[2s+i, 2q+j]
                # C is 2x6. Local c_ij corresponds to C[i, 2q+j]
                
                u_coeffs = strassen_steps[m_idx][0]
                v_coeffs = strassen_steps[m_idx][1]
                w_coeffs = W_coeffs[m_idx]
                
                # Update U
                for la, coeff in u_coeffs.items():
                    # la: 0=a11, 1=a12, 2=a21, 3=a22
                    # i = la // 2, j_local = la % 2
                    # global j = 2s + j_local
                    i = la // 2
                    j_local = la % 2
                    j_global = 2 * s + j_local
                    u_row = i * m + j_global
                    U[u_row][col] += coeff
                
                # Update V
                for lb, coeff in v_coeffs.items():
                    # lb: 0=b11, 1=b12, 2=b21, 3=b22
                    # i_local = lb // 2, j_local = lb % 2
                    # global j = 2s + i_local
                    # global k = 2q + j_local
                    i_local = lb // 2
                    j_local = lb % 2
                    j_global = 2 * s + i_local
                    k_global = 2 * q + j_local
                    v_row = j_global * p + k_global
                    V[v_row][col] += coeff
                
                # Update W
                for lc, coeff in w_coeffs.items():
                    # lc: 0=c11, 1=c12, 2=c21, 3=c22
                    # i = lc // 2, k_local = lc % 2
                    # global k = 2q + k_local
                    i = lc // 2
                    k_local = lc % 2
                    k_global = 2 * q + k_local
                    w_row = k_global * n + i
                    W[w_row][col] += coeff
                
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
