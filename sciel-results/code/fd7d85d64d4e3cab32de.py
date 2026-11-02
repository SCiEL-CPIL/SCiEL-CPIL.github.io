import numpy as np

DIMENSIONS = [4, 4, 9]

def construct():
    n, m, p = DIMENSIONS
    
    # Strassen 2x2x2 rank 7
    # U, V, W for 2x2x2
    # Indices: u in 0..3, v in 0..3, w in 0..3
    # r in 0..6
    S2_U = np.zeros((4, 7), dtype=float)
    S2_V = np.zeros((4, 7), dtype=float)
    S2_W = np.zeros((4, 7), dtype=float)
    
    # M1: (A00+A11)(B00+B11)
    # A00: u=0, A11: u=3
    # B00: v=0, B11: v=3
    S2_U[0, 0] = 1; S2_U[3, 0] = 1
    S2_V[0, 0] = 1; S2_V[3, 0] = 1
    S2_W[0, 0] = 1; S2_W[3, 0] = 1 # C00, C11 get M1
    
    # M2: (A10+A11)B00
    # A10: u=2, A11: u=3
    # B00: v=0
    S2_U[2, 1] = 1; S2_U[3, 1] = 1
    S2_V[0, 1] = 1
    S2_W[2, 1] = 1; S2_W[3, 1] = 1 # C10, C11 get M2 (with sign)
    
    # M3: A00(B01-B11)
    # A00: u=0
    # B01: v=1, B11: v=3
    S2_U[0, 2] = 1
    S2_V[1, 2] = 1; S2_V[3, 2] = -1
    S2_W[1, 2] = 1; S2_W[3, 2] = 1 # C01, C11 get M3
    
    # M4: A11(B10-B00)
    # A11: u=3
    # B10: v=2, B00: v=0
    S2_U[3, 3] = 1
    S2_V[2, 3] = 1; S2_V[0, 3] = -1
    S2_W[0, 3] = 1; S2_W[2, 3] = 1 # C00, C10 get M4
    
    # M5: (A00+A01)B11
    # A00: u=0, A01: u=1
    # B11: v=3
    S2_U[0, 4] = 1; S2_U[1, 4] = 1
    S2_V[3, 4] = 1
    S2_W[0, 4] = -1; S2_W[3, 4] = 1 # C00, C11 get M5 (C00 with -1)
    
    # M6: (A10-A00)(B00+B01)
    # A10: u=2, A00: u=0
    # B00: v=0, B01: v=1
    S2_U[2, 5] = 1; S2_U[0, 5] = -1
    S2_V[0, 5] = 1; S2_V[1, 5] = 1
    S2_W[3, 5] = 1 # C11 gets M6
    
    # M7: (A01-A11)(B10+B11)
    # A01: u=1, A11: u=3
    # B10: v=2, B11: v=3
    S2_U[1, 6] = 1; S2_U[3, 6] = -1
    S2_V[2, 6] = 1; S2_V[3, 6] = 1
    S2_W[0, 6] = 1 # C00 gets M7

    # Check W signs for C
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    
    # Let's re-verify W assignments in S2_W
    # r=0 (M1): C00 (+1), C11 (+1). S2_W[0,0]=1, S2_W[3,0]=1. OK.
    # r=1 (M2): C10 (+1), C11 (-1). S2_W[2,1]=1, S2_W[3,1]=-1. 
    #   My code above: S2_W[2, 1] = 1; S2_W[3, 1] = 1. WRONG. Should be -1 for C11.
    #   Wait, C11 = M1 - M2 + ...
    #   So S2_W[3, 1] should be -1.
    
    # r=2 (M3): C01 (+1), C11 (+1). S2_W[1, 2]=1, S2_W[3, 2]=1. OK.
    # r=3 (M4): C00 (+1), C10 (+1). S2_W[0, 3]=1, S2_W[2, 3]=1. OK.
    # r=4 (M5): C00 (-1), C11 (+1). S2_W[0, 4]=-1, S2_W[3, 4]=1. OK.
    # r=5 (M6): C11 (+1). S2_W[3, 5]=1. OK.
    # r=6 (M7): C00 (+1). S2_W[0, 6]=1. OK.
    
    # Correcting S2_W[3, 1] to -1.
    S2_W[3, 1] = -1

    # Now build 4x4x4 using S2
    # Rank 49
    S4_U = np.zeros((16, 49), dtype=float)
    S4_V = np.zeros((16, 49), dtype=float)
    S4_W = np.zeros((16, 49), dtype=float)
    
    # Coefficients for A, B, C blocks
    # c[r][I][J], d[r][I][J], e[r][I][J]
    # r in 0..6
    # I, J in 0..1
    
    # M1: A: 00(1), 11(1); B: 00(1), 11(1)
    # M2: A: 10(1), 11(1); B: 00(1)
    # M3: A: 00(1); B: 01(1), 11(-1)
    # M4: A: 11(1); B: 10(1), 00(-1)
    # M5: A: 00(1), 01(1); B: 11(1)
    # M6: A: 10(1), 00(-1); B: 00(1), 01(1)
    # M7: A: 01(1), 11(-1); B: 10(1), 11(1)
    
    # C coeffs:
    # C00: M1(1), M4(1), M5(-1), M7(1)
    # C01: M3(1), M5(1)
    # C10: M2(1), M4(1)
    # C11: M1(1), M2(-1), M3(1), M6(1)
    
    c_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]], # M6
        [[0, 1], [0, -1]]  # M7
    ]
    d_coeffs = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]], # M3
        [[-1, 0], [1, 0]], # M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 1], [0, 1]]  # M7
    ]
    e_coeffs = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 0], [1, -1]], # M2 -> C10, C11
        [[0, 1], [0, 1]], # M3 -> C01, C11
        [[1, 0], [1, 0]], # M4 -> C00, C10
        [[-1, 0], [0, 1]], # M5 -> C00, C11
        [[0, 0], [0, 1]], # M6 -> C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    
    for r in range(7):
        for s in range(7):
            R = r * 7 + s
            # For each block in A
            for I in range(2):
                for J in range(2):
                    coeff_A = c_coeffs[r][I][J]
                    if coeff_A != 0:
                        # Map to 4x4 indices
                        # Block (I,J) contains indices (2I+i', 2J+j')
                        for i_prime in range(2):
                            for j_prime in range(2):
                                u_idx = (2*I + i_prime) * 4 + (2*J + j_prime)
                                s2_u_val = S2_U[i_prime * 2 + j_prime, s]
                                S4_U[u_idx, R] += coeff_A * s2_u_val
                                
            for I in range(2):
                for J in range(2):
                    coeff_B = d_coeffs[r][I][J]
                    if coeff_B != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                v_idx = (2*I + i_prime) * 4 + (2*J + j_prime)
                                s2_v_val = S2_V[i_prime * 2 + j_prime, s]
                                S4_V[v_idx, R] += coeff_B * s2_v_val
                                
            for I in range(2):
                for J in range(2):
                    coeff_C = e_coeffs[r][I][J]
                    if coeff_C != 0:
                        for i_prime in range(2):
                            for j_prime in range(2):
                                w_idx = (2*J + j_prime) * 4 + (2*I + i_prime)
                                s2_w_val = S2_W[i_prime * 2 + j_prime, s]
                                S4_W[w_idx, R] += coeff_C * s2_w_val

    # Now build 4x4x9
    # U: 16 x 114
    # V: 36 x 114
    # W: 36 x 114
    
    Rank = 114
    U = np.zeros((16, Rank), dtype=float)
    V = np.zeros((36, Rank), dtype=float)
    W = np.zeros((36, Rank), dtype=float)
    
    # Part 1: B1 (4x4) -> C1 (4x4)
    # Uses S4. Rank offset 0.
    U[:, 0:49] = S4_U
    # V for B1: indices j*9 + k, k in 0..3
    for j in range(4):
        for k in range(4):
            V[j*9 + k, 0:49] = S4_V[j*4 + k, 0:49]
    # W for C1: indices k*4 + i, k in 0..3
    for k in range(4):
        for i in range(4):
            W[k*4 + i, 0:49] = S4_W[k*4 + i, 0:49]
            
    # Part 2: B2 (4x4) -> C2 (4x4)
    # Uses S4. Rank offset 49.
    U[:, 49:98] = S4_U
    # V for B2: indices j*9 + k, k in 4..7
    for j in range(4):
        for k in range(4, 8):
            V[j*9 + k, 49:98] = S4_V[j*4 + (k-4), 0:49]
    # W for C2: indices k*4 + i, k in 4..7
    for k in range(4, 8):
        for i in range(4):
            W[k*4 + i, 49:98] = S4_W[(k-4)*4 + i, 0:49]
            
    # Part 3: B3 (4x1) -> C3 (4x1)
    # Schoolbook. Rank offset 98.
    # Rank 16.
    for i in range(4):
        for j in range(4):
            r = 98 + i*4 + j
            U[i*4 + j, r] = 1
            V[j*9 + 8, r] = 1
            W[8*4 + i, r] = 1
            
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
