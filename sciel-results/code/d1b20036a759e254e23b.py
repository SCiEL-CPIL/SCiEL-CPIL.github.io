"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Total rank is 6 * 7 = 42
    # 6 blocks of 2x2x2 Strassen
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Strassen decomposition for 2x2x2
    # U, V, W are 4x7 matrices
    # Indices: U[i*2+j], V[j*2+k], W[k*2+i]
    U_str = [[0]*7 for _ in range(4)]
    V_str = [[0]*7 for _ in range(4)]
    W_str = [[0]*7 for _ in range(4)]
    
    # M1 = (a+d)(e+h)
    # a=00, d=11, e=00, h=11
    U_str[0][0] = 1; U_str[3][0] = 1
    V_str[0][0] = 1; V_str[3][0] = 1
    W_str[0][0] = 1; W_str[3][0] = 1 # C00, C11
    
    # M2 = (c+d)e
    # c=10, d=11, e=00
    U_str[2][1] = 1; U_str[3][1] = 1
    V_str[0][1] = 1
    W_str[2][1] = 1; W_str[3][1] = -1 # C21, C22 (C10, C11 in 0-indexed k,i? No. W[k*2+i])
    # C indices: C00(0), C01(1), C10(2), C11(3) in W[k*2+i]
    # k=0,i=0 -> 0. k=0,i=1 -> 1. k=1,i=0 -> 2. k=1,i=1 -> 3.
    # M2 contributes to C10 (k=1,i=0) and C11 (k=1,i=1)?
    # C = AB.
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M2 contributes to C10 (+1) and C11 (-1).
    # C10 is k=1, i=0. Index 2.
    # C11 is k=1, i=1. Index 3.
    # So W_str[2][1] = 1, W_str[3][1] = -1. Correct.
    
    # M3 = a(f-h)
    # a=00, f=01, h=11
    U_str[0][2] = 1
    V_str[1][2] = 1; V_str[3][2] = -1
    # M3 contributes to C01 (+1) and C11 (+1).
    # C01 is k=0, i=1. Index 1.
    # C11 is k=1, i=1. Index 3.
    W_str[1][2] = 1; W_str[3][2] = 1
    
    # M4 = d(g-e)
    # d=11, g=10, e=00
    U_str[3][3] = 1
    V_str[2][3] = 1; V_str[0][3] = -1
    # M4 contributes to C00 (+1) and C10 (+1).
    # C00 is index 0. C10 is index 2.
    W_str[0][3] = 1; W_str[2][3] = 1
    
    # M5 = (a+b)h
    # a=00, b=01, h=11
    U_str[0][4] = 1; U_str[1][4] = 1
    V_str[3][4] = 1
    # M5 contributes to C00 (-1) and C01 (+1).
    # C00 index 0. C01 index 1.
    W_str[0][4] = -1; W_str[1][4] = 1
    
    # M6 = (c-a)(e+f)
    # c=10, a=00, e=00, f=01
    U_str[2][5] = 1; U_str[0][5] = -1
    V_str[0][5] = 1; V_str[1][5] = 1
    # M6 contributes to C11 (+1).
    # C11 index 3.
    W_str[3][5] = 1
    
    # M7 = (b-d)(g+h)
    # b=01, d=11, g=10, h=11
    U_str[1][6] = 1; U_str[3][6] = -1
    V_str[2][6] = 1; V_str[3][6] = 1
    # M7 contributes to C00 (+1).
    # C00 index 0.
    W_str[0][6] = 1
    
    # Define the 6 blocks
    # Block (i_offset, j_offset, k_offset)
    # i in {0,1}, j in {0,1}, k in {0,1}
    # Global i = i_offset + i_local
    # Global j = j_offset + j_local
    # Global k = k_offset + k_local
    
    blocks = [
        (0, 0, 0), # A_L, B_T1
        (0, 0, 2), # A_L, B_T2
        (0, 0, 4), # A_L, B_T3
        (0, 2, 0), # A_R, B_B1
        (0, 2, 2), # A_R, B_B2
        (0, 2, 4), # A_R, B_B3
    ]
    
    for b_idx, (i_off, j_off, k_off) in enumerate(blocks):
        col_offset = b_idx * 7
        for i_loc in range(2):
            for j_loc in range(2):
                for k_loc in range(2):
                    # Local indices for U, V, W
                    u_loc = i_loc * 2 + j_loc
                    v_loc = j_loc * 2 + k_loc
                    w_loc = k_loc * 2 + i_loc
                    
                    # Global indices
                    i_glob = i_off + i_loc
                    j_glob = j_off + j_loc
                    k_glob = k_off + k_loc
                    
                    u_glob = i_glob * m + j_glob
                    v_glob = j_glob * p + k_glob
                    w_glob = k_glob * n + i_loc # Wait, w_glob = k_glob * n + i_glob
                    w_glob = k_glob * n + i_glob
                    
                    # Copy values
                    for r_loc in range(7):
                        r_glob = col_offset + r_loc
                        if U_str[u_loc][r_loc] != 0:
                            U[u_glob][r_glob] = U_str[u_loc][r_loc]
                        if V_str[v_loc][r_loc] != 0:
                            V[v_glob][r_glob] = V_str[v_loc][r_loc]
                        if W_str[w_loc][r_loc] != 0:
                            W[w_glob][r_glob] = W_str[w_loc][r_loc]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
