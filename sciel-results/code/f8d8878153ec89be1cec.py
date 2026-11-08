"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 33 decomposition
    # Part 1: 2x1x6 block (rank 12)
    # Part 2: 2x2x6 block (rank 21) using Strassen on three 2x2x2 blocks
    
    rank = 33
    U = [[0] * rank for _ in range(n * m)]  # 6 x 33
    V = [[0] * rank for _ in range(m * p)]  # 18 x 33
    W = [[0] * rank for _ in range(p * n)]  # 12 x 33
    
    # Part 1: A_1 (2x1) * B_1 (1x6)
    # A indices: A[i, 0] for i in 0..1 -> idx i*3
    # B indices: B[0, k] for k in 0..5 -> idx k
    # C indices: C[i, k] -> W[k*n + i]
    
    r = 0
    for i in range(2):
        for k in range(6):
            # U
            U[i * 3][r] = 1
            # V
            V[k][r] = 1
            # W
            W[k * 2 + i][r] = 1
            r += 1
            
    # Part 2: A_2 (2x2) * B_2 (2x6)
    # A_2 indices:
    # A[0, 1] -> idx 1
    # A[0, 2] -> idx 2
    # A[1, 1] -> idx 4
    # A[1, 2] -> idx 5
    
    a_idx = 1
    b_idx = 2
    c_idx = 4
    d_idx = 5
    
    # B_2 is rows 1,2 of B (0-based 1,2)
    # B[1, k] -> idx 6 + k
    # B[2, k] -> idx 12 + k
    
    # Split B_2 into 3 blocks of 2 columns
    for j in range(3):
        # Block j corresponds to B columns 2*j, 2*j+1
        # e = B[1, 2*j]
        # f = B[1, 2*j+1]
        # g = B[2, 2*j]
        # h = B[2, 2*j+1]
        
        e_idx = 6 + 2 * j
        f_idx = 6 + 2 * j + 1
        g_idx = 12 + 2 * j
        h_idx = 12 + 2 * j + 1
        
        # Strassen steps 0..6
        # M_1: (a+d)(e+h)
        # M_2: (c+d)e
        # M_3: a(f-h)
        # M_4: d(e-g)
        # M_5: (a+b)h
        # M_6: (c-a)(e+f)
        # M_7: (b-d)(g+h)
        
        # W indices for C outputs of this block
        # C[0, 2*j]   -> W[2*j*2 + 0] = 4*j
        # C[0, 2*j+1] -> W[(2*j+1)*2 + 0] = 4*j+2
        # C[1, 2*j]   -> W[2*j*2 + 1] = 4*j+1
        # C[1, 2*j+1] -> W[(2*j+1)*2 + 1] = 4*j+3
        
        w_11_idx = 4 * j
        w_12_idx = 4 * j + 2
        w_21_idx = 4 * j + 1
        w_22_idx = 4 * j + 3
        
        for s in range(7):
            r = 12 + j * 7 + s
            
            if s == 0:
                # M_1
                U[a_idx][r] = 1
                U[d_idx][r] = 1
                V[e_idx][r] = 1
                V[h_idx][r] = 1
                W[w_11_idx][r] = 1
                W[w_22_idx][r] = 1
            elif s == 1:
                # M_2
                U[c_idx][r] = 1
                U[d_idx][r] = 1
                V[e_idx][r] = 1
                W[w_21_idx][r] = 1
                W[w_22_idx][r] = -1
            elif s == 2:
                # M_3
                U[a_idx][r] = 1
                V[f_idx][r] = 1
                V[h_idx][r] = -1
                W[w_12_idx][r] = 1
                W[w_22_idx][r] = 1
            elif s == 3:
                # M_4
                U[d_idx][r] = 1
                V[e_idx][r] = 1
                V[g_idx][r] = -1
                W[w_11_idx][r] = 1
                W[w_21_idx][r] = 1
            elif s == 4:
                # M_5
                U[a_idx][r] = 1
                U[b_idx][r] = 1
                V[h_idx][r] = 1
                W[w_11_idx][r] = -1
                W[w_12_idx][r] = 1
            elif s == 5:
                # M_6
                U[c_idx][r] = 1
                U[a_idx][r] = -1
                V[e_idx][r] = 1
                V[f_idx][r] = 1
                W[w_22_idx][r] = 1
            elif s == 6:
                # M_7
                U[b_idx][r] = 1
                U[d_idx][r] = -1
                V[g_idx][r] = 1
                V[h_idx][r] = 1
                W[w_11_idx][r] = 1

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
