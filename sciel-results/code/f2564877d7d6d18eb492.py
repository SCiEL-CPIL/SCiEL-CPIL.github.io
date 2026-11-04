"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    
    # Strassen decomposition for 2x2x2
    # U (4x7), V (4x7), W (4x7)
    # Indices: U[i*2+j, c], V[j*2+k, c], W[k*2+i, c]
    
    U_small = [
        [1, 0, 0, 0, 1, -1, 0],  # A11
        [0, 0, 0, 0, 1, 0, 1],   # A12
        [0, 0, 1, 0, 0, 1, 0],   # A21
        [1, 1, 0, 1, 0, 0, -1]   # A22
    ]
    
    V_small = [
        [1, 1, 0, -1, 0, 1, 0],  # B11
        [1, 0, -1, 0, 0, 1, 1],  # B12
        [0, 0, 1, 1, 0, 0, 1],   # B21
        [0, 0, 0, 0, 1, 0, 1]    # B22
    ]
    
    W_small = [
        [1, 0, 0, 0, -1, 0, 1],  # C11
        [0, 0, 1, 0, 1, 0, 0],   # C12
        [0, 1, 0, 1, 0, 0, 0],   # C21
        [1, -1, 1, 0, 0, 1, 0]   # C22
    ]

    # Block-level Strassen (same structure)
    U_block = U_small
    V_block = V_small
    W_block = W_small

    # Construct 4x4x4 Strassen decomposition (Rank 49)
    rank_strassen = 49
    U_strassen = [[0] * rank_strassen for _ in range(16)]
    V_strassen = [[0] * rank_strassen for _ in range(16)]
    W_strassen = [[0] * rank_strassen for _ in range(16)]

    for I in range(2):
        for J in range(2):
            for i in range(2):
                for j in range(2):
                    u_row_global = (2*I + i)*4 + (2*J + j)
                    u_row_small = i*2 + j
                    for K in range(7):
                        for k in range(7):
                            col = K*7 + k
                            val = U_block[I*2+J][K] * U_small[u_row_small][k]
                            U_strassen[u_row_global][col] += val

    for J in range(2):
        for L in range(2):
            for j in range(2):
                for l in range(2):
                    v_row_global = (2*J + j)*4 + (2*L + l)
                    v_row_small = j*2 + l
                    for K in range(7):
                        for k in range(7):
                            col = K*7 + k
                            val = V_block[J*2+L][K] * V_small[v_row_small][k]
                            V_strassen[v_row_global][col] += val

    for I in range(2):
        for M in range(2):
            for i in range(2):
                for m in range(2):
                    w_row_global = (2*M + m)*4 + (2*I + i)
                    w_row_small = m*2 + i
                    for K in range(7):
                        for k in range(7):
                            col = K*7 + k
                            val = W_block[M*2+I][K] * W_small[w_row_small][k]
                            W_strassen[w_row_global][col] += val

    # Schoolbook decomposition for 4x4x1 (Rank 16)
    rank_school = 16
    U_school = [[0] * rank_school for _ in range(16)]
    V_school = [[0] * rank_school for _ in range(4)]
    W_school = [[0] * rank_school for _ in range(4)]

    for i in range(4):
        for j in range(4):
            c = i*4 + j
            U_school[i*4+j][c] = 1
            V_school[j][c] = 1
            W_school[i][c] = 1

    # Concatenate
    total_rank = rank_strassen + rank_school
    U = [row[:rank_strassen] + row_school for row, row_school in zip(U_strassen, U_school)]
    V = [row[:rank_strassen] + row_school for row, row_school in zip(V_strassen, V_school)]
    W = [row[:rank_strassen] + row_school for row, row_school in zip(W_strassen, W_school)]

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
