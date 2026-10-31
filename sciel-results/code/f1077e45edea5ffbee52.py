"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 49
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define Strassen coefficients for 2x2 matrix multiplication
    # U_2x2[b][i][j], V_2x2[b][j][k], W_2x2[b][k][i]
    # b is the rank index 0..6
    U_2x2 = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    V_2x2 = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[0, 0], [1, -1]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    W_2x2 = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 0], [1, 0]], # M2 -> C10
        [[0, 1], [0, 1]], # M3 -> C01, C11
        [[1, 0], [1, 0]], # M4 -> C00, C10
        [[0, 1], [0, -1]],# M5 -> C01, -C11
        [[0, 0], [0, 1]], # M6 -> C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    
    # Define outer Strassen coefficients for 4x4 (block 2x2)
    # U_out[a][ib][jb], V_out[a][jb][kb], W_out[a][kb][ib]
    U_out = [
        [[1, 0], [0, 1]], # M1
        [[0, 0], [1, 1]], # M2
        [[1, 0], [0, 0]], # M3
        [[0, 0], [0, 1]], # M4
        [[1, 1], [0, 0]], # M5
        [[-1, 0], [1, 0]],# M6
        [[0, 1], [0, -1]] # M7
    ]
    V_out = [
        [[1, 0], [0, 1]], # M1
        [[1, 0], [0, 0]], # M2
        [[0, 1], [0, -1]],# M3
        [[0, 0], [1, -1]],# M4
        [[0, 0], [0, 1]], # M5
        [[1, 1], [0, 0]], # M6
        [[0, 0], [1, 1]]  # M7
    ]
    W_out = [
        [[1, 0], [0, 1]], # M1 -> C00, C11
        [[0, 0], [1, 0]], # M2 -> C10
        [[0, 1], [0, 1]], # M3 -> C01, C11
        [[1, 0], [1, 0]], # M4 -> C00, C10
        [[0, 1], [0, -1]],# M5 -> C01, -C11
        [[0, 0], [0, 1]], # M6 -> C11
        [[1, 0], [0, 0]]  # M7 -> C00
    ]
    
    for a in range(7):
        for b in range(7):
            r = 7 * a + b
            for i in range(4):
                for j in range(4):
                    ib = i // 2
                    il = i % 2
                    jb = j // 2
                    jl = j % 2
                    U[i * 4 + j][r] = U_out[a][ib][jb] * U_2x2[b][il][jl]
            
            for j in range(4):
                for k in range(4):
                    jb = j // 2
                    jl = j % 2
                    kb = k // 2
                    kl = k % 2
                    V[j * 4 + k][r] = V_out[a][jb][kb] * V_2x2[b][jl][kl]
            
            for k in range(4):
                for i in range(4):
                    kb = k // 2
                    kl = k % 2
                    ib = i // 2
                    il = i % 2
                    W[k * 4 + i][r] = W_out[a][kb][ib] * W_2x2[b][kl][il]
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
