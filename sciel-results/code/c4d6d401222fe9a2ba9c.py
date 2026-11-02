"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    rank = 97
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Part 1: Strassen 4x4x4 decomposition (rank 49)
    # Block-level Strassen coefficients for 2x2 blocks
    # U_block[r][I][J], V_block[r][J][K], W_block[r][K][I]
    U_block = [
        [[1, 0], [0, 1]], # r=0 (M1)
        [[0, 0], [1, 1]], # r=1 (M2)
        [[1, 0], [0, 0]], # r=2 (M3)
        [[0, 0], [0, 1]], # r=3 (M4)
        [[1, 1], [0, 0]], # r=4 (M5)
        [[-1, 0], [1, 0]],# r=5 (M6)
        [[0, 1], [0, -1]] # r=6 (M7)
    ]
    V_block = [
        [[1, 0], [0, 1]], # r=0
        [[1, 0], [0, 0]], # r=1
        [[0, 1], [0, -1]],# r=2
        [[-1, 0], [1, 0]],# r=3
        [[0, 0], [0, 1]], # r=4
        [[1, 1], [0, 0]], # r=5
        [[0, 1], [1, 1]]  # r=6
    ]
    W_block = [
        [[1, 0], [0, 1]], # r=0
        [[0, 0], [1, 0]], # r=1
        [[0, 1], [0, 0]], # r=2
        [[0, 0], [1, 0]], # r=3
        [[-1, 0], [0, 0]],# r=4
        [[0, 0], [0, 1]], # r=5
        [[1, 0], [0, 0]]  # r=6
    ]
    
    # Sub-level Strassen coefficients for 2x2x2
    # U_sub[s][i][j], V_sub[s][j][k], W_sub[s][k][i]
    U_sub = [
        [[1, 0], [1, 0]], # s=0 (M1)
        [[0, 0], [1, 1]], # s=1 (M2)
        [[1, 0], [0, 0]], # s=2 (M3)
        [[0, 0], [0, 1]], # s=3 (M4)
        [[1, 1], [0, 0]], # s=4 (M5)
        [[-1, 0], [1, 0]],# s=5 (M6)
        [[0, 1], [0, -1]] # s=6 (M7)
    ]
    V_sub = [
        [[1, 1], [0, 0]], # s=0
        [[1, 0], [0, 0]], # s=1
        [[0, 1], [0, -1]],# s=2
        [[-1, 0], [1, 0]],# s=3
        [[0, 0], [0, 1]], # s=4
        [[1, 1], [0, 0]], # s=5
        [[0, 1], [1, 1]]  # s=6
    ]
    W_sub = [
        [[1, 0], [0, 1]], # s=0
        [[0, 0], [1, 0]], # s=1
        [[0, 1], [0, 0]], # s=2
        [[0, 0], [1, 0]], # s=3
        [[-1, 0], [0, 0]],# s=4
        [[0, 0], [0, 1]], # s=5
        [[1, 0], [0, 0]]  # s=6
    ]
    
    col = 0
    for r in range(7):
        for s in range(7):
            for I in range(2):
                for J in range(2):
                    for K in range(2):
                        for i_prime in range(2):
                            for j_prime in range(2):
                                for k_prime in range(2):
                                    u_val = U_block[r][I][J] * U_sub[s][i_prime][j_prime]
                                    v_val = V_block[r][J][K] * V_sub[s][j_prime][k_prime]
                                    w_val = W_block[r][K][I] * W_sub[s][k_prime][i_prime]
                                    
                                    if u_val != 0:
                                        i = 2 * I + i_prime
                                        j = 2 * J + j_prime
                                        U[i * m + j][col] += u_val
                                    
                                    if v_val != 0:
                                        j = 2 * J + j_prime
                                        k = 2 * K + k_prime
                                        V[j * p + k][col] += v_val
                                    
                                    if w_val != 0:
                                        k = 2 * K + k_prime
                                        i = 2 * I + i_prime
                                        W[k * n + i][col] += w_val
            col += 1
            
    # Part 2: Schoolbook 4x4x3 decomposition (rank 48)
    for i in range(n):
        for j in range(m):
            for k in range(4, p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
                
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
