"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [3, 4, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 3, 4, 8
    
    # Sub-decompositions
    def make_1x2x4():
        U = [[1]]
        V = [[1], [1]]
        W = [[1], [1], [1], [1]]
        return U, V, W
    
    def make_2x2x4_schoolbook():
        rank = 16
        U = [[0]*rank for _ in range(4)]
        V = [[0]*rank for _ in range(8)]
        W = [[0]*rank for _ in range(8)]
        col = 0
        for i in range(2):
            for j in range(2):
                for k in range(4):
                    U[i*2+j][col] = 1
                    V[j*4+k][col] = 1
                    W[k*2+i][col] = 1
                    col += 1
        return U, V, W
    
    U_124, V_124, W_124 = make_1x2x4()
    U_224, V_224, W_224 = make_2x2x4_schoolbook()
    
    rank_124 = len(U_124[0])  # 1
    rank_224 = len(U_224[0])  # 16
    
    total_rank = 4 * rank_124 + 4 * rank_224
    U = [[0] * total_rank for _ in range(n * m)]
    V = [[0] * total_rank for _ in range(m * p)]
    W = [[0] * total_rank for _ in range(p * n)]
    
    col = 0
    for j_off in [0, 2]:
        for k_off in [0, 4]:
            # 1x2x4 block
            for r in range(rank_124):
                for i_l in range(1):
                    for j_l in range(2):
                        U[(0 + i_l) * m + (j_off + j_l)][col] = U_124[i_l * 2 + j_l][r]
                for j_l in range(2):
                    for k_l in range(4):
                        V[(j_off + j_l) * p + (k_off + k_l)][col] = V_124[j_l * 4 + k_l][r]
                for k_l in range(4):
                    for i_l in range(1):
                        W[(k_off + k_l) * n + (0 + i_l)][col] = W_124[k_l * 1 + i_l][r]
                col += 1
    
    for j_off in [0, 2]:
        for k_off in [0, 4]:
            # 2x2x4 block
            for r in range(rank_224):
                for i_l in range(2):
                    for j_l in range(2):
                        U[(1 + i_l) * m + (j_off + j_l)][col] = U_224[i_l * 2 + j_l][r]
                for j_l in range(2):
                    for k_l in range(4):
                        V[(j_off + j_l) * p + (k_off + k_l)][col] = V_224[j_l * 4 + k_l][r]
                for k_l in range(4):
                    for i_l in range(2):
                        W[(k_off + k_l) * n + (1 + i_l)][col] = W_224[k_l * 2 + i_l][r]
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
