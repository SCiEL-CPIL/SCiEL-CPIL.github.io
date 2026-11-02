import time

DIMENSIONS = [6, 6, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Rank 189 decomposition using Strassen for 2x2 blocks of 3x3
    rank = 189
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    # Define the 7 Strassen terms
    # Each term is (L_blocks, R_blocks, C_blocks)
    # Blocks are (br, bc, coeff)
    terms = [
        # 1
        (([(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)], [(0,0,1), (1,1,1)])),
        # 2
        (([(1,0,1), (1,1,1)], [(0,0,1)], [(1,0,1), (1,1,-1)])),
        # 3
        (([(0,0,1)], [(0,1,1), (1,1,-1)], [(0,1,1), (1,1,1)])),
        # 4
        (([(1,1,1)], [(1,0,1), (0,0,-1)], [(0,0,1), (1,0,1)])),
        # 5
        (([(0,0,1), (0,1,1)], [(1,1,1)], [(0,0,-1), (0,1,1)])),
        # 6
        (([(1,0,1), (0,0,-1)], [(0,0,1), (0,1,1)], [(1,1,1)])),
        # 7
        (([(0,1,1), (1,1,-1)], [(1,0,1), (1,1,1)], [(0,0,1)]))
    ]
    
    col = 0
    for L, R, C in terms:
        for r in range(3):
            for b in range(3):
                for c in range(3):
                    # U contributions
                    for lbr, lbc, lcoeff in L:
                        u_row = 3 * lbr + r
                        u_col = 3 * lbc + b
                        U[u_row][col] += lcoeff
                    
                    # V contributions
                    for rbr, rbc, rcoeff in R:
                        v_row = 3 * rbr + b
                        v_col = 3 * rbc + c
                        V[v_row][col] += rcoeff
                        
                    # W contributions
                    for cbr, cbc, ccoeff in C:
                        w_row = 3 * cbr + r
                        w_col = 3 * cbc + c
                        w_idx = w_col * n + w_row
                        W[w_idx][col] += ccoeff
                        
                    col += 1
                    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
