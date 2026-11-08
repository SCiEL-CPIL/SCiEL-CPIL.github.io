"""Matrix multiplication 2x4 x 4x6 — rank-minimal exact tensor decomposition."""
DIMENSIONS = [2, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 4, 6
    rank = 42
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_strassen = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):1},
        {(0,0):1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (0,0):-1},
        {(0,1):1, (1,1):-1},
    ]
    V_strassen = [
        {(0,0):1, (1,1):1},
        {(0,0):1},
        {(0,1):1, (1,1):-1},
        {(1,0):1, (0,0):-1},
        {(1,1):1},
        {(0,0):1, (0,1):1},
        {(1,0):1, (1,1):1},
    ]
    W_strassen = [
        {(0,0):1, (1,1):1},
        {(1,0):1, (1,1):-1},
        {(0,1):1, (1,1):1},
        {(0,0):1, (1,0):1},
        {(0,0):-1, (0,1):1},
        {(1,1):1},
        {(0,0):1},
    ]
    
    col = 0
    for a in range(2):
        for b in range(3):
            for s in range(7):
                for (i_loc, j_loc), coeff in U_strassen[s].items():
                    i = i_loc
                    j = a * 2 + j_loc
                    U[i * m + j][col] += coeff
                for (j_loc, k_loc), coeff in V_strassen[s].items():
                    j = a * 2 + j_loc
                    k = b * 2 + k_loc
                    V[j * p + k][col] += coeff
                for (i_loc, k_loc), coeff in W_strassen[s].items():
                    i = i_loc
                    k = b * 2 + k_loc
                    W[k * n + i][col] += coeff
                col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
