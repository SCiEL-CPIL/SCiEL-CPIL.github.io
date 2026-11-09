DIMENSIONS = [6, 6, 6]

def construct():
    n, m, p = DIMENSIONS
    rank = 108
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for ib in range(3):
        for jb in range(3):
            for kb in range(3):
                for i_loc in range(2):
                    for j_loc in range(2):
                        for k_loc in range(2):
                            i = ib * 2 + i_loc
                            j = jb * 2 + j_loc
                            k = kb * 2 + k_loc
                            U[i * m + j][col] = 1
                            V[j * p + k][col] = 1
                            W[k * n + i][col] = 1
                            col += 1
    
    return {"U": U, "V": V, "W": W}
