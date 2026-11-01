import time

DIMENSIONS = [2, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # 10s margin
    
    n, m, p = DIMENSIONS
    rank = 54
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    J_blocks = [[0, 1], [2, 3], [4]]
    K_blocks = [[0, 1], [2, 3], [4, 5]]
    
    col = 0
    
    def add_term(u_coeff, v_coeff, w_coeff):
        nonlocal col
        for (i, j), val in u_coeff.items():
            if val != 0:
                U[i * m + j][col] += val
        for (j, k), val in v_coeff.items():
            if val != 0:
                V[j * p + k][col] += val
        for (k, i), val in w_coeff.items():
            if val != 0:
                W[k * n + i][col] += val
        col += 1

    for J in J_blocks:
        for K in K_blocks:
            if time.time() > deadline:
                break
            if len(J) == 2 and len(K) == 2:
                ja, jb = J
                ka, kb = K
                
                # M1
                add_term({(0, ja): 1, (1, jb): 1}, {(ja, ka): 1, (jb, kb): 1}, {(ka, 0): 1, (kb, 1): 1})
                # M2
                add_term({(1, ja): 1, (1, jb): 1}, {(ja, ka): 1}, {(ka, 1): 1})
                # M3
                add_term({(0, ja): 1}, {(jb, ka): 1, (jb, kb): -1}, {(kb, 0): 1})
                # M4
                add_term({(1, jb): 1}, {(ja, kb): 1, (ja, ka): -1}, {(ka, 0): 1, (ka, 1): 1})
                # M5
                add_term({(0, ja): 1, (0, jb): 1}, {(jb, kb): 1}, {(ka, 0): -1, (kb, 0): 1})
                # M6
                add_term({(1, ja): 1, (0, ja): -1}, {(ja, ka): 1, (jb, ka): 1}, {(kb, 1): 1})
                # M7
                add_term({(0, jb): 1, (1, jb): -1}, {(jb, kb): 1, (ja, kb): 1}, {(ka, 0): 1, (kb, 1): -1})
            else:
                j0 = J[0]
                ka, kb = K
                add_term({(0, j0): 1}, {(j0, ka): 1}, {(ka, 0): 1})
                add_term({(0, j0): 1}, {(j0, kb): 1}, {(kb, 0): 1})
                add_term({(1, j0): 1}, {(j0, ka): 1}, {(ka, 1): 1})
                add_term({(1, j0): 1}, {(j0, kb): 1}, {(kb, 1): 1})

    return {"U": U, "V": V, "W": W}
