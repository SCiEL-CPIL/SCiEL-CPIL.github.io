import time

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 100.0  # Safety margin
    
    n, m, p = DIMENSIONS
    
    rank = 44
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    terms_2x2 = [
        ([[1, 0], [0, 1]], [[1, 0], [0, 1]], [[1, 0], [0, 1]]),
        ([[0, 1], [0, 1]], [[1, 0], [0, 0]], [[0, 0], [1, 0]]),
        ([[1, 0], [0, 0]], [[0, 1], [0, 0]], [[0, 0], [1, 0]]),
        ([[0, 0], [0, 1]], [[0, 0], [0, 1]], [[0, 0], [0, 1]]),
        ([[1, 1], [0, 0]], [[0, 0], [0, 1]], [[0, 1], [0, 0]]),
        ([[0, 1], [1, 0]], [[1, 1], [0, 0]], [[0, 1], [0, 0]]),
        ([[0, 1], [-1, 0]], [[0, 0], [1, 1]], [[0, 0], [1, 0]])
    ]
    
    blocks = [
        (0, 0, 0),
        (0, 0, 2),
        (0, 2, 0),
        (0, 2, 2)
    ]
    
    for ib, jb, kb in blocks:
        if time.time() > deadline:
            break
        for u_mat, v_mat, w_mat in terms_2x2:
            for i_prime in range(2):
                for j_prime in range(2):
                    for k_prime in range(2):
                        val = u_mat[i_prime][j_prime] * v_mat[j_prime][k_prime] * w_mat[i_prime][k_prime]
                        if val != 0:
                            gi = ib + i_prime
                            gj = jb + j_prime
                            gk = kb + k_prime
                            U[gi * m + gj][col] += val
                            V[gj * p + gk][col] += val
                            W[gk * n + gi][col] += val
            col += 1
    
    for j in range(4):
        for k in range(4):
            U[2 * m + j][col] = 1
            V[j * p + k][col] = 1
            W[k * n + 2][col] = 1
            col += 1
            
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
