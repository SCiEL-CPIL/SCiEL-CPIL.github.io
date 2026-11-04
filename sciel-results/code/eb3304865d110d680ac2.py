"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
DIMENSIONS = [2, 3, 10]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    DEADLINE = 140.0  # safety margin before 150s
    
    n, m, p = DIMENSIONS  # 2, 3, 10
    rank = 55
    
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    U_str = [
        [[1,0],[0,1]],
        [[0,0],[1,1]],
        [[1,0],[0,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[-1,0],[1,0]],
        [[0,1],[0,-1]],
    ]
    V_str = [
        [[1,0],[0,1]],
        [[1,0],[0,0]],
        [[0,1],[0,-1]],
        [[-1,0],[1,0]],
        [[0,0],[0,1]],
        [[1,1],[0,0]],
        [[0,0],[1,1]],
    ]
    W_str = [
        [[1,0],[0,1]],
        [[0,0],[1,-1]],
        [[0,1],[0,1]],
        [[1,0],[1,0]],
        [[-1,1],[0,0]],
        [[0,0],[0,1]],
        [[1,0],[0,0]],
    ]
    
    col = 0
    for s in range(5):
        if time.time() > DEADLINE:
            break
        k0 = 2 * s
        for c_prime in range(7):
            for i in range(2):
                for j in range(2):
                    U[i * m + j][col] = U_str[c_prime][i][j]
            for j in range(2):
                for k in range(2):
                    V[j * p + (k0 + k)][col] = V_str[c_prime][j][k]
            for k in range(2):
                for i in range(2):
                    W[(k0 + k) * n + i][col] = W_str[c_prime][k][i]
            col += 1
    
    for i in range(2):
        for k in range(10):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
