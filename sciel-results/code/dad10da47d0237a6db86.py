"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [3, 3, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 100  # 100s safety margin for 120s limit
    
    n, m, p = DIMENSIONS
    
    # Default: Schoolbook decomposition
    rank_schoolbook = n * m * p
    U = [[0] * rank_schoolbook for _ in range(n * m)]
    V = [[0] * rank_schoolbook for _ in range(m * p)]
    W = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    
    def is_valid(U, V, W):
        n, m, p = DIMENSIONS
        rank = len(U[0])
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    a = i * m + j
                    b = j * p + k
                    o = k * n + i
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[o][c]
                    if val != 1:
                        return False
        import random
        random.seed(42)
        for _ in range(100):
            a = random.randint(0, n*m - 1)
            b = random.randint(0, m*p - 1)
            o = random.randint(0, p*n - 1)
            i_a = a // m
            j_a = a % m
            j_b = b // p
            k_b = b % p
            k_o = o // n
            i_o = o % n
            if i_a == i_o and j_a == j_b:
                if k_b != k_o:
                    val = 0
                    for c in range(rank):
                        val += U[a][c] * V[b][c] * W[o][c]
                    if val != 0:
                        return False
            else:
                val = 0
                for c in range(rank):
                    val += U[a][c] * V[b][c] * W[o][c]
                if val != 0:
                    return False
        return True

    if is_valid(U, V, W):
        return {"U": U, "V": V, "W": W}
    
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
