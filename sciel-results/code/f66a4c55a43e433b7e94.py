import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]

def _verify_fast(U, V, W, n, m, p, trials=100):
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    for _ in range(trials):
        i = random.randint(0, n - 1)
        j = random.randint(0, m - 1)
        k = random.randint(0, p - 1)
        
        a = i * m + j
        b = j * p + k
        c = k * n + i
        
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
            
        if abs(val - 1.0) > 1e-9:
            return False

    for _ in range(trials):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        
        j = a % m
        i = a // m
        k = b % p
        
        if c == k * n + i:
            continue
        
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
            
        if abs(val) > 1e-9:
            return False

    return True

def _is_valid_entries(U, V, W):
    for matrix in [U, V, W]:
        for row in matrix:
            for val in row:
                if abs(val * 2 - round(val * 2)) > 1e-9:
                    return False
    return True

def _build_strassen55(n, m, p):
    rank = 55
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1
            V[v_idx][col] = 1
            W[w_idx][col] = 1
            col += 1

    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1

        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1

        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1

        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1

        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _build_schoolbook(n, m, p):
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    return U, V, W

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 110 

    U_best, V_best, W_best = _build_strassen55(n, m, p)
    best_rank = 55

    if not (_verify_fast(U_best, V_best, W_best, n, m, p, trials=200) and _is_valid_entries(U_best, V_best, W_best)):
        U_best, V_best, W_best = _build_schoolbook(n, m, p)
        best_rank = n * m * p

    return {"U": U_best, "V": V_best, "W": W_best}
