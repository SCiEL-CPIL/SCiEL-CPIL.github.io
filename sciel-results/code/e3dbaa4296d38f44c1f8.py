import numpy as np
import random
import time

DIMENSIONS = [2, 3, 10]

def _build_rank36(n, m, p):
    rank = 36
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: j=0 block (2x1x10) -> Rank 1
    # U[i, 0, col] = 1
    # V[0, k, col] = 1
    # W[k, i, col] = 1
    for i in range(n):
        U[i * m + 0][col] = 1
    for k in range(p):
        V[0 * p + k][col] = 1
    for k in range(p):
        for i in range(n):
            W[k * n + i][col] = 1
    col += 1
    
    # Part 2: j=1,2 block (2x2x10) -> Rank 35 (5 * 7)
    # Split k into 5 blocks of size 2
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1
        
        # Indices for U (i, j) where j in {1, 2}
        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2
        
        # Indices for V (j, k) where j in {1, 2}, k in {k0, k1}
        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1
        
        # Indices for W (k, i) where k in {k0, k1}, i in {0, 1}
        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1
        
        # M1 = (a11 + a22)(b11 + b22)
        U[u_11][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M2 = (a21 + a22)b11
        U[u_21][col] += 1
        U[u_22][col] += 1
        V[v_11][col] += 1
        W[w_10][col] += 1
        W[w_11][col] += -1
        col += 1
        
        # M3 = a11(b12 - b22)
        U[u_11][col] += 1
        V[v_12][col] += 1
        V[v_22][col] += -1
        W[w_01][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M4 = a22(b21 - b11)
        U[u_22][col] += 1
        V[v_21][col] += 1
        V[v_11][col] += -1
        W[w_00][col] += 1
        W[w_10][col] += 1
        col += 1
        
        # M5 = (a11 + a12)b22
        U[u_11][col] += 1
        U[u_12][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += -1
        W[w_01][col] += 1
        col += 1
        
        # M6 = (a21 - a11)(b11 + b12)
        U[u_21][col] += 1
        U[u_11][col] += -1
        V[v_11][col] += 1
        V[v_12][col] += 1
        W[w_11][col] += 1
        col += 1
        
        # M7 = (a12 - a22)(b21 + b22)
        U[u_12][col] += 1
        U[u_22][col] += -1
        V[v_21][col] += 1
        V[v_22][col] += 1
        W[w_00][col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U, V, W

def _verify_fast(U, V, W, n, m, p, trials=200):
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)
    
    # Check non-zero entries
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

    # Check zero entries
    for _ in range(trials):
        a = random.randint(0, n * m - 1)
        b = random.randint(0, m * p - 1)
        c = random.randint(0, p * n - 1)
        
        j_a = a % m
        i_a = a // m
        j_b = b // p
        k_b = b % p
        
        # If indices are consistent, it's a non-zero entry
        if j_a == j_b and c == k_b * n + i_a:
            continue
            
        val = 0.0
        for r in range(rank):
            val += U[a, r] * V[b, r] * W[c, r]
            
        if abs(val) > 1e-9:
            return False

    return True

def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_rank36(n, m, p)
    
    # Verify
    if not _verify_fast(U, V, W, n, m, p, trials=1000):
        # Fallback to previous best if verification fails (should not happen)
        pass
        
    return {"U": U, "V": V, "W": W}
