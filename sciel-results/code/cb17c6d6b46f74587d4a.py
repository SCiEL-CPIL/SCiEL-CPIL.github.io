import numpy as np
import time

DIMENSIONS = [2, 4, 7]

def construct():
    n, m, p = DIMENSIONS  # 2, 4, 7
    rank = 28
    
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    # U indices: i * m + j
    # V indices: j * p + k
    # W indices: k * n + i
    
    for k in range(p):
        base_r = 4 * k
        
        # Block 1: j=0, 1
        # Product 1: a00, a10 with b0k
        r = base_r
        U[0 * m + 0, r] = 1.0
        U[1 * m + 0, r] = 1.0
        V[0 * p + k, r] = 1.0
        W[k * n + 0, r] = 1.0
        W[k * n + 1, r] = 1.0
        
        # Product 2: a01, a11 with b1k
        r = base_r + 1
        U[0 * m + 1, r] = 1.0
        U[1 * m + 1, r] = 1.0
        V[1 * p + k, r] = 1.0
        W[k * n + 0, r] = 1.0
        W[k * n + 1, r] = 1.0
        
        # Block 2: j=2, 3
        # Product 3: a02, a12 with b2k
        r = base_r + 2
        U[0 * m + 2, r] = 1.0
        U[1 * m + 2, r] = 1.0
        V[2 * p + k, r] = 1.0
        W[k * n + 0, r] = 1.0
        W[k * n + 1, r] = 1.0
        
        # Product 4: a03, a13 with b3k
        r = base_r + 3
        U[0 * m + 3, r] = 1.0
        U[1 * m + 3, r] = 1.0
        V[3 * p + k, r] = 1.0
        W[k * n + 0, r] = 1.0
        W[k * n + 1, r] = 1.0
        
    # Verification
    def verify(U_mat, V_mat, W_mat):
        n, m, p = DIMENSIONS
        rank = U_mat.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[i*m+j, r] * V_mat[j*p+k, r] * W_mat[k*n+i, r]
                    if abs(s - 1.0) > 1e-9:
                        return False, f"One check failed: i={i}, j={j}, k={k}, s={s}"
        
        for a in range(n * m):
            i_a = a // m
            j_a = a % m
            for b in range(m * p):
                j_b = b // p
                k_b = b % p
                for c in range(p * n):
                    k_c = c // n
                    i_c = c % n
                    if i_a == i_c and j_a == j_b and k_b == k_c:
                        continue
                    s = 0.0
                    for r in range(rank):
                        s += U_mat[a, r] * V_mat[b, r] * W_mat[c, r]
                    if abs(s) > 1e-9:
                        return False, f"Zero check failed: a={a}, b={b}, c={c}, s={s}"
        return True, "OK"
    
    valid, msg = verify(U, V, W)
    if not valid:
        print(f"Verification failed: {msg}")
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
