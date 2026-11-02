import numpy as np
import time

DIMENSIONS = [2, 3, 10]
TIME_LIMIT = 140  # safety margin before 150s kill

def construct():
    _start = time.time()
    n, m, p = DIMENSIONS
    rank = 55
    
    # U: (n*m, rank) = (6, rank)
    # V: (m*p, rank) = (30, rank)
    # W: (p*n, rank) = (20, rank)
    U = [[0.0] * rank for _ in range(n * m)]
    V = [[0.0] * rank for _ in range(m * p)]
    W = [[0.0] * rank for _ in range(p * n)]
    
    col = 0
    
    # Part 1: A1 (2x2) * B1 (2x10) using 5 Strassen blocks
    # A is 2x3. We use columns j=0,1 for Strassen.
    # B is 3x10. We use rows j=0,1 for Strassen, split into 5 blocks of 2 columns.
    
    for g in range(5):
        k0 = 2 * g
        k1 = 2 * g + 1
        
        # A indices (U): i * m + j, i in {0,1}, j in {0,1}
        a00 = 0 * m + 0  # 0
        a01 = 0 * m + 1  # 1
        a10 = 1 * m + 0  # 3
        a11 = 1 * m + 1  # 4
        
        # B indices (V): j * p + k, j in {0,1}, k in {k0, k1}
        b00 = 0 * p + k0
        b01 = 0 * p + k1
        b10 = 1 * p + k0
        b11 = 1 * p + k1
        
        # C indices (W): k * n + i, k in {k0, k1}, i in {0,1}
        c00 = k0 * n + 0
        c01 = k1 * n + 0
        c10 = k0 * n + 1
        c11 = k1 * n + 1
        
        # Strassen 7 terms for 2x2x2
        
        # M1 = (a00 + a11) * (b00 + b11) -> c00 += 1, c11 += 1
        U[a00][col] = 1.0
        U[a11][col] = 1.0
        V[b00][col] = 1.0
        V[b11][col] = 1.0
        W[c00][col] = 1.0
        W[c11][col] = 1.0
        col += 1
        
        # M2 = (a10 + a11) * b00 -> c10 += 1, c11 -= 1
        U[a10][col] = 1.0
        U[a11][col] = 1.0
        V[b00][col] = 1.0
        W[c10][col] = 1.0
        W[c11][col] = -1.0
        col += 1
        
        # M3 = a00 * (b01 - b11) -> c01 += 1, c11 += 1
        U[a00][col] = 1.0
        V[b01][col] = 1.0
        V[b11][col] = -1.0
        W[c01][col] = 1.0
        W[c11][col] = 1.0
        col += 1
        
        # M4 = a11 * (b10 - b00) -> c00 += 1, c10 += 1
        U[a11][col] = 1.0
        V[b10][col] = 1.0
        V[b00][col] = -1.0
        W[c00][col] = 1.0
        W[c10][col] = 1.0
        col += 1
        
        # M5 = (a00 + a01) * b11 -> c01 += 1, c00 -= 1
        U[a00][col] = 1.0
        U[a01][col] = 1.0
        V[b11][col] = 1.0
        W[c01][col] = 1.0
        W[c00][col] = -1.0
        col += 1
        
        # M6 = (a10 - a00) * (b00 + b01) -> c11 += 1
        U[a10][col] = 1.0
        U[a00][col] = -1.0
        V[b00][col] = 1.0
        V[b01][col] = 1.0
        W[c11][col] = 1.0
        col += 1
        
        # M7 = (a01 - a11) * (b10 + b11) -> c00 += 1
        U[a01][col] = 1.0
        U[a11][col] = -1.0
        V[b10][col] = 1.0
        V[b11][col] = 1.0
        W[c00][col] = 1.0
        col += 1
    
    # Part 2: A2 (2x1) * B2 (1x10) using Schoolbook
    # A column j=2: U indices i * m + 2, i in {0,1}
    # B row j=2: V indices 2 * p + k, k in {0..9}
    # C: W indices k * n + i, i in {0,1}, k in {0..9}
    
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 2
            v_idx = 2 * p + k
            w_idx = k * n + i
            U[u_idx][col] = 1.0
            V[v_idx][col] = 1.0
            W[w_idx][col] = 1.0
            col += 1
    
    # Verify rank
    if col != rank:
        raise ValueError(f"Rank mismatch: got {col}, expected {rank}")
    
    # Quick verification using NumPy
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(rank):
                    s += Ua[i * m + j, c] * Va[j * p + k, c] * Wa[k * n + i, c]
                err = abs(s - 1.0)
                if err > max_err:
                    max_err = err
    
    if max_err > 1e-9:
        raise ValueError(f"Verification failed with max error {max_err}")
    
    elapsed = time.time() - _start
    if elapsed > TIME_LIMIT:
        raise TimeoutError(f"Time exceeded: {elapsed}")
    
    return {"U": U, "V": V, "W": W}
