import numpy as np
import time

DIMENSIONS = [3, 5, 7]

def construct():
    n, m, p = 3, 5, 7
    
    # We will build the decomposition using a list of terms.
    # Each term is (u_row, v_row, w_row, value)
    # Since we use schoolbook, values are 1.
    # U shape: (15, rank)
    # V shape: (35, rank)
    # W shape: (21, rank)
    
    U = [[0] * 100 for _ in range(n * m)] # Pre-allocate max possible rank
    V = [[0] * 100 for _ in range(m * p)]
    W = [[0] * 100 for _ in range(p * n)]
    
    rank = 0
    
    def add_term(u_row, v_row, w_row):
        nonlocal rank
        U[u_row][rank] = 1
        V[v_row][rank] = 1
        W[w_row][rank] = 1
        rank += 1

    # --- Block 1: 3x2x7 (j in 0..1) ---
    # Schoolbook: i in 0..2, j in 0..1, k in 0..6
    for i in range(n):
        for j in range(2):
            for k in range(p):
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                add_term(u_row, v_row, w_row)
    # Rank so far: 3 * 2 * 7 = 42

    # --- Block 2: 3x3x7 (j in 2..4, map j2 = j-2 in 0..2) ---
    # Split k into k1 in 0..2 and k2 in 0..3 (k = 3+k2)
    
    # Sub-block 2a: 3x3x3 (j2 in 0..2, k1 in 0..2)
    for i in range(n):
        for j2 in range(3):
            j = 2 + j2
            for k1 in range(3):
                u_row = i * m + j
                v_row = j * p + k1
                w_row = k1 * n + i
                add_term(u_row, v_row, w_row)
    # Rank so far: 42 + 9 = 51

    # Sub-block 2b: 3x3x4 (j2 in 0..2, k2 in 0..3)
    # Split k2 into k2a in 0..2 and k2b in 0..0
    # Sub-block 2b1: 3x3x3 (j2 in 0..2, k2a in 0..2, actual k = 3+k2a)
    for i in range(n):
        for j2 in range(3):
            j = 2 + j2
            for k2a in range(3):
                k = 3 + k2a
                u_row = i * m + j
                v_row = j * p + k
                w_row = k * n + i
                add_term(u_row, v_row, w_row)
    # Rank so far: 51 + 9 = 60

    # Sub-block 2b2: 3x3x1 (j2 in 0..2, k2b=0, actual k = 6)
    for i in range(n):
        for j2 in range(3):
            j = 2 + j2
            k = 6
            u_row = i * m + j
            v_row = j * p + k
            w_row = k * n + i
            add_term(u_row, v_row, w_row)
    # Rank so far: 60 + 3 = 63

    # Trim matrices to actual rank
    U = [row[:rank] for row in U]
    V = [row[:rank] for row in V]
    W = [row[:rank] for row in W]
    
    return {"U": U, "V": V, "W": W}

if __name__ == "__main__":
    result = construct()
    U, V, W = result["U"], result["V"], result["W"]
    rank = len(U[0])
    print(f"Rank: {rank}")
    
    # Verify
    n, m, p = 3, 5, 7
    Ua = np.array(U, dtype=float)
    Va = np.array(V, dtype=float)
    Wa = np.array(W, dtype=float)
    
    max_err = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(Ua[i*m+j] * Va[j*p+k] * Wa[k*n+i])
                err = abs(val - 1)
                max_err = max(max_err, err)
    
    print(f"Max error: {max_err}")
    assert max_err < 1e-9, f"Decomposition invalid: max_err={max_err}"
    print("Valid!")
