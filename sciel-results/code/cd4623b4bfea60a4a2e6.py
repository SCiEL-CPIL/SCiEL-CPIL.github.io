"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import sys

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
import numpy as np

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin
    
    n, m, p = DIMENSIONS
    # Rank 26 decomposition: 7 (Strassen 2x2) + 19 (remaining)
    rank = 26
    U = np.zeros((n * m, rank), dtype=float)
    V = np.zeros((m * p, rank), dtype=float)
    W = np.zeros((p * n, rank), dtype=float)
    
    col = 0
    
    # 1. Strassen terms for the 2x2 block (indices 0, 1)
    # A_sub = A[0:2, 0:2], B_sub = B[0:2, 0:2]
    # M1 = A00 * (B00 + B11)
    # M2 = (A00 + A10) * B11
    # M3 = (A01 - A11) * (B00 + B01)
    # M4 = A11 * (B10 - B00)
    # M5 = (A00 + A01) * B11
    # M6 = (A10 - A00) * (B00 + B10)
    # M7 = (A01 - A11) * (B01 + B11)
    
    # Helper to add a term
    def add_term(u_indices, v_indices, w_indices, coeff=1.0):
        nonlocal col
        for idx in u_indices:
            U[idx, col] += coeff
        for idx in v_indices:
            V[idx, col] += coeff
        for idx in w_indices:
            W[idx, col] += coeff
        col += 1

    # Indices:
    # A[i,j] -> i*m + j
    # B[j,k] -> j*p + k
    # C[i,k] -> k*n + i (based on prompt W[k*n+i])
    
    # Term 1: A00 * (B00 + B11)
    # U: A[0,0]
    # V: B[0,0], B[1,1]
    # W: C[0,0], C[0,1] (since M1 contributes to C00 and C01? No, M1 is used in C00 and C11?
    # Strassen:
    # C00 = M1 + M4 - M5 + M7
    # C01 = M3 + M5
    # C10 = M2 + M4
    # C11 = M1 - M2 + M3 + M6
    # So M1 contributes to C00 and C11.
    # C[0,0] -> W[0*n + 0] = W[0]
    # C[1,1] -> W[1*n + 1] = W[4] (n=3, 1*3+1=4)
    
    add_term([0*m + 0], [0*p + 0, 1*p + 1], [0*n + 0, 1*n + 1])
    
    # Term 2: (A00 + A10) * B11
    # U: A[0,0], A[1,0]
    # V: B[1,1]
    # W: C[1,0] (M2 in C10), C[1,1] (M2 in C11)
    add_term([0*m + 0, 1*m + 0], [1*p + 1], [0*n + 1, 1*n + 1])
    
    # Term 3: (A01 - A11) * (B00 + B01)
    # U: A[0,1], A[1,1] (with -1)
    # V: B[0,0], B[0,1]
    # W: C[0,1] (M3 in C01), C[1,1] (M3 in C11)
    add_term([0*m + 1, 1*m + 1], [0*p + 0, 0*p + 1], [1*n + 0, 1*n + 1])
    # Note: A[1,1] coefficient is -1.
    U[1*m + 1, col-1] = -1
    
    # Term 4: A11 * (B10 - B00)
    # U: A[1,1]
    # V: B[1,0], B[0,0] (with -1)
    # W: C[0,0] (M4 in C00), C[1,0] (M4 in C10)
    add_term([1*m + 1], [1*p + 0, 0*p + 0], [0*n + 0, 0*n + 1])
    V[0*p + 0, col-1] = -1
    
    # Term 5: (A00 + A01) * B11
    # U: A[0,0], A[0,1]
    # V: B[1,1]
    # W: C[0,0] (M5 in C00), C[0,1] (M5 in C01)
    add_term([0*m + 0, 0*m + 1], [1*p + 1], [0*n + 0, 1*n + 0])
    
    # Term 6: (A10 - A00) * (B00 + B10)
    # U: A[1,0], A[0,0] (with -1)
    # V: B[0,0], B[1,0]
    # W: C[1,1] (M6 in C11)
    add_term([1*m + 0, 0*m + 0], [0*p + 0, 1*p + 0], [1*n + 1])
    U[0*m + 0, col-1] = -1
    
    # Term 7: (A01 - A11) * (B01 + B11)
    # U: A[0,1], A[1,1] (with -1)
    # V: B[0,1], B[1,1]
    # W: C[0,0] (M7 in C00)
    add_term([0*m + 1, 1*m + 1], [0*p + 1, 1*p + 1], [0*n + 0])
    U[1*m + 1, col-1] = -1

    # 2. Remaining 19 terms (involving index 2)
    # Iterate over all i, j, k where i!=0,1 or j!=0,1 or k!=0,1?
    # Actually, just iterate over all 27, and skip the 8 core ones.
    core_indices = {(0,0,0), (0,0,1), (0,1,0), (0,1,1), (1,0,0), (1,0,1), (1,1,0), (1,1,1)}
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                if (i, j, k) in core_indices:
                    continue
                # Term: A[i,j] * B[j,k] -> C[i,k]
                add_term([i*m + j], [j*p + k], [k*n + i])

    # Verify correctness with a random test
    np.random.seed(42)
    A = np.random.randn(n, m)
    B = np.random.randn(m, p)
    C_expected = A @ B
    
    C_computed = np.zeros((n, p))
    for i in range(n):
        for j in range(p):
            val = 0.0
            for r in range(rank):
                inner = 0.0
                for k in range(m):
                    inner += U[i * m + k, r] * V[k * p + j, r]
                val += inner * W[j * n + i, r]
            C_computed[i, j] = val
    
    if not np.allclose(C_computed, C_expected, atol=1e-10):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
