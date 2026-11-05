import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    rank = 26
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    col = 0

    def add_strassen(i1, i2, j1, j2, k1, k2):
        nonlocal col
        # 7 factors for 2x2 * 2x2
        # M1 = (A00+A11)(B00+B11)
        U[i1*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M2 = (A10+A11)B00
        U[i2*m+j1, col] = 1; U[i2*m+j2, col] = 1
        V[j1*p+k1, col] = 1
        W[k1*n+i2, col] = 1; W[k2*n+i2, col] = -1
        col += 1
        # M3 = A00(B01-B11)
        U[i1*m+j1, col] = 1
        V[j1*p+k2, col] = 1; V[j2*p+k2, col] = -1
        W[k2*n+i1, col] = 1; W[k2*n+i2, col] = 1
        col += 1
        # M4 = A11(B10-B00)
        U[i2*m+j2, col] = 1
        V[j2*p+k1, col] = 1; V[j1*p+k1, col] = -1
        W[k1*n+i1, col] = 1; W[k1*n+i2, col] = 1
        col += 1
        # M5 = (A00+A01)B11
        U[i1*m+j1, col] = 1; U[i1*m+j2, col] = 1
        V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = -1; W[k2*n+i1, col] = 1
        col += 1
        # M6 = (A10-A00)(B00+B01)
        U[i2*m+j1, col] = 1; U[i1*m+j1, col] = -1
        V[j1*p+k1, col] = 1; V[j1*p+k2, col] = 1
        W[k2*n+i2, col] = 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[i1*m+j2, col] = 1; U[i2*m+j2, col] = -1
        V[j2*p+k1, col] = 1; V[j2*p+k2, col] = 1
        W[k1*n+i1, col] = 1
        col += 1

    def add_outer(i, j, k):
        nonlocal col
        U[i*m + j, col] = 1
        V[j*p + k, col] = 1
        W[k*n + i, col] = 1
        col += 1

    # Part 1: A[:, 0:2] * B[0:2, 0:2] -> 2x2 * 2x2, Strassen rank 7
    add_strassen(0, 1, 0, 1, 0, 1)  # col 0-6

    # Part 2: A[:, 0:2] * B[0:2, 2:5] -> 2x2 * 2x3, rank 10
    # Using 3x3 Strassen for cols 2,3 + 1 outer for col 4
    # 2x2 * 2x3: A is 2x2 (rows 0,1; cols 0,1), B is 2x3 (rows 0,1; cols 2,3,4)
    # C is 2x3
    # Use Strassen on B[:, 0:2] (cols 2,3) = 7 factors
    # Plus outer product for B[:, 2] (col 4) = 2 factors
    # Total = 9? No, that's not rank 10.
    # Actually 2x2 * 2x3: let me use a different approach.
    # The 2x2x3 tensor: A(2x2) * B(2x3) -> C(2x3)
    # Schoolbook = 12. Strassen on 2x2x2 part (7) + 2 outer = 9.
    # But we can do better with a custom rank-10 or even rank-9.
    # Let's use: Strassen for B cols 2,3 (7 factors) + outer for col 4 (2 factors) = 9
    add_strassen(0, 1, 0, 1, 2, 3)  # col 7-13
    add_outer(0, 0, 4)  # col 14
    add_outer(1, 0, 4)  # col 15
    add_outer(0, 1, 4)  # col 16
    add_outer(1, 1, 4)  # col 17
    # Wait, that's 4 outer products, not 2. Let me reconsider.
    # For 2x2 * 2x1: we need 4 outer products (each A_ij * B_jk)
    # So 7 + 4 = 11, not 10.
    # Hmm, let me reconsider the rank of 2x2x3.
    # Actually, the minimal rank of 2x2x3 is known to be 10 (Bürgisser, Clausen, Shokrollahi).
    # Let me use a known rank-10 decomposition.
    
    # Let me restart with a cleaner approach.
    # Reset
    col = 0

    # Strategy: 
    # 1. A[:, 0:2] * B[0:2, 0:2]: 2x2*2x2, Strassen rank 7
    # 2. A[:, 0:2] * B[0:2, 2:5]: 2x2*2x3, rank 10 (known)
    # 3. A[:, 2] * B[2, 0:5]: 1x3*3x5 = 2x5, rank 10 (5+5 outer)
    # Total: 7 + 10 + 10 = 27? That's not better.
    
    # Alternative:
    # 1. A[:, 0:2] * B[0:2, 0:2]: 2x2*2x2, Strassen rank 7
    # 2. A[:, 0:2] * B[0:2, 2:5]: 2x2*2x3, use 7+4=11 (Strassen+outer) or rank 10
    # 3. A[:, 2] * B[2, 0:5]: 2x5 outer = 10
    # Total: 7+11+10 = 28 (same as before) or 7+10+10 = 27
    
    # Let me try rank 27 with a proper rank-10 for 2x2x3.
    # Known rank-10 decomposition for 2x2 * 2x3:
    # This is non-trivial. Let me use the simpler 7+4=11 approach and see if 27 is achievable differently.
    
    # Actually, let me try a different partition:
    # 1. A[:, 0:3] * B[0:3, 0:2]: 2x3*3x2, can use Strassen-like
    # 2. A[:, 0:3] * B[0:3, 2:5]: 2x3*3x3
    # This gets complicated.
    
    # Let me stick with the proven rank-28 approach but try to shave off 1-2 factors.
    # Reset to rank 28 approach
    rank = 28
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0

    # 1a. Strassen for A[:, 0:2] * B[0:2, 0:2]
    add_strassen(0, 1, 0, 1, 0, 1)

    # 1b. Outer product for A[:, 2] * B[2, 0:2]
    for i in [0, 1]:
        for k in [0, 1]:
            add_outer(i, 2, k)

    # 2a1. Strassen for A[:, 0:2] * B[0:2, 2:4]
    add_strassen(0, 1, 0, 1, 2, 3)

    # 2a2. Outer product for A[:, 0:2] * B[0:2, 4]
    for i in [0, 1]:
        for j in [0, 1]:
            add_outer(i, j, 4)

    # 2b. Outer product for A[:, 2] * B[2, 2:5]
    for i in [0, 1]:
        for k in [2, 3, 4]:
            add_outer(i, 2, k)

    if col != rank:
        raise ValueError(f"Rank mismatch: expected {rank}, got {col}")

    # Verify exactness
    res = np.zeros((n, m, p))
    for c in range(rank):
        U_c = U[:, c].reshape(n, m)
        V_c = V[:, c].reshape(m, p)
        W_c = W[:, c].reshape(p, n)
        res += U_c[:, :, np.newaxis] * V_c[np.newaxis, :, :] * W_c.T[:, np.newaxis, :]

    if not np.allclose(res, 1.0, atol=1e-9):
        # Fallback to schoolbook
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
