import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS
    # n=2 (rows of A), m=2 (cols of A = rows of B), p=16 (cols of B)

    # Strassen's rank-7 decomposition for 2x2 matrix multiplication
    # M_{2,2,2} has rank 7
    # U: (n*m, 7) = (4, 7)  - input A indices
    # V: (m*p_block, 7) = (4, 7) - input B indices (per block of 2 columns)
    # W: (p_block*n, 7) = (4, 7) - output C indices

    # Standard Strassen decomposition for 2x2:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # C = [[ae+bg, af+dh], [ce+dg, cf+bh]]
    #
    # Strassen's 7 products:
    # p1 = (a+d)(e+h)
    # p2 = (c+d)e
    # p3 = a(f-h)
    # p4 = d(g-e)
    # p5 = (a+b)h
    # p6 = (c-a)(e+f)
    # p7 = (b-d)(g+h)
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6

    # U factor: maps A indices (a,b,c,d) at positions (0,1,2,3)
    # a -> idx 0, b -> idx 1, c -> idx 2, d -> idx 3
    U_s = np.array([
        [1, 0, 1, 0, 0, 0, 0],  # a
        [0, 1, 0, 0, 1, 0, 0],  # b
        [0, 0, 1, 1, 0, 1, 0],  # c
        [1, 0, 0, 1, 0, 0, 1],  # d
    ], dtype=int)

    # V factor: maps B indices (e,f,g,h) at positions (0,1,2,3) within a block
    # e -> idx 0, f -> idx 1, g -> idx 2, h -> idx 3
    V_s = np.array([
        [1, 0, 0, 1, 0, 1, 0],  # e
        [0, 1, 0, 0, 1, 0, 0],  # f
        [0, 0, 1, 0, 0, 0, 1],  # g
        [1, -1, 0, -1, 0, 1, 1],  # h
    ], dtype=int)

    # W factor: maps output C indices (C11, C12, C21, C22) at positions (0,1,2,3)
    # C11 -> idx 0, C12 -> idx 1, C21 -> idx 2, C22 -> idx 3
    # Using trace convention: W[k*n + i, r] where i is row of C (0,1), k is col of C within block
    # C[i, k] where i in {0,1}, k in {0,1}
    # W index = k*n + i
    # C11 = C[0,0] -> idx 0
    # C12 = C[0,1] -> idx n+0 = 2
    # C21 = C[1,0] -> idx 1
    # C22 = C[1,1] -> idx n+1 = 3
    #
    # C11 = p1 + p4 - p5 + p7
    # C12 = p3 + p5
    # C21 = p2 + p4
    # C22 = p1 - p2 + p3 + p6
    W_s = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # C11 = p1 + p4 - p5 + p7
        [0, 1, 0, 0, 1, 0, 0],    # C12 = p3 + p5
        [0, 1, 1, 0, 0, 0, 0],    # C21 = p2 + p4
        [1, -1, 1, 0, 0, 1, 0],   # C22 = p1 - p2 + p3 + p6
    ], dtype=int)

    rank = 8 * 7  # 8 blocks, 7 terms each

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    for q in range(8):
        for s in range(7):
            r = q * 7 + s

            # U: same for all blocks (A is shared)
            U[:, r] = U_s[:, s]

            # V: B indices for block q are at columns 2q and 2q+1
            # B[j, k_global] where j in {0,1}, k_global in {0,...,15}
            # V index = j*p + k_global
            # For block q: k_global = 2q + k_local, k_local in {0,1}
            for j in range(m):
                for k_local in range(2):
                    v_idx = j * p + (2 * q + k_local)
                    V[v_idx, r] = V_s[j * 2 + k_local, s]

            # W: C indices for block q
            # C[i, k_global] where i in {0,1}, k_global in {0,...,15}
            # W index = k_global*n + i (trace convention)
            # For block q: k_global = 2q + k_local, k_local in {0,1}
            for k_local in range(2):
                for i in range(n):
                    w_idx = (2 * q + k_local) * n + i
                    W[w_idx, r] = W_s[k_local * n + i, s]

    # Verify correctness with a random test
    if time.time() < deadline - 5:
        np.random.seed(42)
        A = np.random.randint(-5, 6, size=(n, m))
        B = np.random.randint(-5, 6, size=(m, p))
        C_expected = A @ B

        # Compute using tensor decomposition
        C_computed = np.zeros((n, p), dtype=int)
        for r in range(rank):
            a_vals = U[:, r] @ A.ravel()  # sum over A indices
            b_vals = V[:, r] @ B.ravel()  # sum over B indices
            # Add contribution to C
            for k_global in range(p):
                for i in range(n):
                    C_computed[i, k_global] += a_vals * b_vals * W[k_global * n + i, r]

        if not np.array_equal(C_expected, C_computed):
            # Fallback to schoolbook if Strassen is wrong
            rank_fallback = n * m * p
            U_f = np.zeros((n * m, rank_fallback), dtype=int)
            V_f = np.zeros((m * p, rank_fallback), dtype=int)
            W_f = np.zeros((p * n, rank_fallback), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_f[i * m + j, col] = 1
                        V_f[j * p + k, col] = 1
                        W_f[k * n + i, col] = 1
                        col += 1
            return {"U": U_f.tolist(), "V": V_f.tolist(), "W": W_f.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
