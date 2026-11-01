import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 100  # Safety margin before 150s limit

    n, m, p = DIMENSIONS

    # The problem is to decompose the tensor for C = A @ B
    # A is 2x2, B is 2x16, C is 2x16.
    # This is equivalent to 8 independent 2x2 matrix multiplications
    # because the inner dimension m=2 is small and fixed, and the
    # columns of B and C can be grouped into 8 pairs.
    # Rank(M_2,2,2) = 7 (Strassen).
    # Rank(M_2,2,16) = 8 * 7 = 56.

    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for one 2x2 block
    # A indices: 0(A00), 1(A01), 2(A10), 3(A11)
    # B local: 0(B00), 1(B01), 2(B10), 3(B11)
    # C local: 0(C00), 1(C01), 2(C10), 3(C11)

    # U_coeffs[s][a_idx] = coeff
    U_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]

    # V_coeffs[s][b_local_idx] = coeff
    V_coeffs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]

    # W_contribs[s][c_local_idx] = coeff
    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, C11
        {1: 1, 3: 1},      # M3 -> C01, C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> C00, C01
        {3: 1},            # M6 -> C11
        {0: 1}             # M7 -> C00
    ]

    for q in range(num_blocks):
        # Global indices for B in block q
        # B is 2x16. Row-major flatten: j*p + k.
        # Block q covers columns 2q and 2q+1.
        b_00 = 0 * p + 2 * q       # B[0, 2q]
        b_01 = 0 * p + (2 * q + 1) # B[0, 2q+1]
        b_10 = 1 * p + 2 * q       # B[1, 2q]
        b_11 = 1 * p + (2 * q + 1) # B[1, 2q+1]

        # Global indices for C in block q
        # C is 2x16. Convention: W[k*n+i, r] corresponds to C[i, k]
        # C is 2x16. k is the column index (0..15), i is the row index (0..1)
        # W index = k * n + i = k * 2 + i
        # Block q covers columns 2q and 2q+1.
        c_00 = (2 * q) * n + 0     # C[0, 2q] -> W index (2q)*2+0 = 4q
        c_01 = (2 * q + 1) * n + 0 # C[0, 2q+1] -> W index (2q+1)*2+0 = 4q+2
        c_10 = (2 * q) * n + 1     # C[1, 2q] -> W index (2q)*2+1 = 4q+1
        c_11 = (2 * q + 1) * n + 1 # C[1, 2q+1] -> W index (2q+1)*2+1 = 4q+3

        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}

        for s in range(7):
            r = q * 7 + s

            # Fill U
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff

            # Fill V
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff

            # Fill W
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff

    # Verification with exact integer arithmetic (fast path)
    if time.time() < deadline - 2:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        # Compute C_decomp = W @ (U.T @ A_flat) * (V.T @ B_flat)
        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        # C convention: W[k*n+i, r] corresponds to C[i, k]
        # So C_decomp[4q] = C[0, 2q], C_decomp[4q+1] = C[1, 2q], etc.
        # C_test is (n, p) = (2, 16), so C_test[i, k]
        # We need to flatten C_test in the order C_test[0,0], C_test[1,0], C_test[0,1], C_test[1,1], ...
        # That's C_test.T.flatten()
        C_test_flat = C_test.T.flatten()

        # Exact integer comparison
        if not np.array_equal(C_decomp, C_test_flat):
            # Fallback to schoolbook if Strassen fails (should not happen)
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)

            column = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        u_idx = i * m + j
                        v_idx = j * p + k
                        w_idx = k * n + i
                        U[u_idx, column] = 1
                        V[v_idx, column] = 1
                        W[w_idx, column] = 1
                        column += 1

    # Ensure we return before deadline
    if time.time() > deadline - 1:
        pass  # Already built, just return

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
