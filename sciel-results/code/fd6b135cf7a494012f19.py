import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS
    num_blocks = p // 2  # 8 blocks
    R = num_blocks * 7   # 56

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    # Strassen coefficients for 2x2x2
    # A flat: [A00, A01, A10, A11]
    # B local: [B00, B01, B10, B11]
    # C local: [C00, C01, C10, C11]
    
    # M1 = (A00+A11)*(B00+B11)
    # M2 = (A10+A11)*B00
    # M3 = A00*(B01-B11)
    # M4 = A11*(B10-B00)
    # M5 = (A00+A01)*B11
    # M6 = (A10-A00)*(B00+B01)
    # M7 = (A01-A11)*(B10+B11)
    #
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6

    U_coeffs = [
        {0: 1, 3: 1},      # M1: A00 + A11
        {2: 1, 3: 1},      # M2: A10 + A11
        {0: 1},            # M3: A00
        {3: 1},            # M4: A11
        {0: 1, 1: 1},      # M5: A00 + A01
        {2: 1, 0: -1},     # M6: A10 - A00
        {1: 1, 3: -1}      # M7: A01 - A11
    ]

    V_coeffs = [
        {0: 1, 3: 1},      # M1: B00 + B11
        {0: 1},            # M2: B00
        {1: 1, 3: -1},     # M3: B01 - B11
        {2: 1, 0: -1},     # M4: B10 - B00
        {3: 1},            # M5: B11
        {0: 1, 1: 1},      # M6: B00 + B01
        {2: 1, 3: 1}       # M7: B10 + B11
    ]

    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, -C11
        {1: 1, 3: -1},     # M3 -> C01, -C11  -- wait, let me recheck
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> -C00, C01
        {3: 1},            # M6 -> C11
        {0: 1}             # M7 -> C00
    ]
    
    # Let me re-derive carefully:
    # C00 = M1 - M2 + M3 + M4
    # C01 = M5 + M7
    # C10 = M2 + M4
    # C11 = M1 - M3 - M5 + M6
    
    # So W contributions:
    # M1: C00 += 1, C11 += 1
    # M2: C00 += -1, C10 += 1
    # M3: C00 += 1, C11 += -1
    # M4: C00 += 1, C10 += 1
    # M5: C01 += 1, C11 += -1
    # M6: C11 += 1
    # M7: C01 += 1

    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {0: -1, 2: 1},     # M2 -> -C00, C10
        {0: 1, 3: -1},     # M3 -> C00, -C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {1: 1, 3: -1},     # M5 -> C01, -C11
        {3: 1},            # M6 -> C11
        {1: 1}             # M7 -> C01
    ]

    for q in range(num_blocks):
        # B is 2x16, row-major: idx = j*p + k
        b_00 = 0 * p + 2 * q
        b_01 = 0 * p + (2 * q + 1)
        b_10 = 1 * p + 2 * q
        b_11 = 1 * p + (2 * q + 1)

        # C is 2x16, convention k*n + i
        c_00 = (2 * q) * n + 0
        c_01 = (2 * q + 1) * n + 0
        c_10 = (2 * q) * n + 1
        c_11 = (2 * q + 1) * n + 1

        b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
        c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}

        for s in range(7):
            r = q * 7 + s
            for a_idx, coeff in U_coeffs[s].items():
                U[a_idx, r] = coeff
            for local_idx, coeff in V_coeffs[s].items():
                V[b_map[local_idx], r] = coeff
            for local_out_idx, coeff in W_contribs[s].items():
                W[c_map[local_out_idx], r] = coeff

    # Verify
    if time.time() < deadline - 5:
        np.random.seed(42)
        A_test = np.random.randint(-5, 5, size=(n, m))
        B_test = np.random.randint(-5, 5, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        # C_test[i,k], W convention k*n+i, so C_test.T.flatten() gives k*n+i order
        C_test_flat = C_test.T.flatten()

        if not np.allclose(C_decomp, C_test_flat):
            # Fallback: schoolbook rank 64
            rank = n * m * p
            U = np.zeros((n * m, rank), dtype=int)
            V = np.zeros((m * p, rank), dtype=int)
            W = np.zeros((p * n, rank), dtype=int)
            col = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U[i * m + j, col] = 1
                        V[j * p + k, col] = 1
                        W[k * n + i, col] = 1
                        col += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
