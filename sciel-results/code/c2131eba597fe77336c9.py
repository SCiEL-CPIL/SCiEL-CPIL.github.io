import time
import numpy as np

DIMENSIONS = [2, 2, 16]

def _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed_base=42):
    """Verify that U, V, W correctly decompose the matrix multiplication tensor."""
    for trial in range(num_trials):
        np.random.seed(seed_base + trial)
        A_test = np.random.randint(-10, 10, size=(n, m))
        B_test = np.random.randint(-10, 10, size=(m, p))
        C_test = A_test @ B_test

        A_flat = A_test.flatten()
        B_flat = B_test.flatten()

        UA = U.T @ A_flat
        VB = V.T @ B_flat
        terms = UA * VB
        C_decomp = W @ terms

        # Convention: W[k*n+i] where k=col, i=row
        C_test_flat = C_test.T.flatten()

        if not np.allclose(C_decomp, C_test_flat, atol=1e-9):
            return False
    return True

def _build_strassen_block(R_offset, q, n, p, U, V, W):
    """Build one Strassen rank-7 block for columns 2q, 2q+1."""
    U_coeffs = [
        {0: 1, 3: 1},      # M1
        {2: 1, 3: 1},      # M2
        {0: 1},            # M3
        {3: 1},            # M4
        {0: 1, 1: 1},      # M5
        {2: 1, 0: -1},     # M6
        {1: 1, 3: -1}      # M7
    ]

    V_coeffs = [
        {0: 1, 3: 1},      # M1
        {0: 1},            # M2
        {1: 1, 3: -1},     # M3
        {2: 1, 0: -1},     # M4
        {3: 1},            # M5
        {0: 1, 1: 1},      # M6
        {2: 1, 3: 1}       # M7
    ]

    W_contribs = [
        {0: 1, 3: 1},      # M1 -> C00, C11
        {2: 1, 3: -1},     # M2 -> C10, C11
        {1: 1, 3: 1},      # M3 -> C01, C11
        {0: 1, 2: 1},      # M4 -> C00, C10
        {0: -1, 1: 1},     # M5 -> C00, C01
        {3: 1},            # M6 -> C11
        {0: 1}             # M7 -> C00
    ]

    b_00 = 0 * p + 2 * q
    b_01 = 0 * p + (2 * q + 1)
    b_10 = 1 * p + 2 * q
    b_11 = 1 * p + (2 * q + 1)

    c_00 = (2 * q) * n + 0
    c_01 = (2 * q + 1) * n + 0
    c_10 = (2 * q) * n + 1
    c_11 = (2 * q + 1) * n + 1

    b_map = {0: b_00, 1: b_01, 2: b_10, 3: b_11}
    c_map = {0: c_00, 1: c_01, 2: c_10, 3: c_11}

    for s in range(7):
        r = R_offset + s

        for a_idx, coeff in U_coeffs[s].items():
            U[a_idx, r] = coeff

        for local_idx, coeff in V_coeffs[s].items():
            V[b_map[local_idx], r] = coeff

        for local_out_idx, coeff in W_contribs[s].items():
            W[c_map[local_out_idx], r] = coeff

def _build_strassen_56(n, m, p):
    """Build the rank-56 Strassen-based decomposition."""
    num_blocks = p // 2
    R = num_blocks * 7

    U = np.zeros((n * m, R), dtype=int)
    V = np.zeros((m * p, R), dtype=int)
    W = np.zeros((p * n, R), dtype=int)

    for q in range(num_blocks):
        _build_strassen_block(q * 7, q, n, p, U, V, W)

    return U, V, W

def _build_schoolbook(n, m, p):
    """Build the schoolbook rank-n*m*p decomposition."""
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

    return U, V, W

def construct():
    start_time = time.time()
    deadline = start_time + 110

    n, m, p = DIMENSIONS

    # Build rank-56 Strassen decomposition
    U, V, W = _build_strassen_56(n, m, p)

    # Verify correctness
    if time.time() < deadline - 10:
        if _verify_decomposition(U, V, W, n, m, p, num_trials=10, seed_base=42):
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fallback to schoolbook if Strassen verification fails
    U, V, W = _build_schoolbook(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
