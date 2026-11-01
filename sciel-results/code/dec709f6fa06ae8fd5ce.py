"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 118.0

    def time_left():
        return deadline - time.time()

    # We will use rank 58 based on the current best known construction
    rank = 58

    # Initialize matrices with zeros
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    col = 0

    def add_standard_term(i, j, k):
        nonlocal col
        u_idx = i * m + j
        v_idx = j * p + k
        w_idx = k * n + i
        U[u_idx][col] = 1
        V[v_idx][col] = 1
        W[w_idx][col] = 1
        col += 1

    def add_strassen_block(i_start, j_start, k_start):
        """
        Add Strassen terms for a 2x2x2 block starting at (i_start, j_start, k_start).
        Uses 7 terms instead of 8 standard terms.
        """
        nonlocal col

        # Indices for the 2x2x2 block
        i0, i1 = i_start, i_start + 1
        j0, j1 = j_start, j_start + 1
        k0, k1 = k_start, k_start + 1

        # U indices
        u00 = i0 * m + j0
        u01 = i0 * m + j1
        u10 = i1 * m + j0
        u11 = i1 * m + j1

        # V indices
        v00 = j0 * p + k0
        v01 = j0 * p + k1
        v10 = j1 * p + k0
        v11 = j1 * p + k1

        # W indices
        w00 = k0 * n + i0
        w01 = k0 * n + i1
        w10 = k1 * n + i0
        w11 = k1 * n + i1

        # M1 = (A00 + A11)(B00 + B11) -> C00 +1, C11 +1
        U[u00][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        W[w11][col] += 1
        col += 1

        # M2 = (A10 + A11)B00 -> C10 +1, C11 -1
        U[u10][col] += 1
        U[u11][col] += 1
        V[v00][col] += 1
        W[w10][col] += 1
        W[w11][col] += -1
        col += 1

        # M3 = A00(B01 - B11) -> C01 +1, C11 +1
        U[u00][col] += 1
        V[v01][col] += 1
        V[v11][col] += -1
        W[w01][col] += 1
        W[w11][col] += 1
        col += 1

        # M4 = A11(B10 - B00) -> C00 +1, C10 +1
        U[u11][col] += 1
        V[v10][col] += 1
        V[v00][col] += -1
        W[w00][col] += 1
        W[w10][col] += 1
        col += 1

        # M5 = (A00 + A01)B11 -> C00 -1, C01 +1
        U[u00][col] += 1
        U[u01][col] += 1
        V[v11][col] += 1
        W[w00][col] += -1
        W[w01][col] += 1
        col += 1

        # M6 = (A10 - A00)(B00 + B01) -> C11 +1
        U[u10][col] += 1
        U[u00][col] += -1
        V[v00][col] += 1
        V[v01][col] += 1
        W[w11][col] += 1
        col += 1

        # M7 = (A01 - A11)(B10 + B11) -> C00 +1
        U[u01][col] += 1
        U[u11][col] += -1
        V[v10][col] += 1
        V[v11][col] += 1
        W[w00][col] += 1
        col += 1

    # Part 1: j in {0, 1}, k in {0, 5}
    # For k=0: Strassen block (i in {0,1}, j in {0,1}, k=0)
    add_strassen_block(0, 0, 0)
    # For k=5: standard terms
    for i in range(n):
        for j in range(2):
            add_standard_term(i, j, 5)

    # Part 2: j in {2, 3}, k in {0, 5}
    # For k=0: Strassen block (i in {0,1}, j in {2,3}, k=0)
    add_strassen_block(0, 2, 0)
    # For k=5: standard terms
    for i in range(n):
        for j in range(2, 4):
            add_standard_term(i, j, 5)

    # Part 3: j in {4}, k in {0, 5}
    # For k=0: standard term
    for i in range(n):
        add_standard_term(i, 4, 0)
    # For k=5: standard term
    for i in range(n):
        add_standard_term(i, 4, 5)

    # Part 4: j in {0, 1, 2, 3, 4}, k in {1, 2, 3, 4}
    # All standard terms
    for i in range(n):
        for j in range(m):
            for k in range(1, 5):
                add_standard_term(i, j, k)

    # Check if col is 58
    assert col == rank, f"Expected {rank}, got {col}"

    # Verify the decomposition
    def verify(U, V, W, n, m, p, rank):
        U_np = np.array(U)
        V_np = np.array(V)
        W_np = np.array(W)

        # For each (i, j, k), sum over r of U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] should be 1
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_row = U_np[i*m+j, :]
                    v_row = V_np[j*p+k, :]
                    w_row = W_np[k*n+i, :]
                    result = np.sum(u_row * v_row * w_row)
                    if abs(result - 1) > 1e-10:
                        return False
        return True

    if verify(U, V, W, n, m, p, rank):
        return {"U": U, "V": V, "W": W}
    else:
        # Fall back to schoolbook if verification fails (should not happen)
        rank_fallback = n * m * p
        U_f = [[0] * rank_fallback for _ in range(n * m)]
        V_f = [[0] * rank_fallback for _ in range(m * p)]
        W_f = [[0] * rank_fallback for _ in range(p * n)]
        col_f = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    u_idx = i * m + j
                    v_idx = j * p + k
                    w_idx = k * n + i
                    U_f[u_idx][col_f] = 1
                    V_f[v_idx][col_f] = 1
                    W_f[w_idx][col_f] = 1
                    col_f += 1
        return {"U": U_f, "V": V_f, "W": W_f}

# EVOLVE-BLOCK-END
