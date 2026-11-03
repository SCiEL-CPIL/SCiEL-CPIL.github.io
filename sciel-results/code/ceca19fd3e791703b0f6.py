import numpy as np
import time

DIMENSIONS = [2, 3, 10]

def _verify(U, V, W, n, m, p):
    """Verify that U, V, W give an exact decomposition of the mult tensor."""
    rank = len(U[0])
    U = np.array(U, dtype=np.float64)
    V = np.array(V, dtype=np.float64)
    W = np.array(W, dtype=np.float64)

    # Check non-zero entries
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                # Compute T[a,b,c]
                val = np.sum(U[a, :] * V[b, :] * W[c, :])
                if abs(val - 1.0) > 1e-9:
                    return False

    # Check zero entries (sampled)
    nonzero_set = set()
    for i in range(n):
        for j in range(m):
            for k in range(p):
                nonzero_set.add((i * m + j, j * p + k, k * n + i))

    for _ in range(100):
        a = np.random.randint(0, n * m)
        b = np.random.randint(0, m * p)
        c = np.random.randint(0, p * n)
        if (a, b, c) not in nonzero_set:
            val = np.sum(U[a, :] * V[b, :] * W[c, :])
            if abs(val) > 1e-9:
                return False

    return True

def _build_strassen55(n, m, p):
    """
    Build the rank-55 decomposition using Strassen on the 2x2 sub-block.
    """
    rank = 55
    U = np.zeros((n * m, rank), dtype=np.int8)
    V = np.zeros((m * p, rank), dtype=np.int8)
    W = np.zeros((p * n, rank), dtype=np.int8)

    col = 0

    # Part A: j=0 (first column of A, first row of B)
    # This is 2x1x10, rank 20
    for i in range(n):
        for k in range(p):
            u_idx = i * m + 0
            v_idx = 0 * p + k
            w_idx = k * n + i
            U[u_idx, col] = 1
            V[v_idx, col] = 1
            W[w_idx, col] = 1
            col += 1

    # Part B: j=1,2 (last 2 cols of A, last 2 rows of B)
    # This is 2x2x10, decomposed into 5 blocks of 2x2x2
    # Each 2x2x2 uses Strassen (rank 7)
    for b in range(5):
        k0 = 2 * b
        k1 = 2 * b + 1

        u_11 = 0 * m + 1
        u_12 = 0 * m + 2
        u_21 = 1 * m + 1
        u_22 = 1 * m + 2

        v_11 = 1 * p + k0
        v_12 = 1 * p + k1
        v_21 = 2 * p + k0
        v_22 = 2 * p + k1

        w_00 = k0 * n + 0
        w_01 = k1 * n + 0
        w_10 = k0 * n + 1
        w_11 = k1 * n + 1

        # M1: (A11 + A22)(B11 + B22)
        U[u_11, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        W[w_11, col] += 1
        col += 1

        # M2: (A21 + A22)B11
        U[u_21, col] += 1
        U[u_22, col] += 1
        V[v_11, col] += 1
        W[w_10, col] += 1
        W[w_11, col] += -1
        col += 1

        # M3: A11(B12 - B22)
        U[u_11, col] += 1
        V[v_12, col] += 1
        V[v_22, col] += -1
        W[w_01, col] += 1
        W[w_11, col] += 1
        col += 1

        # M4: A22(B21 - B11)
        U[u_22, col] += 1
        V[v_21, col] += 1
        V[v_11, col] += -1
        W[w_00, col] += 1
        W[w_10, col] += 1
        col += 1

        # M5: (A11 + A12)B22
        U[u_11, col] += 1
        U[u_12, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += -1
        W[w_01, col] += 1
        col += 1

        # M6: (A21 - A11)(B11 + B12)
        U[u_21, col] += 1
        U[u_11, col] += -1
        V[v_11, col] += 1
        V[v_12, col] += 1
        W[w_11, col] += 1
        col += 1

        # M7: (A12 - A22)(B21 + B22)
        U[u_12, col] += 1
        U[u_22, col] += -1
        V[v_21, col] += 1
        V[v_22, col] += 1
        W[w_00, col] += 1
        col += 1

    assert col == rank, f"Expected {rank} columns, got {col}"
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    # Return the rank-55 solution
    U, V, W = _build_strassen55(n, m, p)
    return {"U": U, "V": V, "W": W}
