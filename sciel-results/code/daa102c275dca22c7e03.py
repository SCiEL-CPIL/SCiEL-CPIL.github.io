"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np

DIMENSIONS = [2, 4, 4]

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 115.0

    n, m, p = DIMENSIONS  # 2, 4, 4

    # Build the Strassen-based rank-28 decomposition for 2x4 * 4x4.
    # Split A (2x4) into two 2x2 blocks A1 (cols 0-1), A2 (cols 2-3).
    # Split B (4x4) into four 2x2 blocks B11, B12, B21, B22.
    # C (2x4) = [A1*B11 + A2*B21, A1*B12 + A2*B22]
    # Each 2x2 product uses 7 Strassen terms, so 4*7 = 28 terms.

    rank = 28
    U = [[0] * rank for _ in range(n * m)]   # (8, 28)
    V = [[0] * rank for _ in range(m * p)]   # (16, 28)
    W = [[0] * rank for _ in range(p * n)]   # (8, 28)

    # Strassen 2x2 multiplication terms:
    # M1 = (A11)(B12 - B22)
    # M2 = (A11 + A21)(B22)
    # M3 = (A22 - A11)(B11)
    # M4 = (A22)(B21 - B11)
    # M5 = (A11 + A22)(B11 + B22)
    # M6 = (A12 - A22)(B21 + B22)
    # M7 = (A21 - A11)(B11 + B12)
    # C11 = M5 + M4 - M2 + M6
    # C12 = M1 + M5
    # C21 = M3 + M5
    # C22 = M1 - M4 + M2 + M7

    # Block definitions: (a_col_off, b_row_off, b_col_off, c_col_off)
    blocks = [
        (0, 0, 0, 0),  # Block 11: A[:,0:2] * B[0:2,0:2] -> C[:,0:2]
        (0, 0, 2, 2),  # Block 12: A[:,0:2] * B[0:2,2:4] -> C[:,2:4]
        (2, 2, 0, 0),  # Block 21: A[:,2:4] * B[2:4,0:2] -> C[:,0:2]
        (2, 2, 2, 2),  # Block 22: A[:,2:4] * B[2:4,2:4] -> C[:,2:4]
    ]

    col = 0
    for (a_co, b_ro, b_co, c_co) in blocks:
        # Index mappings:
        # A local (lr, lc) -> U index: lr*m + (a_co + lc) = lr*4 + a_co + lc
        # B local (lr, lc) -> V index: (b_ro + lr)*p + (b_co + lc) = (b_ro+lr)*4 + b_co + lc
        # C local (lr, lc) -> W index: (c_co + lc)*n + lr = (c_co+lc)*2 + lr

        def a_idx(lr, lc):
            return lr * m + (a_co + lc)

        def b_idx(lr, lc):
            return (b_ro + lr) * p + (b_co + lc)

        def c_idx(lr, lc):
            return (c_co + lc) * n + lr

        # M1 = (A11)(B12 - B22)
        U[a_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        V[b_idx(1, 1)][col] -= 1
        W[c_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M2 = (A11 + A21)(B22)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(1, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] -= 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M3 = (A22 - A11)(B11)
        U[a_idx(0, 0)][col] -= 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        W[c_idx(1, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M4 = (A22)(B21 - B11)
        U[a_idx(1, 1)][col] += 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(0, 0)][col] -= 1
        W[c_idx(0, 0)][col] -= 1
        W[c_idx(1, 0)][col] -= 1
        col += 1

        # M5 = (A11 + A22)(B11 + B22)
        U[a_idx(0, 0)][col] += 1
        U[a_idx(1, 1)][col] += 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(0, 1)][col] += 1
        W[c_idx(1, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M6 = (A12 - A22)(B21 + B22)
        U[a_idx(0, 1)][col] += 1
        U[a_idx(1, 1)][col] -= 1
        V[b_idx(1, 0)][col] += 1
        V[b_idx(1, 1)][col] += 1
        W[c_idx(0, 0)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

        # M7 = (A21 - A11)(B11 + B12)
        U[a_idx(0, 0)][col] -= 1
        U[a_idx(1, 0)][col] += 1
        V[b_idx(0, 0)][col] += 1
        V[b_idx(0, 1)][col] += 1
        W[c_idx(1, 1)][col] += 1
        col += 1

    # Verify the decomposition with random integer matrices
    rng = np.random.default_rng(12345)
    A_rand = rng.integers(-5, 5, size=(n, m))
    B_rand = rng.integers(-5, 5, size=(m, p))
    C_expected = A_rand @ B_rand

    U_arr = np.array(U, dtype=np.int64)
    V_arr = np.array(V, dtype=np.int64)
    W_arr = np.array(W, dtype=np.int64)

    # Verification: C[i,k] = sum_r W[k*n+i, r] * (sum_j A[i,j]*U[i*m+j, r]) * (sum_j B[j,k]*V[j*p+k, r])
    A_part = np.zeros((n, rank), dtype=np.int64)
    for i in range(n):
        for r in range(rank):
            for j in range(m):
                A_part[i, r] += A_rand[i, j] * U_arr[i * m + j, r]

    B_part = np.zeros((p, rank), dtype=np.int64)
    for k in range(p):
        for r in range(rank):
            for j in range(m):
                B_part[k, r] += B_rand[j, k] * V_arr[j * p + k, r]

    C_computed = np.zeros((n, p), dtype=np.int64)
    for i in range(n):
        for k in range(p):
            for r in range(rank):
                C_computed[i, k] += W_arr[k * n + i, r] * A_part[i, r] * B_part[k, r]

    if not np.array_equal(C_computed, C_expected):
        # Fallback to schoolbook if Strassen verification fails
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        column = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][column] = 1
                    V[j * p + k][column] = 1
                    W[k * n + i][column] = 1
                    column += 1

    # Try to reduce rank by removing linearly dependent columns
    if time.time() < deadline:
        try:
            U_np = np.array(U, dtype=np.float64)
            V_np = np.array(V, dtype=np.float64)
            W_np = np.array(W, dtype=np.float64)

            # Check for columns where all three factors are simultaneously zero or linearly dependent
            # A column r is removable if the rank of the combined tensor doesn't decrease
            # Simple approach: try removing each column and check if the decomposition still works
            # But this is expensive. Instead, check for trivially dependent columns.

            # Check if any column has all-zero entries in one of the matrices
            # (which would make that term contribute nothing)
            keep = []
            for r in range(rank):
                if np.any(U_np[:, r] != 0) and np.any(V_np[:, r] != 0) and np.any(W_np[:, r] != 0):
                    keep.append(r)

            if len(keep) < rank:
                U = [U[r] for r in keep]
                V = [V[r] for r in keep]
                W = [W[r] for r in keep]

        except Exception:
            pass

    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
