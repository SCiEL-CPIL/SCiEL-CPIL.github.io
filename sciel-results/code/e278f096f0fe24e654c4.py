import time
import numpy as np

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    t0 = time.time()
    DEADLINE = t0 + 110.0  # Safety margin before 120s limit

    U_dim = n * m  # 12
    V_dim = m * p  # 24
    W_dim = p * n  # 18

    # Build the matrix multiplication tensor
    T = np.zeros((U_dim, V_dim, W_dim), dtype=np.int64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1

    def verify(U, V, W):
        T_rec = np.einsum('ir,jr,kr->ijk', U, V, W, optimize=True)
        return np.array_equal(T_rec, T)

    def schoolbook():
        rank = n * m * p
        U = np.zeros((U_dim, rank), dtype=np.int64)
        V = np.zeros((V_dim, rank), dtype=np.int64)
        W = np.zeros((W_dim, rank), dtype=np.int64)
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    def strassen_2x2x2(U, V, W, i0, i1, j0, j1, k0, k1, col_offset):
        a00 = i0 * m + j0
        a01 = i0 * m + j1
        a10 = i1 * m + j0
        a11 = i1 * m + j1
        b00 = j0 * p + k0
        b01 = j0 * p + k1
        b10 = j1 * p + k0
        b11 = j1 * p + k1
        c00 = k0 * n + i0
        c01 = k0 * n + i1
        c10 = k1 * n + i0
        c11 = k1 * n + i1

        col = col_offset
        # M1 = (A00+A11)(B00+B11)
        U[a00, col] += 1; U[a11, col] += 1
        V[b00, col] += 1; V[b11, col] += 1
        W[c00, col] += 1; W[c11, col] += 1
        col += 1
        # M2 = (A10+A11)B00
        U[a10, col] += 1; U[a11, col] += 1
        V[b00, col] += 1
        W[c10, col] += 1; W[c11, col] += 1
        col += 1
        # M3 = A00(B10-B00)
        U[a00, col] += 1
        V[b10, col] += 1; V[b00, col] -= 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M4 = A11(B01-B11)
        U[a11, col] += 1
        V[b01, col] += 1; V[b11, col] -= 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M5 = (A00+A01)B11
        U[a00, col] += 1; U[a01, col] += 1
        V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        # M6 = (A10-A00)(B00+B10)
        U[a10, col] += 1; U[a00, col] -= 1
        V[b00, col] += 1; V[b10, col] += 1
        W[c00, col] += 1; W[c10, col] += 1
        col += 1
        # M7 = (A01-A11)(B10+B11)
        U[a01, col] += 1; U[a11, col] -= 1
        V[b10, col] += 1; V[b11, col] += 1
        W[c01, col] += 1; W[c11, col] += 1
        col += 1
        return col

    def schoolbook_block(U, V, W, i_list, j_list, k_list, col_offset):
        col = col_offset
        for i in i_list:
            for j in j_list:
                for k in k_list:
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return col

    # Build Strassen-based decomposition: rank 66
    total_rank = 6 * 7 + 6 * 4  # 42 + 24 = 66
    U = np.zeros((U_dim, total_rank), dtype=np.int64)
    V = np.zeros((V_dim, total_rank), dtype=np.int64)
    W = np.zeros((W_dim, total_rank), dtype=np.int64)

    col = 0
    # Top 2 rows (i=0,1), 2x2x2 Strassen blocks
    for j0 in [0, 2]:
        j1 = j0 + 1
        for k0 in [0, 2, 4]:
            k1 = k0 + 1
            col = strassen_2x2x2(U, V, W, 0, 1, j0, j1, k0, k1, col)

    # Bottom 1 row (i=2), schoolbook blocks
    for j0 in [0, 2]:
        j1 = j0 + 1
        for k0 in [0, 2, 4]:
            k1 = k0 + 1
            col = schoolbook_block(U, V, W, [2], [j0, j1], [k0, k1], col)

    best_rank = total_rank
    best = {"U": U.copy(), "V": V.copy(), "W": W.copy()}

    if verify(best["U"], best["V"], best["W"]):
        pass  # Good
    else:
        best = schoolbook()
        best_rank = n * m * p

    # Try to find a better decomposition via limited search
    if time.time() < DEADLINE - 20:
        # Try a 3x3x3 based decomposition
        # Partition 3x4x6 as 3x(3+1)x(3+3)
        # Two 3x3x3 blocks + two 3x1x3 blocks
        # If we can get 3x3x3 in rank 23, total = 23+23+9+9 = 64
        # For now, use schoolbook for 3x3x3 (rank 27) and see
        # 27+27+9+9 = 72, not better. Skip.
        pass

    result = {
        "U": best["U"].tolist(),
        "V": best["V"].tolist(),
        "W": best["W"].tolist()
    }
    return result
# EVOLVE-BLOCK-END
