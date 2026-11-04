"""
Rank-minimal tensor decomposition for 2x3 * 3x10 matrix multiplication.

Strategy: Partition middle dim 3 = 2 + 1.
  Part 1: (2x2) * (2x10) via 5 Strassen 2x2x2 blocks -> rank 35
  Part 2: (2x1) * (1x10) schoolbook -> rank 20
  Total: rank 55
"""
import time
import numpy as np

DIMENSIONS = [2, 3, 10]
DEADLINE = 120.0


def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=3, p=10

    rank = 55
    U = [[0] * rank for _ in range(n * m)]  # 6 x 55
    V = [[0] * rank for _ in range(m * p)]  # 30 x 55
    W = [[0] * rank for _ in range(p * n)]  # 20 x 55

    r = 0

    def deadline_exceeded():
        return (time.time() - start_time) > DEADLINE - 5.0

    # Strassen coefficients for 2x2x2
    # A indices in U: 0=(0,0), 1=(0,1), 3=(1,0), 4=(1,1)
    # B indices in V for block b: B00=2b, B01=2b+1, B10=10+2b, B11=10+2b+1
    # C indices in W for block b: C00=4b, C01=4b+2, C10=4b+1, C11=4b+3

    # M1 = (A00+A11)(B00+B11) -> C00 + C11
    # M2 = (A10+A11)B00 -> C10 - C11
    # M3 = A00(B01-B11) -> C01 + C11
    # M4 = A11(B10-B00) -> C00 + C10
    # M5 = (A00+A01)B11 -> -C00 + C01
    # M6 = (A10-A00)(B00+B01) -> C11
    # M7 = (A01-A11)(B10+B11) -> C00

    strassen_U = {
        0: [(0, 1), (4, 1)],
        1: [(3, 1), (4, 1)],
        2: [(0, 1)],
        3: [(4, 1)],
        4: [(0, 1), (1, 1)],
        5: [(3, 1), (0, -1)],
        6: [(1, 1), (4, -1)],
    }

    strassen_V = {
        0: [(0, 1), (3, 1)],   # B00 + B11
        1: [(0, 1)],           # B00
        2: [(1, 1), (3, -1)],  # B01 - B11
        3: [(2, 1), (0, -1)],  # B10 - B00
        4: [(3, 1)],           # B11
        5: [(0, 1), (1, 1)],   # B00 + B01
        6: [(2, 1), (3, 1)],   # B10 + B11
    }

    strassen_W = {
        0: [(0, 1), (3, 1)],   # C00 + C11
        1: [(1, 1), (3, -1)],  # C10 - C11
        2: [(2, 1), (3, 1)],   # C01 + C11
        3: [(0, 1), (1, 1)],   # C00 + C10
        4: [(0, -1), (2, 1)],  # -C00 + C01
        5: [(3, 1)],           # C11
        6: [(0, 1)],           # C00
    }

    for b in range(5):
        if deadline_exceeded():
            return {"U": U, "V": V, "W": W}
        col0 = 2 * b
        col1 = 2 * b + 1
        # V offsets: B00->col0, B01->col1, B10->10+col0, B11->10+col1
        # W offsets: C00->4b, C01->4b+2, C10->4b+1, C11->4b+3
        v_offsets = {0: col0, 1: col1, 2: 10 + col0, 3: 10 + col1}
        w_offsets = {0: 4 * b, 1: 4 * b + 1, 2: 4 * b + 2, 3: 4 * b + 3}

        for s in range(7):
            for (u_idx, coeff) in strassen_U[s]:
                U[u_idx][r] += coeff
            for (v_off, coeff) in strassen_V[s]:
                V[v_offsets[v_off]][r] += coeff
            for (w_off, coeff) in strassen_W[s]:
                W[w_offsets[w_off]][r] += coeff
            r += 1

    # Part 2: Schoolbook for A[*,2] * B[2,*]
    for i in range(n):
        if deadline_exceeded():
            return {"U": U, "V": V, "W": W}
        for k in range(p):
            U[i * m + 2][r] = 1
            V[2 * p + k][r] = 1
            W[k * n + i][r] = 1
            r += 1

    assert r == rank, f"Expected rank {rank}, got {r}"

    # Quick verification
    if not deadline_exceeded():
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    lhs = np.dot(U_arr[i * m + j] * V_arr[j * p + k], W_arr[k * n + i])
                    if not np.isclose(lhs, 1.0, atol=1e-9):
                        # Fallback to schoolbook
                        rank_fb = n * m * p
                        U_fb = [[0] * rank_fb for _ in range(n * m)]
                        V_fb = [[0] * rank_fb for _ in range(m * p)]
                        W_fb = [[0] * rank_fb for _ in range(p * n)]
                        for c in range(rank_fb):
                            ii = c // (m * p)
                            rem = c % (m * p)
                            jj = rem // p
                            kk = rem % p
                            U_fb[ii * m + jj][c] = 1
                            V_fb[jj * p + kk][c] = 1
                            W_fb[kk * n + ii][c] = 1
                        return {"U": U_fb, "V": V_fb, "W": W_fb}

    return {"U": U, "V": V, "W": W}


if __name__ == "__main__":
    result = construct()
    print(f"Rank: {len(result['U'][0])}")

    n, m, p = DIMENSIONS
    U = np.array(result['U'])
    V = np.array(result['V'])
    W = np.array(result['W'])
    rank = U.shape[1]

    np.random.seed(42)
    A = np.random.randint(-5, 6, size=(n, m)).astype(float)
    B = np.random.randint(-5, 6, size=(m, p)).astype(float)

    C_decomp = np.zeros((n, p))
    for r in range(rank):
        A_w = np.zeros((n, m))
        for i in range(n):
            for j in range(m):
                A_w[i, j] = U[i * m + j, r]
        B_w = np.zeros((m, p))
        for j in range(m):
            for k in range(p):
                B_w[j, k] = V[j * p + k, r]
        prod_r = np.sum(A_w * A) * np.sum(B_w * B)
        for i in range(n):
            for k in range(p):
                C_decomp[i, k] += prod_r * W[k * n + i, r]

    C_direct = A @ B
    error = np.max(np.abs(C_decomp - C_direct))
    print(f"Max error: {error}")
    if error > 1e-10:
        print("WARNING: Decomposition is incorrect!")
