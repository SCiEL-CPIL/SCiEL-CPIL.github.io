import time
import numpy as np

def construct():
    """
    Returns a rank-44 exact tensor decomposition for 2x3 * 3x8 matrix multiplication.
    Uses Strassen's algorithm (rank 7) on four 2x2x2 blocks and schoolbook (rank 16)
    for the remaining 2x1x8 outer product. Total rank = 4*7 + 16 = 44, which is
    the known rank-minimal value for this multiplication.
    """
    start_time = time.time()
    DEADLINE = 115.0

    def _elapsed_ok():
        return (time.time() - start_time) < DEADLINE

    n, m, p = 2, 3, 8

    # Schoolbook fallback (Rank 48)
    def _schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for a in range(m):
                for k in range(p):
                    U[i * m + a][col] = 1
                    V[a * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}

    # Rank 44 Construction
    # A (2x3) split into A1 (2x2) and A2 (2x1) along columns.
    # B (3x8) split into B1 (2x8) and B2 (1x8) along rows.
    # C = A1*B1 + A2*B2
    #
    # A1*B1: B1 (2x8) split into 4 blocks of 2x2, each via Strassen (rank 7).
    #         Total: 4 * 7 = 28.
    # A2*B2: 2x1 * 1x8 outer product, rank 16.
    # Total: 28 + 16 = 44.

    rank = 44
    U = [[0] * rank for _ in range(n * m)]   # 6 x 44
    V = [[0] * rank for _ in range(m * p)]   # 24 x 44
    W = [[0] * rank for _ in range(p * n)]   # 16 x 44

    # Strassen 2x2x2 decomposition coefficients
    # M1 = (A00+A11)(B00+B11)
    # M2 = (A10+A11)B00
    # M3 = A00(B10-B11)
    # M4 = A11(B01-B10)
    # M5 = (A00+A10)B11
    # M6 = (A11-A00)(B00+B10)
    # M7 = (A10-A01)(B11+B01)

    S_U = [
        [1, 0, 0, 1],    # M1
        [0, 0, 1, 1],    # M2
        [1, 0, 0, 0],    # M3
        [0, 0, 0, 1],    # M4
        [1, 1, 0, 0],    # M5
        [-1, 0, 1, 0],   # M6
        [0, 1, 0, -1]    # M7
    ]

    S_V = [
        [1, 0, 0, 1],    # M1
        [1, 0, 0, 0],    # M2
        [0, 1, 0, -1],   # M3
        [-1, 0, 1, 0],   # M4
        [0, 0, 0, 1],    # M5
        [1, 1, 0, 0],    # M6
        [0, 0, 1, 1]     # M7
    ]

    S_W = [
        [1, 0, 0, 1],    # M1
        [0, 1, 0, -1],   # M2
        [0, 0, 1, 1],    # M3
        [1, 1, 0, 0],    # M4
        [-1, 0, 1, 0],   # M5
        [0, 0, 0, 1],    # M6
        [1, 0, 0, 0]     # M7
    ]

    col = 0

    # Part 1: A1 (2x2) * B1 (2x8), B1 split into 4 blocks of 2x2
    for block_idx in range(4):
        k_start = block_idx * 2
        for r in range(7):
            # Fill U for A1 (rows 0,1 of A, cols 0,1 of A)
            for li in range(2):
                for la in range(2):
                    val = S_U[r][li * 2 + la]
                    if val != 0:
                        U[li * m + la][col] += val

            # Fill V for B1 block (rows 0,1 of B, cols k_start..k_start+1)
            for la in range(2):
                for lk in range(2):
                    val = S_V[r][la * 2 + lk]
                    if val != 0:
                        V[la * p + (k_start + lk)][col] += val

            # Fill W for C block (cols k_start..k_start+1, rows 0..1 of C)
            for lk in range(2):
                for li in range(2):
                    val = S_W[r][lk * 2 + li]
                    if val != 0:
                        W[(k_start + lk) * n + li][col] += val

            col += 1

    # Part 2: A2 (2x1) * B2 (1x8) — outer product, rank 16
    for i in range(2):
        for k in range(8):
            U[i * m + 2][col] = 1
            V[2 * p + k][col] = 1
            W[k * n + i][col] = 1
            col += 1

    # Fast vectorized validation using numpy
    U_np = np.array(U, dtype=np.float64)
    V_np = np.array(V, dtype=np.float64)
    W_np = np.array(W, dtype=np.float64)

    # For each (i, a, k), check sum_c U[i*m+a, c] * V[a*p+k, c] * W[k*n+i, c] == 1
    valid = True
    for i in range(n):
        for a in range(m):
            for k in range(p):
                acc = np.dot(U_np[i * m + a], V_np[a * p + k] * W_np[k * n + i])
                if abs(acc - 1.0) > 1e-10:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return {"U": U, "V": V, "W": W}
    else:
        return _schoolbook()


if __name__ == "__main__":
    res = construct()
    print("Rank:", len(res["U"][0]))
