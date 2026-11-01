"""Matrix multiplication 2,2,16 - Strassen-based decomposition."""
DIMENSIONS = [2, 2, 16]

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    n, m, p = DIMENSIONS  # n=2, m=2, p=16
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    num_blocks = p // 2  # 8 blocks of 2 columns
    rank = num_blocks * 7  # 56

    # Build U (4 x rank), V (32 x rank), W (32 x rank)
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]

    for block in range(num_blocks):
        k0 = block * 2
        k1 = k0 + 1
        base = block * 7

        # M1 = (A00+A11)(B0k0+B1k1)
        U[0][base + 0] = 1
        U[3][base + 0] = 1
        V[k0][base + 0] = 1
        V[16 + k1][base + 0] = 1
        W[k0 * 2 + 0][base + 0] = 1
        W[k1 * 2 + 1][base + 0] = 1

        # M2 = (A10+A11)B0k0
        U[2][base + 1] = 1
        U[3][base + 1] = 1
        V[k0][base + 1] = 1
        W[k0 * 2 + 1][base + 1] = 1
        W[k1 * 2 + 1][base + 1] = -1

        # M3 = A00(B1k0-B1k1)
        U[0][base + 2] = 1
        V[16 + k0][base + 2] = 1
        V[16 + k1][base + 2] = -1
        W[k1 * 2 + 0][base + 2] = 1
        W[k1 * 2 + 1][base + 2] = 1

        # M4 = A11(B0k1-B0k0)
        U[3][base + 3] = 1
        V[k1][base + 3] = 1
        V[k0][base + 3] = -1
        W[k0 * 2 + 0][base + 3] = 1
        W[k0 * 2 + 1][base + 3] = 1

        # M5 = (A00+A01)B1k1
        U[0][base + 4] = 1
        U[1][base + 4] = 1
        V[16 + k1][base + 4] = 1
        W[k0 * 2 + 0][base + 4] = -1
        W[k1 * 2 + 0][base + 4] = 1

        # M6 = (A10-A00)(B0k0+B0k1)
        U[2][base + 5] = 1
        U[0][base + 5] = -1
        V[k0][base + 5] = 1
        V[k1][base + 5] = 1
        W[k1 * 2 + 1][base + 5] = 1

        # M7 = (A01-A11)(B1k0+B1k1)
        U[1][base + 6] = 1
        U[3][base + 6] = -1
        V[16 + k0][base + 6] = 1
        V[16 + k1][base + 6] = 1
        W[k0 * 2 + 0][base + 6] = 1

    decomp = {"U": U, "V": V, "W": W}

    # Verify decomposition
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    valid = True
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U_arr[i * m + j], V_arr[j * p + k] * W_arr[k * n + i]))
                if abs(val - 1.0) > 1e-9:
                    valid = False
                    break
            if not valid:
                break
        if not valid:
            break

    if valid:
        return decomp

    # Fallback: schoolbook (should not be reached)
    school_rank = n * m * p
    U_s = [[0] * school_rank for _ in range(n * m)]
    V_s = [[0] * school_rank for _ in range(m * p)]
    W_s = [[0] * school_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_s[i * m + j][col] = 1
                V_s[j * p + k][col] = 1
                W_s[k * n + i][col] = 1
                col += 1
    return {"U": U_s, "V": V_s, "W": W_s}
# EVOLVE-BLOCK-END
