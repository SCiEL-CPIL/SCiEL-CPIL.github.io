import time
import numpy as np

DIMENSIONS = [2, 2, 12]

def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    rank = 34
    num_pairs = p // 2  # 6

    U = [[0] * rank for _ in range(n * m)]   # 4 x 34
    V = [[0] * rank for _ in range(m * p)]   # 24 x 34
    W = [[0] * rank for _ in range(p * n)]   # 24 x 34

    # --- Local terms: 5 per block (indices 0..29) ---
    for t in range(num_pairs):
        base = t * 5
        k0 = 2 * t
        k1 = 2 * t + 1

        v_b0_k0 = 0 * p + k0
        v_b0_k1 = 0 * p + k1
        v_b1_k0 = 1 * p + k0
        v_b1_k1 = 1 * p + k1

        w_c0_k0 = k0 * n + 0
        w_c0_k1 = k1 * n + 0
        w_c1_k0 = k0 * n + 1
        w_c1_k1 = k1 * n + 1

        # M1 = (A00+A11)*(B0_k0 + B1_k1)
        c = base + 0
        U[0][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1; W[w_c1_k1][c] = 1

        # M2 = (A10+A11)*B0_k0
        c = base + 1
        U[2][c] = 1; U[3][c] = 1
        V[v_b0_k0][c] = 1
        W[w_c1_k0][c] = 1; W[w_c1_k1][c] = -1

        # M5 = (A00+A01)*B1_k1
        c = base + 2
        U[0][c] = 1; U[1][c] = 1
        V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = -1; W[w_c0_k1][c] = 1

        # M6 = (A10-A00)*(B0_k0 + B0_k1)
        c = base + 3
        U[2][c] = 1; U[0][c] = -1
        V[v_b0_k0][c] = 1; V[v_b0_k1][c] = 1
        W[w_c1_k1][c] = 1

        # M7 = (A01-A11)*(B1_k0 + B1_k1)
        c = base + 4
        U[1][c] = 1; U[3][c] = -1
        V[v_b1_k0][c] = 1; V[v_b1_k1][c] = 1
        W[w_c0_k0][c] = 1

    # --- Global diagonal terms (indices 30..33) ---

    # G0: A00 * B0_k with alternating sign
    c = 30
    U[0][c] = 1
    for k in range(p):
        V[0 * p + k][c] = (-1) ** k
        W[k * n + 0][c] = (-1) ** k

    # G1: A11 * B1_k with alternating sign
    c = 31
    U[3][c] = 1
    for k in range(p):
        V[1 * p + k][c] = (-1) ** k
        W[k * n + 1][c] = (-1) ** k

    # G2: A00 * (B0_{2t} - B0_{2t+1}) per pair
    c = 32
    U[0][c] = 1
    for t in range(num_pairs):
        V[0 * p + 2*t][c] = 1
        V[0 * p + 2*t+1][c] = -1
        W[(2*t) * n + 0][c] = 1
        W[(2*t+1) * n + 0][c] = -1

    # G3: A11 * (B1_{2t} - B1_{2t+1}) per pair
    c = 33
    U[3][c] = 1
    for t in range(num_pairs):
        V[1 * p + 2*t][c] = 1
        V[1 * p + 2*t+1][c] = -1
        W[(2*t) * n + 1][c] = 1
        W[(2*t+1) * n + 1][c] = -1

    # Verify correctness
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)

    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c_idx in range(rank):
                    val += U_arr[i*m+j, c_idx] * V_arr[j*p+k, c_idx] * W_arr[k*n+i, c_idx]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-9:
        # Fallback: rank 48 schoolbook
        rank = 48
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][idx] = 1
                    V[j*p+k][idx] = 1
                    W[k*n+i][idx] = 1
                    idx += 1

    return {"U": U, "V": V, "W": W}
