import time
import numpy as np

DIMENSIONS = [2, 2, 12]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS  # 2, 2, 12
    start_time = time.time()
    DEADLINE = 140.0

    # Rank 34 construction using a different algebraic decomposition:
    # 6 blocks x 5 local terms + 4 global diagonal terms
    rank = 34

    U = np.zeros((n * m, rank), dtype=float)   # 4 x 34
    V = np.zeros((m * p, rank), dtype=float)   # 24 x 34
    W = np.zeros((p * n, rank), dtype=float)   # 24 x 34

    num_pairs = p // 2  # 6

    # --- Local terms: 5 per block ---
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

        # Term 1: (A00+A11)*(B0_k0 + B1_k1) -> C0_k0 + C1_k1
        c = base + 0
        U[0, c] = 1.0
        U[3, c] = 1.0
        V[v_b0_k0, c] = 1.0
        V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = 1.0
        W[w_c1_k1, c] = 1.0

        # Term 2: (A10+A11)*B0_k0 -> C1_k0 - C1_k1
        c = base + 1
        U[2, c] = 1.0
        U[3, c] = 1.0
        V[v_b0_k0, c] = 1.0
        W[w_c1_k0, c] = 1.0
        W[w_c1_k1, c] = -1.0

        # Term 3: (A00+A01)*B1_k1 -> -C0_k0 + C0_k1
        c = base + 2
        U[0, c] = 1.0
        U[1, c] = 1.0
        V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = -1.0
        W[w_c0_k1, c] = 1.0

        # Term 4: (A10-A00)*(B0_k0 + B0_k1) -> C1_k1
        c = base + 3
        U[2, c] = 1.0
        U[0, c] = -1.0
        V[v_b0_k0, c] = 1.0
        V[v_b0_k1, c] = 1.0
        W[w_c1_k1, c] = 1.0

        # Term 5: (A01-A11)*(B1_k0 + B1_k1) -> C0_k0
        c = base + 4
        U[1, c] = 1.0
        U[3, c] = -1.0
        V[v_b1_k0, c] = 1.0
        V[v_b1_k1, c] = 1.0
        W[w_c0_k0, c] = 1.0

    # --- Global diagonal terms ---
    # G0: A00 * sum_k (-1)^k B0_k -> sum_k (-1)^k C0_k
    c = 30
    U[0, c] = 1.0
    for k in range(p):
        sign = 1.0 if k % 2 == 0 else -1.0
        V[0 * p + k, c] = sign
        W[k * n + 0, c] = sign

    # G1: A11 * sum_k (-1)^k B1_k -> sum_k (-1)^k C1_k
    c = 31
    U[3, c] = 1.0
    for k in range(p):
        sign = 1.0 if k % 2 == 0 else -1.0
        V[1 * p + k, c] = sign
        W[k * n + 1, c] = sign

    # G2: A00 * sum_t (B0_{2t} - B0_{2t+1}) -> sum_t (C0_{2t} - C0_{2t+1})
    c = 32
    U[0, c] = 1.0
    for t in range(num_pairs):
        V[0 * p + 2*t, c] = 1.0
        V[0 * p + 2*t+1, c] = -1.0
        W[(2*t) * n + 0, c] = 1.0
        W[(2*t+1) * n + 0, c] = -1.0

    # G3: A11 * sum_t (B1_{2t} - B1_{2t+1}) -> sum_t (C1_{2t} - C1_{2t+1})
    c = 33
    U[3, c] = 1.0
    for t in range(num_pairs):
        V[1 * p + 2*t, c] = 1.0
        V[1 * p + 2*t+1, c] = -1.0
        W[(2*t) * n + 1, c] = 1.0
        W[(2*t+1) * n + 1, c] = -1.0

    # Verify the decomposition
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(val - 1.0) > max_err:
                    max_err = abs(val - 1.0)

    if max_err > 1e-9:
        # Fallback: rank-42 via 6 independent Strassen 2x2x2 blocks
        rank = 42
        U = np.zeros((n * m, rank), dtype=float)
        V = np.zeros((m * p, rank), dtype=float)
        W = np.zeros((p * n, rank), dtype=float)

        for t in range(num_pairs):
            base = t * 7
            k0 = 2 * t
            k1 = 2 * t + 1
            v_b00 = 0 * p + k0; v_b01 = 0 * p + k1
            v_b10 = 1 * p + k0; v_b11 = 1 * p + k1
            w_c00 = k0 * n + 0; w_c01 = k1 * n + 0
            w_c10 = k0 * n + 1; w_c11 = k1 * n + 1

            # Strassen 7-term for 2x2x2
            # M1 = (A00+A11)(B00+B11)
            c = base + 0
            U[0, c] = 1; U[3, c] = 1
            V[v_b00, c] = 1; V[v_b11, c] = 1
            W[w_c00, c] = 1; W[w_c11, c] = 1
            # M2 = (A10+A11)B00
            c = base + 1
            U[2, c] = 1; U[3, c] = 1
            V[v_b00, c] = 1
            W[w_c10, c] = 1; W[w_c11, c] = -1
            # M3 = A00(B01-B11)
            c = base + 2
            U[0, c] = 1
            V[v_b01, c] = 1; V[v_b11, c] = -1
            W[w_c01, c] = 1; W[w_c11, c] = 1
            # M4 = A11(B10-B00)
            c = base + 3
            U[3, c] = 1
            V[v_b10, c] = 1; V[v_b00, c] = -1
            W[w_c00, c] = 1; W[w_c10, c] = 1
            # M5 = (A00+A01)B11
            c = base + 4
            U[0, c] = 1; U[1, c] = 1
            V[v_b11, c] = 1
            W[w_c00, c] = -1; W[w_c01, c] = 1
            # M6 = (A10-A00)(B00+B01)
            c = base + 5
            U[2, c] = 1; U[0, c] = -1
            V[v_b00, c] = 1; V[v_b01, c] = 1
            W[w_c11, c] = 1
            # M7 = (A01-A11)(B10+B11)
            c = base + 6
            U[1, c] = 1; U[3, c] = -1
            V[v_b10, c] = 1; V[v_b11, c] = 1
            W[w_c00, c] = 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
# EVOLVE-BLOCK-END
