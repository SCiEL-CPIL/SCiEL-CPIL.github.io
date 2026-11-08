import numpy as np
import time

def construct():
    n, m, p = 2, 4, 8
    _deadline = time.time() + 145  # safety margin before 150s kill

    # We'll use a Strassen-based decomposition.
    # Partition the 2x4 matrix into 1x2 blocks (2 blocks)
    # Partition the 4x8 matrix into 2x4 blocks (2 blocks)
    # Partition the 2x8 output into 1x4 blocks (2 blocks)
    #
    # For each of the 4 column groups (k blocks of size 2):
    #   For each of the 2 row blocks (j blocks of size 2):
    #     Apply 7 Strassen multiplies
    #
    # This gives rank = 4 * 2 * 7 = 56

    # Strassen formulas for 2x2 matrix multiplication:
    # A = [[a, b], [c, d]], B = [[e, f], [g, h]]
    # m1 = (a+d)(e+h)
    # m2 = (c+d)e
    # m3 = a(f-h)
    # m4 = d(g-e)
    # m5 = (a+b)h
    # m6 = (c-a)(e+f)
    # m7 = (b-d)(g+h)
    # C[0,0] = m1 + m4 - m5 + m7
    # C[0,1] = m3 + m5
    # C[1,0] = m2 + m4
    # C[1,1] = m1 - m2 + m3 + m6

    num_groups = 4  # k-blocks of size 2
    num_ablocks = 2  # j-blocks of size 2
    num_strassen = 7

    rank = num_groups * num_ablocks * num_strassen  # 56

    # Build U, V, W matrices
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    # Strassen coefficient tables
    # U side: coefficients for a,b,c,d (i=0,1 and j'=0,1)
    # X_forms[s][i*2+jp] where i in {0,1}, jp in {0,1}
    X_forms = np.array([
        [1, 0, 0, 1],    # m1: a + d
        [0, 0, 1, 1],    # m2: c + d
        [1, 0, 0, 0],    # m3: a
        [0, 0, 0, 1],    # m4: d
        [1, 1, 0, 0],    # m5: a + b
        [-1, 0, 1, 0],   # m6: c - a
        [0, 1, 0, -1],   # m7: b - d
    ], dtype=np.int64)

    # V side: coefficients for e,f,g,h (jp=0,1 and kp=0,1)
    # Y_forms[s][jp*2+kp] where jp in {0,1}, kp in {0,1}
    Y_forms = np.array([
        [1, 0, 0, 1],    # m1: e + h
        [1, 0, 0, 0],    # m2: e
        [0, 1, 0, -1],   # m3: f - h
        [-1, 0, 1, 0],   # m4: g - e
        [0, 0, 0, 1],    # m5: h
        [1, 1, 0, 0],    # m6: e + f
        [0, 0, 1, 1],    # m7: g + h
    ], dtype=np.int64)

    # W side: combinations for output
    # Z_combos[output_idx][s] where output_idx in {0,1,2,3} (c00,c01,c10,c11)
    Z_combos = np.array([
        [1, 0, 0, 1, -1, 0, 1],   # c00 = m1 + m4 - m5 + m7
        [0, 0, 1, 0, 1, 0, 0],    # c01 = m3 + m5
        [0, 1, 0, 1, 0, 0, 0],    # c10 = m2 + m4
        [1, -1, 1, 0, 0, 1, 0],   # c11 = m1 - m2 + m3 + m6
    ], dtype=np.int64)

    r = 0
    for g in range(num_groups):
        for b in range(num_ablocks):
            for s in range(num_strassen):
                if time.time() > _deadline:
                    # Return partial result with what we have so far
                    # Trim to current rank
                    U_trim = U[:, :r].tolist()
                    V_trim = V[:, :r].tolist()
                    W_trim = W[:, :r].tolist()
                    return {"U": U_trim, "V": V_trim, "W": W_trim}

                # U: A[i, j] where i in {0,1}, j = b*2 + jp
                for i in range(n):
                    for jp in range(2):
                        a_idx = i * m + b * 2 + jp
                        U[a_idx, r] = X_forms[s, i * 2 + jp]

                # V: B[j, k] where j = b*2 + jp, k = 2*g + kp
                for jp in range(2):
                    for kp in range(2):
                        b_idx = (b * 2 + jp) * p + (2 * g + kp)
                        V[b_idx, r] = Y_forms[s, jp * 2 + kp]

                # W: C[i, k] where i in {0,1}, k = 2*g + kp
                # Output index in Z_combos: i*2 + kp
                for i in range(n):
                    for kp in range(2):
                        k = 2 * g + kp
                        c_idx = k * n + i
                        W[c_idx, r] = Z_combos[i * 2 + kp, s]

                r += 1

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
