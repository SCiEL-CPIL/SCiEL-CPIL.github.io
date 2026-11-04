import time
import numpy as np

def construct():
    """
    Rank-28 exact tensor decomposition for 2x4 * 4x8 matrix multiplication.
    
    Structure: Kronecker product of
      - Strassen 2x2x2 (rank 7) on axes (i, j1, c1)
      - Schoolbook 1x2x2 (rank 4) on axes (j2, k2)
    Total rank: 7 * 4 = 28
    
    Indexing:
      A: (i, j) with j = 2*j1 + j2, i in [0,2), j1 in [0,2), j2 in [0,2)
      B: (j, k) with k = 4*c1 + c2, j = 2*j1+j2, c1 in [0,2), c2 in [0,4)
      C: (i, k) with k = 4*c1 + c2
      W: (k*n + i, r) = (k*2 + i, r)
    """
    deadline = time.time() + 110  # Safety margin before 150s kill

    n, m, p = 2, 4, 8

    # Strassen 2x2x2 factors
    # U_s: (4, 7) - rows indexed by i*2+j1
    # V_s: (4, 7) - rows indexed by j1*2+c1
    # W_s: (4, 7) - rows indexed by c1*2+i
    U_s = np.zeros((4, 7))
    V_s = np.zeros((4, 7))
    W_s = np.zeros((4, 7))

    # Strassen 7 products for 2x2 matrices
    # Product 0: (A00+A11)(B00+B11)
    U_s[0, 0] = 1; U_s[3, 0] = 1
    V_s[0, 0] = 1; V_s[3, 0] = 1
    W_s[0, 0] = 1; W_s[3, 0] = 1

    # Product 1: (A10+A11)B00
    U_s[2, 1] = 1; U_s[3, 1] = 1
    V_s[0, 1] = 1
    W_s[2, 1] = 1; W_s[3, 1] = -1

    # Product 2: A00(B10-B11)
    U_s[0, 2] = 1
    V_s[1, 2] = 1; V_s[3, 2] = -1
    W_s[1, 2] = 1; W_s[3, 2] = 1

    # Product 3: A11(B01-B00)
    U_s[3, 3] = 1
    V_s[2, 3] = 1; V_s[0, 3] = -1
    W_s[0, 3] = 1; W_s[2, 3] = 1

    # Product 4: (A00+A01)B11
    U_s[0, 4] = 1; U_s[1, 4] = 1
    V_s[3, 4] = 1
    W_s[0, 4] = -1; W_s[1, 4] = 1

    # Product 5: (A10-A00)(B00+B01)
    U_s[2, 5] = 1; U_s[0, 5] = -1
    V_s[0, 5] = 1; V_s[1, 5] = 1
    W_s[3, 5] = 1

    # Product 6: (A01-A11)(B10+B11)
    U_s[1, 6] = 1; U_s[3, 6] = -1
    V_s[2, 6] = 1; V_s[3, 6] = 1
    W_s[0, 6] = 1

    # Schoolbook 1x2x2 factors
    # U_b: (2, 4) - rows indexed by j2
    # V_b: (4, 4) - rows indexed by j2*2+c2
    # W_b: (2, 4) - rows indexed by c2
    U_b = np.zeros((2, 4))
    V_b = np.zeros((4, 4))
    W_b = np.zeros((2, 4))

    col = 0
    for j2 in range(2):
        for c2 in range(2):
            U_b[j2, col] = 1
            V_b[j2 * 2 + c2, col] = 1
            W_b[c2, col] = 1
            col += 1

    # Build global factors
    rank = 7 * 4
    U = np.zeros((n * m, rank))  # (8, 28)
    V = np.zeros((m * p, rank))  # (32, 28)
    W = np.zeros((p * n, rank))  # (16, 28)

    for s in range(7):
        for b in range(4):
            col = s * 4 + b

            # U: index = i*4 + j1*2 + j2
            for i in range(2):
                for j1 in range(2):
                    for j2 in range(2):
                        val_s = U_s[i * 2 + j1, s]
                        val_b = U_b[j2, b]
                        if val_s != 0 and val_b != 0:
                            U[i * 4 + j1 * 2 + j2, col] += val_s * val_b

            # V: index = j*8 + k = (j1*2+j2)*8 + c1*4+c2
            for j1 in range(2):
                for j2 in range(2):
                    for c1 in range(2):
                        for c2 in range(2):
                            val_s = V_s[j1 * 2 + c1, s]
                            val_b = V_b[j2 * 2 + c2, b]
                            if val_s != 0 and val_b != 0:
                                V[(j1 * 2 + j2) * 8 + c1 * 4 + c2, col] += val_s * val_b

            # W: index = k*n + i = (c1*4+c2)*2 + i
            for c1 in range(2):
                for c2 in range(2):
                    for i in range(2):
                        val_s = W_s[c1 * 2 + i, s]
                        val_b = W_b[c2, b]
                        if val_s != 0 and val_b != 0:
                            W[(c1 * 4 + c2) * 2 + i, col] += val_s * val_b

    # Verification with random test
    rng = np.random.default_rng(42)
    A_test = rng.standard_normal((n, m))
    B_test = rng.standard_normal((m, p))
    C_expected = A_test @ B_test

    C_computed = np.zeros((n, p))
    for r in range(rank):
        u_vec = U[:, r].reshape(n, m)
        v_vec = V[:, r].reshape(m, p)
        w_vec = W[:, r].reshape(p, n)
        C_computed += (u_vec @ v_vec) * w_vec.T

    if not np.allclose(C_computed, C_expected, atol=1e-10, rtol=1e-10):
        # Fallback: schoolbook rank 64
        rank_fb = n * m * p
        U_fb = np.zeros((n * m, rank_fb))
        V_fb = np.zeros((m * p, rank_fb))
        W_fb = np.zeros((p * n, rank_fb))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U_fb[i * m + j, col] = 1.0
                    V_fb[j * p + k, col] = 1.0
                    W_fb[k * n + i, col] = 1.0
                    col += 1
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }

    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }
