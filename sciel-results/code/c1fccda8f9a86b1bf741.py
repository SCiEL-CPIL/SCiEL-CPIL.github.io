import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 115  # Safety margin before 150s limit

    n, m, p = 2, 2, 11

    # Rank 39: 5 blocks of Strassen (rank 7 each) + 1 schoolbook column (rank 4)
    rank = 39
    U = np.zeros((n * m, rank), dtype=np.int64)
    V = np.zeros((m * p, rank), dtype=np.int64)
    W = np.zeros((p * n, rank), dtype=np.int64)

    r = 0

    # Add 5 Strassen blocks for column pairs (0,1), (2,3), (4,5), (6,7), (8,9)
    for block in range(5):
        k0 = block * 2
        k1 = block * 2 + 1

        # B indices: B[j, k] -> V[j*p + k]
        v_e = k0         # B[0, k0]
        v_f = k1         # B[0, k1]
        v_g = p + k0     # B[1, k0]
        v_h = p + k1     # B[1, k1]

        # W indices: C[i, k] -> W[k*n + i]
        w_c00 = k0 * n    # C[0, k0]
        w_c10 = k0 * n + 1  # C[1, k0]
        w_c01 = k1 * n    # C[0, k1]
        w_c11 = k1 * n + 1  # C[1, k1]

        # A indices: A[i,j] -> U[i*m + j]
        # u00=0, u01=1, u10=2, u11=3

        # m1 = (a+d)(e+h)
        U[0, r] += 1; U[3, r] += 1
        V[v_e, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1; W[w_c11, r] += 1
        r += 1

        # m2 = (c+d)e
        U[2, r] += 1; U[3, r] += 1
        V[v_e, r] += 1
        W[w_c10, r] += 1; W[w_c11, r] += -1
        r += 1

        # m3 = a(f-h)
        U[0, r] += 1
        V[v_f, r] += 1; V[v_h, r] += -1
        W[w_c01, r] += 1; W[w_c11, r] += 1
        r += 1

        # m4 = d(g-e)
        U[3, r] += 1
        V[v_g, r] += 1; V[v_e, r] += -1
        W[w_c00, r] += 1; W[w_c10, r] += 1
        r += 1

        # m5 = (a+b)h
        U[0, r] += 1; U[1, r] += 1
        V[v_h, r] += 1
        W[w_c00, r] += -1; W[w_c01, r] += 1
        r += 1

        # m6 = (c-a)(e+f)
        U[2, r] += 1; U[0, r] += -1
        V[v_e, r] += 1; V[v_f, r] += 1
        W[w_c11, r] += 1
        r += 1

        # m7 = (b-d)(g+h)
        U[1, r] += 1; U[3, r] += -1
        V[v_g, r] += 1; V[v_h, r] += 1
        W[w_c00, r] += 1
        r += 1

        if time.time() > deadline:
            break

    # Last column k=10: schoolbook (4 multiplications)
    if r < rank:
        k2 = 10
        v_e2 = k2          # B[0, 10]
        v_g2 = p + k2      # B[1, 10]
        w_c02 = k2 * n       # C[0, 10]
        w_c12 = k2 * n + 1   # C[1, 10]

        # a * B[0, 10] -> C[0,10]
        U[0, r] += 1
        V[v_e2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # b * B[1, 10] -> C[0,10]
        U[1, r] += 1
        V[v_g2, r] += 1
        W[w_c02, r] += 1
        r += 1

        # c * B[0, 10] -> C[1,10]
        U[2, r] += 1
        V[v_e2, r] += 1
        W[w_c12, r] += 1
        r += 1

        # d * B[1, 10] -> C[1,10]
        U[3, r] += 1
        V[v_g2, r] += 1
        W[w_c12, r] += 1
        r += 1

    # Fast verification using einsum
    if time.time() < deadline - 5:
        try:
            # Build target tensor: T[i*m+j, j*p+k, k*n+i] = 1
            T = np.zeros((n * m, m * p, p * n), dtype=np.int64)
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        T[i * m + j, j * p + k, k * n + i] = 1

            # Reconstruct via decomposition
            T_recon = np.einsum('ar,br,cr->abc', U, V, W)
            if np.array_equal(T_recon, T):
                return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
        except Exception:
            pass

    # Fallback: schoolbook rank 44
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb), dtype=np.int64)
    V_sb = np.zeros((m * p, rank_sb), dtype=np.int64)
    W_sb = np.zeros((p * n, rank_sb), dtype=np.int64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1

    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
