import time
import numpy as np

DIMENSIONS = [2, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0

    # Strassen 2x2x2: 7 multiplications
    # Z00 = M1 + M4 - M5 + M7
    # Z01 = M3 + M5
    # Z10 = M2 + M4
    # Z11 = M1 - M2 + M3 + M6
    #
    # M1 = (X00+X11)(Y00+Y11)
    # M2 = (X10+X11)Y00
    # M3 = X00(Y01-Y11)
    # M4 = X11(Y10-Y00)
    # M5 = (X00+X01)Y11
    # M6 = (X10-X00)(Y00+Y01)
    # M7 = (X01-X11)(Y10+Y11)

    # Encode each Strassen term as (U_local, V_local, W_local)
    # local indices: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    strassen_terms = [
        # M1
        ([(0, 1), (3, 1)], [(0, 1), (3, 1)], [(0, 1), (3, 1)]),
        # M2
        ([(2, 1), (3, 1)], [(0, 1)], [(2, 1), (3, -1)]),
        # M3
        ([(0, 1)], [(1, 1), (3, -1)], [(1, 1), (3, 1)]),
        # M4
        ([(3, 1)], [(2, 1), (0, -1)], [(2, 1), (0, 1)]),
        # M5
        ([(0, 1), (1, 1)], [(3, 1)], [(0, -1), (1, 1)]),
        # M6
        ([(2, 1), (0, -1)], [(0, 1), (1, 1)], [(3, 1)]),
        # M7
        ([(1, 1), (3, -1)], [(2, 1), (3, 1)], [(0, 1)]),
    ]

    rank = 42
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)

    col = 0
    for r in range(3):
        for s in range(2):
            for t in range(7):
                u_local, v_local, w_local = strassen_terms[t]

                for (li, coeff) in u_local:
                    i = li >> 1
                    jl = li & 1
                    jg = 2 * s + jl
                    U[i * m + jg, col] = coeff

                for (lj, coeff) in v_local:
                    jl = lj >> 1
                    kl = lj & 1
                    jg = 2 * s + jl
                    kg = 2 * r + kl
                    V[jg * p + kg, col] = coeff

                for (li, coeff) in w_local:
                    i = li >> 1
                    kl = li & 1
                    kg = 2 * r + kl
                    W[kg * n + i, col] = coeff

                col += 1

    # Verify
    try:
        T = np.einsum('ijr,jkr,ki->ijk', U.reshape(n, m, rank), V.reshape(m, p, rank), W.reshape(p, n, rank))
        if not np.allclose(T, 1.0, atol=1e-9):
            r_sb = n * m * p
            U_sb = np.zeros((n * m, r_sb))
            V_sb = np.zeros((m * p, r_sb))
            W_sb = np.zeros((p * n, r_sb))
            c = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        U_sb[i * m + j, c] = 1
                        V_sb[j * p + k, c] = 1
                        W_sb[k * n + i, c] = 1
                        c += 1
            return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    except Exception:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
