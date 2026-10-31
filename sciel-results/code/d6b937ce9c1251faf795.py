import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit

    n, m, p = 4, 4, 4
    rank = 49

    U = np.zeros((n * m, rank), dtype=int)
    V = np.zeros((m * p, rank), dtype=int)
    W = np.zeros((p * n, rank), dtype=int)

    # Strassen coefficients for 2x2 multiplication
    # beta[s][I][J]: coefficient of A[I][J] in s-th product
    # gamma[s][J][K]: coefficient of B[J][K] in s-th product
    # delta[s][I][K]: coefficient of C[I][K] contribution from s-th product

    # M0 = (A00+A11)(B00+B11)
    # M1 = (A10+A11)B00
    # M2 = A00(B01-B11)
    # M3 = A11(B10-B00)
    # M4 = (A00+A01)B11
    # M5 = (A10-A00)(B00+B01)
    # M6 = (A01-A11)(B10+B11)
    # C00 = M0+M3-M4+M6
    # C01 = M2+M4
    # C10 = M1+M3
    # C11 = M0-M1+M2+M5

    beta = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, 1]],   # s=1
        [[1, 0], [0, 0]],   # s=2
        [[0, 0], [0, 1]],   # s=3
        [[1, 1], [0, 0]],   # s=4
        [[-1, 0], [1, 0]],  # s=5
        [[0, 1], [0, -1]]   # s=6
    ])

    gamma = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[1, 0], [0, 0]],   # s=1
        [[0, 1], [0, -1]],  # s=2
        [[-1, 0], [1, 0]],  # s=3
        [[0, 0], [0, 1]],   # s=4
        [[1, 1], [0, 0]],   # s=5
        [[0, 1], [1, 1]]    # s=6
    ])

    delta = np.array([
        [[1, 0], [0, 1]],   # s=0
        [[0, 0], [1, -1]],  # s=1
        [[0, 1], [0, 1]],   # s=2
        [[1, 0], [1, 0]],   # s=3
        [[-1, 1], [0, 0]],  # s=4
        [[0, 0], [0, 1]],   # s=5
        [[1, 0], [0, 0]]    # s=6
    ])

    # Reshape to 2x2 blocks
    beta_2d = beta.reshape(7, 2, 2)
    gamma_2d = gamma.reshape(7, 2, 2)
    delta_2d = delta.reshape(7, 2, 2)

    for s in range(7):
        for t in range(7):
            col = s * 7 + t

            # U[i*4+j, col]: i = I*2+ip, j = J*2+jp
            # U coefficient = beta_s[I][J] * beta_t[ip][jp]
            for I in range(2):
                for J in range(2):
                    bs = beta_2d[s, I, J]
                    if bs == 0:
                        continue
                    for ip in range(2):
                        for jp in range(2):
                            bt = beta_2d[t, ip, jp]
                            if bt == 0:
                                continue
                            row_u = (I * 2 + ip) * 4 + (J * 2 + jp)
                            U[row_u, col] = bs * bt

            # V[j*4+k, col]: j = J*2+jp, k = K*2+kp
            # V coefficient = gamma_s[J][K] * gamma_t[jp][kp]
            for J in range(2):
                for K in range(2):
                    gs = gamma_2d[s, J, K]
                    if gs == 0:
                        continue
                    for jp in range(2):
                        for kp in range(2):
                            gt = gamma_2d[t, jp, kp]
                            if gt == 0:
                                continue
                            row_v = (J * 2 + jp) * 4 + (K * 2 + kp)
                            V[row_v, col] = gs * gt

            # W[k*4+i, col]: k = K*2+kp, i = I*2+ip (trace convention)
            # W coefficient = delta_s[I][K] * delta_t[ip][kp]
            for I in range(2):
                for K in range(2):
                    ds = delta_2d[s, I, K]
                    if ds == 0:
                        continue
                    for ip in range(2):
                        for kp in range(2):
                            dt = delta_2d[t, ip, kp]
                            if dt == 0:
                                continue
                            row_w = (K * 2 + kp) * 4 + (I * 2 + ip)
                            W[row_w, col] = ds * dt

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
