import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110

    # Strassen 2x2x2 decomposition (rank 7)
    # Indexing: U[i*2+j], V[j*2+k], W[k*2+i]
    U2 = np.zeros((4, 7), dtype=float)
    V2 = np.zeros((4, 7), dtype=float)
    W2 = np.zeros((4, 7), dtype=float)

    # r=0: M1 = (A00+A11)(B00+B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # r=1: M2 = (A10+A11)B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[1, 1] = 1; W2[3, 1] = -1

    # r=2: M3 = A00(B01-B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[2, 2] = 1; W2[3, 2] = 1

    # r=3: M4 = A11(B10-B00)
    U2[3, 3] = 1
    V2[0, 3] = -1; V2[2, 3] = 1
    W2[0, 3] = 1; W2[1, 3] = 1

    # r=4: M5 = (A00+A01)B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[2, 4] = 1

    # r=5: M6 = (A10-A00)(B00+B01)
    U2[0, 5] = -1; U2[2, 5] = 1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # r=6: M7 = (A01-A11)(B10+B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4x4 Strassen decomposition (two-level 2x2x2, rank 49)
    U4 = np.zeros((16, 49), dtype=float)
    V4 = np.zeros((16, 49), dtype=float)
    W4 = np.zeros((16, 49), dtype=float)

    for ro in range(7):
        for ri in range(7):
            r = ro * 7 + ri
            for uo in range(4):
                uo_val = U2[uo, ro]
                if uo_val != 0:
                    I = uo // 2
                    J = uo % 2
                    for ui in range(4):
                        ui_val = U2[ui, ri]
                        if ui_val != 0:
                            ip = ui // 2
                            jp = ui % 2
                            i = 2 * I + ip
                            j = 2 * J + jp
                            U4[i * 4 + j, r] = uo_val * ui_val
            for vo in range(4):
                vo_val = V2[vo, ro]
                if vo_val != 0:
                    J = vo // 2
                    K = vo % 2
                    for vi in range(4):
                        vi_val = V2[vi, ri]
                        if vi_val != 0:
                            jp = vi // 2
                            kp = vi % 2
                            j = 2 * J + jp
                            k = 2 * K + kp
                            V4[j * 4 + k, r] = vo_val * vi_val
            for wo in range(4):
                wo_val = W2[wo, ro]
                if wo_val != 0:
                    K = wo // 2
                    I = wo % 2
                    for wi in range(4):
                        wi_val = W2[wi, ri]
                        if wi_val != 0:
                            kp = wi // 2
                            ip = wi % 2
                            k = 2 * K + kp
                            i = 2 * I + ip
                            W4[k * 4 + i, r] = wo_val * wi_val

    # Assemble 4x4x8: two 4x4x4 blocks (k=0..3 and k=4..7)
    rank = 98
    U = np.zeros((16, rank), dtype=float)
    V = np.zeros((32, rank), dtype=float)
    W = np.zeros((32, rank), dtype=float)

    # Block 1: k in 0..3
    U[:, 0:49] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + k, 0:49] = V4[j * 4 + k, :]
            W[k * 4 + j, 0:49] = W4[k * 4 + j, :]

    # Block 2: k in 4..7
    U[:, 49:98] = U4
    for j in range(4):
        for k in range(4):
            V[j * 8 + 4 + k, 49:98] = V4[j * 4 + k, :]
            W[(4 + k) * 4 + j, 49:98] = W4[k * 4 + j, :]

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
