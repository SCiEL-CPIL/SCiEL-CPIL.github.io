import time
import numpy as np

DIMENSIONS = [2, 4, 8]

def construct():
    start_time = time.time()
    deadline = 140

    n, m, p = DIMENSIONS  # 2, 4, 8

    # Strassen 2x2x2x2 rank-7 decomposition
    # Local indices: A: (0,1,2,3) = (a00,a01,a10,a11)
    #                B: (0,1,2,3) = (b00,b01,b10,b11)  
    #                C: (0,1,2,3) = (c00,c01,c10,c11)
    #
    # M1 = (a00+a11)(b00+b11) -> c00 += M1, c11 += M1
    # M2 = (a10+a11)b00       -> c10 += M2, c11 -= M2
    # M3 = a00(b01-b11)       -> c01 += M3, c11 += M3
    # M4 = a11(b10-b00)       -> c00 += M4, c10 += M4
    # M5 = (a00+a01)b11       -> c00 -= M5, c01 += M5
    # M6 = (a10-a00)(b00+b01) -> c11 += M6
    # M7 = (a01-a11)(b10+b11) -> c00 += M7

    U_loc = np.zeros((4, 7))
    V_loc = np.zeros((4, 7))
    W_loc = np.zeros((4, 7))

    # r=0: M1
    U_loc[0, 0] = 1; U_loc[3, 0] = 1
    V_loc[0, 0] = 1; V_loc[3, 0] = 1
    W_loc[0, 0] = 1; W_loc[3, 0] = 1

    # r=1: M2
    U_loc[2, 1] = 1; U_loc[3, 1] = 1
    V_loc[0, 1] = 1
    W_loc[2, 1] = 1; W_loc[3, 1] = -1

    # r=2: M3
    U_loc[0, 2] = 1
    V_loc[1, 2] = 1; V_loc[3, 2] = -1
    W_loc[1, 2] = 1; W_loc[3, 2] = 1

    # r=3: M4
    U_loc[3, 3] = 1
    V_loc[2, 3] = 1; V_loc[0, 3] = -1
    W_loc[0, 3] = 1; W_loc[2, 3] = 1

    # r=4: M5
    U_loc[0, 4] = 1; U_loc[1, 4] = 1
    V_loc[3, 4] = 1
    W_loc[0, 4] = -1; W_loc[1, 4] = 1

    # r=5: M6
    U_loc[2, 5] = 1; U_loc[0, 5] = -1
    V_loc[0, 5] = 1; V_loc[1, 5] = 1
    W_loc[3, 5] = 1

    # r=6: M7
    U_loc[1, 6] = 1; U_loc[3, 6] = -1
    V_loc[2, 6] = 1; V_loc[3, 6] = 1
    W_loc[0, 6] = 1

    total_rank = 56
    U = np.zeros((n * m, total_rank))  # 8 x 56
    V = np.zeros((m * p, total_rank))  # 32 x 56
    W = np.zeros((p * n, total_rank))  # 16 x 56

    # 8 blocks: for each column pair j in {0,1,2,3} and row block s in {0,1}
    # Block (j,s): A uses rows 0,1 cols 2s,2s+1; B uses rows 2s,2s+1 cols 2j,2j+1; C uses cols 2j,2j+1
    for j in range(4):
        for s in range(2):
            base_r = (j * 2 + s) * 7

            # A mapping: local (0,1,2,3) -> global indices
            # local 0 = a00 -> A[0, 2s] -> flat 0*4 + 2s
            # local 1 = a01 -> A[0, 2s+1] -> flat 0*4 + 2s+1
            # local 2 = a10 -> A[1, 2s] -> flat 1*4 + 2s
            # local 3 = a11 -> A[1, 2s+1] -> flat 1*4 + 2s+1
            a_idx = [0 * 4 + 2 * s, 0 * 4 + 2 * s + 1, 1 * 4 + 2 * s, 1 * 4 + 2 * s + 1]

            # B mapping: local (0,1,2,3) -> global indices
            # local 0 = b00 -> B[2s, 2j] -> flat 2s*8 + 2j
            # local 1 = b01 -> B[2s, 2j+1] -> flat 2s*8 + 2j+1
            # local 2 = b10 -> B[2s+1, 2j] -> flat (2s+1)*8 + 2j
            # local 3 = b11 -> B[2s+1, 2j+1] -> flat (2s+1)*8 + 2j+1
            b_idx = [2 * s * 8 + 2 * j, 2 * s * 8 + 2 * j + 1, (2 * s + 1) * 8 + 2 * j, (2 * s + 1) * 8 + 2 * j + 1]

            # C mapping: local (0,1,2,3) -> W indices (k*n+i convention)
            # local 0 = c00 -> C[0, 2j] -> W[2j*2 + 0]
            # local 1 = c01 -> C[0, 2j+1] -> W[(2j+1)*2 + 0]
            # local 2 = c10 -> C[1, 2j] -> W[2j*2 + 1]
            # local 3 = c11 -> C[1, 2j+1] -> W[(2j+1)*2 + 1]
            c_idx = [2 * j * 2 + 0, (2 * j + 1) * 2 + 0, 2 * j * 2 + 1, (2 * j + 1) * 2 + 1]

            for l in range(7):
                r = base_r + l
                for la in range(4):
                    val = U_loc[la, l]
                    if val != 0:
                        U[a_idx[la], r] += val
                for lb in range(4):
                    val = V_loc[lb, l]
                    if val != 0:
                        V[b_idx[lb], r] += val
                for lc in range(4):
                    val = W_loc[lc, l]
                    if val != 0:
                        W[c_idx[lc], r] += val

    if time.time() - start_time > deadline:
        pass

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
