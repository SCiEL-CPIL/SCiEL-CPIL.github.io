import time
import numpy as np

DIMENSIONS = [6, 6, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin before 120s limit

    n, m, p = DIMENSIONS  # 6, 6, 6
    dim = n  # 6
    block_size = 3
    num_blocks = 2  # 6 = 2 * 3

    # Total rank: 7 (Strassen) * 27 (schoolbook 3x3) = 189
    total_rank = 7 * 27

    U = np.zeros((dim * dim, total_rank), dtype=np.float64)
    V = np.zeros((dim * dim, total_rank), dtype=np.float64)
    W = np.zeros((dim * dim, total_rank), dtype=np.float64)

    # Strassen steps for 2x2 block multiplication
    # Each step: (A_coeffs, B_coeffs, C_coeffs)
    # A_coeffs: list of (I, J, coeff) for X = sum A[I,J]*coeff
    # B_coeffs: list of (J, K, coeff) for Y = sum B[J,K]*coeff  
    # C_coeffs: list of (I, K, coeff) for C[I,K] += M*coeff
    strassen_steps = [
        # M1 = (A00 + A11) * (B00 + B11)
        ([(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)], [(0,0,1),(1,1,1)]),
        # M2 = (A10 + A11) * B00
        ([(1,0,1),(1,1,1)], [(0,0,1)], [(1,0,1),(1,1,-1)]),
        # M3 = A00 * (B01 - B11)
        ([(0,0,1)], [(0,1,1),(1,1,-1)], [(0,1,1),(1,1,1)]),
        # M4 = A11 * (B10 - B00)
        ([(1,1,1)], [(1,0,1),(0,0,-1)], [(0,0,1),(1,0,1)]),
        # M5 = (A00 + A01) * B11
        ([(0,0,1),(0,1,1)], [(1,1,1)], [(0,0,-1),(0,1,1)]),
        # M6 = (A10 - A00) * (B00 + B01)
        ([(1,0,1),(0,0,-1)], [(0,0,1),(0,1,1)], [(1,1,1)]),
        # M7 = (A01 - A11) * (B10 + B11)
        ([(0,1,1),(1,1,-1)], [(1,0,1),(1,1,1)], [(0,0,1)]),
    ]

    bs = block_size
    col = 0

    for a_list, b_list, c_list in strassen_steps:
        for r in range(bs):
            for s in range(bs):
                for t in range(bs):
                    # U: left factor, index = i*dim + j
                    # X[r,s] = sum_{I,J} a_coeff * A[I*bs+r, J*bs+s]
                    for (I, J, ca) in a_list:
                        u_idx = (I * bs + r) * dim + (J * bs + s)
                        U[u_idx, col] += ca

                    # V: right factor, index = j*dim + k
                    # Y[s,t] = sum_{J,K} b_coeff * B[J*bs+s, K*bs+t]
                    for (J, K, cb) in b_list:
                        v_idx = (J * bs + s) * dim + (K * bs + t)
                        V[v_idx, col] += cb

                    # W: output factor, trace convention W[k*dim + i, r]
                    # C[I*bs+r, K*bs+t] += c_coeff * M[r,t]
                    for (I, K, cc) in c_list:
                        w_idx = (K * bs + t) * dim + (I * bs + r)
                        W[w_idx, col] += cc

                    col += 1

                    # Periodic time check
                    if col % 27 == 0 and time.time() > deadline:
                        break
                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
