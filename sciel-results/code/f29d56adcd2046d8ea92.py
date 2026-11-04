import numpy as np
import time

def construct():
    n, m, p = 4, 4, 8
    start_time = time.time()
    deadline = start_time + 110.0

    # Rank 98 Strassen-based decomposition for 4x4 x 4x8
    # Uses two levels of Strassen: 4x4x4 (rank 49) applied to each half of B
    rank = 98
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))

    # Base 2x2x2 Strassen matrices (rank 7)
    U_2 = np.zeros((4, 7))
    V_2 = np.zeros((4, 7))
    W_2 = np.zeros((4, 7))

    U_2[0, 0] = 1; U_2[3, 0] = 1
    V_2[0, 0] = 1; V_2[3, 0] = 1
    W_2[0, 0] = 1; W_2[3, 0] = 1

    U_2[2, 1] = 1; U_2[3, 1] = 1
    V_2[0, 1] = 1
    W_2[2, 1] = 1; W_2[3, 1] = -1

    U_2[0, 2] = 1
    V_2[1, 2] = 1; V_2[3, 2] = -1
    W_2[1, 2] = 1; W_2[3, 2] = 1

    U_2[3, 3] = 1
    V_2[2, 3] = 1; V_2[0, 3] = -1
    W_2[0, 3] = 1; W_2[2, 3] = 1

    U_2[0, 4] = 1; U_2[1, 4] = 1
    V_2[3, 4] = 1
    W_2[0, 4] = -1; W_2[1, 4] = 1

    U_2[2, 5] = 1; U_2[0, 5] = -1
    V_2[0, 5] = 1; V_2[1, 5] = 1
    W_2[3, 5] = 1

    U_2[1, 6] = 1; U_2[3, 6] = -1
    V_2[2, 6] = 1; V_2[3, 6] = 1
    W_2[0, 6] = 1

    # Build U, V, W for 4x4x8 using Kronecker structure
    for h in range(2):
        for k in range(7):
            for s in range(7):
                col = h * 49 + k * 7 + s

                A_coefs = [
                    [(1,(0,0)), (1,(1,1))],
                    [(1,(1,0)), (1,(1,1))],
                    [(1,(0,0))],
                    [(1,(1,1))],
                    [(1,(0,0)), (1,(0,1))],
                    [(1,(1,0)), (-1,(0,0))],
                    [(1,(0,1)), (-1,(1,1))],
                ][k]

                for coeff, (br, bc) in A_coefs:
                    for r in range(2):
                        for c in range(2):
                            a_idx = (br*2+r) * 4 + (bc*2+c)
                            eff = r*2 + c
                            U[a_idx, col] += coeff * U_2[eff, s]

                B_coefs = [
                    [(1,(0,0)), (1,(1,1))],
                    [(1,(0,0))],
                    [(1,(0,1)), (-1,(1,1))],
                    [(1,(1,0)), (-1,(0,0))],
                    [(1,(1,1))],
                    [(1,(0,0)), (1,(0,1))],
                    [(1,(1,0)), (1,(1,1))],
                ][k]

                for coeff, (br, bc) in B_coefs:
                    for r in range(2):
                        for c in range(2):
                            b_idx = (br*2+r) * 8 + (bc*2+c + h*4)
                            eff = r*2 + c
                            V[b_idx, col] += coeff * V_2[eff, s]

                C_coefs = [
                    [(1,(0,0)), (1,(1,1))],
                    [(1,(1,0)), (-1,(1,1))],
                    [(1,(0,1)), (1,(1,1))],
                    [(1,(0,0)), (1,(1,0))],
                    [(-1,(0,0)), (1,(0,1))],
                    [(1,(1,1))],
                    [(1,(0,0))],
                ][k]

                for coeff, (br, bc) in C_coefs:
                    for r in range(2):
                        for c in range(2):
                            w_idx = (bc*2+c + h*4) * 4 + (br*2+r)
                            eff = r*2 + c
                            W[w_idx, col] += coeff * W_2[eff, s]

                if time.time() > deadline:
                    break
            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
