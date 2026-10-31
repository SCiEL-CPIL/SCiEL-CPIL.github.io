import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 110.0

    n, m, p = 4, 4, 4

    # Build target tensor
    target = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for k in range(m):
            for j in range(p):
                a_idx = i * m + k
                b_idx = k * p + j
                c_idx = j * n + i
                target[a_idx, b_idx, c_idx] = 1.0

    def verify(U, V, W):
        U_arr = np.array(U, dtype=float)
        V_arr = np.array(V, dtype=float)
        W_arr = np.array(W, dtype=float)
        recon = np.einsum('ir,jr,kr->ijk', U_arr, V_arr, W_arr, optimize=True)
        return np.allclose(recon, target, atol=1e-12)

    def strassen_2x2():
        U2 = np.zeros((4, 7), dtype=float)
        V2 = np.zeros((4, 7), dtype=float)
        W2 = np.zeros((4, 7), dtype=float)

        # M1
        U2[0, 0] = 1; U2[3, 0] = 1
        V2[0, 0] = 1; V2[3, 0] = 1
        W2[0, 0] = 1; W2[3, 0] = 1

        # M2
        U2[2, 1] = 1; U2[3, 1] = 1
        V2[0, 1] = 1
        W2[1, 1] = 1; W2[3, 1] = -1

        # M3
        U2[0, 2] = 1
        V2[1, 2] = 1; V2[3, 2] = -1
        W2[2, 2] = 1; W2[3, 2] = 1

        # M4
        U2[3, 3] = 1
        V2[2, 3] = 1; V2[0, 3] = -1
        W2[0, 3] = 1; W2[1, 3] = 1

        # M5
        U2[0, 4] = 1; U2[1, 4] = 1
        V2[3, 4] = 1
        W2[0, 4] = -1; W2[2, 4] = 1

        # M6
        U2[2, 5] = 1; U2[0, 5] = -1
        V2[0, 5] = 1; V2[1, 5] = 1
        W2[3, 5] = 1

        # M7
        U2[1, 6] = 1; U2[3, 6] = -1
        V2[2, 6] = 1; V2[3, 6] = 1
        W2[0, 6] = 1

        return U2, V2, W2

    def strassen_kronecker_4x4():
        U2, V2, W2 = strassen_2x2()
        rank = 49
        U49 = np.zeros((16, rank), dtype=float)
        V49 = np.zeros((16, rank), dtype=float)
        W49 = np.zeros((16, rank), dtype=float)

        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        a_row = (I * 2 + ii) * 4 + (J * 2 + jj)
                        for s in range(7):
                            for t in range(7):
                                coeff = U2[I * 2 + J, s] * U2[ii * 2 + jj, t]
                                if coeff != 0:
                                    U49[a_row, s * 7 + t] = coeff

        for K in range(2):
            for L in range(2):
                for kk in range(2):
                    for ll in range(2):
                        b_row = (K * 2 + kk) * 4 + (L * 2 + ll)
                        for s in range(7):
                            for t in range(7):
                                coeff = V2[K * 2 + L, s] * V2[kk * 2 + ll, t]
                                if coeff != 0:
                                    V49[b_row, s * 7 + t] = coeff

        for I in range(2):
            for J in range(2):
                for ii in range(2):
                    for jj in range(2):
                        w_row = (J * 2 + jj) * 4 + (I * 2 + ii)
                        for s in range(7):
                            for t in range(7):
                                coeff = W2[I * 2 + J, s] * W2[ii * 2 + jj, t]
                                if coeff != 0:
                                    W49[w_row, s * 7 + t] = coeff

        return U49, V49, W49

    def naive():
        rank = 64
        U = np.zeros((16, rank), dtype=float)
        V = np.zeros((16, rank), dtype=float)
        W = np.zeros((16, rank), dtype=float)
        for i in range(4):
            for k in range(4):
                for j in range(4):
                    r = i * 16 + k * 4 + j
                    U[i * 4 + k, r] = 1.0
                    V[k * 4 + j, r] = 1.0
                    W[j * 4 + i, r] = 1.0
        return U, V, W

    U, V, W = strassen_kronecker_4x4()
    if verify(U.tolist(), V.tolist(), W.tolist()):
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U_n, V_n, W_n = naive()
    if verify(U_n.tolist(), V_n.tolist(), W_n.tolist()):
        return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
        
    return {"U": U_n.tolist(), "V": V_n.tolist(), "W": W_n.tolist()}
