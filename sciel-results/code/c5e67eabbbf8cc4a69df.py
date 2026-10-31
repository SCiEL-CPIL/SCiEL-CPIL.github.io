import time
import numpy as np

DIMENSIONS = [4, 4, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 12.0

    n, m, p = DIMENSIONS
    rank = 49

    # Strassen coefficients for 2x2 matrix multiplication
    A_coeff = np.zeros((7, 2, 2), dtype=int)
    B_coeff = np.zeros((7, 2, 2), dtype=int)
    C_coeff = np.zeros((7, 2, 2), dtype=int)

    # s=0: M0 = (A00+A11)(B00+B11)
    A_coeff[0, 0, 0] = 1; A_coeff[0, 1, 1] = 1
    B_coeff[0, 0, 0] = 1; B_coeff[0, 1, 1] = 1
    C_coeff[0, 0, 0] = 1; C_coeff[0, 1, 1] = 1

    # s=1: M1 = (A10+A11)B00
    A_coeff[1, 1, 0] = 1; A_coeff[1, 1, 1] = 1
    B_coeff[1, 0, 0] = 1
    C_coeff[1, 1, 0] = 1; C_coeff[1, 1, 1] = -1

    # s=2: M2 = A00(B01-B11)
    A_coeff[2, 0, 0] = 1
    B_coeff[2, 0, 1] = 1; B_coeff[2, 1, 1] = -1
    C_coeff[2, 0, 1] = 1; C_coeff[2, 1, 1] = 1

    # s=3: M3 = A11(B10-B00)
    A_coeff[3, 1, 1] = 1
    B_coeff[3, 1, 0] = 1; B_coeff[3, 0, 0] = -1
    C_coeff[3, 0, 0] = 1; C_coeff[3, 1, 0] = 1

    # s=4: M4 = (A00+A01)B11
    A_coeff[4, 0, 0] = 1; A_coeff[4, 0, 1] = 1
    B_coeff[4, 1, 1] = 1
    C_coeff[4, 0, 0] = -1; C_coeff[4, 0, 1] = 1

    # s=5: M5 = (A10-A00)(B00+B01)
    A_coeff[5, 1, 0] = 1; A_coeff[5, 0, 0] = -1
    B_coeff[5, 0, 0] = 1; B_coeff[5, 0, 1] = 1
    C_coeff[5, 1, 1] = 1

    # s=6: M6 = (A01-A11)(B10+B11)
    A_coeff[6, 0, 1] = 1; A_coeff[6, 1, 1] = -1
    B_coeff[6, 1, 0] = 1; B_coeff[6, 1, 1] = 1
    C_coeff[6, 0, 0] = 1

    U = np.zeros((16, 49), dtype=float)
    V = np.zeros((16, 49), dtype=float)
    W = np.zeros((16, 49), dtype=float)

    # Precompute all 49 Kronecker-style products for U, V, W
    # For each pair (s, t), compute the 16-element vector
    for s in range(7):
        for t in range(7):
            r = 7 * s + t

            A_outer = A_coeff[s]
            A_inner = A_coeff[t]
            B_outer = B_coeff[s]
            B_inner = B_coeff[t]
            C_outer = C_coeff[s]
            C_inner = C_coeff[t]

            # U: M_U[I, i2, J, j2] = A_outer[I, J] * A_inner[i2, j2]
            # Reshape to (4,4) with row=2*I+i2, col=2*J+j2
            U[:, r] = np.einsum('ij,kl->ikjl', A_outer, A_inner).reshape(16)

            # V: M_V[J, j2, K, k2] = B_outer[J, K] * B_inner[j2, k2]
            # Reshape to (4,4) with row=2*J+j2, col=2*K+k2
            V[:, r] = np.einsum('ij,kl->ikjl', B_outer, B_inner).reshape(16)

            # W: M_W[k, i] where k=2*K+k2, i=2*I+i2
            # M_W[2*K+k2, 2*I+i2] = C_outer[I, K] * C_inner[i2, k2]
            # P[I, K, i2, k2] = C_outer[I, K] * C_inner[i2, k2]
            P = np.einsum('ik,jl->ikjl', C_outer, C_inner)
            # Q[K, k2, I, i2] = P[I, K, i2, k2]
            Q = P.transpose(1, 3, 0, 2)
            W[:, r] = Q.reshape(16)

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
