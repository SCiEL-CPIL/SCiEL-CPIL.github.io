import time
import numpy as np

def construct():
    start_time = time.time()
    deadline = start_time + 8.0

    n, m, p = 4, 4, 4

    # Build the 2x2 Strassen factor matrices (7 multiplications)
    U2 = np.zeros((4, 7), dtype=int)
    V2 = np.zeros((4, 7), dtype=int)
    W2 = np.zeros((4, 7), dtype=int)

    # M0 = (A00 + A11) * (B00 + B11)
    U2[0, 0] = 1; U2[3, 0] = 1
    V2[0, 0] = 1; V2[3, 0] = 1
    W2[0, 0] = 1; W2[3, 0] = 1

    # M1 = (A10 + A11) * B00
    U2[2, 1] = 1; U2[3, 1] = 1
    V2[0, 1] = 1
    W2[2, 1] = 1; W2[3, 1] = -1

    # M2 = A00 * (B01 - B11)
    U2[0, 2] = 1
    V2[1, 2] = 1; V2[3, 2] = -1
    W2[1, 2] = 1; W2[3, 2] = 1

    # M3 = A11 * (B10 - B00)
    U2[3, 3] = 1
    V2[2, 3] = 1; V2[0, 3] = -1
    W2[0, 3] = 1; W2[2, 3] = 1

    # M4 = (A00 + A01) * B11
    U2[0, 4] = 1; U2[1, 4] = 1
    V2[3, 4] = 1
    W2[0, 4] = -1; W2[1, 4] = 1

    # M5 = (A10 - A00) * (B00 + B01)
    U2[2, 5] = 1; U2[0, 5] = -1
    V2[0, 5] = 1; V2[1, 5] = 1
    W2[3, 5] = 1

    # M6 = (A01 - A11) * (B10 + B11)
    U2[1, 6] = 1; U2[3, 6] = -1
    V2[2, 6] = 1; V2[3, 6] = 1
    W2[0, 6] = 1

    # Build 4x4 decomposition via Kronecker product of Strassen factors
    # Rank = 7 * 7 = 49
    U_arr = np.zeros((16, 49), dtype=int)
    V_arr = np.zeros((16, 49), dtype=int)
    W_arr = np.zeros((16, 49), dtype=int)

    # For each pair (s,t) in {0..6}^2, rank index r = s*7+t
    # U[i*4+j, r] = U2[(i//2)*2+(j//2), s] * U2[(i%2)*2+(j%2), t]
    # V[k*4+l, r] = V2[(k//2)*2+(l//2), s] * V2[(k%2)*2+(l%2), t]
    # W[j*4+i, r] = W2[(i//2)*2+(j//2), s] * W2[(i%2)*2+(j%2), t]
    #   (trace convention: W[k*n+i, r] where k=j, n=4, so W[j*4+i, r])

    for s in range(7):
        for t in range(7):
            r = s * 7 + t

            for i in range(4):
                for j in range(4):
                    cb = U2[(i // 2) * 2 + (j // 2), s]
                    ci = U2[(i % 2) * 2 + (j % 2), t]
                    U_arr[i * 4 + j, r] = cb * ci

                    cbv = V2[(i // 2) * 2 + (j // 2), s]
                    civ = V2[(i % 2) * 2 + (j % 2), t]
                    V_arr[i * 4 + j, r] = cbv * civ

                    cbw = W2[(i // 2) * 2 + (j // 2), s]
                    ciw = W2[(i % 2) * 2 + (j % 2), t]
                    W_arr[j * 4 + i, r] = cbw * ciw

            if time.time() > deadline:
                break
        if time.time() > deadline:
            break

    return {
        "U": U_arr.tolist(),
        "V": V_arr.tolist(),
        "W": W_arr.tolist()
    }
