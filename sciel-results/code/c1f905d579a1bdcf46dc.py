import numpy as np
import time

DIMENSIONS = [2, 3, 5]

def construct():
    start = time.time()
    deadline = start + 130

    n, m, p = 2, 3, 5
    rank = 28
    U = np.zeros((n*m, rank))
    V = np.zeros((m*p, rank))
    W = np.zeros((p*n, rank))
    col = 0

    def add_strassen(i_off, j_off, k_off):
        nonlocal col
        # m1 = (a00+a11)(b00+b11)
        U[i_off*2+0, col] += 1
        U[(i_off+1)*2+1, col] += 1
        V[j_off*2+0, col] += 1
        V[(j_off+1)*2+1, col] += 1
        W[k_off*2+0, col] += 1
        W[(k_off+1)*2+1, col] += 1
        col += 1
        # m2 = (a10+a11)b00
        U[i_off*2+1, col] += 1
        U[(i_off+1)*2+1, col] += 1
        V[j_off*2+0, col] += 1
        W[(k_off+1)*2+0, col] += 1
        col += 1
        # m3 = a00(b01-b11)
        U[i_off*2+0, col] += 1
        V[j_off*2+1, col] += 1
        V[(j_off+1)*2+1, col] += -1
        W[k_off*2+1, col] += 1
        col += 1
        # m4 = a11(b10-b00)
        U[(i_off+1)*2+1, col] += 1
        V[(j_off+1)*2+0, col] += 1
        V[j_off*2+0, col] += -1
        W[(k_off+1)*2+0, col] += 1
        col += 1
        # m5 = (a00+a10)b11
        U[i_off*2+0, col] += 1
        U[i_off*2+1, col] += 1
        V[(j_off+1)*2+1, col] += 1
        W[k_off*2+0, col] += -1
        col += 1
        # m6 = (a11-a10)(b00+b10)
        U[(i_off+1)*2+1, col] += 1
        U[i_off*2+1, col] += -1
        V[j_off*2+0, col] += 1
        V[(j_off+1)*2+0, col] += 1
        W[(k_off+1)*2+1, col] += 1
        col += 1
        # m7 = (a00-a01)(b01+b11)
        U[i_off*2+0, col] += 1
        U[i_off*2+1, col] += -1
        V[j_off*2+1, col] += 1
        V[(j_off+1)*2+1, col] += 1
        W[k_off*2+0, col] += 1
        W[(k_off+1)*2+1, col] += -1
        col += 1

    add_strassen(0, 0, 0)
    add_strassen(0, 0, 2)

    for i in range(2):
        for j in range(2):
            U[i*2+j, col] = 1
            V[j*2+0, col] = 1
            W[4*2+i, col] = 1
            col += 1

    for i in range(2):
        for k in range(5):
            U[i*2+2, col] = 1
            V[2*5+k, col] = 1
            W[k*2+i, col] = 1
            col += 1

    assert col == 28

    # Verify
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                if abs(s - 1.0) > 1e-9:
                    # Fallback to schoolbook
                    r2 = n*m*p
                    U2 = np.zeros((n*m, r2))
                    V2 = np.zeros((m*p, r2))
                    W2 = np.zeros((p*n, r2))
                    cc = 0
                    for ii in range(n):
                        for jj in range(m):
                            for kk in range(p):
                                U2[ii*m+jj, cc] = 1
                                V2[jj*p+kk, cc] = 1
                                W2[kk*n+ii, cc] = 1
                                cc += 1
                    return {"U": U2.tolist(), "V": V2.tolist(), "W": W2.tolist()}

    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
