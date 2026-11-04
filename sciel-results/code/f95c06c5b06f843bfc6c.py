import numpy as np
import time

DIMENSIONS = [4, 5, 6]


def _validate(U, V, W, n, m, p, tol=1e-9):
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = float(np.dot(U[i*m+j, :] * V[j*p+k, :], W[k*n+i, :]))
                if abs(val - 1.0) > tol:
                    return False, val
    return True, 0.0


def _strassen_222():
    U = np.zeros((4, 7)); V = np.zeros((4, 7)); W = np.zeros((4, 7))
    U[0,0]=1; U[3,0]=1; V[0,0]=1; V[3,0]=1; W[0,0]=1; W[1,0]=1; W[2,0]=1; W[3,0]=1
    U[2,1]=1; U[3,1]=1; V[0,1]=1; W[2,1]=1; W[3,1]=1
    U[0,2]=1; V[2,2]=1; V[3,2]=-1; W[0,2]=1; W[1,2]=-1
    U[3,3]=1; V[1,3]=1; V[0,3]=-1; W[2,3]=1; W[3,3]=-1
    U[0,4]=1; U[1,4]=1; V[3,4]=1; W[1,4]=1; W[3,4]=1
    U[2,5]=1; U[0,5]=-1; V[0,5]=1; V[1,5]=1; W[2,5]=1; W[3,5]=1
    U[1,6]=1; U[3,6]=-1; V[2,6]=1; V[3,6]=1; W[0,6]=1; W[1,6]=1
    return U, V, W


def _build_444():
    U_S, V_S, W_S = _strassen_222()
    R = 56
    U = np.zeros((16, R)); V = np.zeros((16, R)); W = np.zeros((16, R))
    col = 0
    for i0 in range(2):
        for j0 in range(2):
            for k0 in range(2):
                for c in range(7):
                    for i1 in range(2):
                        for j1 in range(2):
                            U[(i0*2+i1)*4 + (j0*2+j1), col] = U_S[i1*2 + j1, c]
                    for j1 in range(2):
                        for k1 in range(2):
                            V[(j0*2+j1)*4 + (k0*2+k1), col] = V_S[j1*2 + k1, c]
                    for k1 in range(2):
                        for i1 in range(2):
                            W[(k0*2+k1)*4 + (i0*2+i1), col] = W_S[k1*2 + i1, c]
                    col += 1
    return U, V, W


def _build_456():
    n, m, p = 4, 5, 6
    U444, V444, W444 = _build_444()
    r444 = U444.shape[1]
    extras = []
    for i in range(4):
        for k in range(6):
            extras.append((i, 4, k))
    for i in range(4):
        for j in range(4):
            for k in range(4, 6):
                extras.append((i, j, k))
    R = r444 + len(extras)
    U = np.zeros((n*m, R)); V = np.zeros((m*p, R)); W = np.zeros((p*n, R))
    U[:, :r444] = U444
    V[:, :r444] = V444
    W[:, :r444] = W444
    col = r444
    for (i, j, k) in extras:
        U[i*m+j, col] = 1.0
        V[j*p+k, col] = 1.0
        W[k*n+i, col] = 1.0
        col += 1
    return U, V, W


def construct():
    n, m, p = DIMENSIONS
    U, V, W = _build_456()
    valid, err = _validate(U, V, W, n, m, p)
    if not valid:
        rank = n*m*p
        U = np.zeros((n*m, rank)); V = np.zeros((m*p, rank)); W = np.zeros((p*n, rank))
        c = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, c] = 1; V[j*p+k, c] = 1; W[k*n+i, c] = 1; c += 1
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
