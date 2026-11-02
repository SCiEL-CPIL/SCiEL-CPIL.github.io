"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import random

DIMENSIONS = [3, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 100

    def schoolbook():
        rank = n * m * p
        U = [[0]*rank for _ in range(n*m)]
        V = [[0]*rank for _ in range(m*p)]
        W = [[0]*rank for _ in range(p*n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return U, V, W, rank

    U_sb, V_sb, W_sb, rank_sb = schoolbook()

    def check(U, V, W, tol=1e-6):
        nrm = np.array(U, dtype=float)
        vrm = np.array(V, dtype=float)
        wrm = np.array(W, dtype=float)
        for _ in range(3000):
            i = random.randint(0, n-1)
            j = random.randint(0, m-1)
            jp = random.randint(0, m-1)
            k = random.randint(0, p-1)
            ip = random.randint(0, n-1)
            val = float(np.sum(nrm[i*m+j] * vrm[jp*p+k] * wrm[k*n+ip]))
            exp = 1.0 if (i==ip and j==jp) else 0.0
            if abs(val - exp) > tol:
                return False
        return True

    best_U, best_V, best_W, best_rank = U_sb, V_sb, W_sb, rank_sb

    for rank in range(50, rank_sb):
        if time.time() > deadline:
            break
        for trial in range(2):
            if time.time() > deadline:
                break
            np.random.seed(rank*10+trial)
            U = np.random.randn(n*m, rank)*0.3
            V = np.random.randn(m*p, rank)*0.3
            W = np.random.randn(p*n, rank)*0.3

            for it in range(20):
                if time.time() > deadline:
                    break
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((m*p*n, rank))
                        b = np.zeros(m*p*n)
                        for jp in range(m):
                            for k in range(p):
                                for ip in range(n):
                                    idx = jp*p*n + k*n + ip
                                    A[idx] = V[jp*p+k] * W[k*n+ip]
                                    b[idx] = 1.0 if (jp==j and ip==i) else 0.0
                        U[i*m+j],_,_,_ = np.linalg.lstsq(A, b, rcond=1e-12)
                for jp in range(m):
                    for k in range(p):
                        A = np.zeros((n*m*n, rank))
                        b = np.zeros(n*m*n)
                        for i in range(n):
                            for j in range(m):
                                for ip in range(n):
                                    idx = i*m*n + j*n + ip
                                    A[idx] = U[i*m+j] * W[k*n+ip]
                                    b[idx] = 1.0 if (j==jp and ip==i) else 0.0
                        V[jp*p+k],_,_,_ = np.linalg.lstsq(A, b, rcond=1e-12)
                for k in range(p):
                    for ip in range(n):
                        A = np.zeros((n*m*m, rank))
                        b = np.zeros(n*m*m)
                        for i in range(n):
                            for j in range(m):
                                for jp in range(m):
                                    idx = i*m*m + j*m + jp
                                    A[idx] = U[i*m+j] * V[jp*p+k]
                                    b[idx] = 1.0 if (ip==i and jp==j) else 0.0
                        W[k*n+ip],_,_,_ = np.linalg.lstsq(A, b, rcond=1e-12)

            U_r = np.round(U*2)/2
            V_r = np.round(V*2)/2
            W_r = np.round(W*2)/2

            if check(U_r.tolist(), V_r.tolist(), W_r.tolist()):
                best_U, best_V, best_W, best_rank = U_r.tolist(), V_r.tolist(), W_r.tolist(), rank
                break
        if best_rank == rank:
            break

    return {"U": best_U, "V": best_V, "W": best_W}
# EVOLVE-BLOCK-END
