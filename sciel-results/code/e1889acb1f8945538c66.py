import time
import numpy as np

DEADLINE = 150.0
START = time.time()

def _remaining():
    return DEADLINE - (time.time() - START)

def _build_seed_36():
    n, m, p = 2, 4, 5
    strassen_U = [
        [1,0,0,1],[0,0,1,1],[1,0,0,0],[0,0,0,1],
        [1,1,0,0],[-1,0,1,0],[0,1,0,-1],
    ]
    strassen_V = [
        [1,0,0,1],[1,0,0,0],[0,1,0,-1],[-1,0,1,0],
        [0,0,0,1],[1,1,0,0],[0,0,1,1],
    ]
    strassen_W = [
        [1,0,0,1],[0,1,0,-1],[0,0,1,1],[1,1,0,0],
        [-1,0,1,0],[0,0,0,1],[1,0,0,0],
    ]
    rank = 4*7 + 8
    U = [[0]*rank for _ in range(n*m)]
    V = [[0]*rank for _ in range(m*p)]
    W = [[0]*rank for _ in range(p*n)]
    col = 0
    for j0 in (0,2):
        for k0 in (0,2):
            for r in range(7):
                for li in range(2):
                    for lj in range(2):
                        U[li*m + (j0+lj)][col] = strassen_U[r][li*2+lj]
                for lj in range(2):
                    for lk in range(2):
                        V[(j0+lj)*p + (k0+lk)][col] = strassen_V[r][lj*2+lk]
                for lk in range(2):
                    for li in range(2):
                        W[(k0+lk)*n + li][col] = strassen_W[r][lk*2+li]
                col += 1
    for i in range(n):
        for j in range(m):
            U[i*m+j][col] = 1
            V[j*p+4][col] = 1
            W[4*n+i][col] = 1
            col += 1
    return U, V, W

def _verify(U, V, W):
    n, m, p = 2, 4, 5
    rank = len(U[0])
    U = np.array(U, dtype=float); V = np.array(V, dtype=float); W = np.array(W, dtype=float)
    T = np.zeros((n*m, m*p, p*n))
    Texp = np.zeros((n*m, m*p, p*n))
    for r in range(rank):
        u = U[:, r].reshape(n, m)
        v = V[:, r].reshape(m, p)
        w = W[:, r].reshape(p, n)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    T[(i*m+j),(j*p+k),(k*n+i)] += u[i,j]*v[j,k]*w[k,i]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                Texp[(i*m+j),(j*p+k),(k*n+i)] = 1.0
    return np.allclose(T, Texp, atol=1e-9)

def construct():
    n, m, p = 2, 4, 5
    U, V, W = _build_seed_36()
    best_rank = len(U[0])
    best_U, best_V, best_W = U, V, W

    # Try to reduce rank by removing one term from the 28 Strassen terms
    # and verifying.
    # We iterate through the first 28 terms.
    for r_to_remove in range(28):
        if _remaining() < 5.0:
            break
        keep = [x for x in range(best_rank) if x != r_to_remove]
        U2 = [[row[x] for x in keep] for row in best_U]
        V2 = [[row[x] for x in keep] for row in best_V]
        W2 = [[row[x] for x in keep] for row in best_W]
        if _verify(U2, V2, W2):
            best_rank = len(U2[0])
            best_U, best_V, best_W = U2, V2, W2
            break # Found a reduction, stop searching for more? Or continue?
            # Continue to see if we can remove more?
            # But removing one might enable another? Unlikely in this structure.

    return {"U": best_U, "V": best_V, "W": best_W}
