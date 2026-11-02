"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 8]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W
    
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i * m + j, j * p + k, k * n + i] = 1
    
    def check(U, V, W, tol=1e-8):
        Tr = np.einsum('ad,bd,cd->abc', U, V, W)
        return np.allclose(Tr, T, atol=tol)
    
    best_U, best_V, best_W = schoolbook()
    best_rank = n * m * p
    
    for r in range(best_rank - 1, 23, -1):
        if time.time() > deadline:
            break
        for seed in range(3):
            if time.time() > deadline:
                break
            rng = np.random.RandomState(seed)
            U = rng.randn(n * m, r)
            V = rng.randn(m * p, r)
            W = rng.randn(p * n, r)
            for _ in range(30):
                if time.time() > deadline:
                    break
                # Solve U
                M = np.einsum('bd,cd->bc', V, W)
                T2 = T.reshape(n*m, -1)
                M2 = M.reshape(m*p*p*n, 1)  # not quite right
                # Better: T[a,b,c] = sum_d U[a,d] V[b,d] W[c,d]
                # Let B = V * W element-wise per d: B[b,c,d] = V[b,d]*W[c,d]
                # T[a] = U[a] @ B_flat.T where B_flat[d, b*p*n+c] = V[b,d]*W[c,d]
                Bflat = np.zeros((r, m*p*p*n))
                for d in range(r):
                    Bflat[d] = np.outer(V[:,d], W[:,d]).flatten()
                try:
                    U = T.reshape(n*m,-1) @ np.linalg.pinv(Bflat.T)
                except:
                    break
                # Solve V
                Aflat = np.zeros((r, n*m*p*n))
                for d in range(r):
                    Aflat[d] = np.outer(U[:,d], W[:,d]).flatten()
                try:
                    V = T.reshape(m*p,-1) @ np.linalg.pinv(Aflat.T)
                except:
                    break
                # Solve W
                Cflat = np.zeros((r, n*m*m*p))
                for d in range(r):
                    Cflat[d] = np.outer(U[:,d], V[:,d]).flatten()
                try:
                    W = T.reshape(p*n,-1) @ np.linalg.pinv(Cflat.T)
                except:
                    break
            if check(U, V, W, 1e-6):
                Ur = np.round(U*2)/2
                Vr = np.round(V*2)/2
                Wr = np.round(W*2)/2
                if check(Ur, Vr, Wr, 1e-6):
                    best_U, best_V, best_W = Ur, Vr, Wr
                    best_rank = r
                    break
                else:
                    best_U, best_V, best_W = U, V, W
                    best_rank = r
                    break
        if best_rank < n*m*p:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
