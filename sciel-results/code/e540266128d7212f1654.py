import numpy as np
import time

DIMENSIONS = [2, 3, 8]

def construct():
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    M = np.zeros((nm, mp, pn), dtype=float)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                M[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Schoolbook
    rank_sb = n * m * p
    U_sb = np.zeros((nm, rank_sb))
    V_sb = np.zeros((mp, rank_sb))
    W_sb = np.zeros((pn, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    best_rank = rank_sb
    
    start = time.time()
    deadline = start + 100
    
    def try_optimize(U0, V0, W0, R, lr=0.01, max_steps=500):
        U = U0.copy()
        V = V0.copy()
        W = W0.copy()
        for step in range(max_steps):
            if time.time() > deadline:
                return None
            E = np.einsum('ar,br,cr->abc', U, V, W) - M
            err = np.sum(E*E)
            if err < 1e-8:
                return (U, V, W)
            dU = np.einsum('abc,bcr->acr', E, V[:, :, None] * W[None, :, :])
            dV = np.einsum('abc,acr->bcr', E, U[:, :, None] * W[None, :, :])
            dW = np.einsum('abc,abr->cr', E, U[:, :, None] * V[None, :, :])
            U -= lr * dU
            V -= lr * dV
            W -= lr * dW
        err = np.sum((np.einsum('ar,br,cr->abc', U, V, W) - M)**2)
        if err < 1e-6:
            return (U, V, W)
        return None
    
    # Try random initializations for various ranks
    for R in [32, 36, 40, 44, 48]:
        if time.time() > deadline:
            break
        for seed in range(10):
            if time.time() > deadline:
                break
            np.random.seed(seed * 1000 + R)
            U0 = np.random.randn(nm, R) * 0.5
            V0 = np.random.randn(mp, R) * 0.5
            W0 = np.random.randn(pn, R) * 0.5
            result = try_optimize(U0, V0, W0, R, lr=0.02, max_steps=300)
            if result is not None:
                U, V, W = result
                best_U, best_V, best_W = U, V, W
                best_rank = R
                break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
