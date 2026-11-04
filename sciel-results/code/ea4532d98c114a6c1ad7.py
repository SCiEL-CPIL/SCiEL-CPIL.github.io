import numpy as np
import time

DIMENSIONS = [3, 4, 6]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145
    
    n, m, p = DIMENSIONS
    rank_schoolbook = n * m * p
    T = np.ones((n, m, p), dtype=np.float64)
    N, M, P = n, m, p
    
    def verify(U, V, W):
        r = U.shape[1]
        U_r = U.reshape(N, M, r)
        V_r = V.reshape(M, P, r)
        W_r = W.reshape(P, N, r)
        T_rec = np.einsum('ijc,jkc,kic->ijk', U_r, V_r, W_r)
        return np.allclose(T_rec, T, atol=1e-9)
    
    def als_solve(r, n_iter=50, seed=0):
        if time.time() > deadline:
            return None
        np.random.seed(seed)
        U = np.random.randn(N, M, r) * 0.5
        V = np.random.randn(M, P, r) * 0.5
        W = np.random.randn(P, N, r) * 0.5
        
        for it in range(n_iter):
            if time.time() > deadline:
                break
            for i in range(N):
                for j in range(M):
                    A = np.array([V[j, k, :] * W[k, i, :] for k in range(P)])
                    U[i, j, :] = np.linalg.lstsq(A.T, T[i, j, :], rcond=None)[0]
            for j in range(M):
                for k in range(P):
                    A = np.array([U[i, j, :] * W[k, i, :] for i in range(N)])
                    V[j, k, :] = np.linalg.lstsq(A.T, T[:, j, k], rcond=None)[0]
            for k in range(P):
                for i in range(N):
                    A = np.array([U[i, j, :] * V[j, k, :] for j in range(M)])
                    W[k, i, :] = np.linalg.lstsq(A.T, T[i, :, k], rcond=None)[0]
            T_rec = np.einsum('ijc,jkc,kic->ijk', U, V, W)
            if np.linalg.norm(T_rec - T) < 1e-10:
                break
        
        U_flat = U.reshape(N*M, r)
        V_flat = V.reshape(M*P, r)
        W_flat = W.reshape(P*N, r)
        for denom in [1, 2, 3, 4, 6, 12]:
            Us = np.round(U_flat * denom) / denom
            Vs = np.round(V_flat * denom) / denom
            Ws = np.round(W_flat * denom) / denom
            if verify(Us, Vs, Ws):
                return Us, Vs, Ws
        if verify(U_flat, V_flat, W_flat):
            return U_flat, V_flat, W_flat
        return None
    
    # Schoolbook
    U_fb = np.zeros((N*M, rank_schoolbook))
    V_fb = np.zeros((M*P, rank_schoolbook))
    W_fb = np.zeros((P*N, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_fb[i*m+j, col] = 1.0
                V_fb[j*p+k, col] = 1.0
                W_fb[k*n+i, col] = 1.0
                col += 1
    
    best = (U_fb, V_fb, W_fb)
    best_rank = rank_schoolbook
    
    for r in range(40, rank_schoolbook):
        if time.time() > deadline:
            break
        for seed in range(2):
            if time.time() > deadline:
                break
            res = als_solve(r, 30, seed)
            if res:
                best = res
                best_rank = r
                break
    
    U_out, V_out, W_out = best
    return {"U": U_out.tolist(), "V": V_out.tolist(), "W": W_out.tolist()}
# EVOLVE-BLOCK-END
