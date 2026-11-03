import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    start_time = time.time()
    deadline = start_time + 110
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n
    
    def is_consistent(a, b, c):
        i = a // m
        j_a = a % m
        j_b = b // p
        k = b % p
        k_c = c // n
        i_c = c % n
        return j_a == j_b and k == k_c and i == i_c
    
    T = np.zeros((nm, mp, pn))
    for a in range(nm):
        for b in range(mp):
            for c in range(pn):
                if is_consistent(a, b, c):
                    T[a, b, c] = 1.0
    
    best_rank = n * m * p
    best_decomp = None
    
    rng = np.random.default_rng(42)
    
    for rank in range(22, best_rank):
        if time.time() > deadline:
            break
        
        for trial in range(10):
            if time.time() > deadline:
                break
            
            U = rng.standard_normal((nm, rank))
            V = rng.standard_normal((mp, rank))
            W = rng.standard_normal((pn, rank))
            
            success = False
            for iteration in range(150):
                if time.time() > deadline:
                    break
                
                M = np.einsum('br,cr->bcr', V, W).reshape(rank, mp*pn)
                MT = M.T
                for a in range(nm):
                    U[a], _, _, _ = np.linalg.lstsq(MT, T[a].reshape(-1), rcond=None)
                
                M = np.einsum('ar,cr->acr', U, W).reshape(rank, nm*pn)
                MT = M.T
                for b in range(mp):
                    V[b], _, _, _ = np.linalg.lstsq(MT, T[:, b, :].reshape(-1), rcond=None)
                
                M = np.einsum('ar,br->abr', U, V).reshape(rank, nm*mp)
                MT = M.T
                for c in range(pn):
                    W[c], _, _, _ = np.linalg.lstsq(MT, T[:, :, c].reshape(-1), rcond=None)
                
                recon = np.einsum('ar,br,cr->abc', U, V, W)
                error = np.sum((recon - T) ** 2)
                
                if error < 1e-12:
                    success = True
                    break
            
            if success:
                best_rank = rank
                best_decomp = (U.copy(), V.copy(), W.copy())
                break
        
        if best_rank < n * m * p:
            break
    
    if best_decomp is not None:
        U, V, W = best_decomp
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        recon = np.einsum('ar,br,cr->abc', U_r, V_r, W_r)
        if np.sum((recon - T) ** 2) < 1e-10:
            U, V, W = U_r, V_r, W_r
    else:
        rank = n * m * p
        U = np.zeros((nm, rank))
        V = np.zeros((mp, rank))
        W = np.zeros((pn, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
    
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
