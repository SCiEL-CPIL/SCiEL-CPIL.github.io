import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24
    
    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0
    
    # Start with schoolbook decomposition
    rank_schoolbook = n*m*p
    U_sb = np.zeros((nm, rank_schoolbook))
    V_sb = np.zeros((mp, rank_schoolbook))
    W_sb = np.zeros((pn, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i*m+j, col] = 1
                V_sb[j*p+k, col] = 1
                W_sb[k*n+i, col] = 1
                col += 1
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    def verify_decomposition(U, V, W):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])  # nm x mp
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))
    
    # Try various ranks with ALS approach
    np.random.seed(42)
    
    # Try ranks from lower to higher, stop when we find a valid one
    for target_rank in [80, 90, 100, 110, 115]:
        if time.time() > deadline - 20:
            break
        
        for attempt in range(15):
            if time.time() > deadline - 15:
                break
            
            # Random initialization with small variance to help convergence
            scale = 0.5
            U = np.random.randn(nm, target_rank) * scale
            V = np.random.randn(mp, target_rank) * scale
            W = np.random.randn(pn, target_rank) * scale
            
            # ALS iterations with early stopping
            best_err = np.inf
            for it in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Fix V, W: solve for U
                # For each a: sum_t U[a,t] * (V[b,t] * W[c,t]) = T[a,b,c]
                X = np.einsum('bt,ct->bct', V, W).reshape(target_rank, mp*pn).T  # (mp*pn, target_rank)
                for a in range(nm):
                    U[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
                
                # Fix U, W: solve for V
                X = np.einsum('at,ct->act', U, W).reshape(target_rank, nm*pn).T  # (nm*pn, target_rank)
                for b in range(mp):
                    V[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
                
                # Fix U, V: solve for W
                X = np.einsum('at,bt->abt', U, V).reshape(target_rank, nm*mp).T  # (nm*mp, target_rank)
                for c in range(pn):
                    W[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
                
                # Check convergence
                err = verify_decomposition(U, V, W)
                if err < best_err:
                    best_err = err
                if err < 1e-12:
                    break
            
            err = verify_decomposition(U, V, W)
            if err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                break  # Found a valid decomposition for this rank, try lower
    
    if time.time() > deadline - 5:
        U, V, W = best_decomp
        return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
