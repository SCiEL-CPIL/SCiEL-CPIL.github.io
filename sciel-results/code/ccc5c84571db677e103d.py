import numpy as np
import time

DIMENSIONS = [4, 5, 6]

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit

    n, m, p = DIMENSIONS
    nm, mp, pn = n*m, m*p, p*n  # 20, 30, 24

    # Build the target tensor T
    T = np.zeros((nm, mp, pn))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m+j, j*p+k, k*n+i] = 1.0

    # Schoolbook decomposition (rank 120)
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

    def verify_decomposition(U, V, W, tol=1e-10):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))

    def als_optimize(U, V, W, max_iters=200, tol=1e-12):
        """Alternating Least Squares optimization for tensor decomposition."""
        r = U.shape[1]
        
        for it in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Fix V, W: solve for U
            # T[a,:,:] = sum_t U[a,t] * V[:,t] * W[:,t]
            # For each a: T[a,:,:].reshape(-1) = X @ U[a,:]
            # where X[b,c,t] = V[b,t]*W[c,t]
            X = np.einsum('bt,ct->bct', V, W).reshape(mp*pn, r).T
            U_new = np.zeros_like(U)
            for a in range(nm):
                U_new[a, :] = np.linalg.lstsq(X, T[a, :, :].reshape(-1), rcond=None)[0]
            
            # Fix U, W: solve for V
            X = np.einsum('at,ct->act', U, W).reshape(nm*pn, r).T
            V_new = np.zeros_like(V)
            for b in range(mp):
                V_new[b, :] = np.linalg.lstsq(X, T[:, b, :].reshape(-1), rcond=None)[0]
            
            # Fix U, V: solve for W
            X = np.einsum('at,bt->abt', U, V).reshape(nm*mp, r).T
            W_new = np.zeros_like(W)
            for c in range(pn):
                W_new[c, :] = np.linalg.lstsq(X, T[:, :, c].reshape(-1), rcond=None)[0]
            
            # Check convergence
            U_diff = np.max(np.abs(U_new - U))
            V_diff = np.max(np.abs(V_new - V))
            W_diff = np.max(np.abs(W_new - W))
            U, V, W = U_new, V_new, W_new
            if max(U_diff, V_diff, W_diff) < tol:
                break
        
        return U, V, W

    # Try to find a decomposition with rank < 120
    np.random.seed(42)
    
    best_rank = rank_schoolbook
    best_decomp = (U_sb.copy(), V_sb.copy(), W_sb.copy())
    
    # Try various target ranks
    target_ranks = [115, 110, 105, 100, 95, 90]
    
    for target_rank in target_ranks:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(3):
            if time.time() > deadline - 20:
                break
            
            # Initialize with random perturbation of schoolbook structure
            # Use a mixture of schoolbook-like and random initialization
            U = np.random.randn(nm, target_rank) * 0.05
            V = np.random.randn(mp, target_rank) * 0.05
            W = np.random.randn(pn, target_rank) * 0.05
            
            # Also try initializing with some schoolbook structure
            if attempt == 0:
                # Use first min(target_rank, 120) schoolbook terms as initialization
                init_rank = min(target_rank, rank_schoolbook)
                U_init = np.zeros((nm, target_rank))
                V_init = np.zeros((mp, target_rank))
                W_init = np.zeros((pn, target_rank))
                col = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            if col < target_rank:
                                U_init[i*m+j, col] = 1
                                V_init[j*p+k, col] = 1
                                W_init[k*n+i, col] = 1
                                col += 1
                # Add small random perturbation
                U = U_init + np.random.randn(nm, target_rank) * 0.01
                V = V_init + np.random.randn(mp, target_rank) * 0.01
                W = W_init + np.random.randn(pn, target_rank) * 0.01
            
            # ALS optimization
            U, V, W = als_optimize(U, V, W, max_iters=300, tol=1e-12)
            
            # Check if we achieved exact decomposition
            err = verify_decomposition(U, V, W, tol=1e-10)
            if err < 1e-10:
                if target_rank < best_rank:
                    best_rank = target_rank
                    best_decomp = (U.copy(), V.copy(), W.copy())
                    break  # This rank works, try lower
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
