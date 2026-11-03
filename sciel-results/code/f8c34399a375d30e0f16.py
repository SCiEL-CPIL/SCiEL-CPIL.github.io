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

    def verify_decomposition(U, V, W, tol=1e-10):
        r = U.shape[1]
        T_recon = np.zeros((nm, mp, pn))
        for t in range(r):
            uv = np.outer(U[:,t], V[:,t])
            T_recon += np.einsum('ab,c->abc', uv, W[:,t])
        return np.max(np.abs(T_recon - T))

    def find_rank_reduction(U, V, W, tol=1e-8):
        """Find linear dependencies among the decomposition terms to reduce rank."""
        r = U.shape[1]
        if r <= 1:
            return U, V, W, r
        
        # Stack into a single matrix: M = [U; V; W] of shape (nm+mp+pn, r)
        M = np.vstack([U, V, W])  # shape: (nm+mp+pn, r)
        
        # Find null space of M
        try:
            U_mat, s, Vt = np.linalg.svd(M, full_matrices=False)
            # Null space vectors correspond to small singular values
            if len(s) > 0 and s[0] > 0:
                null_space = Vt.T[:, s < tol * max(s)]
            else:
                null_space = np.zeros((r, r))
            null_dim = null_space.shape[1]
        except:
            return U, V, W, r
        
        if null_dim == 0:
            return U, V, W, r
        
        # For each null vector, we can eliminate one term
        # Pick the term with largest norm to eliminate
        for nv in range(null_dim):
            c = null_space[:, nv]
            # Find which term to eliminate (largest |c_t|)
            idx = np.argmax(np.abs(c))
            if abs(c[idx]) < 1e-10:
                continue
            
            # Eliminate term idx by adding c/c[idx] * term_idx to all other terms
            c_norm = c / c[idx]
            U_new = U.copy()
            V_new = V.copy()
            W_new = W.copy()
            for t in range(r):
                if t == idx:
                    continue
                U_new[:, t] += c_norm[t] * U[:, idx]
                V_new[:, t] += c_norm[t] * V[:, idx]
                W_new[:, t] += c_norm[t] * W[:, idx]
            
            # Remove the idx-th column
            mask = [t for t in range(r) if t != idx]
            U_new = U_new[:, mask]
            V_new = V_new[:, mask]
            W_new = W_new[:, mask]
            
            # Verify
            err = verify_decomposition(U_new, V_new, W_new)
            if err < 1e-8:
                return U_new, V_new, W_new, r - 1
        
        return U, V, W, r

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

    # Start with schoolbook and try to reduce rank
    U, V, W = U_sb.copy(), V_sb.copy(), W_sb.copy()
    current_rank = rank_schoolbook
    
    # Iteratively find and remove linear dependencies
    max_iterations = 100
    for it in range(max_iterations):
        if time.time() > deadline - 15:
            break
        U, V, W, new_rank = find_rank_reduction(U, V, W)
        if new_rank >= current_rank:
            break
        current_rank = new_rank
    
    # Try random perturbations to find more dependencies
    np.random.seed(42)
    best_rank = current_rank
    best_decomp = (U.copy(), V.copy(), W.copy())
    
    # Try several random initializations based on the reduced decomposition
    for attempt in range(30):
        if time.time() > deadline - 10:
            break
        
        # Perturb the current best decomposition
        U_p = U.copy() + np.random.randn(nm, best_rank) * 0.01
        V_p = V.copy() + np.random.randn(mp, best_rank) * 0.01
        W_p = W.copy() + np.random.randn(pn, best_rank) * 0.01
        
        # Try to reduce further
        U_r, V_r, W_r, new_rank = find_rank_reduction(U_p, V_p, W_p)
        if new_rank < best_rank:
            # Verify
            err = verify_decomposition(U_r, V_r, W_r)
            if err < 1e-8:
                best_rank = new_rank
                best_decomp = (U_r.copy(), V_r.copy(), W_r.copy())
                U, V, W = U_r, V_r, W_r
    
    U, V, W = best_decomp
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
