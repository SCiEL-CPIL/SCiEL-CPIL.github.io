import numpy as np
import time

DIMENSIONS = [3, 3, 5]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    time_limit = 115  # safety margin before 120s limit
    
    # Schoolbook decomposition
    rank_schoolbook = n * m * p
    U_school = np.zeros((n * m, rank_schoolbook))
    V_school = np.zeros((m * p, rank_schoolbook))
    W_school = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U = U_school.copy()
    best_V = V_school.copy()
    best_W = W_school.copy()
    best_rank = rank_schoolbook
    
    # Build the target tensor T
    # T[a, b, c] where a = i*m+j, b = j'*p+k, c = k'*n+i
    # T[a, b, c] = 1 if a corresponds to (i,j), b to (j,k), c to (k,i) with matching j
    T = np.zeros((n * m, m * p, p * n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c = k * n + i
                T[a, b, c] = 1
    
    # Try to find a lower rank decomposition using alternating minimization
    # with structured initialization and polishing
    
    def try_rank_decomposition(target_rank, max_iters=50, seed=42):
        """Try to find a rank decomposition of given target_rank using alternating minimization."""
        nonlocal best_U, best_V, best_W, best_rank
        
        if time.time() - start_time > time_limit:
            return
        
        rng = np.random.RandomState(seed)
        
        # Initialize U, V, W with random values
        U = rng.randn(n * m, target_rank) * 0.5
        V = rng.randn(m * p, target_rank) * 0.5
        W = rng.randn(p * n, target_rank) * 0.5
        
        # Precompute the flattened tensor
        T_flat = T.reshape(n * m, m * p * p * n)  # not quite right, let me think
        
        # Actually, let's use a more efficient approach
        # For each mode, we solve a linear system
        
        prev_err = np.inf
        
        for iteration in range(max_iters):
            if time.time() - start_time > time_limit - 5:
                break
            
            # Update W: for each c, solve M_W @ W[c,:] = t_c
            # M_W[(a,b), r] = U[a,r] * V[b,r]
            M_W = np.zeros((n * m * m * p, target_rank))
            idx = 0
            for a in range(n * m):
                for b in range(m * p):
                    M_W[idx, :] = U[a, :] * V[b, :]
                    idx += 1
            
            for c in range(p * n):
                t_c = T[:, :, c].flatten()
                W[c, :], _, _, _ = np.linalg.lstsq(M_W, t_c, rcond=None)
            
            # Update U: for each a, solve M_U @ U[a,:] = t_a
            # M_U[(b,c), r] = V[b,r] * W[c,r]
            M_U = np.zeros((m * p * p * n, target_rank))
            idx = 0
            for b in range(m * p):
                for c in range(p * n):
                    M_U[idx, :] = V[b, :] * W[c, :]
                    idx += 1
            
            for a in range(n * m):
                t_a = T[a, :, :].flatten()
                U[a, :], _, _, _ = np.linalg.lstsq(M_U, t_a, rcond=None)
            
            # Update V: for each b, solve M_V @ V[b,:] = t_b
            # M_V[(a,c), r] = U[a,r] * W[c,r]
            M_V = np.zeros((n * m * p * n, target_rank))
            idx = 0
            for a in range(n * m):
                for c in range(p * n):
                    M_V[idx, :] = U[a, :] * W[c, :]
                    idx += 1
            
            for b in range(m * p):
                t_b = T[:, b, :].flatten()
                V[b, :], _, _, _ = np.linalg.lstsq(M_V, t_b, rcond=None)
            
            # Compute error
            E = np.zeros((n * m, m * p, p * n))
            for r in range(target_rank):
                E += np.einsum('a,b,c->abc', U[:, r], V[:, r], W[:, r])
            err = np.sum((T - E) ** 2)
            
            # Check if we found an exact solution
            if err < 1e-10:
                # Try to round to half-integers
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify the rounded solution
                E_rounded = np.zeros((n * m, m * p, p * n))
                for r in range(target_rank):
                    E_rounded += np.einsum('a,b,c->abc', U_rounded[:, r], V_rounded[:, r], W_rounded[:, r])
                err_rounded = np.sum((T - E_rounded) ** 2)
                
                if err_rounded < 1e-8:
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
                    return True
                else:
                    # Use the unrounded solution if it's exact
                    best_U = U
                    best_V = V
                    best_W = W
                    best_rank = target_rank
                    return True
            
            # Early stopping if no improvement
            if abs(prev_err - err) < 1e-12:
                break
            prev_err = err
        
        return False
    
    # Try ranks from 44 down to 35
    for target_rank in range(44, 34, -1):
        if time.time() - start_time > time_limit - 10:
            break
        
        # Try multiple random initializations
        for seed in range(5):
            if time.time() - start_time > time_limit - 10:
                break
            success = try_rank_decomposition(target_rank, max_iters=30, seed=seed)
            if success:
                break
    
    if time.time() - start_time > time_limit:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
