import time
import numpy as np

DIMENSIONS = [2, 2, 9]

def construct():
    deadline = time.time() + 140  # safety margin before 150s limit
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition (rank 36)
    best_rank = n * m * p
    best_U = np.zeros((n*m, best_rank))
    best_V = np.zeros((m*p, best_rank))
    best_W = np.zeros((p*n, best_rank))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                col = i * m * p + j * p + k
                best_U[i*m + j, col] = 1.0
                best_V[j*p + k, col] = 1.0
                best_W[k*n + i, col] = 1.0
    
    # Try to find lower rank decompositions
    # For rank r, we have:
    # U: (n*m) x r = 4 x r
    # V: (m*p) x r = 18 x r
    # W: (p*n) x r = 18 x r
    # Constraint: for all (i,j,k): 1 = sum_r U[2i+j,r] * V[2j+k,r] * W[9k+i,r]
    
    # For fixed U, V, the system is linear in W.
    # Let's try ranks from 35 down to some minimum.
    
    def try_rank(r, max_iters=5000, seed=0):
        """Try to find a rank-r decomposition using random initialization + linear solve."""
        rng = np.random.default_rng(seed)
        
        for attempt in range(max_iters):
            if time.time() > deadline:
                return None
            
            # Random initialization
            U = rng.standard_normal((n*m, r))
            V = rng.standard_normal((m*p, r))
            
            # Build linear system for W
            # For each (i,j,k): 1 = sum_r U[2i+j,r] * V[2j+k,r] * W[9k+i,r]
            # Group by (k,i): for fixed (k,i), unknown is W[9k+i,:] (length r)
            # Equations for fixed (k,i): for j=0,1:
            #   1 = sum_r U[2i+j,r] * V[2j+k,r] * W[9k+i,r]
            
            # Total unknowns: p*n*r = 18*r
            # Total equations: n*m*p = 36
            
            # Build A matrix (36 x 18*r) and b vector (36,)
            A = np.zeros((n*m*p, p*n*r))
            b = np.ones(n*m*p)
            
            eq_idx = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        w_row = k * n + i  # index into W
                        for r_idx in range(r):
                            A[eq_idx, w_row * r + r_idx] = U[i*m + j, r_idx] * V[j*p + k, r_idx]
                        eq_idx += 1
            
            # Solve least squares
            try:
                W_flat, residuals, rank_sys, sv = np.linalg.lstsq(A, b, rcond=None)
                W = W_flat.reshape(p*n, r)
                
                # Check residual
                # Recompute: for each (i,j,k), check if sum_r U[..]*V[..]*W[..] == 1
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-10:
                    return U, V, W
            except:
                pass
        
        return None
    
    # Try ranks from 35 down
    for r in range(best_rank - 1, 1, -1):
        if time.time() > deadline:
            break
        
        result = try_rank(r, max_iters=200, seed=r)
        if result is not None:
            U, V, W = result
            best_rank = r
            best_U, best_V, best_W = U, V, W
            break
    
    # Convert to lists for return
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
