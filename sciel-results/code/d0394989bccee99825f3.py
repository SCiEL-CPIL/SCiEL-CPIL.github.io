import numpy as np
import time
import itertools
from scipy.optimize import minimize

DIMENSIONS = [2, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor T[i, j, k] = 1 for all i, j, k
    T = np.ones((n, m, p), dtype=float)
    
    # Schoolbook fallback
    rank_school = n * m * p
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    best_result = None
    best_rank = rank_school
    
    # Try to find lower rank - search from 35 down to 30
    for r in range(35, 29, -1):
        if time.time() > deadline:
            break
            
        found = False
        # Try multiple random restarts for this rank
        for restart in range(30):
            if time.time() > deadline:
                break
            
            # Initialize U, V, W with random values
            np.random.seed(restart * 1000 + r)
            U_init = np.random.randn(n * m, r) * 0.5
            V_init = np.random.randn(m * p, r) * 0.5
            W_init = np.random.randn(p * n, r) * 0.5
            
            # Reshape to tensors
            U_t = U_init.reshape(n, m, r)
            V_t = V_init.reshape(m, p, r)
            W_t = W_init.reshape(p, n, r)
            
            def objective(params, n=n, m=m, p=p, r=r):
                U_flat, V_flat, W_flat = params[:n*m*r], params[n*m*r : n*m*r + m*p*r], params[n*m*r + m*p*r:]
                U_t = U_flat.reshape(n, m, r)
                V_t = V_flat.reshape(m, p, r)
                W_t = W_flat.reshape(p, n, r)
                
                # Compute sum_c U[i,j,c] V[j,k,c] W[k,i,c]
                prod = np.einsum('ijc,jkc,kic->ijk', U_t, V_t, W_t, optimize=True)
                error = np.sum((prod - T)**2)
                return error
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            try:
                res = minimize(objective, x0, method='L-BFGS-B', options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10, 'maxls': 50})
                if res.fun < 1e-8:
                    # Found a solution, snap to half-integers
                    U_sol = res.x[:n*m*r].reshape(n, m, r)
                    V_sol = res.x[n*m*r : n*m*r + m*p*r].reshape(m, p, r)
                    W_sol = res.x[n*m*r + m*p*r:].reshape(p, n, r)
                    
                    # Snap to nearest half-integer
                    U_snap = np.round(U_sol * 2) / 2
                    V_snap = np.round(V_sol * 2) / 2
                    W_snap = np.round(W_sol * 2) / 2
                    
                    # Verify exactness
                    prod_snap = np.einsum('ijc,jkc,kic->ijk', U_snap, V_snap, W_snap, optimize=True)
                    if np.allclose(prod_snap, T, atol=1e-9):
                        # Success!
                        U_out = U_snap.reshape(n*m, r).tolist()
                        V_out = V_snap.reshape(m*p, r).tolist()
                        W_out = W_snap.reshape(p*n, r).tolist()
                        
                        best_result = {"U": U_out, "V": V_out, "W": W_out}
                        best_rank = r
                        found = True
                        break
            except Exception:
                pass
        
        if found:
            # Found a solution at this rank, but continue to try lower ranks
            # unless we're at the minimum we're trying
            if r == 30:
                break
    
    if best_result is not None:
        return best_result
    
    # Fallback to schoolbook
    return {"U": U_school.tolist(), "V": V_school.tolist(), "W": W_school.tolist()}
# EVOLVE-BLOCK-END
