import numpy as np
import time
from fractions import Fraction
import scipy.optimize

DIMENSIONS = [2, 3, 3]

def construct():
    n, m, p = DIMENSIONS
    R_school = n * m * p  # 18
    
    # Build the target tensor T[i, j, k] = 1 for all (i,j,k)
    T = np.ones((n, m, p))
    
    # Build schoolbook solution as fallback
    U_school = np.zeros((n * m, R_school))
    V_school = np.zeros((m * p, R_school))
    W_school = np.zeros((p * n, R_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = R_school
    
    start_time = time.time()
    deadline = 140  # Safety margin before 150s kill
    
    def compute_error(U, V, W, target_rank):
        """Compute the error of the tensor decomposition."""
        U_r = U.reshape(n, m, target_rank)
        V_r = V.reshape(m, p, target_rank)
        W_r = W.reshape(p, n, target_rank)
        result = np.einsum('ijr,jkr,kir->ijk', U_r, V_r, W_r)
        return np.linalg.norm(result - T)
    
    def objective(x, target_rank):
        """Objective function for scipy optimizer."""
        size_U = n * m * target_rank
        size_V = m * p * target_rank
        U_flat = x[:size_U]
        V_flat = x[size_U:size_U + size_V]
        W_flat = x[size_U + size_V:]
        
        U = U_flat.reshape(n, m, target_rank)
        V = V_flat.reshape(m, p, target_rank)
        W = W_flat.reshape(p, n, target_rank)
        
        result = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        residual = T - result
        return np.sum(residual * residual)

    def try_rank(target_rank):
        """Try to find a valid rank-target_rank decomposition using scipy L-BFGS-B."""
        max_restarts = 20 # Fewer restarts as L-BFGS-B is more powerful per run
        for restart in range(max_restarts):
            if time.time() - start_time > deadline:
                return False
            
            seed = restart * 1000 + target_rank * 7
            rng = np.random.RandomState(seed)
            
            # Initialize with small random values
            U = rng.randn(n * m, target_rank) * 0.5
            V = rng.randn(m * p, target_rank) * 0.5
            W = rng.randn(p * n, target_rank) * 0.5
            
            # Flatten
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            # Optimize
            try:
                res = scipy.optimize.minimize(objective, x0, args=(target_rank,), method='L-BFGS-B', 
                                              options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8})
            except Exception:
                continue
                
            if not res.success:
                continue
                
            # Extract result
            x_opt = res.x
            size_U = n * m * target_rank
            size_V = m * p * target_rank
            U_opt = x_opt[:size_U].reshape(n * m, target_rank)
            V_opt = x_opt[size_U:size_U + size_V].reshape(m * p, target_rank)
            W_opt = x_opt[size_U + size_V:].reshape(p * n, target_rank)
            
            err = compute_error(U_opt, V_opt, W_opt, target_rank)
            
            if err < 1e-6:
                # Try rounding to half-integers
                U_rounded = np.round(U_opt * 2) / 2
                V_rounded = np.round(V_opt * 2) / 2
                W_rounded = np.round(W_opt * 2) / 2
                
                err_rounded = compute_error(U_rounded, V_rounded, W_rounded, target_rank)
                if err_rounded < 1e-6:
                    nonlocal best_U, best_V, best_W, best_rank
                    best_U = U_rounded
                    best_V = V_rounded
                    best_W = W_rounded
                    best_rank = target_rank
                    return True
                
                # Try rounding to integers (in case half-integers are not the minimal representation)
                U_rounded_int = np.round(U_opt)
                V_rounded_int = np.round(V_opt)
                W_rounded_int = np.round(W_opt)
                
                err_rounded_int = compute_error(U_rounded_int, V_rounded_int, W_rounded_int, target_rank)
                if err_rounded_int < 1e-6:
                    nonlocal best_U, best_V, best_W, best_rank
                    best_U = U_rounded_int
                    best_V = V_rounded_int
                    best_W = W_rounded_int
                    best_rank = target_rank
                    return True

        return False
    
    # Try ranks from 14 to 17 (18 is schoolbook)
    for target_rank in range(14, R_school):
        if try_rank(target_rank):
            break
    
    # Final verification
    def verify_solution(U, V, W, rank):
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(rank):
                        val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                    max_err = max(max_err, abs(val - T[i, j, k]))
        return max_err

    if best_rank < R_school:
        err = verify_solution(best_U, best_V, best_W, best_rank)
        if err > 1e-6:
            best_U, best_V, best_W = U_school, V_school, W_school
            best_rank = R_school

    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
