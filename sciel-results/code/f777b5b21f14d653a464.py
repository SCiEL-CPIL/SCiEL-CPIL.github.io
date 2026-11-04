"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize

DIMENSIONS = [3, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin from 150s limit

    n, m, p = DIMENSIONS

    def make_trivial():
        """Create the trivial rank n*m*p decomposition."""
        rank = n * m * p
        U = np.zeros((n * m, rank))
        V = np.zeros((m * p, rank))
        W = np.zeros((p * n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1.0
                    V[j * p + k, col] = 1.0
                    W[k * n + i, col] = 1.0
                    col += 1
        return U, V, W, rank

    U_triv, V_triv, W_triv, trivial_rank = make_trivial()
    best_result = {"U": U_triv.tolist(), "V": V_triv.tolist(), "W": W_triv.tolist()}
    best_rank = trivial_rank

    def residuals(params, rank):
        """Compute residual vector from flattened parameters."""
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        
        U_flat = params[:U_size]
        V_flat = params[U_size:U_size + V_size]
        W_flat = params[U_size + V_size:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        res = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    s = np.dot(U[i * m + j], V[j * p + k] * W[k * n + i])
                    res[idx] = 1.0 - s
                    idx += 1
        return res

    def try_rank(rank, num_restarts=20, seed_base=0):
        """Try to find a decomposition of given rank using scipy.optimize.least_squares."""
        rng = np.random.RandomState(seed_base)
        U_size = n * m * rank
        V_size = m * p * rank
        W_size = p * n * rank
        total_size = U_size + V_size + W_size
        
        for restart in range(num_restarts):
            if time.time() > deadline - 5:
                return None
            
            # Random initialization
            # Scale to avoid overflow/underflow
            init_scale = 1.0 / np.sqrt(rank)
            params0 = rng.randn(total_size) * init_scale
            
            # Solve
            try:
                sol = scipy.optimize.least_squares(
                    residuals, 
                    params0, 
                    args=(rank,),
                    method='lm', # Levenberg-Marquardt, good for underdetermined
                    max_nfev=1000 # Limit function evaluations
                )
            except Exception:
                continue
                
            if sol.success or sol.cost < 1e-12:
                # Verify solution
                res = residuals(sol.x, rank)
                if np.max(np.abs(res)) < 1e-9:
                    U_flat = sol.x[:U_size]
                    V_flat = sol.x[U_size:U_size + V_size]
                    W_flat = sol.x[U_size + V_size:]
                    
                    U = U_flat.reshape(n * m, rank)
                    V = V_flat.reshape(m * p, rank)
                    W = W_flat.reshape(p * n, rank)
                    
                    return U, V, W, rank
        
        return None

    # Try ranks from 60 to 83
    # 60 is a known lower bound, 84 is trivial
    # Start from higher ranks to increase chance of finding a solution
    for rank in range(83, 59, -1):
        if time.time() > deadline - 10:
            break
        
        result = try_rank(rank, num_restarts=10, seed_base=rank * 100)
        
        if result is not None:
            U, V, W, r = result
            # Verify again
            res = residuals(np.concatenate([U.flatten(), V.flatten(), W.flatten()]), r)
            if np.max(np.abs(res)) < 1e-9:
                best_result = {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
                best_rank = r
                break  # Found a valid decomposition, return it
    
    return best_result
# EVOLVE-BLOCK-END
