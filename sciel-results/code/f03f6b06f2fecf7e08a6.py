import time
import numpy as np
from scipy.optimize import least_squares, differential_evolution
import sys

DIMENSIONS = [2, 4, 8]

def construct():
    """
    Find a rank-minimal exact tensor decomposition for multiplying a 2x4 by 4x8 matrix.
    We try to find a decomposition of rank r where:
    T[i,j,k] = sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    """
    start_time = time.time()
    deadline = start_time + 130.0  # 10s safety margin
    
    n, m, p = DIMENSIONS
    nmp = n * m * p  # 64
    
    # Try ranks from low to high
    # The theoretical lower bound for matrix multiplication tensor T_{n,m,p} is 
    # at least max(n*m, m*p, n*p) and at least n+m+p-2 (Strassen bound)
    # For 2x4x8: lower bound is max(8, 32, 16) = 32, and 2+4+8-2 = 12
    # So rank >= 32
    
    best_result = None
    
    # First, provide a valid schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_school = np.zeros((n*m, rank_schoolbook))
    V_school = np.zeros((m*p, rank_schoolbook))
    W_school = np.zeros((p*n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i*m+j, col] = 1
                V_school[j*p+k, col] = 1
                W_school[k*n+i, col] = 1
                col += 1
    best_result = (rank_schoolbook, U_school.copy(), V_school.copy(), W_school.copy())
    
    # Try to find lower rank decompositions
    # We'll use a randomized approach with local optimization
    # The constraint is that sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if the decomposition is exact (within tolerance)."""
        rank = U.shape[1]
        # Compute T[i,j,k] for all i,j,k
        max_val = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] for c in range(rank))
                    max_val = max(max_val, abs(val - 1.0))
        return max_val < tol, max_val
    
    def random_init(r):
        """Generate random initial guess for rank r."""
        U = np.random.randn(n*m, r)
        V = np.random.randn(m*p, r)
        W = np.random.randn(p*n, r)
        # Normalize
        for c in range(r):
            norm = np.sqrt(np.sum(U[:,c]**2) * np.sum(V[:,c]**2) * np.sum(W[:,c]**2))
            if norm > 1e-10:
                scale = (1.0 / norm) ** (1.0/3.0)
                U[:,c] *= scale
                V[:,c] *= scale
                W[:,c] *= scale
        return U, V, W
    
    def objective(params, r):
        """Residual of the tensor decomposition."""
        U = params[0:n*m*r].reshape(n*m, r)
        V = params[n*m*r:2*n*m*r + m*p*r - n*m*r].reshape(m*p, r) if False else None
        # Let me restructure this
        return None  # Won't use this approach due to complexity
    
    # Better approach: use differential evolution to optimize the residual
    def build_residual(r, U, V, W):
        """Build residual vector for the decomposition."""
        residual = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    residual.append(val - 1.0)
        return np.array(residual)
    
    def flatten_and_unflatten(params, r):
        """Unflatten parameters into U, V, W."""
        U_size = n*m*r
        V_size = m*p*r
        W_size = p*n*r
        
        U = params[0:U_size].reshape(n*m, r)
        V = params[U_size:U_size+V_size].reshape(m*p, r)
        W = params[U_size+V_size:U_size+V_size+W_size].reshape(p*n, r)
        return U, V, W
    
    def objective_flattened(params, r):
        """Objective function for optimization."""
        U, V, W = flatten_and_unflatten(params, r)
        return build_residual(r, U, V, W)
    
    # Try various ranks
    # For 2x4x8, let's try ranks from 32 up
    tried_ranks = []
    
    for r in range(32, 65):
        if time.time() > deadline - 30:
            break
        
        tried_ranks.append(r)
        
        # Try multiple random restarts
        best_residual = float('inf')
        best_params = None
        
        n_restarts = min(5, max(1, int((deadline - 30 - time.time()) / 5)))
        
        for restart in range(n_restarts):
            if time.time() > deadline - 20:
                break
            
            # Random initialization
            U_init, V_init, W_init = random_init(r)
            params_init = np.concatenate([
                U_init.flatten(),
                V_init.flatten(),
                W_init.flatten()
            ])
            
            # Use least squares to minimize the residual
            try:
                result = least_squares(
                    objective_flattened,
                    params_init,
                    args=(r,),
                    method='lm',
                    max_nfev=500
                )
                
                residual = np.max(np.abs(result.fun))
                
                if residual < best_residual:
                    best_residual = residual
                    best_params = result.x
                
                if residual < 1e-9:
                    break  # Found exact solution
            
            except Exception:
                pass
        
        if best_params is not None and best_residual < 1e-9:
            U_sol, V_sol, W_sol = flatten_and_unflatten(best_params, r)
            best_result = (r, U_sol, V_sol, W_sol)
            break  # Found the minimum rank
        
        elif best_params is not None and best_residual < best_result[1] if isinstance(best_result[1], float) else True:
            # Keep the best so far as potential fallback
            pass
    
    # Return the best result
    rank, U, V, W = best_result
    
    # Convert to lists as expected
    U_list = U.tolist()
    V_list = V.tolist()
    W_list = W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
