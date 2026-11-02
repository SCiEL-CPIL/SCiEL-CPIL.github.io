"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time
import itertools

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Target tensor: T[i,j,k] = 1 for all i,j,k (matrix multiplication)
    # U shape: (n*m, rank), V shape: (m*p, rank), W shape: (p*n, rank)
    # Constraint: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    deadline = time.time() + 110  # safety margin
    
    # Try ranks from low to high
    for rank in range(15, 46):
        if time.time() > deadline:
            break
        if _try_find_decomposition(n, m, p, rank, deadline):
            return _build_result(n, m, p, rank)
    
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}

def _try_find_decomposition(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition. Returns True if found."""
    # Build the constraint matrix structure
    # For each (i,j,k), we need: sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    # We'll use a randomized local search approach
    # Start with random initial values and minimize the sum of squared residuals
    
    num_constraints = n * m * p  # 45
    num_vars = rank * (n * m + m * p + p * n)  # total variables
    
    def objective_and_grad(params):
        # Unpack parameters
        U_flat = params[:rank * n * m]
        V_flat = params[rank * n * m:rank * n * m + rank * m * p]
        W_flat = params[rank * n * m + rank * m * p:]
        
        U = U_flat.reshape(n * m, rank)
        V = V_flat.reshape(m * p, rank)
        W = W_flat.reshape(p * n, rank)
        
        # Compute residuals
        residual_sum_sq = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                    # Actually: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    residual = val - 1.0
                    residual_sum_sq += residual * residual
        
        return residual_sum_sq
    
    # Try multiple random restarts
    max_restarts = 20
    rng = np.random.default_rng(42)
    
    for restart in range(max_restarts):
        if time.time() > deadline:
            return False
        
        # Initialize with small random values
        scale = 0.5
        params = rng.uniform(-scale, scale, size=num_vars)
        
        # Use Nelder-Mead or BFGS
        try:
            result = minimize(objective_and_grad, params, method='L-BFGS-B', 
                            options={'maxiter': 500, 'maxfun': 5000})
            
            if result.fun < 1e-6:
                # Verify the solution
                U_flat = result.x[:rank * n * m]
                V_flat = result.x[rank * n * m:rank * n * m + rank * m * p]
                W_flat = result.x[rank * n * m + rank * m * p:]
                
                U = U_flat.reshape(n * m, rank)
                V = V_flat.reshape(m * p, rank)
                W = W_flat.reshape(p * n, rank)
                
                # Check all constraints
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            max_err = max(max_err, err)
                
                if max_err < 1e-4:
                    # Store the solution globally
                    global _found_U, _found_V, _found_W
                    _found_U = U
                    _found_V = V
                    _found_W = W
                    return True
        except Exception:
            continue
    
    return False

def _build_result(n, m, p, rank):
    U = _found_U.tolist()
    V = _found_V.tolist()
    W = _found_W.tolist()
    return {"U": U, "V": V, "W": W}

# Global storage for found solution
_found_U = None
_found_V = None
_found_W = None
# EVOLVE-BLOCK-END
