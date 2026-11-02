"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 9]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

def _compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the tensor decomposition."""
    err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i*m+j, :], V[j*p+k, :]) * W[k*n+i, :]
                val = np.sum(val)
                err += (val - 1.0) ** 2
    return err

def _verify(U, V, W, n, m, p, tol=1e-9):
    """Check if the decomposition is exact."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook
    rank_schoolbook = n * m * p
    U_sb = [[0] * rank_schoolbook for _ in range(n * m)]
    V_sb = [[0] * rank_schoolbook for _ in range(m * p)]
    W_sb = [[0] * rank_schoolbook for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][column] = 1
                V_sb[j * p + k][column] = 1
                W_sb[k * n + i][column] = 1
                column += 1
    
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = rank_schoolbook
    
    # Lower bound is max(nm, mp, pn) = 27
    for rank in range(27, rank_schoolbook):
        if time.time() > deadline - 20.0:
            break
        
        # Try a few random initializations with optimization
        for trial in range(3):
            if time.time() > deadline - 10.0:
                break
            
            np.random.seed(rank * 100 + trial)
            
            # Initialize with small random values
            U = np.random.randn(n*m, rank) * 0.1
            V = np.random.randn(m*p, rank) * 0.1
            W = np.random.randn(p*n, rank) * 0.1
            
            # Flatten all variables
            def flatten_params():
                return np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            
            def unpack_params(x):
                U_flat = x[:n*m*rank]
                V_flat = x[n*m*rank:n*m*rank + m*p*rank]
                W_flat = x[n*m*rank + m*p*rank:]
                return U_flat.reshape(n*m, rank), V_flat.reshape(m*p, rank), W_flat.reshape(p*n, rank)
            
            def objective(x):
                U_opt, V_opt, W_opt = unpack_params(x)
                err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_opt[i*m+j, :] * V_opt[j*p+k, :] * W_opt[k*n+i, :])
                            err += (val - 1.0) ** 2
                return err
            
            # Use L-BFGS-B optimization
            x0 = flatten_params()
            
            try:
                result = minimize(
                    objective,
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-8:
                    U_opt, V_opt, W_opt = unpack_params(result.x)
                    
                    if _verify(U_opt, V_opt, W_opt, n, m, p, tol=1e-6):
                        best_U, best_V, best_W = U_opt, V_opt, W_opt
                        best_rank = rank
                        break
                        
            except Exception:
                continue
    
    U_out = best_U.tolist() if isinstance(best_U, np.ndarray) else best_U
    V_out = best_V.tolist() if isinstance(best_V, np.ndarray) else best_V
    W_out = best_W.tolist() if isinstance(best_W, np.ndarray) else best_W
    
    return {"U": U_out, "V": V_out, "W": W_out}
# EVOLVE-BLOCK-END
