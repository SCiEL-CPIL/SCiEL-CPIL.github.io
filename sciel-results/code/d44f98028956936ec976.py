import time
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 6]

def construct():
    start_time = time.time()
    deadline = start_time + 140
    
    n, m, p = DIMENSIONS
    
    # Schoolbook fallback
    rank_sb = n * m * p
    U_sb = np.zeros((n * m, rank_sb))
    V_sb = np.zeros((m * p, rank_sb))
    W_sb = np.zeros((p * n, rank_sb))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb
    
    def verify(U, V, W, tol=1e-9):
        """Verify that the decomposition exactly decomposes the matrix multiplication tensor."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(U.shape[1]):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    if abs(s - 1) > tol:
                        return False
        return True
    
    def make_decomposition_from_params(params, rank):
        """Convert flat params to U, V, W matrices."""
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return U, V, W
    
    def objective(params, rank):
        """Objective: sum of squared errors in the tensor decomposition."""
        U, V, W = make_decomposition_from_params(params, rank)
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    error += (s - 1) ** 2
        return error
    
    # Try to find a rank < 54 decomposition using numerical optimization
    # Start with lower ranks and work up
    for rank in range(46, min(54, rank_sb)):
        if time.time() > deadline - 30:
            break
            
        # Try a few random initializations
        for trial in range(5):
            if time.time() > deadline - 30:
                break
                
            # Initialize with small random values
            np.random.seed(42 + trial)
            params0 = np.random.randn(n*m*rank + m*p*rank + p*n*rank) * 0.1
            
            # Use L-BFGS-B for optimization
            try:
                result = minimize(
                    objective,
                    params0,
                    args=(rank,),
                    method='L-BFGS-B',
                    options={'maxiter': 1000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                
                if result.fun < 1e-8:
                    U, V, W = make_decomposition_from_params(result.x, rank)
                    if verify(U, V, W, tol=1e-7):
                        best_rank = rank
                        best_U = U
                        best_V = V
                        best_W = W
                        break  # Found a valid decomposition for this rank
            except Exception:
                continue
    
    # If we found a better rank, use it
    if best_rank < rank_sb:
        return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
    
    # Fall back to schoolbook
    return {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
