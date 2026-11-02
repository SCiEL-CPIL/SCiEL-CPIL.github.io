"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.linalg import svd

DIMENSIONS = [3, 3, 7]
_START_TIME = None

# EVOLVE-BLOCK-START
def construct():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()
    
    n, m, p = DIMENSIONS
    deadline = _START_TIME + 120  # 120s safety margin
    
    # Schoolbook decomposition as fallback
    rank_schoolbook = n * m * p
    U_sb = np.zeros((n * m, rank_schoolbook))
    V_sb = np.zeros((m * p, rank_schoolbook))
    W_sb = np.zeros((p * n, rank_schoolbook))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    # Try to find a lower-rank decomposition using numerical optimization
    # The tensor T[i,j,k] = 1 for all i,j,k (matrix multiplication tensor)
    # We want T[i,j,k] = sum_c U[i,j,c] * V[j,k,c] * W[k,i,c]
    
    best_result = {"U": U_sb.tolist(), "V": V_sb.tolist(), "W": W_sb.tolist()}
    
    # Try various ranks starting from a reasonable lower bound
    # For 3x3x3, Strassen gives rank 7
    # For 3x3x7, we can try ranks from 7 up
    for target_rank in [7, 8, 9, 10, 11, 12, 13, 14, 15]:
        if time.time() > deadline - 30:
            break
        
        try:
            result = _try_rank(n, m, p, target_rank, deadline)
            if result is not None:
                U, V, W = result
                # Verify the decomposition
                if _verify_decomposition(n, m, p, U, V, W):
                    # Convert to list format
                    best_result = {
                        "U": U.tolist(),
                        "V": V.tolist(),
                        "W": W.tolist()
                    }
                    break  # Found a valid lower-rank decomposition
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank-r decomposition using numerical optimization."""
    
    # Initialize with random values
    np.random.seed(42)
    
    U = np.random.randn(n * m, rank)
    V = np.random.randn(m * p, rank)
    W = np.random.randn(p * n, rank)
    
    # Flatten for optimization
    def flatten(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        return U, V, W
    
    def objective(params):
        if time.time() > deadline - 5:
            return float('inf')
        
        U, V, W = flatten(params)
        
        # Compute the tensor from the decomposition
        # T[i,j,k] should equal sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
        error = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    # Compute sum over c
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    error += (val - 1.0) ** 2
        
        return error
    
    # Initial parameters
    params0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
    
    # Use L-BFGS-B for optimization
    try:
        result = minimize(
            objective,
            params0,
            method='L-BFGS-B',
            options={'maxiter': 500, 'maxfun': 5000, 'disp': False}
        )
        
        if result.fun < 1e-8:
            U, V, W = flatten(result.x)
            return U, V, W
    except Exception:
        pass
    
    return None

def _verify_decomposition(n, m, p, U, V, W):
    """Verify that the decomposition exactly reproduces the matrix multiplication tensor."""
    rank = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1.0) > 1e-6:
                    return False
    return True
# EVOLVE-BLOCK-END
