"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK. Maintaining schoolbook rank 27."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [3, 3, 3]

# EVOLVE-BLOCK-START
def _verify_decomposition(U, V, W, n=3, m=3, p=3):
    """Verify that the decomposition exactly computes matrix multiplication."""
    rank = len(U[0]) if U else 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                total = 0.0
                for c in range(rank):
                    total += U[i*m+j][c] * V[j*p+k][c] * W[k*n+i][c]
                if abs(total - 1) > 1e-9:
                    return False
    return True

def _try_rank23_decomposition():
    """
    Attempt to construct a rank-23 decomposition for 3x3 matrix multiplication.
    """
    import numpy as np
    
    n, m, p = 3, 3, 3
    rank = 23
    
    # We need U (9x23), V (9x23), W (9x23)
    # such that sum_c U[i,j,c] V[j,k,c] W[k,i,c] = 1
    
    # Use a random initialization and gradient descent
    # Or use a known structure.
    # Let's try to find a solution using scipy.optimize
    
    def error_func(params):
        # params: flattened U, V, W
        U_flat = params[:9*rank]
        V_flat = params[9*rank:18*rank]
        W_flat = params[18*rank:27*rank]
        
        U = U_flat.reshape(9, rank)
        V = V_flat.reshape(9, rank)
        W = W_flat.reshape(9, rank)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    total = 0.0
                    for c in range(rank):
                        total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += (total - 1.0)**2
        return err

    # Initial guess: Random
    np.random.seed(42)
    U0 = np.random.randn(9, rank)
    V0 = np.random.randn(9, rank)
    W0 = np.random.randn(9, rank)
    
    # Scale to reduce error initially?
    # The product of three random numbers is small.
    # We need sum of products to be 1.
    # If we scale U, V, W by 1/sqrt(rank), the expected sum is 0.
    # We need non-zero mean.
    
    # Better init: Start with schoolbook and merge?
    # Or just use random and hope.
    
    # Let's try a few random restarts
    best_params = None
    best_err = float('inf')
    
    for restart in range(5):
        U0 = np.random.randn(9, rank)
        V0 = np.random.randn(9, rank)
        W0 = np.random.randn(9, rank)
        
        # Scale up to ensure magnitudes are reasonable
        U0 /= np.linalg.norm(U0) * 0.1
        V0 /= np.linalg.norm(V0) * 0.1
        W0 /= np.linalg.norm(W0) * 0.1
        
        x0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            from scipy.optimize import least_squares
            # Use least_squares for better stability
            def residuals(params):
                U_flat = params[:9*rank]
                V_flat = params[9*rank:18*rank]
                W_flat = params[18*rank:27*rank]
                U = U_flat.reshape(9, rank)
                V = V_flat.reshape(9, rank)
                W = W_flat.reshape(9, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            total = 0.0
                            for c in range(rank):
                                total += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            res.append(total - 1.0)
                return np.array(res)
            
            result = least_squares(residuals, x0, method='lm', max_nfev=10000)
            if result.success or result.cost < best_err:
                if result.cost < 1e-10:
                    best_err = result.cost
                    best_params = result.x
                elif best_err > result.cost:
                    best_err = result.cost
                    best_params = result.x
        except Exception:
            pass
            
    if best_params is not None and best_err < 1e-10:
        U = best_params[:9*rank].reshape(9, rank)
        V = best_params[9*rank:18*rank].reshape(9, rank)
        W = best_params[18*rank:27*rank].reshape(9, rank)
        
        # Convert to list of lists
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
        
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Leave 10s safety margin
    
    n, m, p = DIMENSIONS
    
    # Try rank-23 first
    if time.time() < deadline - 10:
        result = _try_rank23_decomposition()
        if result is not None:
            return result
    
    # Fall back to schoolbook rank 27
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
# EVOLVE-BLOCK-END
