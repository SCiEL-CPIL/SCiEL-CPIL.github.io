"""Program for 3x3 x 3x5 matrix multiplication tensor decomposition."""
import time
import random
import numpy as np
from scipy.optimize import minimize

DIMENSIONS = [3, 3, 5]
DEADLINE = 150.0
SAFETY_MARGIN = 10.0

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    n, m, p = DIMENSIONS  # 3, 3, 5
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - SAFETY_MARGIN - elapsed()
    
    def verify(U, V, W, rank):
        """Verify that the decomposition exactly reconstructs the multiplication tensor."""
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :rank], V[j*p+k, :rank] * W[k*n+i, :rank])
                    if abs(val - 1) > 1e-9:
                        return False
        return True
        
    def make_schoolbook():
        """Return the schoolbook decomposition of rank n*m*p = 45."""
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j][col] = 1
                    V[j*p+k][col] = 1
                    W[k*n+i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p  # 45
    
    def tensor_error(params, R):
        """Compute the sum of squared errors for the decomposition given flattened params."""
        # Unpack params into U, V, W
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        
        U_flat = params[:size_U]
        V_flat = params[size_U : size_U + size_V]
        W_flat = params[size_U + size_V : size_U + size_V + size_W]
        
        U = U_flat.reshape(n * m, R)
        V = V_flat.reshape(m * p, R)
        W = W_flat.reshape(p * n, R)
        
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    err += (val - 1.0) ** 2
        return err

    def try_rank(R, max_attempts=5):
        """Try to find a rank-R decomposition using gradient-based optimization."""
        if remaining() < 20:
            return None
        
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        total_size = size_U + size_V + size_W
        
        for attempt in range(max_attempts):
            if remaining() < 10:
                break
            
            # Random initialization
            params = np.random.randn(total_size) * 0.1
            
            # Optimize
            try:
                res = minimize(tensor_error, params, args=(R,), method='L-BFGS-B', 
                               options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 50})
            except Exception:
                continue
                
            if res.fun < 1e-10:
                # Extract U, V, W
                U_flat = res.x[:size_U]
                V_flat = res.x[size_U : size_U + size_V]
                W_flat = res.x[size_U + size_V : size_U + size_V + size_W]
                
                U = U_flat.reshape(n * m, R)
                V = V_flat.reshape(m * p, R)
                W = W_flat.reshape(p * n, R)
                
                # Try to round to half-integers
                U_round = np.round(U * 2) / 2
                V_round = np.round(V * 2) / 2
                W_round = np.round(W * 2) / 2
                if verify(U_round, V_round, W_round, R):
                    return {
                        "U": U_round.tolist(),
                        "V": V_round.tolist(),
                        "W": W_round.tolist()
                    }
                
                # Also try integer rounding
                U_int = np.round(U)
                V_int = np.round(V)
                W_int = np.round(W)
                if verify(U_int, V_int, W_int, R):
                    return {
                        "U": U_int.tolist(),
                        "V": V_int.tolist(),
                        "W": W_int.tolist()
                    }
                    
                # If not rounded, return the float version? 
                # The task says "integer/half-integer entries".
                # If we can't round, it's invalid.
        
        return None
    
    # Try ranks from 44 downward? Or just 44, 43?
    # 45 is the baseline.
    # If 44 is possible, we want it.
    for target_rank in [44, 43]:
        if remaining() < 30:
            break
        
        result = try_rank(target_rank, max_attempts=3)
        if result is not None:
            best_rank = target_rank
            best_result = result
            break
            
    return best_result
# EVOLVE-BLOCK-END
