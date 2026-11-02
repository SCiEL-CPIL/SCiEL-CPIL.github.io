"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import least_squares

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    schoolbook_rank = n * m * p  # 24
    
    # Schoolbook solution as fallback
    U_sb = [[0] * schoolbook_rank for _ in range(n * m)]
    V_sb = [[0] * schoolbook_rank for _ in range(m * p)]
    W_sb = [[0] * schoolbook_rank for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    best_rank = schoolbook_rank
    
    # Try ranks from schoolbook-1 down to some minimum
    # We limit the range to save time, focusing on likely improvements
    for rank in range(schoolbook_rank - 1, 17, -1):
        if time.time() > deadline:
            break
        
        try:
            result = _try_rank(n, m, p, rank, deadline)
            if result is not None:
                best_result = result
                best_rank = rank
                # If we found a good rank, we can break if we hit a known lower bound or just continue
                # To maximize score, we try to find the lowest possible rank
                # But if we find rank 23, we should try 22.
                continue
        except Exception:
            continue
    
    return best_result

def _try_rank(n, m, p, rank, deadline):
    """Try to find a rank decomposition with the given rank using least_squares."""
    num_params = (n * m + m * p + p * n) * rank
    
    # Residual function for least_squares
    def residual_function(params):
        U = params[:n*m*rank].reshape(n*m, rank)
        V = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
        W = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
        
        residuals = np.zeros(n * m * p)
        idx = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j], V[j*p+k] * W[k*n+i])
                    residuals[idx] = val - 1.0
                    idx += 1
        return residuals

    # Multiple restarts to escape local minima
    np.random.seed(int(time.time() * 1000) % 10000)
    
    for restart in range(5):
        if time.time() > deadline:
            break
            
        # Random initialization
        U0 = np.random.randn(n * m, rank)
        V0 = np.random.randn(m * p, rank)
        W0 = np.random.randn(p * n, rank)
        
        # Scale initialization to be closer to expected magnitude
        # The product of three random numbers needs to sum to 1.
        # If entries are ~1, sum of rank terms is ~rank. So scale by 1/sqrt(rank)
        scale = 1.0 / np.sqrt(rank)
        U0 *= scale
        V0 *= scale
        W0 *= scale
        
        params0 = np.concatenate([U0.flatten(), V0.flatten(), W0.flatten()])
        
        try:
            result = least_squares(residual_function, params0, method='trf', max_nfev=100)
        except Exception:
            continue
            
        if result.cost < 1e-8:
            U_opt = result.x[:n*m*rank].reshape(n*m, rank)
            V_opt = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
            W_opt = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
            
            # Try snapping to half-integers
            for denom in [1, 2, 3, 4]:
                U_snap = np.round(U_opt * denom) / denom
                V_snap = np.round(V_opt * denom) / denom
                W_snap = np.round(W_opt * denom) / denom
                
                # Verify exactness
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_snap[i*m+j], V_snap[j*p+k] * W_snap[k*n+i])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-9:
                    return {
                        "U": U_snap.tolist(),
                        "V": V_snap.tolist(),
                        "W": W_snap.tolist()
                    }
    
    return None

# EVOLVE-BLOCK-END
