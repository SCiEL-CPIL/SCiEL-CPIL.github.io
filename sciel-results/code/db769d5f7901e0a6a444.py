"""Matrix multiplication 2,2,13 - rank-minimal exact tensor decomposition."""
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

DIMENSIONS = [2, 2, 13]
DEADLINE = 120.0  # seconds

def construct():
    """Return a valid tensor decomposition for 2x2 * 2x13 matrix multiplication."""
    start_time = time.time()
    n, m, p = DIMENSIONS  # n=2, m=2, p=13
    
    def elapsed():
        return time.time() - start_time
    
    def remaining():
        return DEADLINE - 5 - elapsed()
    
    def check_decomposition(U, V, W, tol=1e-9):
        """Check if U, V, W form a valid decomposition. U,V,W are numpy arrays."""
        rank = U.shape[1]
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    def residual(U, V, W):
        """Compute sum of squared residuals."""
        rank = U.shape[1]
        total = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m + j, :] * V[j*p + k, :] * W[k*n + i, :])
                    total += (val - 1.0) ** 2
        return total
    
    def unpack(x, R):
        """Unpack flat vector into U, V, W arrays."""
        size_U = n * m * R
        size_V = m * p * R
        size_W = p * n * R
        U = x[:size_U].reshape(n * m, R)
        V = x[size_U:size_U + size_V].reshape(m * p, R)
        W = x[size_U + size_V:].reshape(p * n, R)
        return U, V, W
    
    def objective(x, R):
        """Objective function for optimizer."""
        U, V, W = unpack(x, R)
        return residual(U, V, W)
    
    # Schoolbook decomposition as fallback
    schoolbook_rank = n * m * p  # 52
    U_sb = np.zeros((n * m, schoolbook_rank))
    V_sb = np.zeros((m * p, schoolbook_rank))
    W_sb = np.zeros((p * n, schoolbook_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1
                V_sb[j * p + k, col] = 1
                W_sb[k * n + i, col] = 1
                col += 1
    
    best_rank = schoolbook_rank
    best_U = U_sb.copy()
    best_V = V_sb.copy()
    best_W = W_sb.copy()
    best_resid = 0.0
    
    # Try to find lower rank decompositions using optimization
    # Start with smaller ranks and work up
    for target_rank in [26, 28, 30, 32, 34, 36, 40, 44, 48]:
        if remaining() < 10:
            break
        
        # Multiple random restarts for this rank
        max_restarts = max(1, int(remaining() / 15))
        
        for attempt in range(max_restarts):
            if remaining() < 5:
                break
            
            np.random.seed(attempt * 7919 + target_rank * 31)
            
            # Initialize with small random values
            x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank) * 0.3
            
            # Use Nelder-Mead for local optimization
            try:
                result = minimize(
                    objective, x0, args=(target_rank,),
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
                )
                
                U, V, W = unpack(result.x, target_rank)
                resid = result.fun
                
                if resid < 1e-8:
                    if check_decomposition(U, V, W, tol=1e-9):
                        if target_rank < best_rank:
                            best_rank = target_rank
                            best_U = U.copy()
                            best_V = V.copy()
                            best_W = W.copy()
                            best_resid = resid
                        break  # Found valid decomposition for this rank, try smaller
                elif resid < best_resid and target_rank < best_rank:
                    # Keep track of best even if not fully converged
                    pass
                    
            except Exception:
                continue
        
        if best_rank <= target_rank:
            break
    
    # Try differential evolution for a few ranks if time permits
    if remaining() > 20 and best_rank > 26:
        for target_rank in [26, 28, 30]:
            if remaining() < 15:
                break
            
            bounds = [(-2, 2)] * (n*m*target_rank + m*p*target_rank + p*n*target_rank)
            
            try:
                result = differential_evolution(
                    objective, bounds, args=(target_rank,),
                    seed=42, maxiter=100, tol=1e-10,
                    polish=True, updating='immediate',
                    init='sobol'
                )
                
                U, V, W = unpack(result.x, target_rank)
                resid = result.fun
                
                if resid < 1e-8 and check_decomposition(U, V, W, tol=1e-9):
                    if target_rank < best_rank:
                        best_rank = target_rank
                        best_U = U.copy()
                        best_V = V.copy()
                        best_W = W.copy()
                        best_resid = resid
                        break
                        
            except Exception:
                continue
    
    # Convert to lists for return
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
