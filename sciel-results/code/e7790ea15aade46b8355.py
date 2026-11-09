"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
import scipy.optimize as opt

DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
def construct():
    """
    Find a rank-minimal exact tensor decomposition for 2x3 * 3x4 matrix multiplication.
    We search for a rank 19 decomposition, which is the known minimum.
    If not found within time limits, we fall back to schoolbook rank 24.
    """
    start_time = time.time()
    DEADLINE = 145  # seconds, safety margin before 150s limit
    
    n, m, p = DIMENSIONS  # 2, 3, 4
    
    # Schoolbook decomposition as fallback
    def schoolbook():
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
    
    # Target rank 19
    target_rank = 19
    
    # Build target tensor T
    # T[i, j, k] = 1
    T = np.ones((n, m, p))
    
    def unpack(x):
        """Unpack flat vector into U, V, W"""
        size_u = n * m * target_rank
        size_v = m * p * target_rank
        size_w = p * n * target_rank
        U = x[:size_u].reshape(n, m, target_rank)
        V = x[size_u:size_u+size_v].reshape(m, p, target_rank)
        W = x[size_u+size_v:].reshape(p, n, target_rank)
        return U, V, W
        
    def residuals(x):
        """Compute residual for least squares"""
        U, V, W = unpack(x)
        # Compute sum_r U[i,j,r] * V[j,k,r] * W[k,i,r]
        # Use einsum for efficiency
        approx = np.einsum('ijr,jkr,kir->ijk', U, V, W)
        # Residual is approx - T
        # Flatten for least_squares
        return (approx - T).flatten()
        
    best_result = None
    best_error = float('inf')
    
    # Try multiple random initializations
    np.random.seed(42)
    
    max_trials = 100
    for trial in range(max_trials):
        if time.time() - start_time > DEADLINE:
            break
            
        # Initialize with small random values
        x0 = np.random.randn(n*m*target_rank + m*p*target_rank + p*n*target_rank) * 0.1
        
        # Run optimizer
        try:
            res = opt.least_squares(residuals, x0, method='trf', max_nfev=1000, xtol=1e-12, ftol=1e-12, gtol=1e-12)
        except Exception:
            continue
            
        error = res.cost * 2 # cost is 0.5 * sum(residuals^2)
        
        if error < best_error:
            best_error = error
            best_x = res.x
            
        # Check if we found an exact solution (within tolerance)
        if error < 1e-10:
            U, V, W = unpack(best_x)
            
            # Round to nearest 0.5
            U_rounded = np.round(U * 2) / 2
            V_rounded = np.round(V * 2) / 2
            W_rounded = np.round(W * 2) / 2
            
            # Verify rounded solution
            approx_rounded = np.einsum('ijr,jkr,kir->ijk', U_rounded, V_rounded, W_rounded)
            error_rounded = np.sum((approx_rounded - T)**2)
            
            if error_rounded < 1e-8:
                # Convert to lists
                U_list = U_rounded.reshape(n*m, target_rank).tolist()
                V_list = V_rounded.reshape(m*p, target_rank).tolist()
                W_list = W_rounded.reshape(p*n, target_rank).tolist()
                
                # Ensure integers where possible
                for i in range(len(U_list)):
                    for j in range(len(U_list[i])):
                        if abs(U_list[i][j] - round(U_list[i][j])) < 1e-9:
                            U_list[i][j] = int(round(U_list[i][j]))
                for i in range(len(V_list)):
                    for j in range(len(V_list[i])):
                        if abs(V_list[i][j] - round(V_list[i][j])) < 1e-9:
                            V_list[i][j] = int(round(V_list[i][j]))
                for i in range(len(W_list)):
                    for j in range(len(W_list[i])):
                        if abs(W_list[i][j] - round(W_list[i][j])) < 1e-9:
                            W_list[i][j] = int(round(W_list[i][j]))
                            
                return {"U": U_list, "V": V_list, "W": W_list}
                
    # If best_error is very small but rounding failed, try harder rounding or return schoolbook
    # For now, if we didn't return, we fall back
    return schoolbook()
# EVOLVE-BLOCK-END
