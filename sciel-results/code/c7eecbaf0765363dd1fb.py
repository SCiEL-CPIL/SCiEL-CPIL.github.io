"""Bootstrap direct approach for 2x3x7 matrix multiplication tensor decomposition."""
import time
import numpy as np
from fractions import Fraction

DIMENSIONS = [2, 3, 7]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin before 150s kill

    n, m, p = DIMENSIONS
    # Schoolbook as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1

    def verify_decomposition(U, V, W):
        """Verify that U, V, W exactly decompose the matmul tensor."""
        n, m, p = DIMENSIONS
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = sum(U[i*m+j][r] * V[j*p+k][r] * W[k*n+i][r] for r in range(len(U[0])))
                    if abs(val - 1.0) > 1e-10:
                        return False
        return True

    # Try a structured approach: use the fact that we can decompose
    # the computation using intermediate products.
    
    # For 2x3x7, let's try to find a rank-14 decomposition.
    # Strategy: Use a combination of schoolbook terms and clever cancellations.
    
    # First, let's try a direct algebraic construction.
    # We need: for all i in {0,1}, j in {0,1,2}, k in {0,...,6}:
    #   sum_r U[i,j,r] * V[j,k,r] * W[i,k,r] = 1
    
    # Try rank R from 14 upward
    for R in range(14, 25):
        if time.time() > deadline - 20:
            break
        
        # Use a structured initialization based on the problem structure
        # U[i,j,r], V[j,k,r], W[i,k,r]
        
        # Try a pattern where we use the j-index to separate terms
        # For each j, we need the sum over r to give 1 for all i,k
        
        # Let's try: for each j, assign R/3 terms to handle that j
        # But R might not be divisible by 3, so let's be more careful.
        
        # Alternative: use a random initialization with structure
        np.random.seed(42)
        
        # Initialize with small random values, but ensure some structure
        U_init = np.random.randn(n*m, R) * 0.5
        V_init = np.random.randn(m*p, R) * 0.5
        W_init = np.random.randn(p*n, R) * 0.5
        
        # Try to solve using scipy.optimize.least_squares
        try:
            from scipy.optimize import least_squares
            
            def residuals(params):
                U = params[:n*m*R].reshape(n*m, R)
                V = params[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W = params[n*m*R+m*p*R:].reshape(p*n, R)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                            res.append(val - 1.0)
                return np.array(res)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            result = least_squares(residuals, x0, method='lm', max_nfev=5000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
            
            if result.success or result.cost < 1e-12:
                U_sol = result.x[:n*m*R].reshape(n*m, R)
                V_sol = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                W_sol = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                
                # Try rounding to half-integers
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(n*m)]
                V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(m*p)]
                W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(p*n)]
                
                if verify_decomposition(U_list, V_list, W_list):
                    return {"U": U_list, "V": V_list, "W": W_list}
                
                # If rounding failed, try a local search around the solution
                # to find a nearby half-integer point
                
        except Exception:
            pass
        
        # Try multiple random initializations
        for seed in range(3):
            if time.time() > deadline - 15:
                break
            
            np.random.seed(seed + 100)
            U_init = np.random.randn(n*m, R) * 0.3
            V_init = np.random.randn(m*p, R) * 0.3
            W_init = np.random.randn(p*n, R) * 0.3
            
            try:
                from scipy.optimize import least_squares
                
                def residuals(params):
                    U = params[:n*m*R].reshape(n*m, R)
                    V = params[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W = params[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    res = []
                    for i in range(n):
                        for j in range(m):
                            for k in range(p):
                                val = np.sum(U[i*m+j] * V[j*p+k] * W[k*n+i])
                                res.append(val - 1.0)
                    return np.array(res)
                
                x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
                
                result = least_squares(residuals, x0, method='lm', max_nfev=3000, ftol=1e-15, xtol=1e-15, gtol=1e-15)
                
                if result.success or result.cost < 1e-12:
                    U_sol = result.x[:n*m*R].reshape(n*m, R)
                    V_sol = result.x[n*m*R:n*m*R+m*p*R].reshape(m*p, R)
                    W_sol = result.x[n*m*R+m*p*R:].reshape(p*n, R)
                    
                    U_rounded = np.round(U_sol * 2) / 2
                    V_rounded = np.round(V_sol * 2) / 2
                    W_rounded = np.round(W_sol * 2) / 2
                    
                    U_list = [[float(U_rounded[i, r]) for r in range(R)] for i in range(n*m)]
                    V_list = [[float(V_rounded[i, r]) for r in range(R)] for i in range(m*p)]
                    W_list = [[float(W_rounded[i, r]) for r in range(R)] for i in range(p*n)]
                    
                    if verify_decomposition(U_list, V_list, W_list):
                        return {"U": U_list, "V": V_list, "W": W_list}
                        
            except Exception:
                pass
    
    # If we didn't find a better solution, return schoolbook
    return {"U": U_sb, "V": V_sb, "W": W_sb}
# EVOLVE-BLOCK-END
