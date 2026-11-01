import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [2, 3, 9]

def construct():
    start_time = time.time()
    deadline = start_time + 140  # safety margin
    
    n, m, p = DIMENSIONS  # 2, 3, 9
    
    # Tensor T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # Decomposition: T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    
    # Try ranks from low to high
    # Lower bound: at least max(n*m, m*p, p*n) is NOT a valid lower bound for tensor rank
    # But we need R >= 1 at minimum
    # A better lower bound: the "subspace" bounds
    
    # Let's try to find the minimum rank by optimization
    best_result = None
    
    # Start with schoolbook as fallback
    rank_school = n * m * p  # 54
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    best_result = {"U": U_school, "V": V_school, "W": W_school}
    
    # Try to find lower rank decompositions
    # Use optimization
    
    def make_residuals(U, V, W, n, m, p):
        """Return residuals for all (i,j,k)."""
        residuals = []
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = 0.0
                    for r in range(U.shape[1]):
                        val += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    residuals.append(val - 1.0)
        return np.array(residuals)
    
    def objective(x, R, n, m, p):
        """x is flattened [U | V | W], each of shape (dim, R)."""
        U = x[:n*m*R].reshape(n*m, R)
        V = x[n*m*R:(n*m+m*p)*R].reshape(m*p, R)
        W = x[(n*m+m*p)*R:].reshape(p*n, R)
        res = make_residuals(U, V, W, n, m, p)
        return np.sum(res**2)
    
    # Try ranks from 27 down (or some reasonable range)
    # Actually, let's start with a reasonable estimate
    # The tensor is 2x3x9, schoolbook is 54
    # Try to find if we can do better
    
    # Let's try a few ranks
    for R in range(27, 55):
        if time.time() > deadline - 20:
            break
        
        # Random initialization
        np.random.seed(42)
        x0 = np.random.randn((n*m + m*p + p*n) * R) * 0.1
        
        try:
            result = minimize(objective, x0, args=(R, n, m, p), method='L-BFGS-B', 
                            options={'maxiter': 200, 'ftol': 1e-12})
            
            if result.fun < 1e-10:
                # Found a valid decomposition
                U = result.x[:n*m*R].reshape(n*m, R)
                V = result.x[n*m*R:(n*m+m*p)*R].reshape(m*p, R)
                W = result.x[(n*m+m*p)*R:].reshape(p*n, R)
                
                # Convert to lists
                U_list = [[float(U[i,j]) for j in range(R)] for i in range(n*m)]
                V_list = [[float(V[i,j]) for j in range(R)] for i in range(m*p)]
                W_list = [[float(W[i,j]) for j in range(R)] for i in range(p*n)]
                
                best_result = {"U": U_list, "V": V_list, "W": W_list}
                break  # Found the lowest R, stop
                
        except Exception:
            continue
    
    return best_result
