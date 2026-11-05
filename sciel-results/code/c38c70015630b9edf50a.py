import numpy as np
from scipy.optimize import minimize
import time

DIMENSIONS = [3, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # 10s safety margin
    
    # Naive decomposition
    rank_naive = n * m * p
    U_naive = np.zeros((n*m, rank_naive))
    V_naive = np.zeros((m*p, rank_naive))
    W_naive = np.zeros((p*n, rank_naive))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_naive[i*m+j, col] = 1
                V_naive[j*p+k, col] = 1
                W_naive[k*n+i, col] = 1
                col += 1
    
    def verify(U, V, W, n, m, p):
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1) > 1e-8:
                        return False
        return True
    
    # Verify naive
    if not verify(U_naive, V_naive, W_naive, n, m, p):
        raise ValueError("Naive decomposition failed")
    
    best_rank = rank_naive
    best_U, best_V, best_W = U_naive, V_naive, W_naive
    
    # Try to find rank-23 decomposition for 3x3x3
    if time.time() < deadline - 60:
        try:
            U23, V23, W23 = find_rank23_decomp()
            if U23 is not None:
                # Build 59-rank decomposition
                U59 = np.zeros((n*m, 59))
                V59 = np.zeros((m*p, 59))
                W59 = np.zeros((p*n, 59))
                
                # First 23 terms: 3x3x3 for k=0,1,2
                for r in range(23):
                    for i in range(n):
                        for j in range(m):
                            U59[i*m+j, r] = U23[i, j, r]
                    for j in range(m):
                        for k in range(3):
                            V59[j*p+k, r] = V23[j, k, r]
                    for k in range(3):
                        for i in range(n):
                            W59[k*n+i, r] = W23[k, i, r]
                
                # Next 36 terms: naive 3x3x4 for k=3,4,5,6
                col = 23
                for i in range(n):
                    for j in range(m):
                        for k in range(3, p):
                            U59[i*m+j, col] = 1
                            V59[j*p+k, col] = 1
                            W59[k*n+i, col] = 1
                            col += 1
                
                if verify(U59, V59, W59, n, m, p):
                    best_rank = 59
                    best_U, best_V, best_W = U59, V59, W59
        except Exception as e:
            pass
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}

def find_rank23_decomp():
    # Try to find rank-23 decomposition for 3x3x3
    n = 3
    rank = 23
    
    # Try multiple random initializations
    best_err = np.inf
    best_params = None
    
    for trial in range(5):
        np.random.seed(trial)
        U = np.random.randn(n, n, rank)
        V = np.random.randn(n, n, rank)
        W = np.random.randn(n, n, rank)
        
        params = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
        
        def objective(params):
            U = params[:n*n*rank].reshape(n, n, rank)
            V = params[n*n*rank:2*n*n*rank].reshape(n, n, rank)
            W = params[2*n*n*rank:].reshape(n, n, rank)
            
            err = 0
            for i in range(n):
                for j in range(n):
                    for k in range(n):
                        val = np.sum(U[i,j,:] * V[j,k,:] * W[k,i,:])
                        err += (val - 1)**2
            return err
        
        result = minimize(objective, params, method='BFGS', options={'maxiter': 200, 'gtol': 1e-8})
        
        if result.fun < best_err:
            best_err = result.fun
            best_params = result.x
    
    if best_err < 1e-6:
        U = best_params[:n*n*rank].reshape(n, n, rank)
        V = best_params[n*n*rank:2*n*n*rank].reshape(n, n, rank)
        W = best_params[2*n*n*rank:].reshape(n, n, rank)
        return U, V, W
    else:
        return None, None, None
