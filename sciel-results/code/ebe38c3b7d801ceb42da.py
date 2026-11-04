"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import least_squares

DIMENSIONS = [4, 5, 6]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140
    
    def is_half_integer(arr, tol=1e-9):
        doubled = arr * 2
        return np.allclose(doubled, np.round(doubled), atol=tol)
    
    def to_list(arr):
        result = []
        for row in arr:
            r = []
            for val in row:
                dval = val * 2
                if abs(dval - round(dval)) < 1e-9:
                    r.append(int(round(dval)) / 2.0)
                else:
                    r.append(val)  # shouldn't happen if is_half_integer passed
            result.append(r)
        return result
    
    # Schoolbook
    rank_school = n * m * p
    U_sb = np.zeros((n * m, rank_school))
    V_sb = np.zeros((m * p, rank_school))
    W_sb = np.zeros((p * n, rank_school))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j, col] = 1.0
                V_sb[j * p + k, col] = 1.0
                W_sb[k * n + i, col] = 1.0
                col += 1
    
    best_result = {"U": to_list(U_sb), "V": to_list(V_sb), "W": to_list(W_sb)}
    best_rank = rank_school
    
    # Try to find lower rank decompositions
    # The tensor equation: for all i,j,k: sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    # This is nonlinear. Let's try a randomized approach with low rank.
    
    np.random.seed(42)
    
    for rank in range(100, rank_school):  # Try ranks from 100 to 119
        if time.time() > deadline - 10:
            break
        
        found = False
        for trial in range(20):
            if time.time() > deadline - 5:
                break
            
            # Random initialization
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            # Define the residual: for each (i,j,k), the product should equal 1
            def residual(params):
                Uf = params[:n*m*rank].reshape(n*m, rank)
                Vf = params[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                Wf = params[n*m*rank + m*p*rank:].reshape(p*n, rank)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(Uf[i*m+j] * Vf[j*p+k] * Wf[k*n+i])
                            res.append(val - 1.0)
                return np.array(res)
            
            try:
                result = least_squares(residual, params, method='lm', max_nfev=1000)
                if result.success and np.max(np.abs(result.fun)) < 1e-8:
                    Uf = result.x[:n*m*rank].reshape(n*m, rank)
                    Vf = result.x[n*m*rank:n*m*rank + m*p*rank].reshape(m*p, rank)
                    Wf = result.x[n*m*rank + m*p*rank:].reshape(p*n, rank)
                    
                    # Check half-integer
                    if is_half_integer(Uf) and is_half_integer(Vf) and is_half_integer(Wf):
                        best_result = {"U": to_list(Uf), "V": to_list(Vf), "W": to_list(Wf)}
                        best_rank = rank
                        found = True
                        break
            except:
                pass
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
