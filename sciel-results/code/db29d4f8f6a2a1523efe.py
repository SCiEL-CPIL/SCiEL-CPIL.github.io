"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 4]

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time

def construct():
    n, m, p = DIMENSIONS
    deadline = time.time() + 140  # safety margin before 150s limit
    
    # The constraint: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    # U: (n*m, r), V: (m*p, r), W: (p*n, r)
    
    # Start with schoolbook rank and try to reduce
    best_rank = n * m * p
    best_result = None
    
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
    best_result = {"U": U_sb, "V": V_sb, "W": W_sb}
    
    # Try to find lower rank decompositions
    # For each candidate rank r, try to solve the nonlinear system
    for r in range(1, best_rank):
        if time.time() > deadline:
            break
        
        # We need to find U (n*m x r), V (m*p x r), W (p*n x r)
        # such that for all (i,j,k): sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
        
        # Use random restarts with optimization
        num_restarts = 5
        found = False
        
        for restart in range(num_restarts):
            if time.time() > deadline:
                break
            
            # Initialize randomly
            np.random.seed(42 + restart + r)
            U_init = np.random.randn(n * m, r)
            V_init = np.random.randn(m * p, r)
            W_init = np.random.randn(p * n, r)
            
            x0 = np.concatenate([U_init.flatten(), V_init.flatten(), W_init.flatten()])
            
            def objective(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Compute residual: for each (i,j,k), check if sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
                residual = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            residual += (val - 1.0) ** 2
                return residual
            
            def gradient(x):
                U = x[:n*m*r].reshape(n*m, r)
                V = x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W = x[n*m*r+m*p*r:].reshape(p*n, r)
                
                grad_U = np.zeros_like(U)
                grad_V = np.zeros_like(V)
                grad_W = np.zeros_like(W)
                
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                            err = val - 1.0
                            for c in range(r):
                                grad_U[i*m+j, c] += 2*err * V[j*p+k, c] * W[k*n+i, c]
                                grad_V[j*p+k, c] += 2*err * U[i*m+j, c] * W[k*n+i, c]
                                grad_W[k*n+i, c] += 2*err * U[i*m+j, c] * V[j*p+k, c]
                
                return np.concatenate([grad_U.flatten(), grad_V.flatten(), grad_W.flatten()])
            
            res = minimize(objective, x0, method='L-BFGS-B', jac=gradient, 
                          options={'maxiter': 500, 'gtol': 1e-10, 'ftol': 1e-12})
            
            if res.fun < 1e-8:
                # Found a solution! Round to nearest half-integer and verify
                U_sol = res.x[:n*m*r].reshape(n*m, r)
                V_sol = res.x[n*m*r:n*m*r+m*p*r].reshape(m*p, r)
                W_sol = res.x[n*m*r+m*p*r:].reshape(p*n, r)
                
                # Round to nearest 0.5
                U_rounded = np.round(U_sol * 2) / 2
                V_rounded = np.round(V_sol * 2) / 2
                W_rounded = np.round(W_sol * 2) / 2
                
                # Verify the rounded solution
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = 0.0
                            for c in range(r):
                                val += U_rounded[i*m+j, c] * V_rounded[j*p+k, c] * W_rounded[k*n+i, c]
                            if abs(val - 1.0) > 1e-6:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    found = True
                    # Convert to list of lists with integer/half-integer entries
                    U_list = [[float(U_rounded[i, c]) for c in range(r)] for i in range(n*m)]
                    V_list = [[float(V_rounded[i, c]) for c in range(r)] for i in range(m*p)]
                    W_list = [[float(W_rounded[i, c]) for c in range(r)] for i in range(p*n)]
                    best_result = {"U": U_list, "V": V_list, "W": W_list}
                    best_rank = r
                    break
        
        if found:
            break
    
    return best_result
# EVOLVE-BLOCK-END
