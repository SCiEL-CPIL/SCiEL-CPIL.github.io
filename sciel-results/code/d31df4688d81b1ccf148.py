import scipy
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 7]

def construct():
    n, m, p = DIMENSIONS
    # Target rank to try
    target_ranks = [41, 40, 39, 38]
    
    best_result = None
    best_rank = n * m * p
    
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 100  # 100 seconds buffer
    
    for r in target_ranks:
        if time.time() > deadline:
            break
            
        # Try a few random restarts for this rank
        found = False
        max_restarts = 50
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
                
            # Random initialization
            # Scale factors to help convergence
            U = np.random.randn(n * m, r) * 0.1
            V = np.random.randn(m * p, r) * 0.1
            W = np.random.randn(p * n, r) * 0.1
            
            # Define the residual function
            def residuals(x):
                # x is flattened U, V, W
                U_flat = x[:n*m*r].reshape(n*m, r)
                V_flat = x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                W_flat = x[n*m*r + m*p*r :].reshape(p*n, r)
                
                res = []
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            # T_ijk is 1 if it's a valid multiplication term?
                            # The tensor T is defined by the multiplication.
                            # T_{i, j, k} corresponds to the term A_{i,j} B_{j,k} contributing to C_{i,k}.
                            # In the decomposition sum_l U_{im+j, l} V_{jp+k, l} W_{kn+i, l} = T_{i,j,k}
                            # The target value T_{i,j,k} is 1 for all i,j,k?
                            # Wait. The multiplication tensor T has entries T_{i,j,k} = 1 if the term A_{i,j}B_{j,k} is part of the sum for C_{i,k}.
                            # Yes, for matrix multiplication, every pair (i,j) and (j,k) contributes to C_{i,k} with coefficient 1.
                            # So T_{i,j,k} = 1 for all i in [0,n), j in [0,m), k in [0,p).
                            target = 1.0
                            val = 0.0
                            for l in range(r):
                                val += U_flat[i*m + j, l] * V_flat[j*p + k, l] * W_flat[k*n + i, l]
                            res.append(val - target)
                return np.array(res)
                
            # Initial residual
            x0 = np.concatenate([U.flatten(), V.flatten(), W.flatten()])
            try:
                sol = scipy.optimize.least_squares(residuals, x0, method='lm', max_nfev=1000, ftol=1e-12, xtol=1e-12, gtol=1e-12)
                if sol.success and np.max(np.abs(sol.fun)) < 1e-9:
                    # Check if entries are "nice" (integers or half-integers)
                    # The problem asks for integer/half-integer entries.
                    # We might need to round or scale.
                    # Often the solution is integer.
                    U_sol = sol.x[:n*m*r].reshape(n*m, r)
                    V_sol = sol.x[n*m*r : n*m*r + m*p*r].reshape(m*p, r)
                    W_sol = sol.x[n*m*r + m*p*r :].reshape(p*n, r)
                    
                    # Try to round to nearest integer or half-integer
                    # If the solution is close to integers, round them.
                    U_int = np.round(U_sol * 2) / 2
                    V_int = np.round(V_sol * 2) / 2
                    W_int = np.round(W_sol * 2) / 2
                    
                    # Verify the rounded solution
                    def verify(U, V, W, r):
                        for i in range(n):
                            for j in range(m):
                                for k in range(p):
                                    val = 0.0
                                    for l in range(r):
                                        val += U[i*m + j, l] * V[j*p + k, l] * W[k*n + i, l]
                                    if abs(val - 1.0) > 1e-9:
                                        return False
                        return True
                        
                    if verify(U_int, V_int, W_int, r):
                        return {"U": U_int.tolist(), "V": V_int.tolist(), "W": W_int.tolist()}
                    elif verify(U_sol, V_sol, W_sol, r):
                        # If not integer, we might still return it if the evaluator accepts floats?
                        # The prompt says "integer/half-integer entries".
                        # So we must find an integer solution.
                        pass
                        
            except Exception:
                pass
                
    # Fallback to schoolbook
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
