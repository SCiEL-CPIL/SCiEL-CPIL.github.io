"""Matrix multiplication 2x3 x 3x6 - rank-minimal tensor decomposition."""
import numpy as np
import time
import random

DIMENSIONS = [2, 3, 6]
n, m, p = DIMENSIONS

def construct():
    start_time = time.time()
    deadline = start_time + 130
    
    def verify(U, V, W, rank):
        U = np.array(U, dtype=float)
        V = np.array(V, dtype=float)
        W = np.array(W, dtype=float)
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    if abs(val - 1.0) > 1e-8:
                        return False
        return True
    
    # Schoolbook
    def schoolbook():
        rank = n * m * p
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1
                    V[j*p+k, col] = 1
                    W[k*n+i, col] = 1
                    col += 1
        return U.tolist(), V.tolist(), W.tolist(), rank
    
    best_rank = n * m * p
    best_solution = schoolbook()
    
    # Try to find lower rank solutions
    # Use iterative refinement: fix V and W, solve for U (and vice versa)
    
    for target_rank in range(18, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        max_attempts = 30
        if time.time() > deadline - 20:
            max_attempts = min(max_attempts, 5)
        
        found = False
        for attempt in range(max_attempts):
            if time.time() > deadline - 10:
                break
            
            # Initialize randomly
            U = np.random.randn(n*m, target_rank) * 0.5
            V = np.random.randn(m*p, target_rank) * 0.5
            W = np.random.randn(p*n, target_rank) * 0.5
            
            # Alternating least squares
            for iteration in range(300):
                if time.time() > deadline - 10:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for i in range(n):
                    for j in range(m):
                        # Need: U[i*m+j,:] dot (V[j*p+k,:] * W[k*n+i,:]) = 1 for all k
                        A = np.array([V[j*p+k, :] * W[k*n+i, :] for k in range(p)])  # p x rank
                        b = np.ones(p)
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            U[i*m+j, :] = sol
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for j in range(m):
                    for k in range(p):
                        A = np.array([U[i*m+j, :] * W[k*n+i, :] for i in range(n)])  # n x rank
                        b = np.ones(n)
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            V[j*p+k, :] = sol
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for k in range(p):
                    for i in range(n):
                        A = np.array([U[i*m+j, :] * V[j*p+k, :] for j in range(m)])  # m x rank
                        b = np.ones(m)
                        try:
                            sol = np.linalg.lstsq(A, b, rcond=None)[0]
                            W[k*n+i, :] = sol
                        except:
                            pass
                
                # Check error
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                            max_err = max(max_err, abs(val - 1.0))
                
                if max_err < 1e-7:
                    U_list = U.tolist()
                    V_list = V.tolist()
                    W_list = W.tolist()
                    best_rank = target_rank
                    best_solution = (U_list, V_list, W_list, target_rank)
                    found = True
                    break
            
            if found:
                break
        
        if found:
            break
    
    U_list, V_list, W_list, rank = best_solution
    return {"U": U_list, "V": V_list, "W": W_list}
