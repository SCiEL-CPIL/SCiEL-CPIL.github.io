import numpy as np
import time

def construct():
    n, m, p = 2, 4, 4
    deadline = time.time() + 120
    
    # The tensor T[i,j,k] = 1 for all valid (i,j,k)
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1 for all i,j,k
    
    # Start with naive rank 32
    rank_naive = n * m * p  # 32
    U = np.zeros((n * m, rank_naive), dtype=float)
    V = np.zeros((m * p, rank_naive), dtype=float)
    W = np.zeros((p * n, rank_naive), dtype=float)
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1
                V[j * p + k, idx] = 1
                W[k * n + i, idx] = 1
                idx += 1
    
    best_rank = rank_naive
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to find lower rank decompositions
    # Strategy: use random initialization and iterative refinement
    # For a given rank r, we want to solve:
    # U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] summed over r = 1 for all i,j,k
    
    # Try ranks from 16 down to 1
    for target_rank in range(16, 0, -1):
        if time.time() > deadline - 30:
            break
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            # Random initialization
            np.random.seed(attempt * 1000 + target_rank)
            U_r = np.random.randn(n * m, target_rank)
            V_r = np.random.randn(m * p, target_rank)
            W_r = np.random.randn(p * n, target_rank)
            
            # Iterative refinement using ALS (Alternating Least Squares)
            # For fixed V and W, solve for U
            # For each (i,j), we want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
            # This is linear in U for fixed V, W
            
            converged = False
            for iteration in range(100):
                if time.time() > deadline - 30:
                    break
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        # For each k, we have: sum_r U[i*m+j, r] * (V[j*p+k, r] * W[k*n+i, r]) = 1
                        # Let A[r, k] = V[j*p+k, r] * W[k*n+i, r]
                        # Then A @ U[i*m+j, :] should equal ones(p)
                        A = np.array([V[j * p + k, :] * W[k * n + i, :] for k in range(p)])
                        # Solve A @ x = ones(p)
                        try:
                            U[i * m + j, :] = np.linalg.lstsq(A, np.ones(p), rcond=None)[0]
                        except:
                            pass
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        # For each i, we have: sum_r V[j*p+k, r] * (U[i*m+j, r] * W[k*n+i, r]) = 1
                        # Let A[r, i] = U[i*m+j, r] * W[k*n+i, r]
                        # Then A @ V[j*p+k, :] should equal ones(n)
                        A = np.array([U[i * m + j, :] * W[k * n + i, :] for i in range(n)])
                        try:
                            V[j * p + k, :] = np.linalg.lstsq(A, np.ones(n), rcond=None)[0]
                        except:
                            pass
                
                # Update W
                for k in range(p):
                    for i in range(n):
                        # For each j, we have: sum_r W[k*n+i, r] * (U[i*m+j, r] * V[j*p+k, r]) = 1
                        # Let A[r, j] = U[i*m+j, r] * V[j*p+k, r]
                        # Then A @ W[k*n+i, :] should equal ones(m)
                        A = np.array([U[i * m + j, :] * V[j * p + k, :] for j in range(m)])
                        try:
                            W[k * n + i, :] = np.linalg.lstsq(A, np.ones(m), rcond=None)[0]
                        except:
                            pass
                
                # Check convergence
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-6:
                    converged = True
                    break
            
            if converged:
                # Verify the solution
                max_err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.sum(U_r[i * m + j, :] * V_r[j * p + k, :] * W_r[k * n + i, :])
                            max_err = max(max_err, abs(val - 1))
                
                if max_err < 1e-4:
                    # Found a valid decomposition
                    best_rank = target_rank
                    best_U, best_V, best_W = U_r.copy(), V_r.copy(), W_r.copy()
                    break
        
        if best_rank == target_rank:
            break
    
    # Convert to lists
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
