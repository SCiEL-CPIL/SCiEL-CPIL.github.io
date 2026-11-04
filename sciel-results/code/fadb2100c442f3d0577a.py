import numpy as np
import time
from itertools import product

DIMENSIONS = [3, 4, 6]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Start with schoolbook rank 72
    schoolbook_rank = n * m * p  # 72
    
    # Try to find a better decomposition using structured approach
    # Key insight: We can use Strassen-style block decomposition
    # Split m=4 into two 2x2 blocks, and p=6 into two 3x3 blocks
    
    # For a 3x4 * 4x6 multiplication:
    # A is 3x4, B is 4x6, C is 3x6
    # Split A = [A1 | A2] where A1, A2 are 3x2
    # Split B = [B1; B2] where B1, B2 are 2x6
    # C = A1*B1 + A2*B2
    
    # Now A1*B1 is 3x2 * 2x6 = 3x6
    # A2*B2 is 3x2 * 2x6 = 3x6
    
    # Each 3x2 * 2x6 can be decomposed using Strassen-like methods
    # For 2x2 * 2x2, Strassen uses 7 multiplications instead of 8
    
    # Let's try a different approach: use the fact that we can share
    # intermediate computations between the two products
    
    # Actually, let me try a greedy set-cover approach with structured candidates
    
    # Generate candidate rank-1 tensors that are "structured"
    # A rank-1 tensor is defined by vectors u, v, w
    # For matrix multiplication, we want u[i*m+j] * v[j*p+k] = w[k*n+i]
    # This means u, v, w must have special structure
    
    # Let's try to find a decomposition by solving a linear system
    # for a given rank r
    
    def try_rank(r, timeout=None):
        """Try to find a rank-r decomposition"""
        if timeout is None:
            timeout = deadline - time.time() - 1.0
        if timeout < 0.5:
            return None
        
        # We need to find U (12xr), V (24xr), W (18xr) such that
        # for all i,j,k: sum_t U[i*m+j, t] * V[j*p+k, t] = W[k*n+i, t]
        # Wait, that's not right. The tensor decomposition is:
        # T[i,j,k] = sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t]
        # where T[i,j,k] = 1 if this is a valid multiplication term
        
        # Actually, for matrix multiplication tensor:
        # (A*B)[i,k] = sum_j A[i,j] * B[j,k]
        # The tensor T has T[i,j,k] = 1 for all i,j,k
        # We want: sum_t U[i*m+j, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all i,j,k
        
        # This is a system of n*m*p = 72 equations in r*12 + r*24 + r*18 = 54r unknowns
        # But it's nonlinear (cubic).
        
        # Let's use a different approach: fix U and V, solve for W
        # Or use alternating minimization
        
        # For small r, we can try random initialization and ALS
        
        best_result = None
        
        # Try multiple random initializations
        n_trials = max(1, int(timeout * 2))
        
        for trial in range(n_trials):
            if time.time() > deadline - 1.0:
                break
            
            # Initialize U, V randomly
            U = np.random.randn(n * m, r)
            V = np.random.randn(m * p, r)
            W = np.random.randn(p * n, r)
            
            # ALS iterations
            for it in range(50):
                if time.time() > deadline - 1.0:
                    break
                
                # Update U: for each (i,j), solve for U[i*m+j, :]
                for idx in range(n * m):
                    i, j = divmod(idx, m)
                    # We need: sum_t U[idx, t] * V[j*p+k, t] * W[k*n+i, t] = 1 for all k
                    # This is: sum_t U[idx, t] * (V[j*p+k, t] * W[k*n+i, t]) = 1
                    # Let M[k, t] = V[j*p+k, t] * W[k*n+i, t]
                    # Then: U[idx, :] @ M.T = ones(p)
                    # This is underdetermined if r < p, overdetermined if r >= p
                    
                    M = np.zeros((p, r))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    
                    # Solve U[idx, :] @ M.T = ones(p)
                    # Using least squares
                    if r >= p:
                        # Overdetermined, use lstsq
                        try:
                            U[idx, :], residuals, rank_m, s = np.linalg.lstsq(M.T, np.ones(p), rcond=None)
                        except:
                            pass
                    else:
                        # Underdetermined, use minimum norm solution
                        try:
                            U[idx, :] = np.linalg.pinv(M.T) @ np.ones(p)
                        except:
                            pass
                
                # Update V: for each (j,k), solve for V[j*p+k, :]
                for idx in range(m * p):
                    j, k = divmod(idx, p)
                    # We need: sum_t U[i*m+j, t] * V[idx, t] * W[k*n+i, t] = 1 for all i
                    # Let M[i, t] = U[i*m+j, t] * W[k*n+i, t]
                    # Then: V[idx, :] @ M.T = ones(n)
                    
                    M = np.zeros((n, r))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    
                    if r >= n:
                        try:
                            V[idx, :], residuals, rank_m, s = np.linalg.lstsq(M.T, np.ones(n), rcond=None)
                        except:
                            pass
                    else:
                        try:
                            V[idx, :] = np.linalg.pinv(M.T) @ np.ones(n)
                        except:
                            pass
                
                # Update W: for each (k,i), solve for W[k*n+i, :]
                for idx in range(p * n):
                    k, i = divmod(idx, n)
                    # We need: sum_t U[i*m+j, t] * V[j*p+k, t] * W[idx, t] = 1 for all j
                    # Let M[j, t] = U[i*m+j, t] * V[j*p+k, t]
                    # Then: W[idx, :] @ M.T = ones(m)
                    
                    M = np.zeros((m, r))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    
                    if r >= m:
                        try:
                            W[idx, :], residuals, rank_m, s = np.linalg.lstsq(M.T, np.ones(m), rcond=None)
                        except:
                            pass
                    else:
                        try:
                            W[idx, :] = np.linalg.pinv(M.T) @ np.ones(m)
                        except:
                            pass
            
            # Check if the decomposition is valid
            error = 0.0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                        error += (val - 1.0) ** 2
            
            if error < 1e-6:
                # Convert to list format
                U_list = [[float(U[i, j]) for j in range(r)] for i in range(n * m)]
                V_list = [[float(V[i, j]) for j in range(r)] for i in range(m * p)]
                W_list = [[float(W[i, j]) for j in range(r)] for i in range(p * n)]
                return (r, U_list, V_list, W_list)
        
        return None
    
    # Try ranks from low to high
    best_result = None
    
    # The minimum possible rank is at least max(n, m, p) = 6
    # But for matrix multiplication, it's known to be higher
    # Let's try from 36 upward (half of schoolbook)
    
    for r in range(30, 73):
        if time.time() > deadline - 5.0:
            break
        
        result = try_rank(r)
        if result is not None:
            best_result = result
            break
    
    # If we found a better result, use it
    if best_result is not None:
        r, U, V, W = best_result
        return {"U": U, "V": V, "W": W}
    
    # Otherwise return schoolbook
    rank = schoolbook_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][col] = 1
                V[j * p + k][col] = 1
                W[k * n + i][col] = 1
                col += 1
    
    return {"U": U, "V": V, "W": W}
