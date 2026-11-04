import numpy as np
import time
import random

DIMENSIONS = [2, 3, 9]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    timeout = 110  # Safety margin
    
    # Target tensor is all 1s
    # We search for rank r
    # Start from a lower bound estimate and go up, or start from schoolbook and go down?
    # Usually finding a low rank decomposition is harder.
    # Let's try ranks from 30 to 54.
    # Actually, if we find one at 54, we return it. But we want minimal.
    # Let's try a few specific ranks that are likely candidates for improvement.
    # 54 is trivial.
    # Can we do 53? 52?
    
    # Let's try to find any decomposition with rank < 54.
    # We'll iterate ranks from 30 to 53.
    # If we find one, we return it.
    # If not, we return the schoolbook one (rank 54).
    
    best_rank = n * m * p
    best_decomp = None
    
    # Pre-compute schoolbook as fallback
    U_school = np.zeros((n * m, best_rank))
    V_school = np.zeros((m * p, best_rank))
    W_school = np.zeros((p * n, best_rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    best_decomp = (U_school, V_school, W_school)
    
    # Try to find lower rank
    # We'll use ALS
    # Number of restarts per rank depends on time
    ranks_to_try = list(range(30, best_rank))
    
    for r in ranks_to_try:
        if time.time() - start_time > timeout:
            break
            
        # Try a few restarts for this rank
        # More restarts for smaller ranks?
        max_restarts = 50  # Increased from 20
        found = False
        
        for restart in range(max_restarts):
            if time.time() - start_time > timeout:
                break
                
            # Initialize with different strategies
            strategy = restart % 3
            if strategy == 0:
                U = np.random.randn(n * m, r) * 0.1
                V = np.random.randn(m * p, r) * 0.1
                W = np.random.randn(p * n, r) * 0.1
            elif strategy == 1:
                U = np.random.randn(n * m, r) * 0.5
                V = np.random.randn(m * p, r) * 0.5
                W = np.random.randn(p * n, r) * 0.5
            else:
                U = np.random.randint(0, 2, (n * m, r)) * 0.5
                V = np.random.randint(0, 2, (m * p, r)) * 0.5
                W = np.random.randint(0, 2, (p * n, r)) * 0.5
            
            # ALS iterations
            converged = False
            for it in range(100):  # Increased from 50
                if time.time() - start_time > timeout:
                    break
                    
                # Update U
                # For each i, j, solve A u = 1
                for i in range(n):
                    for j in range(m):
                        # A is p x r
                        A = np.zeros((p, r))
                        for k in range(p):
                            # A[k, c] = V[j*p+k, c] * W[k*n+i, c]
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        # Solve A u = 1
                        # Use lstsq
                        u, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(p), rcond=None)
                        U[i * m + j, :] = u
                        
                # Update V
                for j in range(m):
                    for k in range(p):
                        # A is n x r
                        A = np.zeros((n, r))
                        for i in range(n):
                            # A[i, c] = U[i*m+j, c] * W[k*n+i, c]
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        v, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(n), rcond=None)
                        V[j * p + k, :] = v
                        
                # Update W
                for k in range(p):
                    for i in range(n):
                        # A is m x r
                        A = np.zeros((m, r))
                        for j in range(m):
                            # A[j, c] = U[i*m+j, c] * V[j*p+k, c]
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        w, residuals, rank_A, s = np.linalg.lstsq(A, np.ones(m), rcond=None)
                        W[k * n + i, :] = w
                        
                # Check error
                err = 0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err += (val - 1.0) ** 2
                if err < 1e-6:
                    converged = True
                    break
                    
            if converged:
                # Try rounding to nearest 0.5
                U_rounded = np.round(U * 2) / 2
                V_rounded = np.round(V * 2) / 2
                W_rounded = np.round(W * 2) / 2
                
                # Verify exactness
                exact = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_rounded[i * m + j, :], V_rounded[j * p + k, :] * W_rounded[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                exact = False
                                break
                        if not exact:
                            break
                    if not exact:
                        break
                
                if exact:
                    best_rank = r
                    best_decomp = (U_rounded, V_rounded, W_rounded)
                    found = True
                    break # Found a decomposition for this rank, try lower ranks
            
        if found:
            # We can break early if we want the absolute minimum, but we don't know if lower is possible.
            # Given the time limit, if we find one at 40, we should try 39, 38...
            # So we don't break the outer loop, we just update best_rank and continue.
            pass

    # Return the best found
    U, V, W = best_decomp
    # Convert to lists of lists as required by interface? 
    # "factor matrices of shapes ... with integer/half-integer entries"
    # The interface example returns lists of lists.
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
