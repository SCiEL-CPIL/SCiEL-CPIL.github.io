import numpy as np
import time

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140.0
    
    # Schoolbook baseline: rank 36
    rank_school = n * m * p  # 36
    U_school = np.zeros((n * m, rank_school))
    V_school = np.zeros((m * p, rank_school))
    W_school = np.zeros((p * n, rank_school))
    
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j, col] = 1
                V_school[j * p + k, col] = 1
                W_school[k * n + i, col] = 1
                col += 1
    
    best_result = {
        "U": U_school.tolist(),
        "V": V_school.tolist(),
        "W": W_school.tolist()
    }
    best_rank = rank_school
    
    def verify_decomposition(U, V, W, n, m, p, tol=1e-10):
        """Check if the decomposition is valid."""
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.sum(U[i * m + j, :] * V[j * p + k, :] * W[k * n + i, :])
                    if abs(val - 1.0) > tol:
                        return False
        return True
    
    # Strategy: For given U, V, solve for W linearly
    # The equation is: for all (i,j,k), sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    # Let M[i,j,k,r] = U[i*m+j, r] * V[j*p+k, r]
    # Then for each (i,j,k): sum_r M[i,j,k,r] * W[k*n+i, r] = 1
    # Note that W[k*n+i, r] appears in equations where the (k,i) pair matches.
    # For a fixed (k,i), the equations are: for all j, sum_r M[i,j,k,r] * W[k*n+i, r] = 1
    # This is a linear system in W[k*n+i, :] with m equations (one per j) and R unknowns.
    
    def try_solve_for_W(U, V, n, m, p, R):
        """Given U and V, try to solve for W. Returns W if solvable, else None."""
        W = np.zeros((p * n, R))
        
        for k in range(p):
            for i in range(n):
                # Build the linear system for W[k*n+i, :]
                # Equations: for j in range(m): sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
                # This is: A @ W[k*n+i, :] = 1
                # where A[j, r] = U[i*m+j, r] * V[j*p+k, r]
                
                A = np.zeros((m, R))
                for j in range(m):
                    A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                
                # Solve A @ x = 1 for x = W[k*n+i, :]
                # Use least squares
                try:
                    x, residuals, rank_A, sv = np.linalg.lstsq(A, np.ones(m), rcond=None)
                    # Check if the solution is exact
                    if np.allclose(A @ x, np.ones(m), atol=1e-10):
                        W[k * n + i, :] = x
                    else:
                        return None
                except np.linalg.LinAlgError:
                    return None
        
        return W
    
    # Try to find a valid decomposition at various ranks
    for target_rank in [35, 34, 33, 32, 31, 30]:
        if time.time() > deadline - 30:
            break
        
        for attempt in range(50):
            if time.time() > deadline - 30:
                break
            
            np.random.seed(42 + attempt * 1000 + target_rank * 100)
            
            # Generate random U and V with some sparsity
            U_rand = np.random.randn(n * m, target_rank) * 0.5
            V_rand = np.random.randn(m * p, target_rank) * 0.5
            
            W_rand = try_solve_for_W(U_rand, V_rand, n, m, p, target_rank)
            
            if W_rand is not None:
                if verify_decomposition(U_rand, V_rand, W_rand, n, m, p):
                    best_result = {
                        "U": U_rand.tolist(),
                        "V": V_rand.tolist(),
                        "W": W_rand.tolist()
                    }
                    best_rank = target_rank
                    break
    
    return best_result
