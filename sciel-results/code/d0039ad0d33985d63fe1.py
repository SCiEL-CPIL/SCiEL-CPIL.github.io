import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # U: (n*m, rank) = (9, rank)
    # V: (m*p, rank) = (12, rank)
    # W: (p*n, rank) = (12, rank)
    # Constraint: for all i in [0,n), j in [0,m), k in [0,p):
    #   sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1

    # Try to find a low-rank decomposition using an improved alternating least squares
    # with better initialization and convergence criteria.
    
    best_rank = n * m * p  # fallback to schoolbook (36)
    best_solution = None
    
    # We know that for 3x3x4, the schoolbook rank is 36.
    # There are known lower bounds and upper bounds.
    # Let's try to find a rank below 36.
    # A common approach is to use random initialization and ALS.
    
    # To improve upon the current program's approach, we will:
    # 1. Use a more robust initialization strategy (e.g., orthogonal initialization).
    # 2. Use a more aggressive convergence check.
    # 3. Try more trials for lower ranks.
    
    # The current program starts at rank 16. Let's start a bit higher to ensure we find something,
    # but still try to beat 36.
    
    # Actually, let's stick to the approach but make it more robust.
    # The main issue with the previous program might be getting stuck in local minima or not converging.
    # We can use a "restart" strategy within the ALS loop if it stalls.
    
    for rank in range(20, n * m * p + 1):
        found = False
        for trial in range(100):
            np.random.seed(trial * 7 + rank * 13)
            # Initialize with small random values
            U = np.random.randn(n * m, rank) * 0.5
            V = np.random.randn(m * p, rank) * 0.5
            W = np.random.randn(p * n, rank) * 0.5
            
            converged = False
            for it in range(500):
                prev_err = None
                
                # Update W
                for i in range(n):
                    for k in range(p):
                        A = np.zeros((m, rank))
                        for j in range(m):
                            A[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        b = np.ones(m)
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        W[k * n + i, :] = sol
                
                # Update U
                for i in range(n):
                    for j in range(m):
                        A = np.zeros((p, rank))
                        for k in range(p):
                            A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        b = np.ones(p)
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        U[i * m + j, :] = sol
                
                # Update V
                for j in range(m):
                    for k in range(p):
                        A = np.zeros((n, rank))
                        for i in range(n):
                            A[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        b = np.ones(n)
                        sol, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                        V[j * p + k, :] = sol
                
                # Check residual
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            err = abs(val - 1.0)
                            if err > max_err:
                                max_err = err
                
                if max_err < 1e-8:
                    converged = True
                    break
                
                if prev_err is not None and abs(prev_err - max_err) < 1e-12:
                    # Stalled, break early to try another trial
                    break
                prev_err = max_err
            
            if converged:
                # Round to nearest 0.5
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                # Verify exact decomposition with rounded values
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U_r[i * m + j, :], V_r[j * p + k, :] * W_r[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid:
                            break
                    if not valid:
                        break
                
                if valid:
                    best_rank = rank
                    best_solution = (U_r, V_r, W_r)
                    found = True
                    break
        
        if found:
            break
    
    if best_solution is not None:
        U, V, W = best_solution
        U_list = [[float(x) for x in row] for row in U]
        V_list = [[float(x) for x in row] for row in V]
        W_list = [[float(x) for x in row] for row in W]
        return {"U": U_list, "V": V_list, "W": W_list}
    else:
        # Fallback: schoolbook
        rank = n * m * p
        U = [[0.0] * rank for _ in range(n * m)]
        V = [[0.0] * rank for _ in range(m * p)]
        W = [[0.0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1.0
                    V[j * p + k][col] = 1.0
                    W[k * n + i][col] = 1.0
                    col += 1
        return {"U": U, "V": V, "W": W}
