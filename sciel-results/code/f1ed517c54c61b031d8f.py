import numpy as np

DIMENSIONS = [3, 3, 4]

def construct():
    n, m, p = DIMENSIONS  # 3, 3, 4
    # Target: sum_s U[i*m+j, s] * V[j*p+k, s] * W[k*n+i, s] = 1
    # We use integer variables U', V', W' such that U = U'/2, etc?
    # If U, V, W are half-integers, 2U, 2V, 2W are integers.
    # (2U/2)(2V/2)(2W/2) = (2U)(2V)(2W)/8 = 1 => (2U)(2V)(2W) = 8.
    # So we solve for integer matrices A, B, C such that sum A*B*C = 8.
    # Then U = A/2, V = B/2, W = C/2.
    
    best_rank = n * m * p
    best_solution = None
    
    # Try ranks from low to high
    # Known rank for 3x3x4 is 18.
    for rank in range(18, n * m * p + 1):
        found = False
        # Try multiple random initializations
        for trial in range(200):
            np.random.seed(trial * 1000 + rank)
            # Initialize with small integers
            A = np.random.randint(-2, 3, size=(n * m, rank))
            B = np.random.randint(-2, 3, size=(m * p, rank))
            C = np.random.randint(-2, 3, size=(p * n, rank))
            
            converged = False
            prev_err = float('inf')
            
            for it in range(50):
                # Update A (U')
                for i in range(n):
                    for j in range(m):
                        # Equations for k=0..p-1:
                        # sum_s A[i*m+j, s] * (B[j*p+k, s] * C[k*n+i, s]) = 8
                        M = np.zeros((p, rank))
                        for k in range(p):
                            M[k, :] = B[j * p + k, :] * C[k * n + i, :]
                        b = np.ones(p) * 8
                        # Solve M @ A[i*m+j, :] = b
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        except:
                            sol = np.zeros(rank)
                        A[i * m + j, :] = np.round(sol)
                
                # Update B (V')
                for j in range(m):
                    for k in range(p):
                        # Equations for i=0..n-1:
                        # sum_s B[j*p+k, s] * (A[i*m+j, s] * C[k*n+i, s]) = 8
                        M = np.zeros((n, rank))
                        for i in range(n):
                            M[i, :] = A[i * m + j, :] * C[k * n + i, :]
                        b = np.ones(n) * 8
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        except:
                            sol = np.zeros(rank)
                        B[j * p + k, :] = np.round(sol)
                        
                # Update C (W')
                for k in range(p):
                    for i in range(n):
                        # Equations for j=0..m-1:
                        # sum_s C[k*n+i, s] * (A[i*m+j, s] * B[j*p+k, s]) = 8
                        M = np.zeros((m, rank))
                        for j in range(m):
                            M[j, :] = A[i * m + j, :] * B[j * p + k, :]
                        b = np.ones(m) * 8
                        try:
                            sol, _, _, _ = np.linalg.lstsq(M, b, rcond=None)
                        except:
                            sol = np.zeros(rank)
                        C[k * n + i, :] = np.round(sol)
                
                # Check residual
                max_err = 0.0
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(A[i * m + j, :], B[j * p + k, :] * C[k * n + i, :])
                            err = abs(val - 8.0)
                            if err > max_err:
                                max_err = err
                            
                if max_err < 1e-6:
                    converged = True
                    break
                
                if max_err > prev_err + 1e-6:
                    # If error increases significantly, might be stuck
                    # But we round, so it's discrete.
                    pass
                prev_err = max_err

            if converged:
                # Convert back to U, V, W (floats)
                U = A / 2.0
                V = B / 2.0
                W = C / 2.0
                
                # Final verification
                valid = True
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                            if abs(val - 1.0) > 1e-9:
                                valid = False
                                break
                        if not valid: break
                    if not valid: break
                
                if valid:
                    best_rank = rank
                    best_solution = (U, V, W)
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
