import numpy as np
import time

DIMENSIONS = [2, 2, 3]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook rank
    rank_school = n * m * p
    
    U_school = [[0] * rank_school for _ in range(n * m)]
    V_school = [[0] * rank_school for _ in range(m * p)]
    W_school = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_school[i * m + j][col] = 1
                V_school[j * p + k][col] = 1
                W_school[k * n + i][col] = 1
                col += 1
    schoolbook = {"U": U_school, "V": V_school, "W": W_school}

    # Try to find rank 10
    target_rank = 10
    start_time = time.time()
    deadline = 10.0 # 10 seconds for search
    
    # Initialize random U, V
    # Use simple random integers or floats
    # We want half-integers, so scale by 2
    # Let's work in integers for the solver variables? No, floats are easier.
    
    # Constraint: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta[j, j']
    
    # We will use an iterative approach
    # Fix U, V, solve W. Fix U, W, solve V. Fix V, W, solve U.
    
    U = np.random.rand(n * m, target_rank)
    V = np.random.rand(m * p, target_rank)
    
    best_U, best_V, best_W = None, None, None
    best_error = float('inf')
    
    while time.time() - start_time < deadline:
        # Solve for W
        # Equation: sum_r U[i*m+j, r] V[j'*p+k, r] W[k*n+i, r] = delta[j, j']
        # For fixed i, k, j, j', we have:
        # sum_r (U[i*m+j, r] * V[j'*p+k, r]) * W[k*n+i, r] = delta[j, j']
        # This is a linear system for W?
        # W has indices (k*n+i, r).
        # For a fixed k, i, the variable is W[k*n+i, :] (vector of length R).
        # The equation involves W[k*n+i, r] multiplied by coefficients depending on r.
        # So for each (k, i), we have a system:
        # Let A_{ji, r} = U[i*m+j, r] * V[j'*p+k, r]
        # We want sum_r A_{ji, r} W_{ki, r} = delta_{j, j'}
        # This is not a standard linear system for W because W is a vector and the RHS is a scalar for each (j, j').
        # Actually, for fixed k, i, we have 4 equations (j=0,1; j'=0,1) and R unknowns (W_{ki, 0} ... W_{ki, R-1}).
        # We can solve for W_{ki, :} using least squares.
        
        W = np.zeros((p * n, target_rank))
        for k in range(p):
            for i in range(n):
                # Build matrix M of shape (4, R)
                M = np.zeros((m * m, target_rank))
                for j in range(m):
                    for jp in range(m): # j'
                        idx = j * m + jp
                        M[idx, :] = U[i * m + j, :] * V[jp * p + k, :]
                # RHS vector b of shape (4,)
                b = np.zeros(m * m)
                for j in range(m):
                    b[j * m + j] = 1.0 # delta[j, j']
                
                # Solve M x = b
                # x is W[k*n+i, :]
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    W[k * n + i, :] = sol
                except:
                    pass
        
        # Now solve for V, fixing U, W
        V = np.zeros((m * p, target_rank))
        for j in range(m):
            for k in range(p):
                # Equation: sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = delta[j, j']
                # This equation must hold for all i, j'.
                # For fixed j, k, the variable is V[j*p+k, :] (vector of length R).
                # We have equations for each i, j'.
                # Number of equations: n * m = 4.
                # Build M of shape (n*m, R)
                M = np.zeros((n * m, target_rank))
                b = np.zeros(n * m)
                for i in range(n):
                    for jp in range(m): # j'
                        idx = i * m + jp
                        # Coefficient for V[j*p+k, r] is U[i*m+j, r] * W[k*n+i, r]
                        M[idx, :] = U[i * m + j, :] * W[k * n + i, :]
                        # RHS is delta[j, jp]
                        if j == jp:
                            b[idx] = 1.0
                        else:
                            b[idx] = 0.0
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    V[j * p + k, :] = sol
                except:
                    pass
        
        # Now solve for U, fixing V, W
        U = np.zeros((n * m, target_rank))
        for i in range(n):
            for j in range(m):
                # Variable U[i*m+j, :]
                # Equations for each k, j'
                M = np.zeros((p * m, target_rank))
                b = np.zeros(p * m)
                for k in range(p):
                    for jp in range(m): # j'
                        idx = k * m + jp
                        M[idx, :] = V[jp * p + k, :] * W[k * n + i, :]
                        if j == jp:
                            b[idx] = 1.0
                        else:
                            b[idx] = 0.0
                
                try:
                    sol, residuals, rank, sv = np.linalg.lstsq(M, b, rcond=None)
                    U[i * m + j, :] = sol
                except:
                    pass
        
        # Check error
        error = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.sum(U[i * m + j, :] * V[jp * p + k, :] * W[k * n + i, :])
                        target = 1.0 if j == jp else 0.0
                        error += (val - target) ** 2
        
        if error < best_error:
            best_error = error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < 1e-6:
            break
            
    # If we found a good solution, round to half-integers and verify
    if best_error < 1e-6:
        U_r = np.round(best_U * 2) / 2.0
        V_r = np.round(best_V * 2) / 2.0
        W_r = np.round(best_W * 2) / 2.0
        
        # Verify rounded solution
        err_r = 0.0
        for i in range(n):
            for k in range(p):
                for j in range(m):
                    for jp in range(m):
                        val = np.sum(U_r[i * m + j, :] * V_r[jp * p + k, :] * W_r[k * n + i, :])
                        target = 1.0 if j == jp else 0.0
                        err_r += (val - target) ** 2
        
        if err_r < 1e-6:
            # Convert to list of lists
            return {
                "U": U_r.tolist(),
                "V": V_r.tolist(),
                "W": W_r.tolist()
            }
            
    return schoolbook
# EVOLVE-BLOCK-END
