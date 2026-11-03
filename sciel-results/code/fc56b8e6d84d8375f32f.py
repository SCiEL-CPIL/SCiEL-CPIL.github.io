import numpy as np
import time

def construct():
    n, m, p = 3, 5, 5
    t0 = time.time()
    deadline = t0 + 110.0
    
    # Schoolbook rank
    rank0 = n * m * p
    U0 = np.zeros((n*m, rank0))
    V0 = np.zeros((m*p, rank0))
    W0 = np.zeros((p*n, rank0))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U0[i*m+j, col] = 1
                V0[j*p+k, col] = 1
                W0[k*n+i, col] = 1
                col += 1
                
    best_rank = rank0
    best_U = U0.copy()
    best_V = V0.copy()
    best_W = W0.copy()
    
    # Target tensor identity: delta_{j1, j2}
    # We want sum_r W[k*n+i, r] U[i*m+j1, r] V[j2*p+k, r] == 1 if j1==j2 else 0
    
    # Strategy: 
    # 1. Try to reduce rank from 75 downwards.
    # 2. Use a "structured" initialization: take the schoolbook solution and 
    #    apply random linear combinations of columns to break symmetry, 
    #    then project to lower rank.
    
    # We will try ranks: 74, 73, ..., down to 50.
    # For each rank, we will do a few restarts.
    
    # Helper to verify
    def _verify(U, V, W, n, m, p, tol=1e-6):
        r = U.shape[1]
        U = np.asarray(U, dtype=float)
        V = np.asarray(V, dtype=float)
        W = np.asarray(W, dtype=float)
        for i in range(n):
            for k in range(p):
                for j1 in range(m):
                    for j2 in range(m):
                        s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                        target = 1.0 if j1 == j2 else 0.0
                        if abs(s - target) > tol:
                            return False
        return True

    # Helper to perform ALS
    def _als(U, V, W, n, m, p, max_iters=100, tol=1e-8):
        r = U.shape[1]
        prev_err = float('inf')
        for it in range(max_iters):
            if time.time() > deadline - 10.0:
                break
                
            # Update W
            for k in range(p):
                for i in range(n):
                    # A shape (m*m, r)
                    A = np.zeros((m*m, r))
                    for j1 in range(m):
                        for j2 in range(m):
                            A[j1*m+j2, :] = U[i*m+j1, :] * V[j2*p+k, :]
                    b = np.zeros(m*m)
                    for j in range(m):
                        b[j*m+j] = 1.0
                    w, *_ = np.linalg.lstsq(A, b, rcond=None)
                    W[k*n+i, :] = w
                    
            # Update U
            for i in range(n):
                for j1 in range(m):
                    A = np.zeros((p*m, r))
                    b = np.zeros(p*m)
                    for k in range(p):
                        for j2 in range(m):
                            A[k*m+j2, :] = W[k*n+i, :] * V[j2*p+k, :]
                            b[k*m+j2] = 1.0 if j1 == j2 else 0.0
                    u, *_ = np.linalg.lstsq(A, b, rcond=None)
                    U[i*m+j1, :] = u
                    
            # Update V
            for j2 in range(m):
                for k in range(p):
                    A = np.zeros((n*m, r))
                    b = np.zeros(n*m)
                    for i in range(n):
                        for j1 in range(m):
                            A[i*m+j1, :] = W[k*n+i, :] * U[i*m+j1, :]
                            b[i*m+j1] = 1.0 if j1 == j2 else 0.0
                    v, *_ = np.linalg.lstsq(A, b, rcond=None)
                    V[j2*p+k, :] = v
                    
            # Calculate error
            err = 0.0
            for i in range(n):
                for k in range(p):
                    for j1 in range(m):
                        for j2 in range(m):
                            s = np.dot(W[k*n+i, :], U[i*m+j1, :] * V[j2*p+k, :])
                            target = 1.0 if j1 == j2 else 0.0
                            err += (s - target)**2
                            
            if err < tol:
                break
            if err > prev_err * 1.01 and it > 10:
                break
            prev_err = err
            
        return U, V, W, err

    # Try ranks
    # We start from 74 and go down.
    # We use the schoolbook solution as a base and "merge" columns.
    # A simple way to get a rank r solution from rank r+1 is to pick r random columns 
    # from the r+1 solution and re-solve. But this is brittle.
    # Better: Start with random init for rank r, but bias towards the schoolbook structure.
    
    for r in range(rank0 - 1, 45, -1):
        if time.time() > deadline - 20.0:
            break
            
        # Number of restarts: more for lower ranks
        n_restarts = 3 if r > 65 else 5 if r > 55 else 8
        
        found = False
        for restart in range(n_restarts):
            if time.time() > deadline - 15.0:
                break
                
            # Initialization
            # Use a mix of random and schoolbook-like structure
            np.random.seed(42 + r * 100 + restart)
            
            # Create a random projection of the schoolbook solution
            # This helps break the symmetry and find non-trivial solutions
            # We want U, V, W of shape (n*m, r), (m*p, r), (p*n, r)
            
            # Simple random init
            U = np.random.randn(n*m, r) * 0.5
            V = np.random.randn(m*p, r) * 0.5
            W = np.random.randn(p*n, r) * 0.5
            
            # Run ALS
            U, V, W, err = _als(U, V, W, n, m, p, max_iters=150)
            
            # Check if converged to a valid solution
            if err < 1e-8:
                # Snap to half-integers? The problem says integer/half-integer entries.
                # But the evaluator checks exact decomposition. 
                # If the ALS converges to a valid solution, we can return it.
                # The prompt says "integer/half-integer entries". 
                # Standard ALS might produce floats. 
                # We should try to round to nearest 0.5 and verify.
                U_r = np.round(U * 2) / 2
                V_r = np.round(V * 2) / 2
                W_r = np.round(W * 2) / 2
                
                if _verify(U_r, V_r, W_r, n, m, p):
                    best_rank = r
                    best_U = U_r.copy()
                    best_V = V_r.copy()
                    best_W = W_r.copy()
                    found = True
                    break
                else:
                    # If rounding failed, check if the unrounded solution is valid enough?
                    # The task requires integer/half-integer. 
                    # If ALS converges to a valid solution with floats, 
                    # it might be a rational solution with small denominator.
                    # We can try to find the rational approximation.
                    # For now, if rounding fails, we discard this restart.
                    pass
        
        # If found a solution for rank r, we can stop trying this rank 
        # and move to r-1.
        # If not found, we continue to next lower rank.
        
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }
