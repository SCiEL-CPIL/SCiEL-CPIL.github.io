import numpy as np
import time

DIMENSIONS = [3, 3, 7]

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin
    
    n, m, p = DIMENSIONS
    
    # Tensor T[i, j, k] = 1 for all i,j,k
    # We want: sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] = 1
    
    best_rank = n * m * p
    best_factors = None
    
    # Schoolbook fallback
    def schoolbook():
        R = n * m * p
        U = np.zeros((n * m, R))
        V = np.zeros((m * p, R))
        W = np.zeros((p * n, R))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j, col] = 1
                    V[j * p + k, col] = 1
                    W[k * n + i, col] = 1
                    col += 1
        return U, V, W, R

    U_fb, V_fb, W_fb, R_fb = schoolbook()
    
    np.random.seed(42)
    
    def compute_error(U, V, W, R):
        """Compute sum of squared errors efficiently using vectorization"""
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0) ** 2
        return err
    
    def try_snap(U, V, W, R, tol=1e-6):
        """Try to snap to half-integers and verify exactness"""
        U_snap = np.round(U * 2) / 2
        V_snap = np.round(V * 2) / 2
        W_snap = np.round(W * 2) / 2
        
        snap_err = compute_error(U_snap, V_snap, W_snap, R)
        
        if snap_err < tol:
            return U_snap.copy(), V_snap.copy(), W_snap.copy()
        return None
    
    # Try ranks from 54 upward
    for R in range(54, 63):
        if time.time() > deadline:
            break
        
        best_err_for_R = float('inf')
        best_U_V_W_for_R = None
        
        n_restarts = 25 if R <= 55 else 15
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Multiple initialization strategies
            if restart % 4 == 0:
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
            elif restart % 4 == 1:
                U = np.random.randn(n * m, R) * 0.5
                V = np.random.randn(m * p, R) * 0.5
                W = np.random.randn(p * n, R) * 0.5
            elif restart % 4 == 2:
                U = np.ones((n * m, R)) * 0.1
                V = np.ones((m * p, R)) * 0.1
                W = np.ones((p * n, R)) * 0.1
            else:
                # Sparse initialization
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                for r in range(R):
                    idx_u = np.random.randint(0, n*m, 3)
                    idx_v = np.random.randint(0, m*p, 3)
                    idx_w = np.random.randint(0, p*n, 3)
                    U[idx_u, r] = np.random.choice([-1, 1])
                    V[idx_v, r] = np.random.choice([-1, 1])
                    W[idx_w, r] = np.random.choice([-1, 1])
            
            # ALS iterations
            max_iters = 100
            prev_err = float('inf')
            converged = False
            
            for it in range(max_iters):
                if time.time() > deadline:
                    break
                
                # Update U: for each (i,j), solve M @ U_col = ones(p)
                for i in range(n):
                    for j in range(m):
                        M = np.zeros((p, R))
                        for k in range(p):
                            M[k, :] = V[j * p + k, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(p)
                            # Solve (M M^T) x = ones, then U_col = M^T x
                            x = np.linalg.solve(MMt, np.ones(p))
                            U[i * m + j, :] = M.T @ x
                        except:
                            pass
                
                # Update V: for each (j,k), solve M @ V_col = ones(n)
                for j in range(m):
                    for k in range(p):
                        M = np.zeros((n, R))
                        for i in range(n):
                            M[i, :] = U[i * m + j, :] * W[k * n + i, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(n)
                            x = np.linalg.solve(MMt, np.ones(n))
                            V[j * p + k, :] = M.T @ x
                        except:
                            pass
                        
                # Update W: for each (k,i), solve M @ W_col = ones(m)
                for k in range(p):
                    for i in range(n):
                        M = np.zeros((m, R))
                        for j in range(m):
                            M[j, :] = U[i * m + j, :] * V[j * p + k, :]
                        try:
                            MMt = M @ M.T
                            MMt += 1e-10 * np.eye(m)
                            x = np.linalg.solve(MMt, np.ones(m))
                            W[k * n + i, :] = M.T @ x
                        except:
                            pass
                
                # Calculate error
                err = compute_error(U, V, W, R)
                
                if err < 1e-10:
                    converged = True
                    break
                
                # Allow some increase but break if significantly worse
                if err > prev_err * 1.05 and it > 5:
                    break
                prev_err = err
            
            # Check for snap-able solution
            result = try_snap(U, V, W, R)
            if result is not None:
                U_snap, V_snap, W_snap = result
                snap_err = compute_error(U_snap, V_snap, W_snap, R)
                if snap_err < best_err_for_R:
                    best_err_for_R = snap_err
                    best_U_V_W_for_R = (U_snap, V_snap, W_snap)
                
                if R == 54:
                    best_rank = R
                    best_factors = best_U_V_W_for_R
                    break
    
    if best_factors is None and best_U_V_W_for_R is not None:
        best_rank = R
        best_factors = best_U_V_W_for_R

    if best_factors:
        U, V, W = best_factors
        return {
            "U": U.tolist(),
            "V": V.tolist(),
            "W": W.tolist()
        }
    else:
        # Fallback to schoolbook
        return {
            "U": U_fb.tolist(),
            "V": V_fb.tolist(),
            "W": W_fb.tolist()
        }
