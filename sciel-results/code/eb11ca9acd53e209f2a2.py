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
    
    # Try ranks from 49 upward (exploiting block structure potential)
    for R in range(49, 63):
        if time.time() > deadline:
            break
        
        best_err_for_R = float('inf')
        best_U_V_W_for_R = None
        
        # Allow more restarts for lower ranks as they are faster
        n_restarts = 50 if R <= 50 else (30 if R <= 55 else 15)
        for restart in range(n_restarts):
            if time.time() > deadline:
                break
            
            # Multiple initialization strategies
            # Strategy 0: Random
            if restart % 4 == 0:
                U = np.random.randn(n * m, R)
                V = np.random.randn(m * p, R)
                W = np.random.randn(p * n, R)
            elif restart % 4 == 1:
                # Strategy 1: Block-structured initialization
                # Pre-fill the 3x1 block (k=6) with schoolbook rank 3
                U = np.zeros((n * m, R))
                V = np.zeros((m * p, R))
                W = np.zeros((p * n, R))
                
                # Block 3: k=6 (3x1 mult, rank 3)
                # C_i6 = sum_j A_ij B_j6
                # Terms for i=0..2, j=0..2? No, rank 3 means 3 terms.
                # Schoolbook for 3x1: 3 terms (one for each i? or j?)
                # C_i6 = A_i0 B_06 + A_i1 B_16 + A_i2 B_26
                # Term r: U[i, r]=1, V[j, r]=1, W[6, i]=1?
                # We need sum_r U[i*m+j, r] V[j*p+6, r] W[6*n+i, r] = 1 for all i,j?
                # No, the tensor T[i,j,6] must be 1 for all i,j.
                # Wait, T[i,j,k]=1 for all i,j,k.
                # So for k=6, T[i,j,6]=1 for all i,j.
                # We need sum_r U[i*m+j, r] V[j*p+6, r] W[6*n+i, r] = 1.
                # If we use 3 terms, can we do it?
                # Let r=0,1,2.
                # U[i*m+j, r] V[j*p+6, r] W[6*n+i, r]
                # This looks like it depends on j.
                # If we set V[j*p+6, r] = 1 for all j?
                # Then sum_r U[i*m+j, r] W[6*n+i, r] = 1.
                # This must hold for all i,j.
                # If U[i*m+j, r] does not depend on j, say U[i*m+j, r] = U_i_r.
                # Then sum_r U_i_r W[6*n+i, r] = 1.
                # We can set U_i_r = delta_{ir}, W[6*n+i, r] = 1.
                # Then U[i*m+j, r] = delta_{ir}.
                # V[j*p+6, r] = 1.
                # Check: sum_r delta_{ir} * 1 * 1 = 1 if i=r? No, sum over r.
                # For fixed i, sum_r delta_{ir} = 1.
                # So this works for rank 3.
                for r in range(3):
                    for i in range(n):
                        U[i * m + 0, r] = 1 if i == r else 0 # m=3, so index is i*3+j. We need it for all j?
                        # If U depends on j, it's hard.
                        # Let's just use random for now, but bias the block.
                        pass
                
                # Simpler Block Init:
                # Random for all, but force the k=6 rows of V and k=6 cols of W to be non-zero?
                U = np.random.randn(n * m, R) * 0.1
                V = np.random.randn(m * p, R) * 0.1
                W = np.random.randn(p * n, R) * 0.1
                
                # Boost the k=6 block
                V[:, 6*n:(6+1)*n] *= 5.0 # V index is j*p+k. k=6 is last 3 cols? No, j*p+k.
                # V shape (m*p, R). Rows are j*p+k.
                # k=6 corresponds to rows j*7+6 for j=0,1,2.
                V[6, :] *= 5.0
                V[13, :] *= 5.0
                V[20, :] *= 5.0
                
                W[6*n:6*n+n, :] *= 5.0 # W index k*n+i. k=6 is last 3 rows.
                
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
