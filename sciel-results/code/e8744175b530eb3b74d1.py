import numpy as np
import time
import sys

DIMENSIONS = [2, 5, 6]

def construct():
    n, m, p = DIMENSIONS
    # Shapes: U (n*m, R), V (m*p, R), W (p*n, R)
    # Tensor T (n*m, m*p, p*n) is not explicitly built, we check error.
    
    def check_rank(R, max_iters=1000, tol=1e-8):
        # Initialize U, V, W randomly
        np.random.seed(42)
        U = np.random.randn(n * m, R)
        V = np.random.randn(m * p, R)
        W = np.random.randn(p * n, R)
        
        # Target tensor T
        # T[i*m+j, j*p+k, k*n+i] = 1 for all i,j,k
        # We want sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] == 1
        
        # Unfoldings
        # T_(1) (n*m, m*p * p*n)
        # T_(2) (m*p, n*m * p*n)
        # T_(3) (p*n, n*m * m*p)
        
        # We can use ALS:
        # Fix V, W. Update U: U = T_(1) @ (V @ W^T)^-1 ? No.
        # T_(1) = U (V @ W^T)
        # So U = T_(1) @ (V @ W^T)^+ (pseudo-inverse)
        
        # Construct T_(1)
        # T_(1)[i*m+j, (j*p+k) * (p*n) + (k*n+i)] = 1
        # This is sparse.
        # Instead of building huge matrices, we can do:
        # U_new[i*m+j, r] = sum_{j',k} T[i*m+j, j'*p+k, k*n+i] V[j'*p+k, r] W[k*n+i, r]
        # Note: T[i*m+j, j'*p+k, k*n+i] is 1 if i=i, j=j', k=k?
        # The tensor is sum_{i,j,k} e_{ij} \otimes e_{jk} \otimes e_{ki}.
        # So T[i*m+j, j'*p+k, k'*n+i'] = 1 if i=i', j=j', k=k'.
        # Wait, the indices are linked.
        # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i.
        # So for a fixed a (i,j) and c (k,i), b must be j*p+k.
        # And i in a must match i in c.
        
        # Let's implement the update step directly.
        
        best_err = float('inf')
        best_U, best_V, best_W = U, V, W
        
        for it in range(max_iters):
            # Update U
            # U[:, r] = sum_{j,k} V[j*p+k, r] W[k*n+i, r] e_{ij} ... 
            # Actually, T = sum_r u_r v_r w_r.
            # T[i*m+j, j*p+k, k*n+i] = sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r]
            # We want this to be 1.
            
            # Fix V, W. Solve for U.
            # For each r, U[:, r] is a vector in R^(n*m).
            # The equation is linear in U.
            # sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] = 1
            
            # This is a system of n*m * m * p * p*n equations? No.
            # It's one equation for each (i,j,k).
            # Total n*m*p equations.
            # Unknowns: R * n*m.
            
            # We can set up a least squares problem.
            # Let A be a matrix of size (n*m*p) x (R * n*m).
            # A[idx, col] = V[j*p+k, r] W[k*n+i, r]
            # where idx = i*m*p + j*p + k
            # col = r * n*m + i*m + j
            
            # Build A
            A = np.zeros((n * m * p, R * n * m))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        for r in range(R):
                            col = r * n * m + i * m + j
                            A[idx, col] = V[j * p + k, r] * W[k * n + i, r]
            
            b = np.ones(n * m * p)
            
            try:
                sol, residuals, rank, s = np.linalg.lstsq(A, b, rcond=None)
                U_new = sol.reshape(R, n * m).T
            except:
                break
            
            # Update V
            A_V = np.zeros((n * m * p, R * m * p))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        for r in range(R):
                            col = r * m * p + j * p + k
                            A_V[idx, col] = U_new[i * m + j, r] * W[k * n + i, r]
            try:
                sol_V, _, _, _ = np.linalg.lstsq(A_V, b, rcond=None)
                V_new = sol_V.reshape(R, m * p).T
            except:
                break
            
            # Update W
            A_W = np.zeros((n * m * p, R * p * n))
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        idx = i * m * p + j * p + k
                        for r in range(R):
                            col = r * p * n + k * n + i
                            A_W[idx, col] = U_new[i * m + j, r] * V_new[j * p + k, r]
            try:
                sol_W, _, _, _ = np.linalg.lstsq(A_W, b, rcond=None)
                W_new = sol_W.reshape(R, p * n).T
            except:
                break
            
            U, V, W = U_new, V_new, W_new
            
            # Check error
            # err = max | sum_r U[i*m+j, r] V[j*p+k, r] W[k*n+i, r] - 1 |
            err = 0
            for i in range(n):
                for j in range(m):
                    for k in range(p):
                        val = 0
                        for r in range(R):
                            val += U[i * m + j, r] * V[j * p + k, r] * W[k * n + i, r]
                        err = max(err, abs(val - 1))
            
            if err < tol:
                return U, V, W, err
                
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U, V, W
                
            if time.time() - start_time > 100:
                break
                
        return best_U, best_V, best_W, best_err

    start_time = time.time()
    
    # Try ranks from 50 up to 60
    for R in range(50, 61):
        U, V, W, err = check_rank(R, max_iters=200)
        if err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
            
    # Fallback to schoolbook
    rank = n * m * p
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
