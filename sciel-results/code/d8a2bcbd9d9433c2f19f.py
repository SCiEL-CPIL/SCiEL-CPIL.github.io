"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
DIMENSIONS = [2, 3, 5]

# EVOLVE-BLOCK-START
import numpy as np
import time
from itertools import product

def _verify(U, V, W, n, m, p, tol=1e-6):
    """Verify tensor decomposition: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should be 1 for all i,j,k."""
    U = np.array(U, dtype=float)
    V = np.array(V, dtype=float)
    W = np.array(W, dtype=float)
    r = U.shape[1]
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for c in range(r):
                    s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                if abs(s - 1.0) > tol:
                    return False, s
    return True, None

def _try_rank(r, n, m, p, max_iter=200, seed=42):
    """Try to find a decomposition of rank r using alternating least squares."""
    np.random.seed(seed)
    U = np.random.randn(n*m, r)
    V = np.random.randn(m*p, r)
    W = np.random.randn(p*n, r)
    
    for it in range(max_iter):
        # Update U: for each (i,j), solve least squares
        for i in range(n):
            for j in range(m):
                # U[i*m+j, c] should satisfy: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1 for all k
                # This is: U[i*m+j,:] @ (V[j*p+0:p,:].T * W[0:n,:].T) ... not quite
                # Let's build the system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                # Define M_k = V[j*p+k,:] * W[k*n+i,:] elementwise
                # Then U[i*m+j,:] @ M_k = 1 for all k
                # Stack: A[i*m+j,:] @ [M_0; M_1; ...] = [1; 1; ...]
                A_mat = np.zeros((p, r))
                for k in range(p):
                    A_mat[k, :] = V[j*p+k, :] * W[k*n+i, :]
                # Solve A_mat.T @ u = ones(p) for u = U[i*m+j,:]
                # Least squares: u = pinv(A_mat.T) @ ones(p)
                try:
                    U[i*m+j, :] = np.linalg.lstsq(A_mat.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: for each (j,k), similar
        for j in range(m):
            for k in range(p):
                A_mat = np.zeros((n, r))
                for i in range(n):
                    A_mat[i, :] = U[i*m+j, :] * W[k*n+i, :]
                try:
                    V[j*p+k, :] = np.linalg.lstsq(A_mat.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: for each (k,i), similar
        for k in range(p):
            for i in range(n):
                A_mat = np.zeros((m, r))
                for j in range(m):
                    A_mat[j, :] = U[i*m+j, :] * V[j*p+k, :]
                try:
                    W[k*n+i, :] = np.linalg.lstsq(A_mat.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for c in range(r):
                        s += U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c]
                    err += abs(s - 1.0)
        if err < 1e-8:
            return U, V, W
    
    return None

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin
    
    # Start with schoolbook
    rank_school = n * m * p
    U = [[0] * rank_school for _ in range(n * m)]
    V = [[0] * rank_school for _ in range(m * p)]
    W = [[0] * rank_school for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i*m+j][col] = 1
                V[j*p+k][col] = 1
                W[k*n+i][col] = 1
                col += 1
    
    best = (rank_school, U, V, W)
    
    # Try lower ranks
    for r in range(rank_school - 1, 14, -1):
        if time.time() > deadline:
            break
        result = _try_rank(r, n, m, p, max_iter=100, seed=42)
        if result is not None:
            U_r, V_r, W_r = result
            # Verify
            ok, _ = _verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), n, m, p)
            if ok:
                best = (r, U_r.tolist(), V_r.tolist(), W_r.tolist())
                break
    
    r, U, V, W = best
    return {"U": U, "V": V, "W": W}
# EVOLVE-BLOCK-END
