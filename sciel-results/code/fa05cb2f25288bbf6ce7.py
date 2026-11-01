"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

DIMENSIONS = [2, 3, 3]

# EVOLVE-BLOCK-START
def _build_schoolbook(n, m, p):
    """Build the schoolbook rank n*m*p decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1
                V[j * p + k, col] = 1
                W[k * n + i, col] = 1
                col += 1
    return U, V, W

def _compute_residual(U, V, W, n, m, p, target_rank):
    """Compute the residual tensor T[i,j,k] - sum_r U[i*m+j,r]*V[j*p+k,r]*W[k*n+i,r]."""
    T = np.ones((n, m, p))
    residual = np.zeros((n, m, p))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                residual[i, j, k] = T[i, j, k] - val
    return residual

def _als_solve(n, m, p, target_rank, max_iters=500, tol=1e-10, deadline=None):
    """
    Use Alternating Least Squares to find a rank-`target_rank` decomposition.
    Returns (U, V, W, converged) where converged is True if residual < tol.
    """
    T = np.ones((n, m, p))
    
    # Random initialization with multiple restarts handled externally
    U = np.random.randn(n * m, target_rank)
    V = np.random.randn(m * p, target_rank)
    W = np.random.randn(p * n, target_rank)
    
    # Normalize columns to avoid scale drift
    for r in range(target_rank):
        nu = np.linalg.norm(U[:, r])
        nv = np.linalg.norm(V[:, r])
        nw = np.linalg.norm(W[:, r])
        if nu < 1e-12 or nv < 1e-12 or nw < 1e-12:
            # Reinitialize this column
            U[:, r] = np.random.randn(n * m)
            V[:, r] = np.random.randn(m * p)
            W[:, r] = np.random.randn(p * n)
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
        # Balance norms: make each have norm ~1
        scale = (nu * nv * nw) ** (1.0 / 3.0)
        U[:, r] /= (nu / scale)
        V[:, r] /= (nv / scale)
        W[:, r] /= (nw / scale)
    
    for it in range(max_iters):
        if deadline is not None and time.time() > deadline:
            return U, V, W, False
        
        # --- Update U ---
        # For each (i, j): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let A[k, r] = V[j*p+k, r] * W[k*n+i, r]  (shape p x target_rank)
        # Then T[i,j,:] = U[i*m+j,:] @ A.T
        # Solve: U[i*m+j,:] = T[i,j,:] @ pinv(A.T)
        for i in range(n):
            for j in range(m):
                A = np.zeros((p, target_rank))
                for k in range(p):
                    A[k, :] = V[j * p + k, :] * W[k * n + i, :]
                # A.T has shape (target_rank, p)
                # We want x @ A.T = T[i,j,:] where x has shape (target_rank,)
                # This is: A.T.T @ x.T = T[i,j,:].T  => A @ x.T = T[i,j,:].T
                # Actually: x @ A.T = b means A.T.T @ x.T = b.T, i.e., A @ x.T = b.T
                # So x.T = lstsq(A, b.T)
                try:
                    x = np.linalg.lstsq(A, T[i, j, :].reshape(-1, 1), rcond=None)[0].flatten()
                    U[i * m + j, :] = x
                except:
                    pass
        
        # --- Update V ---
        # For each (j, k): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let B[i, r] = U[i*m+j, r] * W[k*n+i, r]  (shape n x target_rank)
        # Then T[:,j,k] = V[j*p+k,:] @ B.T
        # Solve: V[j*p+k,:] = T[:,j,k] @ pinv(B.T)
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, target_rank))
                for i in range(n):
                    B[i, :] = U[i * m + j, :] * W[k * n + i, :]
                try:
                    x = np.linalg.lstsq(B, T[:, j, k].reshape(-1, 1), rcond=None)[0].flatten()
                    V[j * p + k, :] = x
                except:
                    pass
        
        # --- Update W ---
        # For each (k, i): T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
        # Let C[j, r] = U[i*m+j, r] * V[j*p+k, r]  (shape m x target_rank)
        # Then T[i,:,k] = W[k*n+i,:] @ C.T
        # Solve: W[k*n+i,:] = T[i,:,k] @ pinv(C.T)
        for k in range(p):
            for i in range(n):
                C = np.zeros((m, target_rank))
                for j in range(m):
                    C[j, :] = U[i * m + j, :] * V[j * p + k, :]
                try:
                    x = np.linalg.lstsq(C, T[i, :, k].reshape(-1, 1), rcond=None)[0].flatten()
                    W[k * n + i, :] = x
                except:
                    pass
        
        # Check convergence
        residual = _compute_residual(U, V, W, n, m, p, target_rank)
        if np.max(np.abs(residual)) < tol:
            return U, V, W, True
    
    return U, V, W, False

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 130  # Safety margin before 150s kill
    
    # Build schoolbook as fallback
    U_school, V_school, W_school = _build_schoolbook(n, m, p)
    best_U, best_V, best_W = U_school, V_school, W_school
    best_rank = n * m * p
    
    # Try ranks from 14 upward (R(2,3,3) is known to be 14)
    for target_rank in range(14, best_rank + 1):
        if time.time() > deadline:
            break
        
        found = False
        # More restarts for lower ranks
        max_restarts = 80 if target_rank <= 15 else 40 if target_rank <= 16 else 20
        
        for restart in range(max_restarts):
            if time.time() > deadline:
                break
            
            np.random.seed(restart * 7919 + target_rank * 104729)
            
            U, V, W, converged = _als_solve(n, m, p, target_rank, max_iters=300, tol=1e-9, deadline=deadline)
            
            if converged:
                # Verify with higher precision
                residual = _compute_residual(U, V, W, n, m, p, target_rank)
                if np.max(np.abs(residual)) < 1e-8:
                    found = True
                    best_U, best_V, best_W = U, V, W
                    best_rank = target_rank
                    break
        
        if found:
            break
    
    return {"U": best_U.tolist(), "V": best_V.tolist(), "W": best_W.tolist()}
# EVOLVE-BLOCK-END
