"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # 5s margin before 150s limit
    margin = 10.0
    
    n, m, p = 3, 3, 5
    
    def schoolbook_decomposition():
        """Return rank-45 schoolbook decomposition."""
        rank = n * m * p  # 45
        U = np.zeros((n*m, rank))
        V = np.zeros((m*p, rank))
        W = np.zeros((p*n, rank))
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i*m+j, col] = 1.0
                    V[j*p+k, col] = 1.0
                    W[k*n+i, col] = 1.0
                    col += 1
        return U, V, W
    
    def check_decomposition(U, V, W, tol=1e-6):
        """Check if the decomposition is exact. Returns max error."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    max_err = max(max_err, abs(s - 1.0))
        return max_err
    
    def check_valid(U, V, W, tol=1e-6):
        """Check if decomposition is valid (error < tol)."""
        return check_decomposition(U, V, W, tol) < tol
    
    def polish_decomposition(U, V, W, max_iters=100, tol=1e-12):
        """
        High-precision polish pass using alternating least squares with tight tolerance.
        """
        rank = U.shape[1]
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            if check_decomposition(U, V, W, tol=tol) < tol:
                return U, V, W, True
        
        return U, V, W, False
    
    def alternating_minimization(rank, max_iters=300, tol=1e-10, seed=42):
        """
        Alternating least squares for tensor decomposition.
        Fix two factors, solve for the third via linear least squares.
        """
        np.random.seed(seed)
        
        # Initialize with structured random
        U = np.random.randn(n*m, rank)
        V = np.random.randn(m*p, rank)
        W = np.random.randn(p*n, rank)
        
        # Normalize columns to avoid scale issues
        for r in range(rank):
            nu = np.linalg.norm(U[:, r])
            nv = np.linalg.norm(V[:, r])
            nw = np.linalg.norm(W[:, r])
            if nu > 1e-10 and nv > 1e-10 and nw > 1e-10:
                # Scale so that product of norms is ~1
                scale = (nu * nv * nw) ** (1.0/3.0)
                U[:, r] /= scale
                V[:, r] /= scale
                W[:, r] /= scale
        
        prev_err = float('inf')
        stagnation_count = 0
        
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U: fix V and W
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, rank))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            err = check_decomposition(U, V, W, tol=float('inf'))
            
            if err < tol:
                # Polish
                U_p, V_p, W_p, success = polish_decomposition(U, V, W, max_iters=50, tol=1e-12)
                if success:
                    return U_p, V_p, W_p
                else:
                    return U, V, W
            
            if abs(err - prev_err) < 1e-12:
                stagnation_count += 1
                if stagnation_count > 10:
                    break
            else:
                stagnation_count = 0
            prev_err = err
        
        # Final check
        if check_decomposition(U, V, W, tol=1e-6) < 1e-6:
            U_p, V_p, W_p, success = polish_decomposition(U, V, W, max_iters=100, tol=1e-12)
            if success:
                return U_p, V_p, W_p
            elif check_decomposition(U, V, W, tol=1e-4) < 1e-4:
                return U, V, W
        
        return None, None, None
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    
    # Try ranks from 35 up to 44
    for rank in range(35, min(45, best_rank)):
        if time.time() > deadline - margin:
            break
        
        # Try multiple random seeds
        found = False
        num_seeds = 50 if rank >= 40 else 30
        for seed in range(num_seeds):
            if time.time() > deadline - margin:
                break
            
            U_try, V_try, W_try = alternating_minimization(rank, max_iters=200, seed=seed)
            if U_try is not None and check_valid(U_try, V_try, W_try, tol=1e-6):
                best_U, best_V, best_W = U_try, V_try, W_try
                best_rank = rank
                found = True
                break
        
        if found:
            break
    
    # If we found something, convert to list of lists
    if best_U is not None:
        return {
            "U": best_U.tolist(),
            "V": best_V.tolist(),
            "W": best_W.tolist()
        }
    
    # Fallback: Schoolbook
    U, V, W = schoolbook_decomposition()
    return {
        "U": U.tolist(),
        "V": V.tolist(),
        "W": W.tolist()
    }

# EVOLVE-BLOCK-END
