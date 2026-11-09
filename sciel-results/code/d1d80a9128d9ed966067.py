"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    margin = 5.0
    
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
        """Check if the decomposition is exact."""
        rank = U.shape[1]
        max_err = 0.0
        # Vectorized check
        # U: (nm, R), V: (mp, R), W: (pn, R)
        # We need T[i, j, k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] == 1
        # Reshape U to (n, m, R), V to (m, p, R), W to (p, n, R)
        U_resh = U.reshape(n, m, rank)
        V_resh = V.reshape(m, p, rank)
        W_resh = W.reshape(p, n, rank)
        
        # Product: U[i, j, r] * V[j, k, r] * W[k, i, r]
        # Sum over r
        # We can do this by iterating over i, j, k or using einsum
        # T[i, j, k] = sum_r U[i, j, r] V[j, k, r] W[k, i, r]
        # einsum('ijr,jkr,kir->ijk', U_resh, V_resh, W_resh)
        T = np.einsum('ijr,jkr,kir->ijk', U_resh, V_resh, W_resh, optimize=True)
        max_err = np.max(np.abs(T - 1.0))
        return max_err < tol
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def polish_decomposition(U, V, W, max_iters=20, tol=1e-10):
        """
        High-precision polish pass using alternating least squares.
        """
        for it in range(max_iters):
            if time.time() > deadline - margin:
                break
            
            # Solve for U
            U_new = np.zeros_like(U)
            for i in range(n):
                for j in range(m):
                    A = np.zeros((p, U.shape[1]))
                    for k in range(p):
                        A[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    b = np.ones(p)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, V.shape[1]))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, W.shape[1]))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            if check_decomposition(U, V, W, tol=tol):
                return U, V, W
        
        return U, V, W
    
    def alternating_minimization(rank, max_iters=300, tol=1e-8, seed=42):
        """
        Alternating least squares for tensor decomposition.
        """
        np.random.seed(seed)
        
        # Initialize
        U = np.random.randn(n*m, rank) * 0.5
        V = np.random.randn(m*p, rank) * 0.5
        W = np.random.randn(p*n, rank) * 0.5
        
        prev_err = float('inf')
        
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
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    U_new[i*m+j, :] = x
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    V_new[j*p+k, :] = x
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                    W_new[k*n+i, :] = x
            
            U, V, W = U_new, V_new, W_new
            
            err = check_decomposition(U, V, W, tol=float('inf'))
            if err < tol:
                # Try rounding
                U_r, V_r, W_r = round_to_half_integers(U, V, W)
                if check_decomposition(U_r, V_r, W_r, tol=tol):
                    return U_r, V_r, W_r
                else:
                    return U, V, W
            
            if abs(err - prev_err) < 1e-10:
                break
            prev_err = err
        
        # Final check with polish
        if check_decomposition(U, V, W, tol=1e-2):
            U_p, V_p, W_p = polish_decomposition(U, V, W, max_iters=50, tol=1e-10)
            if check_decomposition(U_p, V_p, W_p, tol=1e-8):
                U_r, V_r, W_r = round_to_half_integers(U_p, V_p, W_p)
                if check_decomposition(U_r, V_r, W_r, tol=1e-4):
                    return U_r, V_r, W_r
                else:
                    return U_p, V_p, W_p
        return None, None, None
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    
    # Try ranks from 37 up to 44
    for rank in range(37, 45):
        if time.time() > deadline - margin:
            break
        
        found = False
        for seed in range(20):
            if time.time() > deadline - margin:
                break
            
            U_try, V_try, W_try = alternating_minimization(rank, max_iters=200, seed=seed)
            if U_try is not None and check_decomposition(U_try, V_try, W_try, tol=1e-4):
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
