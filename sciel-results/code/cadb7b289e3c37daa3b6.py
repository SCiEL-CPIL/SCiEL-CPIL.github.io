"""Program to find a low-rank exact decomposition for 3x3 * 3x5 matrix multiplication."""
import time
import numpy as np
from scipy.optimize import nnls
import random

DIMENSIONS = [3, 3, 5]

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s
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
    
    def check_decomposition(U, V, W, tol=1e-10):
        """Check if the decomposition is exact. Returns max error."""
        rank = U.shape[1]
        max_err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for r in range(rank):
                        s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                    err = abs(s - 1.0)
                    if err > max_err:
                        max_err = err
        return max_err
    
    def is_valid(U, V, W, tol=1e-10):
        """Check if decomposition is within tolerance."""
        return check_decomposition(U, V, W, tol) < tol
    
    def round_to_half_integers(U, V, W):
        """Round entries to nearest half-integers."""
        U_r = np.round(U * 2) / 2
        V_r = np.round(V * 2) / 2
        W_r = np.round(W * 2) / 2
        return U_r, V_r, W_r
    
    def polish_decomposition(U, V, W, max_iters=200, tol=1e-14):
        """
        High-precision polish pass using alternating least squares with tight tolerance.
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
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = x
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Solve for V
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, V.shape[1]))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = x
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Solve for W
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, W.shape[1]))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = x
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            U, V, W = U_new, V_new, W_new
            
            if check_decomposition(U, V, W, tol) < tol:
                return U, V, W
        
        return U, V, W
    
    def alternating_minimization(rank, max_iters=500, seed=42, init_scale=1.0):
        """
        Alternating least squares for tensor decomposition.
        Fix two factors, solve for the third via linear least squares.
        Returns (U, V, W, max_error) or (None, None, None, inf) on failure.
        """
        np.random.seed(seed)
        
        # Initialize with random values
        U = np.random.randn(n*m, rank) * init_scale
        V = np.random.randn(m*p, rank) * init_scale
        W = np.random.randn(p*n, rank) * init_scale
        
        prev_err = float('inf')
        best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
        best_err = float('inf')
        
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
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        U_new[i*m+j, :] = x
                    except:
                        U_new[i*m+j, :] = U[i*m+j, :]
            
            # Solve for V: fix U and W
            V_new = np.zeros_like(V)
            for j in range(m):
                for k in range(p):
                    A = np.zeros((n, rank))
                    for i in range(n):
                        A[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    b = np.ones(n)
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        V_new[j*p+k, :] = x
                    except:
                        V_new[j*p+k, :] = V[j*p+k, :]
            
            # Solve for W: fix U and V
            W_new = np.zeros_like(W)
            for k in range(p):
                for i in range(n):
                    A = np.zeros((m, rank))
                    for j in range(m):
                        A[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    b = np.ones(m)
                    try:
                        x, residuals, rank_ls, sv = np.linalg.lstsq(A, b, rcond=None)
                        W_new[k*n+i, :] = x
                    except:
                        W_new[k*n+i, :] = W[k*n+i, :]
            
            U, V, W = U_new, V_new, W_new
            
            # Track best solution
            err = check_decomposition(U, V, W, tol=float('inf'))
            if err < best_err:
                best_err = err
                best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
            # Check convergence
            if err < 1e-10:
                break
            
            if abs(err - prev_err) < 1e-14:
                break
            prev_err = err
        
        # If we got close, try to polish
        if best_err < 1e-2:
            U_p, V_p, W_p = polish_decomposition(best_U, best_V, W_p, max_iters=300, tol=1e-14)
            err_p = check_decomposition(U_p, V_p, W_p, tol=float('inf'))
            if err_p < best_err:
                best_U, best_V, best_W, best_err = U_p, V_p, W_p, err_p
        
        # Try rounding if close enough
        if best_err < 0.5:
            U_r, V_r, W_r = round_to_half_integers(best_U, best_V, best_W)
            err_r = check_decomposition(U_r, V_r, W_r, tol=float('inf'))
            if err_r < 1e-8:
                return U_r, V_r, W_r, err_r
        
        if best_err < 1e-10:
            return best_U, best_V, best_W, best_err
        
        return None, None, None, float('inf')
    
    # Try to find low-rank decompositions
    best_U, best_V, best_W = None, None, None
    best_rank = n * m * p  # 45
    best_err = float('inf')
    
    # Try ranks from lower to higher
    # Theoretical lower bound: for 3x3x3, rank is 23. For 3x3x5, we need at least 23 + some
    # Let's try from 25 up to 44
    min_rank_to_try = 25
    max_rank_to_try = min(44, best_rank)
    
    for rank in range(min_rank_to_try, max_rank_to_try + 1):
        if time.time() > deadline - margin:
            break
        
        # Try multiple random seeds with different strategies
        found = False
        
        # Strategy 1: Standard random initialization with various scales
        for seed in range(100):
            if time.time() > deadline - margin:
                break
            
            scale = 0.5 + 0.5 * (seed % 3)  # Vary scale
            U_try, V_try, W_try, err = alternating_minimization(rank, max_iters=300, seed=seed, init_scale=scale)
            if U_try is not None and err < 1e-8:
                if rank < best_rank or (rank == best_rank and err < best_err):
                    best_U, best_V, best_W = U_try, V_try, W_try
                    best_rank = rank
                    best_err = err
                found = True
                break
        
        if found:
            break
        
        # Strategy 2: If standard didn't work, try with schoolbook-like initialization
        if time.time() < deadline - margin:
            U_sb, V_sb, W_sb = schoolbook_decomposition()
            U_init = U_sb[:, :rank].copy()
            V_init = V_sb[:, :rank].copy()
            W_init = W_sb[:, :rank].copy()
            
            # Add perturbations to escape local minima
            for seed in range(50):
                if time.time() > deadline - margin:
                    break
                
                np.random.seed(seed)
                U_pert = U_init + np.random.randn(n*m, rank) * 0.2
                V_pert = V_init + np.random.randn(m*p, rank) * 0.2
                W_pert = W_init + np.random.randn(p*n, rank) * 0.2
                
                # Run alternating minimization from this start
                U_cur, V_cur, W_cur = U_pert.copy(), V_pert.copy(), W_pert.copy()
                prev_err = float('inf')
                
                for it in range(300):
                    if time.time() > deadline - margin:
                        break
                    
                    # Solve for U
                    U_new = np.zeros_like(U_cur)
                    for i in range(n):
                        for j in range(m):
                            A = np.zeros((p, rank))
                            for k in range(p):
                                A[k, :] = V_cur[j*p+k, :] * W_cur[k*n+i, :]
                            b = np.ones(p)
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                U_new[i*m+j, :] = x
                            except:
                                U_new[i*m+j, :] = U_cur[i*m+j, :]
                    
                    # Solve for V
                    V_new = np.zeros_like(V_cur)
                    for j in range(m):
                        for k in range(p):
                            A = np.zeros((n, rank))
                            for i in range(n):
                                A[i, :] = U_cur[i*m+j, :] * W_cur[k*n+i, :]
                            b = np.ones(n)
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                V_new[j*p+k, :] = x
                            except:
                                V_new[j*p+k, :] = V_cur[j*p+k, :]
                    
                    # Solve for W
                    W_new = np.zeros_like(W_cur)
                    for k in range(p):
                        for i in range(n):
                            A = np.zeros((m, rank))
                            for j in range(m):
                                A[j, :] = U_cur[i*m+j, :] * V_cur[j*p+k, :]
                            b = np.ones(m)
                            try:
                                x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
                                W_new[k*n+i, :] = x
                            except:
                                W_new[k*n+i, :] = W_cur[k*n+i, :]
                    
                    U_cur, V_cur, W_cur = U_new, V_new, W_new
                    
                    err = check_decomposition(U_cur, V_cur, W_cur, tol=float('inf'))
                    if err < 1e-10:
                        break
                    if abs(err - prev_err) < 1e-14:
                        break
                    prev_err = err
                
                err_final = check_decomposition(U_cur, V_cur, W_cur, tol=float('inf'))
                if err_final < 1e-8:
                    # Polish
                    U_p, V_p, W_p = polish_decomposition(U_cur, V_cur, W_cur, max_iters=200, tol=1e-14)
                    err_p = check_decomposition(U_p, V_p, W_p, tol=float('inf'))
                    if err_p < 1e-8:
                        if rank < best_rank or (rank == best_rank and err_p < best_err):
                            best_U, best_V, best_W = U_p, V_p, W_p
                            best_rank = rank
                            best_err = err_p
                        found = True
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
