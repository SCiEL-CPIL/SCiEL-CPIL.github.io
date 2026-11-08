"""Program to find a rank-minimal tensor decomposition for 3x3 by 3x6 matrix multiplication."""
import time
import random
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 6]
START_TIME = time.time()
TIME_LIMIT = 115  # Safety margin from 120s limit

# EVOLVE-BLOCK-START
def _check_time():
    """Return True if we're still within time limits."""
    return time.time() - START_TIME < TIME_LIMIT

def _compute_decomposition_error(U, V, W, n, m, p):
    """
    Compute the error of a tensor decomposition.
    
    The decomposition must satisfy for all i in [0,n), j in [0,m), k in [0,p):
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1
    
    Returns the max absolute error.
    """
    rank = U.shape[1]
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = 0.0
                for c in range(rank):
                    val += U[i*m + j, c] * V[j*p + k, c] * W[k*n + i, c]
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err

def _als_step(U, V, W, n, m, p, max_iter=200, tol=1e-10):
    """
    Alternating Least Squares to refine a tensor decomposition.
    
    We want to minimize the error where:
        sum_c U[i*m+j, c] * V[j*p+k, c] * W[k*n+i, c] = 1 for all i,j,k
    
    This is a bilinear problem in each factor when the other two are fixed.
    """
    rank = U.shape[1]
    U = U.copy()
    V = V.copy()
    W = W.copy()
    
    for iteration in range(max_iter):
        if not _check_time():
            break
        
        # Update U: For each (i,j), we need sum_c U[i*m+j,c] * (V[j*p+k,c] * W[k*n+i,c]) = 1 for all k
        # This is: U[i*m+j,:] @ diag(V[j*p+k,c]*W[k*n+i,c] for all k) = 1
        # Actually, for fixed V,W, the constraint for U[i*m+j,:] is:
        # For each k: sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
        # Let A_{k,c} = V[j*p+k,c] * W[k*n+i,c]
        # Then U[i*m+j,:] @ A^T = 1 (vector of ones)
        
        for i in range(n):
            for j in range(m):
                # Build the system: for each k, sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
                A = np.zeros((p, rank))
                for k in range(p):
                    for c in range(rank):
                        A[k, c] = V[j*p + k, c] * W[k*n + i, c]
                # Solve A^T @ U[i*m+j,:] = ones(p)
                # i.e., U[i*m+j,:] = (A^T)^{-1} @ ones(p) if possible
                # Use least squares
                try:
                    U[i*m + j, :] = np.linalg.lstsq(A.T, np.ones(p), rcond=None)[0]
                except:
                    pass
        
        # Update V: For each j, we need sum_c V[j*p+k,c] * (U[i*m+j,c] * W[k*n+i,c]) = 1 for all i,k
        # For fixed U,W, the constraint for V[j*p+k,:] is:
        # For each i: sum_c V[j*p+k,c] * U[i*m+j,c] * W[k*n+i,c] = 1
        # Let B_{i,c} = U[i*m+j,c] * W[k*n+i,c]
        # Then V[j*p+k,:] @ B^T = 1 (vector of ones over i)
        
        for j in range(m):
            for k in range(p):
                B = np.zeros((n, rank))
                for i in range(n):
                    for c in range(rank):
                        B[i, c] = U[i*m + j, c] * W[k*n + i, c]
                try:
                    V[j*p + k, :] = np.linalg.lstsq(B.T, np.ones(n), rcond=None)[0]
                except:
                    pass
        
        # Update W: For each (i,k), we need sum_c W[k*n+i,c] * (U[i*m+j,c] * V[j*p+k,c]) = 1 for all j
        # For fixed U,V, the constraint for W[k*n+i,:] is:
        # For each j: sum_c W[k*n+i,c] * U[i*m+j,c] * V[j*p+k,c] = 1
        # Let C_{j,c} = U[i*m+j,c] * V[j*p+k,c]
        # Then W[k*n+i,:] @ C^T = 1 (vector of ones over j)
        
        for i in range(n):
            for k in range(p):
                C = np.zeros((m, rank))
                for j in range(m):
                    for c in range(rank):
                        C[j, c] = U[i*m + j, c] * V[j*p + k, c]
                try:
                    W[k*n + i, :] = np.linalg.lstsq(C.T, np.ones(m), rcond=None)[0]
                except:
                    pass
        
        # Check convergence
        err = _compute_decomposition_error(U, V, W, n, m, p)
        if err < tol:
            break
    
    return U, V, W, err

def _try_rank_als(n, m, p, target_rank, num_restarts=5):
    """
    Try to find a decomposition of given rank using ALS.
    Returns (U, V, W, error) if successful, else None.
    """
    best_U, best_V, best_W, best_err = None, None, None, float('inf')
    
    for restart in range(num_restarts):
        if not _check_time():
            break
        
        # Random initialization
        U_init = np.random.randn(n * m, target_rank)
        V_init = np.random.randn(m * p, target_rank)
        W_init = np.random.randn(p * n, target_rank)
        
        # Scale to help convergence
        U_init /= np.linalg.norm(U_init) + 1e-10
        V_init /= np.linalg.norm(V_init) + 1e-10
        W_init /= np.linalg.norm(W_init) + 1e-10
        
        U_out, V_out, W_out, err = _als_step(U_init, V_init, W_init, n, m, p, max_iter=100, tol=1e-12)
        
        if err < best_err:
            best_err = err
            best_U, best_V, best_W = U_out, V_out, W_out
        
        if best_err < 1e-10:
            break
    
    if best_err < 1e-8:
        return best_U, best_V, best_W, best_err
    return None

def construct():
    """
    Construct a tensor decomposition for multiplying a 3x3 matrix by a 3x6 matrix.
    
    Returns:
        dict: {'U': (9, rank), 'V': (18, rank), 'W': (18, rank)}
              U[i*m + j, c], V[j*p + k, c], W[k*n + i, c]
              where n=3, m=3, p=6
    """
    n, m, p = DIMENSIONS
    
    # Schoolbook decomposition of rank n*m*p = 54
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, column] = 1
                V[j * p + k, column] = 1
                W[k * n + i, column] = 1
                column += 1
    
    best_rank = rank
    best_U, best_V, best_W = U, V, W
    
    # Try to find a better decomposition using ALS
    # Start from lower ranks and work up
    for target_rank in range(30, 54):
        if not _check_time():
            break
        
        result = _try_rank_als(n, m, p, target_rank, num_restarts=3)
        if result is not None:
            U_cand, V_cand, W_cand, err = result
            best_rank = target_rank
            best_U, best_V, best_W = U_cand, V_cand, W_cand
            # If we found a valid decomposition, try even lower ranks
            continue
        else:
            # If we can't find one at this rank, try higher ranks
            # But first check if we're running out of time
            if not _check_time():
                break
    
    # Verify the best solution
    final_err = _compute_decomposition_error(best_U, best_V, best_W, n, m, p)
    if final_err > 1e-6:
        # Fall back to schoolbook
        best_U, best_V, best_W = U, V, W
        best_rank = rank
    
    # Convert to lists for the return interface
    return {
        "U": best_U.tolist(),
        "V": best_V.tolist(),
        "W": best_W.tolist()
    }

# EVOLVE-BLOCK-END
