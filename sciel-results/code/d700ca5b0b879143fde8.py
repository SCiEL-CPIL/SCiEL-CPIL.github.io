"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from fractions import Fraction

DIMENSIONS = [4, 4, 7]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s kill
    
    def make_schoolbook():
        rank = n * m * p
        U = [[0] * rank for _ in range(n * m)]
        V = [[0] * rank for _ in range(m * p)]
        W = [[0] * rank for _ in range(p * n)]
        col = 0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    U[i * m + j][col] = 1
                    V[j * p + k][col] = 1
                    W[k * n + i][col] = 1
                    col += 1
        return {"U": U, "V": V, "W": W}
    
    best_result = make_schoolbook()
    best_rank = n * m * p
    
    # Precompute index arrays for efficient verification
    i_idx = np.repeat(np.arange(n), m * p)
    j_idx = np.tile(np.repeat(np.arange(m), p), n)
    k_idx = np.tile(np.arange(p), n * m)
    u_idx = i_idx * m + j_idx
    v_idx = j_idx * p + k_idx
    w_idx = k_idx * n + i_idx
    
    def verify(U, V, W, r, tol=1e-9):
        """Check if U, V, W form a valid tensor decomposition. Returns max error."""
        Ua = np.array(U, dtype=float)
        Va = np.array(V, dtype=float)
        Wa = np.array(W, dtype=float)
        
        u_vals = Ua[u_idx, :]  # (n*m*p, r)
        v_vals = Va[v_idx, :]  # (n*m*p, r)
        w_vals = Wa[w_idx, :]  # (n*m*p, r)
        
        prod = u_vals * v_vals * w_vals
        result = prod.sum(axis=1)
        
        return np.max(np.abs(result - 1.0))
    
    def als_refine(U, V, W, r, max_iter=50, tol=1e-10):
        """Alternating Least Squares refinement for tensor decomposition.
        
        For each factor, fix the other two and solve a linear least squares problem.
        """
        Ua = np.array(U, dtype=float).copy()
        Va = np.array(V, dtype=float).copy()
        Wa = np.array(W, dtype=float).copy()
        
        # Build the tensor T[i,j,k] = 1 for all (i,j,k)
        T = np.ones((n, m, p))
        
        for it in range(max_iter):
            if time.time() > deadline - 5:
                break
            
            # Update U: for each (i,j), solve min ||T[i,j,:] - (V[j,:]*W[i,:])^T @ U[i,j,:]||
            # T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
            # Fix V and W, solve for each row of U
            
            for i in range(n):
                for j in range(m):
                    # Build matrix A where A[k,c] = V[j*p+k,c] * W[k*n+i,c]
                    A = np.zeros((p, r))
                    for k in range(p):
                        A[k, :] = Va[j*p+k, :] * Wa[k*n+i, :]
                    # b[k] = T[i,j,k] = 1
                    b = np.ones(p)
                    # Solve A^T A x = A^T b
                    try:
                        x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = Ua[i*m+j, :]
                    Ua[i*m+j, :] = x
            
            # Update V: for each (j,k), solve min ||T[:,j,k] - (U[:,j]*W[k,:])^T @ V[j,k,:]||
            for j in range(m):
                for k in range(p):
                    # Build matrix A where A[i,c] = U[i*m+j,c] * W[k*n+i,c]
                    A = np.zeros((n, r))
                    for i in range(n):
                        A[i, :] = Ua[i*m+j, :] * Wa[k*n+i, :]
                    # b[i] = T[i,j,k] = 1
                    b = np.ones(n)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = Va[j*p+k, :]
                    Va[j*p+k, :] = x
            
            # Update W: for each (k,i), solve min ||T[:,:,k] - (U[:,k]*V[:,k])^T @ W[k,i,:]||
            for k in range(p):
                for i in range(n):
                    # Build matrix A where A[j,c] = U[i*m+j,c] * V[j*p+k,c]
                    A = np.zeros((m, r))
                    for j in range(m):
                        A[j, :] = Ua[i*m+j, :] * Va[j*p+k, :]
                    # b[j] = T[i,j,k] = 1
                    b = np.ones(m)
                    try:
                        x, _, _, _ = np.linalg.lstsq(A.T, b, rcond=None)
                    except np.linalg.LinAlgError:
                        x = Wa[k*n+i, :]
                    Wa[k*n+i, :] = x
            
            # Check convergence
            err = verify(Ua.tolist(), Va.tolist(), Wa.tolist(), r)
            if err < tol:
                break
        
        return Ua, Va, Wa
    
    def round_to_half_int(arr):
        """Round array entries to nearest half-integer."""
        return np.round(2.0 * arr) / 2.0
    
    def try_rank(r, num_starts=10):
        """Try to find a decomposition of rank r using multiple starts."""
        for start in range(num_starts):
            if time.time() > deadline - 10:
                break
            
            np.random.seed(start * 10000 + r * 100)
            
            # Structured initialization: use Kronecker-like patterns
            U = np.random.randn(n * m, r) * 0.5
            V = np.random.randn(m * p, r) * 0.5
            W = np.random.randn(p * n, r) * 0.5
            
            # Add some structure: initialize with partial schoolbook patterns
            if r >= n * m:
                for i in range(n):
                    for j in range(m):
                        col = (i * m + j) % r
                        U[i*m+j, col] += 1.0
                        V[j*p:(j+1)*p, col] += 0.1
                        W[0:p*n, col] += 0.1
            
            # ALS refinement
            U_ref, V_ref, W_ref = als_refine(U, V, W, r, max_iter=40)
            
            # Round to half-integers and verify
            U_rounded = round_to_half_int(U_ref)
            V_rounded = round_to_half_int(V_ref)
            W_rounded = round_to_half_int(W_ref)
            
            err_rounded = verify(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), r)
            err_unrounded = verify(U_ref.tolist(), V_ref.tolist(), W_ref.tolist(), r)
            
            if err_rounded < 1e-6:
                return U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), r
            
            # If unrounded is very close, try finer rounding
            if err_unrounded < 1e-4:
                # Try rounding with different strategies
                for scale in [2, 4, 8]:
                    U_r = np.round(scale * U_ref) / scale
                    V_r = np.round(scale * V_ref) / scale
                    W_r = np.round(scale * W_ref) / scale
                    err = verify(U_r.tolist(), V_r.tolist(), W_r.tolist(), r)
                    if err < 1e-6:
                        return U_r.tolist(), V_r.tolist(), W_r.tolist(), r
        
        return None
    
    # Try ranks from a reasonable lower bound upward
    # Lower bound for matrix multiplication rank: max(n*m, m*p, p*n) = max(16, 28, 28) = 28
    # But practical lower bounds are higher. Start from 80 and go up.
    lower_bound = 80
    for r in range(lower_bound, best_rank + 1):
        if time.time() > deadline - 15:
            break
        
        result = try_rank(r, num_starts=8)
        if result is not None:
            U, V, W, r_found = result
            best_rank = r_found
            best_result = {"U": U, "V": V, "W": W}
            break
    
    if time.time() > deadline:
        return best_result
    
    return best_result
# EVOLVE-BLOCK-END
