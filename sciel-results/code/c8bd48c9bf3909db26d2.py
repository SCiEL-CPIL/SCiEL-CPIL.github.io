"""Tensor decomposition search for 2x5x6 matrix multiplication."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 6]
START_TIME = time.time()
TIME_LIMIT = 115.0  # safety margin before 120s

def _elapsed():
    return time.time() - START_TIME

def _schoolbook():
    """Return the standard rank-60 decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank))
    V = np.zeros((m * p, rank))
    W = np.zeros((p * n, rank))
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, tol=1e-9):
    """Check if U,V,W form a valid tensor decomposition of matmul 2x5x6."""
    n, m, p = DIMENSIONS
    U = np.asarray(U, dtype=float)
    V = np.asarray(V, dtype=float)
    W = np.asarray(W, dtype=float)
    rank = U.shape[1]
    # Build the tensor: T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c]
    # Should equal 1 if i,j,k are valid indices for the matmul tensor
    # The matmul tensor T[i,j,k] = 1 for all valid (i,j,k), 0 otherwise
    # Actually for matrix multiplication C = A*B where A is n x m, B is m x p:
    # C[i,k] = sum_j A[i,j] B[j,k]
    # The tensor T[i,j,k] has T[i,j,k] = 1 for all i in [0,n), j in [0,m), k in [0,p)
    # This is the standard convention where the tensor entries are all 1.
    # Wait, that's not right. The tensor for matrix multiplication has:
    # T[i, j, k] = 1 if the product involves A[i,j]*B[j,k]*C[k,i]... 
    # Actually the standard tensor: T[i,j,k] = 1 for all valid indices means
    # sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] should equal 1 for all (i,j,k).
    max_err = 0.0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :])
                val = np.dot(val, W[k * n + i, :])
                err = abs(val - 1.0)
                if err > max_err:
                    max_err = err
    return max_err < tol

def _is_half_integer(x, tol=1e-9):
    """Check if x is within tol of a half-integer (n/2 for integer n)."""
    # Half-integer means x*2 is an integer
    x2 = x * 2.0
    return abs(x2 - round(x2)) < tol

def _check_half_integer(U, V, W, tol=1e-9):
    """Check all entries are half-integers."""
    for M in [U, V, W]:
        M = np.asarray(M, dtype=float)
        M2 = M * 2.0
        if np.max(np.abs(M2 - np.round(M2))) > tol:
            return False
    return True

def _round_to_half_int(M, tol=1e-6):
    """Round a matrix to nearest half-integers if close enough."""
    M = np.asarray(M, dtype=float)
    M2 = M * 2.0
    M2r = np.round(M2)
    if np.max(np.abs(M2 - M2r)) > tol:
        return None
    return M2r / 2.0

def _construct_rank_attempt(target_rank, max_iter=5000, seed=None):
    """
    Try to find a rank-`target_rank` decomposition using Newton-type iterations.
    Initialize randomly and optimize.
    """
    n, m, p = DIMENSIONS
    rng = np.random.RandomState(seed)
    
    # The tensor T[i,j,k] = 1 for all i,j,k (the matmul tensor in this convention)
    # We need: for all (i,j,k), sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] = 1
    
    # Parameterization: U is (n*m) x r, V is (m*p) x r, W is (p*n) x r
    # We have r rank-1 terms.
    
    U = rng.randn(n * m, target_rank) * 0.5
    V = rng.randn(m * p, target_rank) * 0.5
    W = rng.randn(p * n, target_rank) * 0.5
    
    # We want to minimize sum over (i,j,k) of (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
    # This is a non-convex problem. Use gradient-based approach.
    
    # Precompute index mappings
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    
    # For efficiency, precompute which (i*m+j), (j*p+k), (k*n+i) correspond to each index
    idx_U = np.array([i * m + j for i, j, k in indices])
    idx_V = np.array([j * p + k for i, j, k in indices])
    idx_W = np.array([k * n + i for i, j, k in indices])
    
    lr = 0.01
    for iteration in range(max_iter):
        if _elapsed() > TIME_LIMIT - 5:
            return None
        
        # Compute residuals
        # r_c = sum_c U[idx_U,c] * V[idx_V,c] * W[idx_W,c] for each index
        # This is: for each c, outer product of U[idx_U,c], V[idx_V,c], W[idx_W,c] summed over c
        # Efficiently: for each c, compute element-wise product and sum
        residual = np.zeros(len(indices))
        for c in range(target_rank):
            residual += U[idx_U, c] * V[idx_V, c] * W[idx_W, c]
        
        # Residual error: residual - 1
        err = residual - 1.0
        loss = np.sum(err ** 2)
        
        if loss < 1e-14:
            return U, V, W
        
        # Compute gradients
        # dL/dU[idx_U,c] = 2 * sum over indices where idx_U matches... 
        # Actually: dL/dU[a,c] where a = i*m+j
        # L = sum_idx (sum_c' U[idx_U,idx',c'] * V[idx_V,idx',c'] * W[idx_W,idx',c'] - 1)^2
        # dL/dU[a,c] = 2 * sum_{idx: idx_U[idx]=a} err[idx] * V[idx_V[idx],c] * W[idx_W[idx],c]
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(target_rank):
            # For each index, the contribution to grad
            # grad_U[idx_U,c] += 2 * err * V[idx_V,c] * W[idx_W,c]
            # But multiple indices can map to same idx_U value
            # Use np.add.at or bincount
            contrib_U = 2.0 * err * V[idx_V, c] * W[idx_W, c]
            contrib_V = 2.0 * err * U[idx_U, c] * W[idx_W, c]
            contrib_W = 2.0 * err * U[idx_U, c] * V[idx_V, c]
            
            # Sum contributions by index
            np.add.at(grad_U, (idx_U, c), contrib_U)
            np.add.at(grad_V, (idx_V, c), contrib_V)
            np.add.at(grad_W, (idx_W, c), contrib_W)
        
        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        # Adaptive learning rate
        if loss < 1e-10:
            lr *= 0.5
        elif iteration % 500 == 0 and iteration > 0:
            # Check if we're making progress
            pass
    
    # Check if we converged
    residual = np.zeros(len(indices))
    for c in range(target_rank):
        residual += U[idx_U, c] * V[idx_V, c] * W[idx_W, c]
    err = np.max(np.abs(residual - 1.0))
    if err < 1e-9:
        return U, V, W
    return None

def _try_polish(U, V, W, max_iter=2000):
    """Polish an approximate decomposition to high precision."""
    n, m, p = DIMENSIONS
    rank = U.shape[1]
    
    indices = [(i, j, k) for i in range(n) for j in range(m) for k in range(p)]
    idx_U = np.array([i * m + j for i, j, k in indices])
    idx_V = np.array([j * p + k for i, j, k in indices])
    idx_W = np.array([k * n + i for i, j, k in indices])
    
    lr = 0.001
    prev_loss = np.inf
    for iteration in range(max_iter):
        if _elapsed() > TIME_LIMIT - 3:
            break
        
        residual = np.zeros(len(indices))
        for c in range(rank):
            residual += U[idx_U, c] * V[idx_V, c] * W[idx_W, c]
        err = residual - 1.0
        loss = np.sum(err ** 2)
        
        if loss < 1e-20:
            break
        if abs(prev_loss - loss) < 1e-16:
            lr *= 0.5
        prev_loss = loss
        
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        for c in range(rank):
            contrib_U = 2.0 * err * V[idx_V, c] * W[idx_W, c]
            contrib_V = 2.0 * err * U[idx_U, c] * W[idx_W, c]
            contrib_W = 2.0 * err * U[idx_U, c] * V[idx_V, c]
            np.add.at(grad_U, (idx_U, c), contrib_U)
            np.add.at(grad_V, (idx_V, c), contrib_V)
            np.add.at(grad_W, (idx_W, c), contrib_W)
        
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
    
    return U, V, W

def _to_output_matrices(U, V, W):
    """Convert numpy arrays to lists of half-integers if possible, else floats."""
    U_out = _round_to_half_int(U, tol=1e-6)
    V_out = _round_to_half_int(V, tol=1e-6)
    W_out = _round_to_half_int(W, tol=1e-6)
    
    if U_out is not None and V_out is not None and W_out is not None:
        return U_out.tolist(), V_out.tolist(), W_out.tolist()
    
    # If not half-integers, return as-is (floats)
    return U.tolist(), V.tolist(), W.tolist()

def construct():
    n, m, p = DIMENSIONS
    
    # Start with schoolbook as fallback
    U_sb, V_sb, W_sb = _schoolbook()
    best_U, best_V, best_W = U_sb, V_sb, W_sb
    best_rank = n * m * p  # 60
    
    # Try to find better ranks using iterative optimization
    # Theoretical lower bound for 2xm x mxp: at least 2*max(m,p) or so
    # Known: rank(2xm x mxm) can be less than 2m^2
    # For 2x5x6, let's try ranks from 30 upward
    
    # Actually, let me think about what's achievable.
    # The tensor has 60 entries all equal to 1.
    # A rank-r decomposition means we can write the all-ones tensor on {0,1}x{0..4}x{0..5}
    # as a sum of r rank-1 tensors.
    
    # This is equivalent to: find r vectors u_c in R^10, v_c in R^30, w_c in R^12
    # such that for all (i,j,k): sum_c u_c[i*m+j] * v_c[j*p+k] * w_c[k*n+i] = 1
    
    # Let's try various ranks
    for target_rank in range(20, best_rank):
        if _elapsed() > TIME_LIMIT - 15:
            break
        
        # Try multiple random initializations
        found = False
        for attempt in range(3):
            if _elapsed() > TIME_LIMIT - 15:
                break
            
            seed = 42 + target_rank * 100 + attempt
            result = _construct_rank_attempt(target_rank, max_iter=2000, seed=seed)
            if result is not None:
                U_try, V_try, W_try = result
                # Verify
                if _verify(U_try, V_try, W_try, tol=1e-8):
                    # Try to polish
                    U_pol, V_pol, W_pol = _try_polish(U_try, V_try, W_try, max_iter=1000)
                    if _verify(U_pol, V_pol, W_pol, tol=1e-8):
                        best_U, best_V, best_W = U_pol, V_pol, W_pol
                        best_rank = target_rank
                        found = True
                        break
                    elif _verify(U_try, V_try, W_try, tol=1e-8):
                        best_U, best_V, best_W = U_try, V_try, W_try
                        best_rank = target_rank
                        found = True
                        break
        if found:
            # Found a better rank, continue searching for even better
            continue
    
    # Try to round to half-integers
    U_out, V_out, W_out = _to_output_matrices(best_U, best_V, best_W)
    
    # Final verification
    if not _verify(U_out, V_out, W_out, tol=1e-8):
        # Fall back to schoolbook
        U_out, V_out, W_out = U_sb.tolist(), V_sb.tolist(), W_sb.tolist()
    
    return {"U": U_out, "V": V_out, "W": W_out}

# EVOLVE-BLOCK-END

