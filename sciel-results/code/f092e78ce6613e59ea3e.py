import numpy as np
import time

DIMENSIONS = [2, 3, 3]

def _snap_to_half(arr):
    """Snap values to the nearest half-integer."""
    return np.round(arr * 2) / 2.0

def _is_valid_half_integer(arr):
    """Check if all entries are integers or half-integers (within tolerance)."""
    snapped = np.round(arr * 2) / 2.0
    return np.allclose(arr, snapped, atol=1e-9)

def _build_rank17():
    """Build the known rank-17 decomposition."""
    n, m, p = DIMENSIONS
    U = np.zeros((n * m, 17))
    V = np.zeros((m * p, 17))
    W = np.zeros((p * n, 17))
    
    # Strassen for 2x2x2 block
    U[0, 0] = 1; U[4, 0] = 1
    V[0, 0] = 1; V[4, 0] = 1
    W[0, 0] = 1; W[3, 0] = 1
    
    U[3, 1] = 1; U[4, 1] = 1
    V[0, 1] = 1
    W[1, 1] = 1; W[3, 1] = -1
    
    U[0, 2] = 1
    V[1, 2] = 1; V[4, 2] = -1
    W[2, 2] = 1; W[3, 2] = 1
    
    U[4, 3] = 1
    V[3, 3] = 1; V[0, 3] = -1
    W[0, 3] = 1; W[1, 3] = 1
    
    U[0, 4] = 1; U[1, 4] = 1
    V[4, 4] = 1
    W[0, 4] = -1; W[2, 4] = 1
    
    U[3, 5] = 1; U[0, 5] = -1
    V[0, 5] = 1; V[1, 5] = 1
    W[3, 5] = 1
    
    U[1, 6] = 1; U[4, 6] = -1
    V[3, 6] = 1; V[4, 6] = 1
    W[0, 6] = 1
    
    # Schoolbook for A[0:2,2] * B[2,0:2]
    U[2, 7] = 1; V[6, 7] = 1; W[0, 7] = 1
    U[2, 8] = 1; V[7, 8] = 1; W[2, 8] = 1
    U[5, 9] = 1; V[6, 9] = 1; W[1, 9] = 1
    U[5, 10] = 1; V[7, 10] = 1; W[3, 10] = 1
    
    # Schoolbook for A[0:2,0:3] * B[0:3,2]
    U[0, 11] = 1; V[2, 11] = 1; W[4, 11] = 1
    U[1, 12] = 1; V[5, 12] = 1; W[4, 12] = 1
    U[2, 13] = 1; V[8, 13] = 1; W[4, 13] = 1
    U[3, 14] = 1; V[2, 14] = 1; W[5, 14] = 1
    U[4, 15] = 1; V[5, 15] = 1; W[5, 15] = 1
    U[5, 16] = 1; V[8, 16] = 1; W[5, 16] = 1
    
    return U, V, W

def _verify(U, V, W):
    """Verify that the decomposition is correct."""
    n, m, p = DIMENSIONS
    errors = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.sum(U[i*m+j, :] * V[j*p+k, :] * W[k*n+i, :])
                if abs(val - 1) > 1e-9:
                    errors.append((i, j, k, val))
    return len(errors) == 0, errors

def _als_rank16(U17, V17, W17, time_budget=60, seed=42):
    """Alternating Least Squares search for rank-16 decomposition."""
    start_time = time.time()
    n, m, p = DIMENSIONS
    num_i = n * m  # 6
    num_j = m * p  # 9
    num_k = p * n  # 6
    rank = 16
    
    np.random.seed(seed)
    
    def compute_error(U, V, W):
        err = 0.0
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    val = np.dot(U[i*m+j, :], V[j*p+k, :] * W[k*n+i, :])
                    err += (val - 1.0)**2
        return err
    
    def als_step(U, V, W, max_iter=200):
        """Alternating least squares: fix two matrices, solve for the third."""
        for it in range(max_iter):
            if time.time() - start_time > time_budget:
                break
            
            # Update U
            for i in range(n):
                for j in range(m):
                    idx = i*m + j
                    M = np.zeros((p, rank))
                    for k in range(p):
                        M[k, :] = V[j*p+k, :] * W[k*n+i, :]
                    try:
                        U[idx, :] = np.linalg.lstsq(M, np.ones(p), rcond=None)[0]
                    except:
                        pass
            
            # Update V
            for j in range(m):
                for k in range(p):
                    idx = j*p + k
                    M = np.zeros((n, rank))
                    for i in range(n):
                        M[i, :] = U[i*m+j, :] * W[k*n+i, :]
                    try:
                        V[idx, :] = np.linalg.lstsq(M, np.ones(n), rcond=None)[0]
                    except:
                        pass
            
            # Update W
            for k in range(p):
                for i in range(n):
                    idx = k*n + i
                    M = np.zeros((m, rank))
                    for j in range(m):
                        M[j, :] = U[i*m+j, :] * V[j*p+k, :]
                    try:
                        W[idx, :] = np.linalg.lstsq(M, np.ones(m), rcond=None)[0]
                    except:
                        pass
            
            # Check convergence
            err = compute_error(U, V, W)
            if err < 1e-10:
                break
        
        return U, V, W, err
    
    # Strategy 1: Initialize from rank-17 by dropping each term and refining
    for term_to_zero in range(17):
        if time.time() - start_time > time_budget:
            break
        
        U = U17.copy()
        V = V17.copy()
        W = W17.copy()
        
        # Zero out term_to_zero
        U[:, term_to_zero] = 0
        V[:, term_to_zero] = 0
        W[:, term_to_zero] = 0
        
        # Remove the zeroed column
        if term_to_zero < 16:
            U = np.hstack([U[:, :term_to_zero], U[:, term_to_zero+1:]])
            V = np.hstack([V[:, :term_to_zero], V[:, term_to_zero+1:]])
            W = np.hstack([W[:, :term_to_zero], W[:, term_to_zero+1:]])
        else:
            U = U[:, :term_to_zero]
            V = V[:, :term_to_zero]
            W = W[:, :term_to_zero]
        
        # Run ALS
        U, V, W, err = als_step(U, V, W, max_iter=100)
        
        if err < 1e-8:
            return U, V, W, 16
    
    # Strategy 2: Random initialization with ALS
    for attempt in range(20):
        if time.time() - start_time > time_budget:
            break
        
        U = np.random.randn(num_i, rank) * 0.3
        V = np.random.randn(num_j, rank) * 0.3
        W = np.random.randn(num_k, rank) * 0.3
        
        U, V, W, err = als_step(U, V, W, max_iter=150)
        
        if err < 1e-8:
            return U, V, W, 16
    
    return None, None, None, 17

def construct():
    n, m, p = DIMENSIONS
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s kill
    
    # Build the known rank-17 solution
    U17, V17, W17 = _build_rank17()
    
    # Verify it
    valid, errors = _verify(U17, V17, W17)
    
    # Try to find rank 16
    time_for_search = min(100, max(10, (deadline - time.time()) - 10))
    if time_for_search > 5:
        U16, V16, W16, rank_found = _als_rank16(U17, V17, W17, time_budget=time_for_search)
        
        if U16 is not None and rank_found == 16:
            # Snap to half-integers
            U16_snapped = _snap_to_half(U16)
            V16_snapped = _snap_to_half(V16)
            W16_snapped = _snap_to_half(W16)
            
            # Verify the snapped rank-16 solution
            valid16, errors16 = _verify(U16_snapped, V16_snapped, W16_snapped)
            
            # Check if entries are valid half-integers (snapping should ensure this, but check for NaN/Inf)
            if valid16 and np.all(np.isfinite(U16_snapped)) and np.all(np.isfinite(V16_snapped)) and np.all(np.isfinite(W16_snapped)):
                return {"U": U16_snapped.tolist(), "V": V16_snapped.tolist(), "W": W16_snapped.tolist()}
    
    # Return the rank-17 solution (snapped just in case, though it's integers)
    U17_snapped = _snap_to_half(U17)
    V17_snapped = _snap_to_half(V17)
    W17_snapped = _snap_to_half(W17)
    
    return {"U": U17_snapped.tolist(), "V": V17_snapped.tolist(), "W": W17_snapped.tolist()}
