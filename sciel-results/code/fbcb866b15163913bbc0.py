"""
Matrix multiplication 5x5x5: rank-minimal tensor decomposition search.
"""

import time
import numpy as np

DIMENSIONS = [5, 5, 5]
DEADLINE = 130.0  # safety margin before 150s kill

def _schoolbook():
    """Return the standard rank-125 schoolbook decomposition."""
    n, m, p = DIMENSIONS
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, col] = 1.0
                V[j * p + k, col] = 1.0
                W[k * n + i, col] = 1.0
                col += 1
    return U, V, W

def _verify(U, V, W, n=5, m=5, p=5, tol=1e-9):
    """
    Verify that U, V, W decompose the matrix multiplication tensor.
    T[i,j,k] = sum_c U[i*m+j,c] * V[j*p+k,c] * W[k*n+i,c] should equal 1 for all i,j,k.
    """
    U = np.asarray(U, dtype=np.float64)
    V = np.asarray(V, dtype=np.float64)
    W = np.asarray(W, dtype=np.float64)
    
    if U.shape[0] != n * m or V.shape[0] != m * p or W.shape[0] != p * n:
        return False
    if U.shape[1] != V.shape[1] or V.shape[1] != W.shape[1]:
        return False
    
    rank = U.shape[1]
    if rank == 0:
        return False
    
    # Check tensor identity: T[i,j,k] = sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] = 1
    for i in range(n):
        for j in range(m):
            for k in range(p):
                val = np.dot(U[i * m + j, :], V[j * p + k, :] * W[k * n + i, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_int(arr, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2
    if np.max(np.abs(arr - snapped)) < tol:
        return snapped
    return arr

def construct():
    """
    Construct a rank-minimal exact tensor decomposition for 5x5 matrix multiplication.
    Returns {'U': U, 'V': V, 'W': W} with integer/half-integer entries.
    """
    start_time = time.time()
    
    n, m, p = DIMENSIONS
    nm, mp, pn = n * m, m * p, p * n
    
    # Start with schoolbook
    U, V, W = _schoolbook()
    if not _verify(U, V, W):
        raise ValueError("Schoolbook verification failed")
    
    best_rank = U.shape[1]
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Try to reduce rank by searching for lower-rank decompositions
    # Strategy: Try ranks from best_rank-1 downward, attempting to find valid decompositions
    
    def _try_rank(r):
        """Try to find a rank-r decomposition using iterative refinement."""
        if time.time() - start_time > DEADLINE - 5:
            return None
        
        # Initialize with random perturbation of schoolbook truncated to r columns
        # or use a structured initialization
        np.random.seed(42)
        
        # Try multiple random initializations
        for attempt in range(50):
            if time.time() - start_time > DEADLINE - 5:
                break
            
            # Initialize: take first r columns of schoolbook and perturb
            U_init = np.zeros((nm, r), dtype=np.float64)
            V_init = np.zeros((mp, r), dtype=np.float64)
            W_init = np.zeros((pn, r), dtype=np.float64)
            
            # Use a subset of schoolbook terms as initialization
            idx = np.random.choice(nm * mp * pn, size=min(r, nm * mp * pn), replace=False)
            for c, idx_val in enumerate(idx):
                i = idx_val // (mp * pn)
                rem = idx_val % (mp * pn)
                j = rem // pn
                k = rem % pn
                U_init[i * m + j, c] = 1.0
                V_init[j * p + k, c] = 1.0
                W_init[k * n + i, c] = 1.0
            
            # Add small random noise to break symmetry
            U_init += np.random.randn(nm, r) * 0.1
            V_init += np.random.randn(mp, r) * 0.1
            W_init += np.random.randn(pn, r) * 0.1
            
            # Iteratively refine to satisfy the tensor identity
            # Objective: minimize sum_{i,j,k} (sum_c U[i*m+j,c]*V[j*p+k,c]*W[k*n+i,c] - 1)^2
            # Use gradient descent
            
            U_cur = U_init.copy()
            V_cur = V_init.copy()
            W_cur = W_init.copy()
            
            lr = 0.01
            for it in range(200):
                if time.time() - start_time > DEADLINE - 5:
                    break
                
                # Compute residuals
                residuals = np.zeros((n, m, p), dtype=np.float64)
                for i in range(n):
                    for j in range(m):
                        for k in range(p):
                            residuals[i, j, k] = np.dot(
                                U_cur[i * m + j, :], 
                                V_cur[j * p + k, :] * W_cur[k * n + i, :]
                            ) - 1.0
                
                err = np.sum(residuals ** 2)
                if err < 1e-10:
                    # Check if it's valid
                    if _verify(U_cur, V_cur, W_cur):
                        # Snap to half-integers
                        U_snap = _snap_to_half_int(U_cur)
                        V_snap = _snap_to_half_int(V_cur)
                        W_snap = _snap_to_half_int(W_cur)
                        if _verify(U_snap, V_snap, W_snap):
                            return U_snap, V_snap, W_snap
                    break
                
                # Compute gradients
                # d/dU[i*m+j,c] = sum_{i',j',k'} residuals[i',j',k'] * V[j'*p+k',c] * W[k'*n+i',c] * delta(i'=i, j'=j)
                # This is expensive; use a simplified update
                
                # For efficiency, only update a few elements per step
                idx_u = np.random.randint(0, nm, size=min(10, nm))
                idx_v = np.random.randint(0, mp, size=min(10, mp))
                idx_w = np.random.randint(0, pn, size=min(10, pn))
                
                for ii in idx_u:
                    i = ii // m
                    j = ii % m
                    for c in range(r):
                        grad = 0.0
                        for k in range(p):
                            grad += residuals[i, j, k] * V_cur[j * p + k, c] * W_cur[k * n + i, c]
                        U_cur[ii, c] -= lr * grad
                
                for ii in idx_v:
                    j = ii // p
                    k = ii % p
                    for c in range(r):
                        grad = 0.0
                        for i in range(n):
                            grad += residuals[i, j, k] * U_cur[i * m + j, c] * W_cur[k * n + i, c]
                        V_cur[ii, c] -= lr * grad
                
                for ii in idx_w:
                    k = ii // n
                    i = ii % n
                    for c in range(r):
                        grad = 0.0
                        for j in range(m):
                            grad += residuals[i, j, k] * U_cur[i * m + j, c] * V_cur[j * p + k, c]
                        W_cur[ii, c] -= lr * grad
                
                # Decay learning rate
                lr *= 0.99
        
        return None
    
    # Try ranks from best_rank-1 downward, with a time budget
    for r in range(best_rank - 1, max(50, best_rank - 20), -1):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        result = _try_rank(r)
        if result is not None:
            U_new, V_new, W_new = result
            if _verify(U_new, V_new, W_new):
                best_rank = r
                best_U, best_V, best_W = U_new, V_new, W_new
                break  # Found a lower rank, stop searching
    
    # Return the best found
    U_list = best_U.tolist()
    V_list = best_V.tolist()
    W_list = best_W.tolist()
    
    return {"U": U_list, "V": V_list, "W": W_list}
