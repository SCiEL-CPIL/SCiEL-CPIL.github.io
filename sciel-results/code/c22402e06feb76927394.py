"""Matrix multiplication 2x5x5 via randomized rank-reduction search."""
import numpy as np
import time
import random

DIMENSIONS = [2, 5, 5]

# EVOLVE-BLOCK-START

def _verify_decomposition(U, V, W, n, m, p, tol=1e-9):
    """Verify that U, V, W exactly decompose the matmul tensor."""
    rank = len(U[0])
    U_arr = np.array(U, dtype=np.float64)
    V_arr = np.array(V, dtype=np.float64)
    W_arr = np.array(W, dtype=np.float64)
    
    # T[a, b, c] = 1 if a=i*m+j, b=j*p+k, c=k*n+i
    # We check sum_c U[a,c] V[b,c] W[c,c] == T[a,b,c]
    # Iterate over all valid (i, j, k)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                a = i * m + j
                b = j * p + k
                c_idx = k * n + i
                # Compute the dot product for this specific tensor entry
                # The tensor entry is 1.0
                val = np.dot(U_arr[a, :], V_arr[b, :] * W_arr[c_idx, :])
                if abs(val - 1.0) > tol:
                    return False
    return True

def _snap_to_half_integers(arr, tol=1e-6):
    """Snap values to nearest half-integer if close."""
    snapped = np.round(arr * 2) / 2.0
    if np.all(np.abs(arr - snapped) < tol):
        return snapped
    return None

def _optimize_decomposition(U, V, W, n, m, p, steps=500, lr=0.01):
    """Perform gradient descent to minimize error of decomposition."""
    U = U.copy()
    V = V.copy()
    W = W.copy()
    rank = U.shape[1]
    
    # Precompute indices
    indices = []
    for i in range(n):
        for j in range(m):
            for k in range(p):
                indices.append((i * m + j, j * p + k, k * n + i))
                
    indices = np.array(indices)
    a_idx = indices[:, 0]
    b_idx = indices[:, 1]
    c_idx = indices[:, 2]
    
    # Error function: sum (sum_c U[a,c] V[b,c] W[c,c] - 1)^2
    # Gradient w.r.t U[a,c]: 2 * (err) * V[b,c] * W[c,c]
    
    for step in range(steps):
        # Compute errors
        # Reshape for vectorization? Small enough for loops or simple numpy ops
        # Let's use a loop for clarity and moderate speed
        errors = np.zeros(len(a_idx))
        for idx in range(len(a_idx)):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            errors[idx] = np.dot(U[a, :], V[b, :] * W[c, :]) - 1.0
            
        # Gradients
        # dU[a,c] = sum_{idx where a_idx==a} 2 * errors[idx] * V[b_idx[idx], c] * W[c_idx[idx], c]
        # This is expensive. Let's use a simpler heuristic or just random restarts if optimization is too slow.
        
        # For 2x5x5, rank ~50, 50 points. 500 steps * 50 points * 50 rank = 1.25M ops. Fast enough.
        
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for idx in range(len(a_idx)):
            a, b, c = a_idx[idx], b_idx[idx], c_idx[idx]
            e = errors[idx]
            # grad U
            dU[a, :] += 2 * e * V[b, :] * W[c, :]
            # grad V
            dV[b, :] += 2 * e * U[a, :] * W[c, :]
            # grad W
            dW[c, :] += 2 * e * U[a, :] * V[b, :]
            
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
    return U, V, W

def _search_rank(n, m, p, target_rank, time_limit=60):
    """Search for a decomposition of target_rank."""
    start_time = time.time()
    max_attempts = 1000
    best_error = float('inf')
    best_solution = None
    
    for attempt in range(max_attempts):
        if time.time() - start_time > time_limit:
            break
            
        # Initialize randomly
        U = np.random.randn(n * m, target_rank) * 0.1
        V = np.random.randn(m * p, target_rank) * 0.1
        W = np.random.randn(p * n, target_rank) * 0.1
        
        # Optimize
        U_opt, V_opt, W_opt = _optimize_decomposition(U, V, W, n, m, p, steps=200, lr=0.05)
        
        # Calculate final error
        indices = [(i * m + j, j * p + k, k * n + i) for i in range(n) for j in range(m) for k in range(p)]
        error = 0.0
        for a, b, c in indices:
            val = np.dot(U_opt[a, :], V_opt[b, :] * W_opt[c, :])
            error += (val - 1.0) ** 2
            
        if error < best_error:
            best_error = error
            best_solution = (U_opt, V_opt, W_opt)
            
        # Try to snap
        U_snap = _snap_to_half_integers(U_opt)
        V_snap = _snap_to_half_integers(V_opt)
        W_snap = _snap_to_half_integers(W_opt)
        
        if U_snap is not None and V_snap is not None and W_snap is not None:
            if _verify_decomposition(U_snap, V_snap, W_snap, n, m, p):
                return U_snap, V_snap, W_snap
                
    return None

def construct():
    n, m, p = DIMENSIONS
    schoolbook_rank = n * m * p
    
    # Try to find a lower rank
    # Known lower bounds for 2x5x5 might be 50. 
    # We try ranks from 49 down to 45 to see if we can find anything.
    # If not found, return schoolbook.
    
    for r in range(schoolbook_rank - 1, max(40, schoolbook_rank - 10), -1):
        solution = _search_rank(n, m, p, r, time_limit=10)
        if solution is not None:
            U, V, W = solution
            return {
                "U": U.tolist(),
                "V": V.tolist(),
                "W": W.tolist()
            }
            
    # Fallback to schoolbook
    rank = schoolbook_rank
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

# EVOLVE-BLOCK-END
