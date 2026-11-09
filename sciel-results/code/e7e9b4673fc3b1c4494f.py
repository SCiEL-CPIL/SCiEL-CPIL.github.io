"""Find a rank-minimal exact tensor decomposition for 3x3 * 3x7 matrix multiplication."""
import time
import numpy as np
from itertools import product

DIMENSIONS = [3, 3, 7]

def _build_target_tensor(n, m, p):
    """Build the matrix multiplication tensor T[i,j,k] = 1 for all i,j,k."""
    return np.ones((n, m, p), dtype=np.float64)

def _decompose_to_factors(T, n, m, p, rank, U, V, W):
    """
    Given rank and factor matrices, verify exact decomposition.
    T[i,j,k] = sum_r U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
    """
    # Reshape to check
    nmp = n * m * p
    # Build the full tensor from factors
    # For each r, we add outer product U[:,r] * V[:,r] * W[:,r]
    # U has shape (n*m, rank), V (m*p, rank), W (p*n, rank)
    
    result = np.zeros((n, m, p), dtype=np.float64)
    for r in range(rank):
        u_vec = U[:, r]  # (n*m,)
        v_vec = V[:, r]  # (m*p,)
        w_vec = W[:, r]  # (p*n,)
        
        # Reshape
        u_mat = u_vec.reshape(n, m)
        v_mat = v_vec.reshape(m, p)
        w_mat = w_vec.reshape(p, n)
        
        # Contract: sum over j of u_mat[i,j] * v_mat[j,k] * w_mat[k,i]
        # For each i, k: sum_j u[i,j] * v[j,k] * w[k,i]
        for i in range(n):
            for k in range(p):
                s = 0.0
                for j in range(m):
                    s += u_mat[i, j] * v_mat[j, k] * w_mat[k, i]
                result[i, k]  # Wait, the indices are T[i,j,k]
                # Actually T[i,j,k] = sum_r U[i*m+j,r] V[j*p+k,r] W[k*n+i,r]
                # So for fixed i,j,k: sum_r U[i*m+j,r] * V[j*p+k,r] * W[k*n+i,r]
    # Let me redo this more carefully
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                result[i, j, k] = s
    return result

def _snap_to_half_int(x):
    """Snap to nearest half-integer."""
    return np.round(x * 2) / 2

def _check_exact(T, n, m, p, rank, U, V, W):
    """Check if the decomposition is exact (within floating point tolerance)."""
    result = np.zeros((n, m, p), dtype=np.float64)
    for i in range(n):
        for j in range(m):
            for k in range(p):
                s = 0.0
                for r in range(rank):
                    s += U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r]
                result[i, j, k] = s
    # Check if result matches T (all ones) within tolerance
    diff = np.abs(result - T)
    return np.max(diff) < 1e-9

def _try_rank(R, n, m, p, T, max_iters=5000, tol=1e-8):
    """
    Try to find a rank-R decomposition using gradient descent.
    Returns (U, V, W) if successful, None otherwise.
    """
    np.random.seed(R)  # Deterministic for reproducibility
    U = np.random.randn(n*m, R) * 0.1
    V = np.random.randn(m*p, R) * 0.1
    W = np.random.randn(p*n, R) * 0.1
    
    # Simple gradient descent on the squared error
    lr = 0.01
    prev_err = float('inf')
    
    for it in range(max_iters):
        # Compute current error
        err = 0.0
        # Gradients
        dU = np.zeros_like(U)
        dV = np.zeros_like(V)
        dW = np.zeros_like(W)
        
        for i in range(n):
            for j in range(m):
                for k in range(p):
                    target = T[i, j, k]
                    current = sum(U[i*m+j, r] * V[j*p+k, r] * W[k*n+i, r] for r in range(R))
                    diff = current - target
                    err += diff * diff
                    # Gradient w.r.t. U[i*m+j, r]: diff * V[j*p+k, r] * W[k*n+i, r]
                    for r in range(R):
                        dU[i*m+j, r] += diff * V[j*p+k, r] * W[k*n+i, r]
                        dV[j*p+k, r] += diff * U[i*m+j, r] * W[k*n+i, r]
                        dW[k*n+i, r] += diff * U[i*m+j, r] * V[j*p+k, r]
        
        err *= 0.5  # Because of squared error, derivative is 2x
        
        # Update
        U -= lr * dU
        V -= lr * dV
        W -= lr * dW
        
        if err < tol:
            # Try snapping to half-integers
            Us = _snap_to_half_int(U)
            Vs = _snap_to_half_int(V)
            Ws = _snap_to_half_int(W)
            if _check_exact(T, n, m, p, R, Us, Vs, Ws):
                return Us, Vs, Ws
            # Also try without snapping if already close enough
            if _check_exact(T, n, m, p, R, U, V, W):
                return U, V, W
        
        if abs(prev_err - err) < 1e-12:
            break
        prev_err = err
    
    # Final snap attempt
    Us = _snap_to_half_int(U)
    Vs = _snap_to_half_int(V)
    Ws = _snap_to_half_int(W)
    if _check_exact(T, n, m, p, R, Us, Vs, Ws):
        return Us, Vs, Ws
    if _check_exact(T, n, m, p, R, U, V, W):
        return U, V, W
    
    return None

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    n, m, p = DIMENSIONS
    T = _build_target_tensor(n, m, p)
    full_rank = n * m * p  # 63
    
    # Try descending from full_rank - 1 down to some minimum
    # Start from 62, 61, ... but limit total time
    for R in range(full_rank - 1, 20, -1):
        if time.time() > deadline - 10:
            break
        result = _try_rank(R, n, m, p, T, max_iters=2000, tol=1e-8)
        if result is not None:
            U, V, W = result
            # Convert to lists
            U_list = U.tolist()
            V_list = V.tolist()
            W_list = W.tolist()
            return {"U": U_list, "V": V_list, "W": W_list}
    
    # Fallback: schoolbook rank
    rank = full_rank
    U = [[0] * rank for _ in range(n * m)]
    V = [[0] * rank for _ in range(m * p)]
    W = [[0] * rank for _ in range(p * n)]
    column = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j][column] = 1
                V[j * p + k][column] = 1
                W[k * n + i][column] = 1
                column += 1
    return {"U": U, "V": V, "W": W}
