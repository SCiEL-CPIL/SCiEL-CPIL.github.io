"""Rewritten program for Matrix Multiplication 3,4,4 rank minimization."""

DIMENSIONS = [3, 4, 4]

# EVOLVE-BLOCK-START
import numpy as np
import time
import sys

TIME_LIMIT = 145.0
START_TIME = time.time()

def time_left():
    return TIME_LIMIT - (time.time() - START_TIME)

def schoolbook_solution(n, m, p):
    """Return the schoolbook (rank n*m*p) decomposition."""
    rank = n * m * p
    U = np.zeros((n * m, rank), dtype=np.float64)
    V = np.zeros((m * p, rank), dtype=np.float64)
    W = np.zeros((p * n, rank), dtype=np.float64)
    
    idx = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U[i * m + j, idx] = 1.0
                V[j * p + k, idx] = 1.0
                W[k * n + i, idx] = 1.0
                idx += 1
    
    return U, V, W

def compute_error(U, V, W, n, m, p):
    """Compute the sum of squared errors for the decomposition."""
    # Reshape U, V, W to 3D tensors for efficient computation
    # U: (n*m, R) -> (n, m, R)
    # V: (m*p, R) -> (m, p, R)
    # W: (p*n, R) -> (p, n, R)
    
    U_3d = U.T.reshape(n, m, -1) # (n, m, R)
    V_3d = V.T.reshape(m, p, -1) # (m, p, R)
    W_3d = W.T.reshape(p, n, -1) # (p, n, R)
    
    # We want to compute C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    # Note: W index is k*n + i, so W_3d[k, i, r] is correct.
    
    # Let's compute the tensor product
    # Term[i, j, k, r] = U[i, j, r] * V[j, k, r] * W[k, i, r]
    # This is too large to store directly if R is large, but R is small (47).
    # n=3, m=4, p=4. 3*4*4*47 = 2256 floats. Very small.
    
    # Efficient way:
    # Sum over r of (U[:, :, r] * V[:, :, r] * W[:, :, r])
    # But dimensions don't align directly for element-wise multiplication in one go without broadcasting.
    
    # Let's use einsum or manual loops.
    # C[i, k] = sum_j sum_r U[i, j, r] * V[j, k, r] * W[k, i, r]
    
    # We can compute this as:
    # 1. For each r, compute M_r = U[:, :, r] @ V[:, :, r] (n x p matrix)
    #    M_r[i, k] = sum_j U[i, j, r] * V[j, k, r]
    # 2. Then C[i, k] = sum_r M_r[i, k] * W[k, i, r]
    
    C = np.zeros((n, p), dtype=np.float64)
    for r in range(U.shape[1]):
        # M_r[i, k] = sum_j U[i*m+j, r] * V[j*p+k, r]
        # U_col = U[:, r] reshaped to (n, m)
        # V_col = V[:, r] reshaped to (m, p)
        U_col = U[:, r].reshape(n, m)
        V_col = V[:, r].reshape(m, p)
        M_r = U_col @ V_col
        
        # W_col = W[:, r] reshaped to (p, n)
        W_col = W[:, r].reshape(p, n)
        
        # C[i, k] += M_r[i, k] * W_col[k, i]
        # Note: W_col[k, i] is W[k*n+i, r]
        C += M_r * W_col.T
        
    # Error is sum (C[i, k] - 1)^2
    error = np.sum((C - 1.0)**2)
    return error

def try_rank_reduction(n, m, p, target_rank, max_iterations=1000, lr=0.01, tol=1e-6):
    """
    Try to find a decomposition of target_rank using gradient descent.
    """
    if time_left() < 10:
        return None
        
    # Initialize with random values
    # Using small random values to start close to zero might be bad, 
    # maybe start near schoolbook?
    # Schoolbook has 48 terms. If target_rank < 48, we can't start directly from schoolbook.
    # We can start with random Gaussian noise.
    
    U = np.random.randn(n * m, target_rank) * 0.1
    V = np.random.randn(m * p, target_rank) * 0.1
    W = np.random.randn(p * n, target_rank) * 0.1
    
    current_error = compute_error(U, V, W, n, m, p)
    best_error = current_error
    best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
    
    # Simple gradient descent
    # We need gradients.
    # dE/dU[i, j, r] = 2 * (C[i, k] - 1) * V[j, k, r] * W[k, i, r] summed over k?
    # No, C[i, k] = sum_j sum_r U[i, j, r] V[j, k, r] W[k, i, r]
    # dC[i, k]/dU[i', j', r'] = delta(i, i') delta(r, r') V[j', k, r'] W[k, i, r']
    # So dE/dU[i', j', r'] = 2 * sum_k (C[i', k] - 1) * V[j', k, r'] * W[k, i', r']
    
    # Similarly for V and W.
    
    for it in range(max_iterations):
        if time_left() < 5:
            break
            
        # Compute gradients
        grad_U = np.zeros_like(U)
        grad_V = np.zeros_like(V)
        grad_W = np.zeros_like(W)
        
        # C is (n, p)
        # We need C for the current step.
        # Recomputing C is O(R * n * m * p). 47 * 48 = 2256 ops. Very fast.
        C = np.zeros((n, p), dtype=np.float64)
        for r in range(target_rank):
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            M_r = U_col @ V_col
            W_col = W[:, r].reshape(p, n)
            C += M_r * W_col.T
            
        error_vec = 2.0 * (C - 1.0) # (n, p)
        
        for r in range(target_rank):
            U_col = U[:, r].reshape(n, m)
            V_col = V[:, r].reshape(m, p)
            W_col = W[:, r].reshape(p, n)
            
            # grad_U[i, j, r] = sum_k error_vec[i, k] * V[j, k, r] * W[k, i, r]
            # V[j, k, r] is V_col[j, k]
            # W[k, i, r] is W_col[k, i]
            
            # For each i, j:
            # grad_U[i, j, r] = sum_k error_vec[i, k] * V_col[j, k] * W_col[k, i]
            
            # We can compute this as:
            # Let A = error_vec (n x p)
            # Let B = V_col (m x p)
            # Let C_mat = W_col (p x n)
            
            # grad_U_col[i, j] = sum_k A[i, k] * B[j, k] * C_mat[k, i]
            
            # This is not a simple matrix multiplication.
            # We can loop over i, j, k.
            # n=3, m=4, p=4. 3*4*4 = 48 ops per r. 47*48 = 2256 ops.
            
            for i in range(n):
                for j in range(m):
                    s = 0.0
                    for k in range(p):
                        s += error_vec[i, k] * V_col[j, k] * W_col[k, i]
                    grad_U[i * m + j, r] = s
                    
            # grad_V[j, k, r] = sum_i error_vec[i, k] * U[i, j, r] * W[k, i, r]
            # U[i, j, r] is U_col[i, j]
            # W[k, i, r] is W_col[k, i]
            
            for j in range(m):
                for k in range(p):
                    s = 0.0
                    for i in range(n):
                        s += error_vec[i, k] * U_col[i, j] * W_col[k, i]
                    grad_V[j * p + k, r] = s
                    
            # grad_W[k, i, r] = sum_j error_vec[i, k] * U[i, j, r] * V[j, k, r]
            # U[i, j, r] is U_col[i, j]
            # V[j, k, r] is V_col[j, k]
            
            for k in range(p):
                for i in range(n):
                    s = 0.0
                    for j in range(m):
                        s += error_vec[i, k] * U_col[i, j] * V_col[j, k]
                    grad_W[k * n + i, r] = s

        # Update
        U -= lr * grad_U
        V -= lr * grad_V
        W -= lr * grad_W
        
        current_error = compute_error(U, V, W, n, m, p)
        
        if current_error < best_error:
            best_error = current_error
            best_U, best_V, best_W = U.copy(), V.copy(), W.copy()
            
        if best_error < tol:
            return best_U, best_V, best_W
            
    if best_error < 1e-3: # Relaxed tolerance for early termination check
         return best_U, best_V, best_W
         
    return None

def construct():
    n, m, p = DIMENSIONS
    
    # Try rank 47
    result = try_rank_reduction(n, m, p, 47, max_iterations=500, lr=0.05, tol=1e-8)
    if result is not None:
        U, V, W = result
        # Verify
        err = compute_error(U, V, W, n, m, p)
        if err < 1e-6:
            return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}
    
    # Try rank 47 with different seeds/params if time permits
    if time_left() > 20:
         result = try_rank_reduction(n, m, p, 47, max_iterations=500, lr=0.1, tol=1e-8)
         if result is not None:
             U, V, W = result
             err = compute_error(U, V, W, n, m, p)
             if err < 1e-6:
                 return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

    # Fall back to schoolbook
    U, V, W = schoolbook_solution(n, m, p)
    return {"U": U.tolist(), "V": V.tolist(), "W": W.tolist()}

# EVOLVE-BLOCK-END
