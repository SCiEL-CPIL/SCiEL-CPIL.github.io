import numpy as np
import time
import sys

DIMENSIONS = [4, 5, 5]

# EVOLVE-BLOCK-START
def construct():
    n, m, p = DIMENSIONS
    # Schoolbook decomposition as fallback
    rank_sb = n * m * p
    U_sb = [[0] * rank_sb for _ in range(n * m)]
    V_sb = [[0] * rank_sb for _ in range(m * p)]
    W_sb = [[0] * rank_sb for _ in range(p * n)]
    col = 0
    for i in range(n):
        for j in range(m):
            for k in range(p):
                U_sb[i * m + j][col] = 1
                V_sb[j * p + k][col] = 1
                W_sb[k * n + i][col] = 1
                col += 1
    
    # Time limit
    start_time = time.time()
    deadline = start_time + 145  # 145 seconds limit (5s safety margin)
    
    # Construct the target tensor T of shape (n*m, m*p, p*n)
    # T[i*m+j, j*p+k, k*n+i] = 1
    T = np.zeros((n*m, m*p, p*n))
    for i in range(n):
        for j in range(m):
            for k in range(p):
                T[i*m + j, j*p + k, k*n + i] = 1.0

    best_rank = rank_sb
    best_U = U_sb
    best_V = V_sb
    best_W = W_sb

    # Try to find lower rank decompositions using ALS optimization
    # We'll try ranks from 99 down to 80
    for target_rank in range(99, 79, -1):
        if time.time() > deadline:
            break
        
        result = try_als_decomposition(T, n, m, p, target_rank, deadline)
        if result is not None:
            U_opt, V_opt, W_opt = result
            # Verify the decomposition
            if verify_decomposition(U_opt, V_opt, W_opt, T, n, m, p):
                best_rank = target_rank
                best_U = U_opt
                best_V = V_opt
                best_W = W_opt
                break
    
    # Return the best decomposition found
    return {
        "U": best_U,
        "V": best_V,
        "W": best_W
    }

def try_als_decomposition(T, n, m, p, rank, deadline):
    """Try to find a rank decomposition using Alternating Least Squares"""
    np.random.seed(42)
    
    # Initialize with random values
    U = np.random.randn(n*m, rank) * 0.5
    V = np.random.randn(m*p, rank) * 0.5
    W = np.random.randn(p*n, rank) * 0.5
    
    # ALS iterations
    for it in range(200):
        if time.time() > deadline:
            return None
        
        # Update U: for each row a, solve linear system
        # T[a, :, :] = U[a, :] @ (V[:, r] ⊗ W[:, r])^T
        # Actually: T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed V, W: T[a, :, :] = sum_r U[a,r] * (V[:, r] ⊗ W[:, r])^T
        # Let M_r = V[:, r] ⊗ W[:, r] (flattened)
        # Then T[a, :, :].flatten() = sum_r U[a,r] * M_r
        # So we need: T_flat[a, :] = U[a, :] @ M^T where M is (rank, m*p*p*n)
        
        # Precompute M matrix: M[r, b, c] = V[b,r] * W[c,r]
        # Reshape to (rank, m*p*p*n)
        M = np.einsum('br,cr->bcr', V, W).reshape(rank, -1)
        
        # Solve for U: T_flat[a, :] = U[a, :] @ M.T
        # This is: T_flat = U @ M.T
        # So U = T_flat @ pinv(M.T)
        try:
            U_new = np.linalg.lstsq(M.T, T.reshape(n*m, -1).T, rcond=None)[0].T
            U = U_new
        except:
            pass
        
        # Update V: similarly
        # T[:, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U, W: T[:, b, c] = sum_r V[b,r] * (U[:, r] ⊗ W[:, r])^T... 
        # Actually: T[a, b, c] = V[b,r] * (U[:, r] ⊗ W[:, r])... no
        # T[a, b, c] = sum_r U[a,r] * V[b,r] * W[c,r]
        # For fixed U, W: T[:, b, :] = sum_r V[b,r] * (U[:, r] ⊗ W[:, r])... 
        # Let's define N_r = U[:, r] ⊗ W[:, r]
        # Then T[:, b, c] = sum_r V[b,r] * N_r[a, c]... this doesn't work directly
        # Better: T.reshape(n*m, m*p*p*n)[:, b*p*n+c] = sum_r V[b,r] * (U[:, r] ⊗ W[:, r])[a, c]
        # This is complex. Let me use a different approach.
        
        # Alternative: Use gradient descent instead
        if it % 10 == 0:
            # Compute gradient and do a simple GD step
            # Loss = ||T - U⊗V⊗W||^2
            # grad_U = 2 * (T - U⊗V⊗W) ⊗ (V ⊗ W)
            current = np.einsum('ar,br,cr->abc', U, V, W)
            residual = T - current
            grad_U = np.einsum('abc,br,cr->ar', residual, V, W)
            grad_V = np.einsum('abc,ar,cr->br', residual, U, W)
            grad_W = np.einsum('abc,ar,br->cr', residual, U, V)
            
            lr = 0.01
            U -= lr * grad_U
            V -= lr * grad_V
            W -= lr * grad_W
    
    # After optimization, try to round to integers/half-integers
    # Scale and round
    U_rounded = np.round(U * 2) / 2
    V_rounded = np.round(V * 2) / 2
    W_rounded = np.round(W * 2) / 2
    
    # Verify rounded version
    if verify_decomposition(U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist(), T, n, m, p):
        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
    
    # If rounding doesn't work, try with original floats (but they need to be integer/half-integer)
    # Check if close to half-integers
    if np.allclose(U, U_rounded) and np.allclose(V, V_rounded) and np.allclose(W, W_rounded):
        return (U_rounded.tolist(), V_rounded.tolist(), W_rounded.tolist())
    
    return None

def verify_decomposition(U, V, W, T, n, m, p, tol=1e-6):
    """Verify that the decomposition exactly reconstructs T"""
    U_arr = np.array(U, dtype=float)
    V_arr = np.array(V, dtype=float)
    W_arr = np.array(W, dtype=float)
    
    # Check shapes
    if U_arr.shape != (n*m, W_arr.shape[1]):
        return False
    if V_arr.shape != (m*p, W_arr.shape[1]):
        return False
    if W_arr.shape != (p*n, U_arr.shape[1]):
        return False
    
    # Compute reconstruction
    reconstructed = np.einsum('ar,br,cr->abc', U_arr, V_arr, W_arr)
    
    # Check if close to T
    return np.allclose(reconstructed, T, atol=tol)

# EVOLVE-BLOCK-END
