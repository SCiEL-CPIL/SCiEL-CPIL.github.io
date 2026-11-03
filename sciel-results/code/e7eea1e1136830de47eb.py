import numpy as np
import time
import math

N = 100

def construct():
    # Deadline guard
    start_time = time.time()
    timeout = 110.0  # Leave 10s margin for evaluation overhead
    
    # 1. Initial points: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Optimization loop
    # We use a simple gradient descent with re-normalization.
    # Energy E = -sum_{i<j} log(||xi - xj||)
    # Grad_i = sum_{j!=i} (xi - xj) / ||xi - xj||^2
    
    # Adaptive step size or fixed small step size.
    # Let's use a decreasing step size or a fixed one with a check.
    # A good strategy is to use a line search or a fixed step with decay.
    
    eta = 0.01
    min_eta = 1e-6
    decay = 0.999
    
    best_energy = np.inf
    best_X = X.copy()
    
    # Precompute indices for efficiency? No, vectorized numpy is fast enough for N=100.
    
    for step in range(100000): # Max iterations, but will break on time
        if time.time() - start_time > timeout:
            break
            
        # Compute pairwise differences
        # Diff[i, j] = X[i] - X[j]
        Diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        
        # Distances
        # D[i, j] = ||X[i] - X[j]||
        D = np.linalg.norm(Diff, axis=2)
        
        # Avoid division by zero
        # If points are too close, energy is high (log(1/d) -> inf). 
        # We should ensure no points are identical.
        # D[i, i] is 0. We need to handle the diagonal.
        np.fill_diagonal(D, 1.0) # Temporary fix for division
        
        # Gradient computation
        # Grad[i] = sum_j (X[i] - X[j]) / D[i, j]^2
        # Note: The term for j=i is 0/1 = 0, so it doesn't matter.
        
        # We want to minimize E.
        # E = - sum_{i<j} log(D[i,j])
        # dE/dX[i] = - sum_{j!=i} (1/D[i,j]) * d(D[i,j])/dX[i]
        # d(D[i,j])/dX[i] = (X[i] - X[j]) / D[i,j]
        # So dE/dX[i] = - sum_{j!=i} (X[i] - X[j]) / D[i,j]^2
        
        # Let's verify the sign.
        # If we move X[i] away from X[j], D increases, log(1/D) decreases, Energy decreases.
        # So we want to move in the direction of (X[i] - X[j]).
        # Gradient of Energy w.r.t X[i] should point in direction of increasing Energy.
        # Increasing Energy means moving CLOSER (D decreases).
        # So Gradient should point towards other points.
        # Formula: - sum (Xi - Xj) / D^2.
        # If Xi > Xj (in 1D), Xi-Xj is positive. Gradient is negative.
        # Moving in direction of Gradient (descent) means moving in negative direction.
        # So Xi decreases, moving towards Xj. This increases Energy.
        # Wait. We want to MINIMIZE energy.
        # Descent step: X_new = X - eta * Grad.
        # If Grad points "towards" other points (increasing energy direction), then -Grad points "away".
        # So X_new moves away. This decreases Energy. Correct.
        
        # Vectorized Gradient
        # G[i, j] = (X[i] - X[j]) / D[i, j]^2
        # We need sum over j for each i.
        
        # To avoid memory blowup for larger N, we can do it in chunks, but N=100 is small.
        # D shape (N, N). Diff shape (N, N, 3).
        
        # Inverse squared distances
        D2 = D**2
        np.fill_diagonal(D2, 1.0) # Avoid div by zero, term is 0 anyway
        
        # Weight matrix W[i, j] = 1 / D[i, j]^2
        W = 1.0 / D2
        np.fill_diagonal(W, 0.0) # Ensure diagonal is 0
        
        # Grad[i] = sum_j W[i, j] * (X[i] - X[j])
        # = sum_j W[i, j] * X[i] - sum_j W[i, j] * X[j]
        # = X[i] * sum_j W[i, j] - sum_j (W[i, j] * X[j])
        
        W_sum = np.sum(W, axis=1, keepdims=True) # Shape (N, 1)
        
        # Term 2: sum_j W[i, j] * X[j]
        # W shape (N, N), X shape (N, 3).
        # Result shape (N, 3).
        WX = W @ X
        
        Grad = X * W_sum - WX
        
        # Update
        X = X - eta * Grad
        
        # Re-normalize to sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        # Avoid division by zero
        norms[norms == 0] = 1.0
        X = X / norms
        
        # Calculate current energy for tracking
        # E = - sum_{i<j} log(D[i,j])
        # Use upper triangle to avoid double counting and self-interaction
        iu = np.triu_indices(N, k=1)
        D_ij = D[iu]
        # Ensure no zeros in D_ij for log
        D_ij_safe = np.maximum(D_ij, 1e-15)
        current_energy = -np.sum(np.log(D_ij_safe))
        
        if current_energy < best_energy:
            best_energy = current_energy
            best_X = X.copy()
            
        # Decay step size
        eta *= decay
        
        # Convergence check (optional, time limit is primary)
        if eta < min_eta:
            break
            
    # Final check and return
    # Ensure the returned points are valid (non-zero, finite)
    if not np.all(np.isfinite(best_X)):
        # Fallback to initial if something went wrong
        return X
        
    return best_X
