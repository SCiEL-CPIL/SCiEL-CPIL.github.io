import numpy as np
import time

N = 150

def construct():
    start_time = time.time()
    deadline = 115.0  # seconds, safety margin
    
    # 1. Initial Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Normalize to ensure unit length
    X /= np.linalg.norm(X, axis=1, keepdims=True)
    
    def compute_energy(X):
        # Calculate pairwise distances
        # D_ij = ||x_i - x_j||^2 = 2 - 2 x_i . x_j
        D = 2.0 - 2.0 * (X @ X.T)
        np.fill_diagonal(D, 1.0) # Avoid log(0)
        # Energy is sum_{i<j} log(1/sqrt(D_ij)) = -0.5 * sum_{i!=j} log(D_ij)
        # Wait, log(1/d) = -log(d). d = sqrt(D).
        # sum_{i<j} -log(sqrt(D_ij)) = -0.5 * sum_{i<j} log(D_ij)
        # sum_{i<j} log(D_ij) = 0.5 * (sum_{i!=j} log(D_ij) - sum_{i=j} log(D_ii))
        # Since we set diagonal to 1, log(1)=0.
        # So sum_{i<j} log(D_ij) = 0.5 * sum_{i!=j} log(D_ij)
        # Total Energy = -0.5 * (0.5 * sum_{i!=j} log(D_ij)) = -0.25 * sum_{i!=j} log(D_ij)
        
        # Let's just sum the upper triangle to be safe and direct.
        # But vectorized sum is faster.
        # Sum over all i!=j of log(D_ij) is 2 * sum_{i<j} log(D_ij).
        # E = -0.5 * sum_{i<j} log(D_ij) = -0.25 * sum_{i!=j} log(D_ij).
        
        # Clip D to avoid log(0) if any points coincide (shouldn't happen with init)
        D = np.maximum(D, 1e-12)
        log_D = np.log(D)
        # Zero out diagonal
        np.fill_diagonal(log_D, 0.0)
        total_sum = np.sum(log_D)
        return -0.25 * total_sum

    current_energy = compute_energy(X)
    
    # Gradient Descent on Sphere
    # F_k = sum_{j!=k} (x_k - x_j) / ||x_k - x_j||^2
    # M_ij = 1/D_ij for i!=j, 0 for i=j
    # F = S[:,None] * X - M @ X
    
    alpha = 0.05  # Initial step size
    max_iters = 5000
    
    for it in range(max_iters):
        if time.time() - start_time > deadline:
            break
            
        # Compute D
        D = 2.0 - 2.0 * (X @ X.T)
        np.fill_diagonal(D, 1.0)
        D = np.maximum(D, 1e-12)
        
        # Compute M
        M = 1.0 / D
        np.fill_diagonal(M, 0.0)
        
        # Compute S (row sums)
        S = np.sum(M, axis=1)
        
        # Compute Y = M @ X
        Y = M @ X
        
        # Compute Force F
        F = S[:, None] * X - Y
        
        # Project to tangent space
        # F_tan = F - (F . X) X
        dot_FX = np.sum(F * X, axis=1)
        F_tan = F - dot_FX[:, None] * X
        
        # Update X
        # X_new = X - alpha * F_tan
        X_new = X - alpha * F_tan
        
        # Normalize
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new /= norms
        
        # Check energy change to adjust step size
        new_energy = compute_energy(X_new)
        
        if new_energy < current_energy:
            X = X_new
            current_energy = new_energy
            # Optionally increase alpha slightly if progress is good?
            # alpha = min(alpha * 1.01, 0.1)
        else:
            # Reduce step size
            alpha *= 0.5
            if alpha < 1e-6:
                break
                
    # Final Polish: Maybe a few more iterations with very small alpha?
    # The adaptive step should handle it.
    
    return X
