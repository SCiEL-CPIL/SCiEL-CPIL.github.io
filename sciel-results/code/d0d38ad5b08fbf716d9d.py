import numpy as np
import time

N = 150

def _energy_and_grad(X):
    # X is (N, 3)
    # Distances
    # d_ij^2 = |x_i - x_j|^2
    # Using broadcasting
    # X[:, np.newaxis, :] - X[np.newaxis, :, :] -> (N, N, 3)
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero for i=j
    # We only need i < j for energy, but gradient needs all j != i
    # Set diagonal to 1 to avoid inf, then mask out
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/sqrt(dist_sq)) = -0.5 * sum_{i!=j} log(dist_sq)
    # Wait, log(1/d) = -log(d). d = sqrt(dist_sq).
    # log(1/d) = -0.5 log(dist_sq).
    # Sum over i<j.
    # Sum over i!=j is 2 * Sum over i<j.
    # So E = -0.25 * sum_{i!=j} log(dist_sq).
    # Let's calculate sum of log(dist_sq) for i != j.
    log_dist_sq = np.log(dist_sq)
    # Mask diagonal
    log_dist_sq = np.where(np.eye(N, dtype=bool), 0.0, log_dist_sq)
    
    E = -0.25 * np.sum(log_dist_sq)
    
    # Gradient
    # grad_xi E = - sum_{j != i} (x_i - x_j) / |x_i - x_j|^2
    # diff is (x_i - x_j)
    # We need (x_i - x_j) / dist_sq_ij
    # diff / dist_sq[..., np.newaxis]
    # But dist_sq is (N, N).
    # grad = - sum_j (x_i - x_j) / dist_sq_ij
    
    # Create matrix of 1/dist_sq
    inv_dist_sq = 1.0 / dist_sq
    np.fill_diagonal(inv_dist_sq, 0.0)
    
    # grad[i] = - sum_j inv_dist_sq[i,j] * diff[i,j]
    # diff[i,j] = x_i - x_j
    # So grad = - (inv_dist_sq @ diff)
    # Wait, diff is 3D.
    # grad = - np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    G = -np.sum(inv_dist_sq[:, :, np.newaxis] * diff, axis=1)
    
    return E, G

def construct():
    start_time = time.time()
    timeout = 140.0 # Safety margin
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # L-BFGS state
    m = 10 # memory size
    s_list = []
    y_list = []
    
    E, G = _energy_and_grad(X)
    best_E = E
    best_X = X.copy()
    
    # Simple L-BFGS loop
    # We will use a fixed number of iterations or time limit
    # Each iteration is O(N^2)
    # N=150, N^2=22500. Fast.
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > timeout:
            break
            
        # Project gradient to tangent space
        # G is (N, 3). X is (N, 3).
        # G_dot_X = sum(G * X, axis=1) -> (N,)
        G_dot_X = np.sum(G * X, axis=1)
        G_tan = G - G_dot_X[:, np.newaxis] * X
        
        # L-BFGS two-loop recursion to find direction P
        # Start with P = -G_tan
        P = -G_tan
        
        # Apply two-loop recursion
        # We need to store s, y.
        # y = G_tan_new - G_tan_old? No, y = grad_new - grad_old.
        # s = x_new - x_old.
        
        # For the first step, P = -G_tan.
        
        # Line search
        # We want to minimize E(X + alpha P) after normalization.
        # Backtracking line search
        alpha = 1.0
        c1 = 1e-4
        slope = np.sum(G_tan * P) # Should be negative
        
        for _ in range(20):
            X_trial = X + alpha * P
            # Normalize
            norms = np.linalg.norm(X_trial, axis=1)
            # Avoid zero norms
            norms[norms < 1e-10] = 1.0
            X_trial = X_trial / norms[:, np.newaxis]
            
            E_trial, G_trial = _energy_and_grad(X_trial)
            
            if E_trial < E + c1 * alpha * slope:
                # Accept step
                break
            alpha *= 0.5
            
        else:
            # Failed to find step, try smaller alpha or break
            alpha = 1e-5
            X_trial = X + alpha * P
            norms = np.linalg.norm(X_trial, axis=1)
            norms[norms < 1e-10] = 1.0
            X_trial = X_trial / norms[:, np.newaxis]
            E_trial, G_trial = _energy_and_grad(X_trial)
            
        # Update L-BFGS memory
        s = X_trial - X
        y = G_trial - G
        
        # Check curvature
        s_dot_y = np.sum(s * y)
        if s_dot_y > 1e-10:
            s_list.append(s)
            y_list.append(y)
            if len(s_list) > m:
                s_list.pop(0)
                y_list.pop(0)
        
        X = X_trial
        E = E_trial
        G = G_trial
        
        if E < best_E:
            best_E = E
            best_X = X.copy()
            
        # Convergence check
        if np.linalg.norm(G_tan) < 1e-8:
            break

    # Final polish with tighter tolerance if time remains
    elapsed = time.time() - start_time
    if elapsed < timeout - 10:
         # Do a few more steps with smaller alpha?
         pass
         
    return best_X

