import numpy as np
import time
import math

N = 150

def _energy_and_grad_fast(X):
    """
    Compute logarithmic energy and gradient using Gram matrix for speed.
    E = sum_{i<j} log(1/d_ij) where d_ij = ||x_i - x_j||
    Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    """
    # Gram matrix: G[i,j] = x_i . x_j
    G = X @ X.T
    # d_ij^2 = 2 - 2*G[i,j]
    dist_sq = 2.0 - 2.0 * G
    # Ensure positive (numerical safety)
    np.maximum(dist_sq, 1e-30, out=dist_sq)
    # Zero diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: E = -sum_{i<j} log(d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # log(d_ij) = 0.5 * log(d_ij^2)
    with np.errstate(divide='ignore'):
        log_dist_sq = np.log(dist_sq)
    np.fill_diagonal(log_dist_sq, 0.0)
    E = -0.25 * np.sum(log_dist_sq)
    
    # Gradient: grad_i = sum_{j!=i} (x_j - x_i) / d_ij^2
    # = sum_{j!=i} x_j / d_ij^2 - x_i * sum_{j!=i} 1/d_ij^2
    # Let W[i,j] = 1/d_ij^2 for i!=j, W[i,i]=0
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # sum_j x_j / d_ij^2 = X.T @ W (but careful: (X.T @ W)[i] = sum_j X[j] * W[j,i] = sum_j x_j * W[i,j])
    # Actually: (W @ X)[i] = sum_j W[i,j] * X[j] = sum_j x_j / d_ij^2  -- this is what we want for first term
    # Wait: grad_i = sum_j (x_j - x_i)/d_ij^2 = sum_j x_j/d_ij^2 - x_i * sum_j 1/d_ij^2
    # First term: sum_j x_j / d_ij^2 = (W @ X)[i]  where W[i,j] = 1/d_ij^2
    # Second term: x_i * sum_j 1/d_ij^2 = x_i * (W @ ones)[i] = x_i * row_sum[i]
    row_sum = W.sum(axis=1)
    Grad = W @ X - row_sum[:, np.newaxis] * X
    
    return E, Grad

def _polish(X, max_time, tol=1e-14):
    """
    High-precision Riemannian gradient descent with backtracking line search.
    Uses the fast Gram-matrix based energy/gradient.
    """
    E, Grad = _energy_and_grad_fast(X)
    best_E = E
    best_X = X.copy()
    
    lr = 0.01
    min_lr = 1e-16
    max_rejects = 30
    reject_count = 0
    start = time.time()
    
    while time.time() - start < max_time:
        # Project gradient to tangent space
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Grad_tan = Grad - proj * X
        
        # Skip if gradient is tiny
        grad_norm = np.linalg.norm(Grad_tan)
        if grad_norm < tol:
            break
        
        # Backtracking line search
        accepted = False
        for _ in range(max_rejects):
            X_new = X + lr * Grad_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            E_new, Grad_new = _energy_and_grad_fast(X_new)
            
            if E_new < E:
                # Accept
                X = X_new
                E = E_new
                Grad = Grad_new
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                accepted = True
                # Slightly increase lr on success
                lr = min(lr * 1.02, 1.0)
                reject_count = 0
                break
            else:
                lr *= 0.5
        
        if not accepted:
            # Reset learning rate and try a small random perturbation
            lr = 0.01
            # Small random kick in tangent space
            rng = np.random.default_rng(42 + int(time.time() * 1000) % 10000)
            kick = rng.standard_normal((N, 3)) * 1e-4
            kick_proj = np.sum(kick * X, axis=1, keepdims=True)
            kick = kick - kick_proj * X
            X = X + kick
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
            E, Grad = _energy_and_grad_fast(X)
            reject_count += 1
            if reject_count > 10:
                break
    
    return best_X, best_E

def construct():
    start_time = time.time()
    deadline = 140.0  # 10s buffer before 150s kill
    
    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Phase 1: Coarse optimization with moderate tolerances
    E, Grad = _energy_and_grad_fast(X)
    best_E = E
    best_X = X.copy()
    
    lr = 0.05
    min_lr = 1e-10
    max_rejects = 20
    reject_count = 0
    phase1_deadline = start_time + 60.0  # 60s for coarse phase
    
    while time.time() < phase1_deadline:
        # Project gradient to tangent space
        proj = np.sum(Grad * X, axis=1, keepdims=True)
        Grad_tan = Grad - proj * X
        
        grad_norm = np.linalg.norm(Grad_tan)
        if grad_norm < 1e-12:
            break
        
        # Backtracking line search
        accepted = False
        for _ in range(max_rejects):
            X_new = X + lr * Grad_tan
            norms = np.linalg.norm(X_new, axis=1, keepdims=True)
            X_new = X_new / norms
            
            E_new, Grad_new = _energy_and_grad_fast(X_new)
            
            if E_new < E:
                X = X_new
                E = E_new
                Grad = Grad_new
                if E < best_E:
                    best_E = E
                    best_X = X.copy()
                accepted = True
                lr = min(lr * 1.03, 1.0)
                reject_count = 0
                break
            else:
                lr *= 0.5
        
        if not accepted:
            lr = 0.05
            # Random perturbation to escape local minima
            rng = np.random.default_rng(int(time.time() * 1000) % 100000)
            kick = rng.standard_normal((N, 3)) * 1e-3
            kick_proj = np.sum(kick * X, axis=1, keepdims=True)
            kick = kick - kick_proj * X
            X = X + kick
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            X = X / norms
            E, Grad = _energy_and_grad_fast(X)
            reject_count += 1
            if reject_count > 15:
                break
    
    # 3. Phase 2: High-precision polish
    remaining = deadline - time.time()
    if remaining > 5.0:
        best_X, best_E = _polish(best_X, max_time=remaining - 5.0, tol=1e-14)
    
    # Final check: ensure points are on unit sphere and valid
    norms = np.linalg.norm(best_X, axis=1, keepdims=True)
    best_X = best_X / norms
    
    # Verify no points are too close
    D = best_X[:, np.newaxis, :] - best_X[np.newaxis, :, :]
    dist = np.linalg.norm(D, axis=2)
    np.fill_diagonal(dist, 1.0)
    if np.min(dist) < 1e-9:
        # Fallback to Fibonacci if collision (unlikely)
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        best_X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    return best_X
