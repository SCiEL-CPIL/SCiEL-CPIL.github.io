import numpy as np
import time

N = 100

def _energy_and_grad(X):
    """
    Compute the logarithmic energy and its gradient projected onto the tangent space
    for points X on the unit sphere.
    X: (N, 3) array of points.
    Returns: (energy, gradient)
    """
    # Compute pairwise squared distances
    # diff[i, j] = X[i] - X[j]
    diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
    dist_sq = np.sum(diff * diff, axis=2)
    
    # Avoid division by zero on diagonal
    np.fill_diagonal(dist_sq, 1.0)
    
    # Energy: sum_{i<j} log(1/d_ij) = -0.5 * sum_{i!=j} log(d_ij)
    # Using log(d_ij) = 0.5 * log(d_ij^2)
    # E = -0.5 * sum_{i!=j} 0.5 * log(d_ij^2) = -0.25 * sum_{i!=j} log(d_ij^2)
    # Since diagonal is 1, log(1)=0, we can sum the whole matrix.
    E = -0.25 * np.sum(np.log(dist_sq))
    
    # Gradient of E w.r.t X_i
    # dE/dX_i = - sum_{j!=i} (X_i - X_j) / ||X_i - X_j||^2
    # Let W_ij = 1 / dist_sq_ij (with diagonal 0)
    W = 1.0 / dist_sq
    np.fill_diagonal(W, 0.0)
    
    # Grad_i = - sum_j W_ij (X_i - X_j) = -X_i sum_j W_ij + sum_j W_ij X_j
    S = np.sum(W, axis=1)  # Shape (N,)
    WX = W @ X             # Shape (N, 3)
    Grad = -S[:, np.newaxis] * X + WX
    
    # Project gradient onto tangent space at X_i
    # Tangent space is perpendicular to X_i (normal vector)
    # Proj(G) = G - (G . X_i) X_i
    grad_dot_x = np.sum(Grad * X, axis=1)  # Shape (N,)
    Grad_proj = Grad - grad_dot_x[:, np.newaxis] * X
    
    return E, Grad_proj

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin for 150s limit
    
    # Initial configuration: Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm (should be already)
    X = X / np.linalg.norm(X, axis=1, keepdims=True)
    
    best_X = X.copy()
    best_E, _ = _energy_and_grad(X)
    
    lr = 0.05
    # We can try to adapt lr or just use a fixed one with many steps
    # A fixed lr of 0.05 is usually safe for this scale
    
    for it in range(20000):
        if time.time() > deadline:
            break
            
        E, G = _energy_and_grad(X)
        
        # Update
        X_new = X - lr * G
        # Project back to sphere
        norms = np.linalg.norm(X_new, axis=1, keepdims=True)
        X_new = X_new / norms
        
        # Check if improved
        E_new, _ = _energy_and_grad(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            X = X_new
        else:
            # If no improvement, reduce learning rate or break?
            # For now, just continue, maybe oscillating.
            # A simple strategy: if no improvement for many steps, reduce lr.
            # But checking "many steps" adds overhead.
            # Let's just keep X as is if no improvement? No, that stops progress.
            # Better: accept the step even if E is slightly higher (momentum-like), 
            # or just reduce lr.
            # Let's reduce lr if we get stuck.
            # To keep it simple, we'll just accept the step if it's not much worse, 
            # but for minimization, rejecting bad steps is safer.
            # However, with a small lr, steps are small.
            # Let's just always update X to X_new, but keep best_X.
            # This might wander.
            # Better: only update X if it improves, otherwise reduce lr.
            lr *= 0.9
            if lr < 1e-6:
                break
            # Do not update X, so next iteration tries again with smaller lr?
            # Or update X to X_new anyway?
            # Let's update X to X_new to keep moving, but track best.
            # Actually, standard gradient descent accepts the step.
            # Let's accept the step.
            X = X_new

    return best_X
