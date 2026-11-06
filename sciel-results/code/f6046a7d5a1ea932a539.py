import numpy as np
import time

N = 100

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 5s safety margin

    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # Ensure unit norm
    X = X / np.linalg.norm(X, axis=1)[:, np.newaxis]

    def compute_energy(X):
        sq_dist = np.sum(X**2, axis=1)[:, np.newaxis] + np.sum(X**2, axis=1)[np.newaxis, :] - 2 * X @ X.T
        sq_dist = np.maximum(sq_dist, 1e-12)
        dist = np.sqrt(sq_dist)
        np.fill_diagonal(dist, 1.0)
        return -0.5 * np.sum(np.log(dist))

    best_X = X.copy()
    best_E = compute_energy(X)
    current_E = best_E
    
    eta = 0.01
    no_improve_count = 0
    
    while time.time() < deadline:
        # Compute gradient
        diff = X[:, np.newaxis, :] - X[np.newaxis, :, :]
        sq_dist = np.sum(diff**2, axis=2)
        np.fill_diagonal(sq_dist, 1.0)
        inv_sq_dist = 1.0 / sq_dist
        np.fill_diagonal(inv_sq_dist, 0.0)
        
        grad = np.sum(diff * inv_sq_dist[:, :, np.newaxis], axis=1)
        
        # Update
        X_new = X + eta * grad
        X_new = X_new / np.linalg.norm(X_new, axis=1)[:, np.newaxis]
        
        E_new = compute_energy(X_new)
        
        if E_new < best_E:
            best_E = E_new
            best_X = X_new.copy()
            X = X_new
            no_improve_count = 0
        else:
            # If no improvement, reduce step size
            no_improve_count += 1
            if no_improve_count > 100:
                eta *= 0.9
                no_improve_count = 0
            # Keep X as is, or revert?
            # Usually better to keep X if it's not worse, but if it's worse, maybe revert?
            # Gradient descent usually assumes small steps. If it jumps over, we might want to reduce eta and retry.
            # For simplicity, we just update X to X_new if it's not infinitely bad, but if it's worse, we might be stuck.
            # Let's just set X = X_new anyway to keep moving, but track best.
            # Actually, if E_new > current_E, we should probably revert and reduce eta.
            # But we don't store current_E separately from best_E logic easily in this loop structure.
            # Let's just set X = X_new. The projection keeps it on sphere.
            X = X_new
            current_E = E_new
            
    return best_X
