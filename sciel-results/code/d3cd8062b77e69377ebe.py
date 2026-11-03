"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def construct():
    start_time = time.time()
    deadline = 145.0  # 5 seconds margin before 150s kill

    # 1. Initialization: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    X = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    # 2. Generate Probe Points for Objective Function
    # We need a dense set of points to approximate the integral over the sphere.
    # 10,000 points is a good balance between accuracy and speed.
    n_probes = 10000
    j = np.arange(n_probes) + 0.5
    z_p = 1.0 - 2.0 * j / n_probes
    phi_p = np.pi * (1.0 + np.sqrt(5.0)) * j
    r_p = np.sqrt(np.maximum(0.0, 1.0 - z_p * z_p))
    Probes = np.column_stack([r_p * np.cos(phi_p), r_p * np.sin(phi_p), z_p])
    Probes = Probes / np.linalg.norm(Probes, axis=1, keepdims=True)

    best_X = X.copy()
    best_score = np.inf

    # 3. Lloyd's Algorithm Iterations
    # We want to minimize the sum of squared distances to the nearest neighbor.
    # This drives points to become centroids of their Voronoi cells.
    
    lr = 0.05  # Learning rate
    max_iters = 500
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
            
        # Find nearest neighbor for each probe
        # Distances: (n_probes, N)
        # d^2 = |p|^2 + |x|^2 - 2 p.x = 1 + 1 - 2 p.x = 2 - 2 p.x
        # Minimizing distance is equivalent to maximizing p.x
        # So we just need argmax of Probes @ X.T
        
        dists_sq = 2.0 - 2.0 * (Probes @ X.T)
        nearest_idx = np.argmin(dists_sq, axis=1)
        
        # Calculate gradient for each point
        # grad_i = sum_{p: nearest(p)=i} 2 * (X_i - p)
        # We can accumulate this using bincount or similar logic, but N is small.
        # Let's just loop over points or use advanced indexing.
        
        grad = np.zeros_like(X)
        for i in range(N):
            mask = (nearest_idx == i)
            if np.any(mask):
                # Sum of (X_i - p) for all p in mask
                # X_i is constant, so sum(X_i - p) = count(X_i) - sum(p)
                count = np.sum(mask)
                sum_p = np.sum(Probes[mask], axis=0)
                grad[i] = 2.0 * (count * X[i] - sum_p)
                
        # Update X
        X = X - lr * grad
        
        # Project back to sphere
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        X = X / norms
        
        # Evaluate current score (Covering Radius approximation)
        # The actual covering radius is the max distance to nearest neighbor.
        # We can estimate it using the probes.
        max_dist_sq = np.max(dists_sq)
        current_radius = np.degrees(np.arccos(np.clip(1.0 - max_dist_sq / 2.0, -1.0, 1.0)))
        
        if current_radius < best_score:
            best_score = current_radius
            best_X = X.copy()

    # Final check: Return the best configuration found
    return best_X
# EVOLVE-BLOCK-END
