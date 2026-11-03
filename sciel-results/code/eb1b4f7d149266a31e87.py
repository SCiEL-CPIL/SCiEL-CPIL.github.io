"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = 145.0  # 5s safety margin

    def compute_radius(centers):
        # centers is shape (N, 2)
        # Compute pairwise distances
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        # Square is [-0.5, 0.5]^2
        dist_to_boundary = np.min(0.5 - np.abs(centers), axis=1)
        min_boundary_dist = np.min(dist_to_boundary)
        
        r = min(min_pair_dist / 2.0, min_boundary_dist)
        return r

    def negative_radius(centers):
        return -compute_radius(centers)

    best_r = -np.inf
    best_centers = None

    # Initialization strategies
    
    # 1. 7x7 Grid (49 points) + 1 "guest" point
    # This is a strong candidate because 49 is a perfect square packing with r=1/14 ~ 0.0714
    ticks = (np.arange(7) + 0.5) / 7 - 0.5
    grid_7x7 = np.array([(x, y) for y in ticks for x in ticks])
    
    # 2. 5x10 Grid (50 points)
    ticks_x = (np.arange(10) + 0.5) / 10 - 0.5
    ticks_y = (np.arange(5) + 0.5) / 5 - 0.5
    grid_5x10 = np.array([(x, y) for y in ticks_y for x in ticks_x])
    
    # 3. Halton Sequence
    def halton(n, base):
        seq = np.zeros(n)
        f = 1.0
        for i in range(1, n+1):
            x = 0.0
            j = i
            while j > 0:
                f /= base
                x += f * (j % base)
                j //= base
            seq[i-1] = x
        return seq
    h_x = halton(N, 2) - 0.5
    h_y = halton(N, 3) - 0.5
    halton_grid = np.column_stack((h_x, h_y))

    inits = [grid_5x10, halton_grid]
    
    # Add 7x7 + guest variations
    np.random.seed(42)
    # Random guests
    for _ in range(4):
        p50 = np.random.uniform(-0.4, 0.4, size=2)
        inits.append(np.vstack([grid_7x7, p50]))
        
    # Symmetric guests near center
    spots = [(0.02, 0.02), (-0.02, 0.02), (0.02, -0.02), (-0.02, -0.02), 
             (0.05, 0.05), (-0.05, 0.05), (0.05, -0.05), (-0.05, -0.05)]
    for s in spots:
        inits.append(np.vstack([grid_7x7, np.array(s)]))

    # Random start
    np.random.seed(123)
    inits.append(np.random.uniform(-0.4, 0.4, size=(N, 2)))

    # Optimize each initialization
    for i, init in enumerate(inits):
        if time.time() > deadline - 15:
            break
            
        x0 = init.flatten()
        
        # Use Nelder-Mead
        # adaptive=True helps with non-convex landscapes
        try:
            result = minimize(
                negative_radius, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9, 'adaptive': True}
            )
        except TypeError:
            # Fallback for older scipy versions
            result = minimize(
                negative_radius, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9}
            )
        
        centers = result.x.reshape(-1, 2)
        r = compute_radius(centers)
        
        if r > best_r:
            best_r = r
            best_centers = centers
            
        # Early exit if we hit a very good score (close to theoretical max)
        if best_r > 0.07140:
            break

    if best_centers is None:
        return grid_5x10
    return best_centers
# EVOLVE-BLOCK-END
