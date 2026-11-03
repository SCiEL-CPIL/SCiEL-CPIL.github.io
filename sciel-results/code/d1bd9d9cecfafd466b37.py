import numpy as np
from scipy.optimize import minimize
import time

N = 50

# EVOLVE-BLOCK-START
def construct():
    # Deadline guard
    start_time = time.time()
    deadline = start_time + 110.0  # Leave 10s margin
    
    def hexagonal_initial_guess(spacing):
        # Create a hexagonal grid
        points = []
        # Generate points in a large enough area
        for i in range(-15, 16):
            for j in range(-15, 16):
                x = i * spacing * np.sqrt(3)
                y = j * spacing * 1.5 + (i % 2) * spacing * 0.75 * np.sqrt(3)
                # Wait, standard hex grid:
                # x = i * s * sqrt(3)
                # y = j * s * 1.5 + (i % 2) * s * 0.75 * sqrt(3) ? No.
                # Let's use the standard axial coordinates or just simple offset.
                # x = i * s * sqrt(3)
                # y = j * s * 1.5
                # if i is odd, y += s * 0.75 * sqrt(3) ?
                # Let's stick to the previous logic but fix the spacing.
                # Previous: x = i * spacing * 1.732, y = j * spacing + (i % 2) * spacing * 0.5 * 1.732
                # This is a valid hex grid.
                
                x = i * spacing * np.sqrt(3)
                y = j * spacing + (i % 2) * spacing * 0.5 * np.sqrt(3)
                
                if np.hypot(x, y) < 0.95:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < N:
            # Fallback to random if grid is too sparse
            angles = np.random.uniform(0, 2*np.pi, N)
            radii = np.sqrt(np.random.uniform(0, 1, N)) * 0.9
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
            return points

        # Select N points closest to center
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)[:N]
        return points[indices]

    def energy(x):
        """
        Objective function to minimize.
        Uses exponential soft-min surrogate to approximate max-min distance.
        """
        x = x.reshape(-1, 2)
        
        # Pairwise distances
        diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        
        # Mask out diagonal
        mask = ~np.eye(N, dtype=bool)
        pair_dists = dists[mask]
        
        # Boundary distances
        center_dists = np.sqrt(np.sum(x**2, axis=1))
        boundary_dists = 1.0 - center_dists
        
        # Exponential penalty
        # Large k makes it behave like min
        k = 20.0
        pair_energy = np.sum(np.exp(-k * pair_dists))
        boundary_energy = np.sum(np.exp(-k * boundary_dists))
        
        return pair_energy + boundary_energy

    best_centers = None
    best_radius = 0.0
    
    # Initial guess
    # Start with a spacing that is roughly the expected optimal diameter
    # r ~ 0.126, so d ~ 0.252
    x0 = hexagonal_initial_guess(0.25)
    
    # Run optimization with multiple restarts/jitters
    # L-BFGS-B is good for this
    options = {
        'maxiter': 300,
        'ftol': 1e-12,
        'gtol': 1e-10
    }
    
    # We will try a few variations
    num_restarts = 10
    
    for i in range(num_restarts):
        if time.time() > deadline:
            break
            
        # Jitter the initial guess
        # Small random perturbation
        jitter = np.random.normal(0, 0.02, x0.shape)
        x_init = x0 + jitter
        
        # Ensure initial points are not exactly on top of each other
        # L-BFGS-B might struggle if starting from exact duplicates, but jitter helps
        
        try:
            res = minimize(energy, x_init, method='L-BFGS-B', options=options)
            x_opt = res.x.reshape(-1, 2)
            
            # Calculate the actual radius for this configuration
            # r = min( min_{i<j} d_ij / 2, min_i (1 - ||x_i||) )
            
            # Pairwise distances
            diff = x_opt[:, np.newaxis, :] - x_opt[np.newaxis, :, :]
            dists = np.sqrt(np.sum(diff**2, axis=-1))
            np.fill_diagonal(dists, np.inf)
            min_pair_dist = np.min(dists)
            
            # Boundary distances
            center_dists = np.sqrt(np.sum(x_opt**2, axis=1))
            min_boundary_dist = np.min(1.0 - center_dists)
            
            current_radius = min(min_pair_dist / 2.0, min_boundary_dist)
            
            # Check validity: all points must be inside
            if np.all(center_dists <= 1.0) and current_radius > best_radius:
                best_radius = current_radius
                best_centers = x_opt
                
        except Exception:
            continue

    # If no valid center found (should not happen), return the initial guess
    if best_centers is None:
        best_centers = x0
        # Calculate radius for x0 just in case
        diff = best_centers[:, np.newaxis, :] - best_centers[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff**2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        center_dists = np.sqrt(np.sum(best_centers**2, axis=1))
        min_boundary_dist = np.min(1.0 - center_dists)
        best_radius = min(min_pair_dist / 2.0, min_boundary_dist)

    return best_centers
# EVOLVE-BLOCK-END
