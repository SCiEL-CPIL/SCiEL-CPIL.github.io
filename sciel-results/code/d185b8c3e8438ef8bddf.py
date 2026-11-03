"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to minimize a potential energy function
    that approximates maximizing the minimum circle radius.
    """
    n = N
    
    def generate_hex_lattice(n_points, max_radius=0.95):
        spacing = 0.105
        points = []
        range_val = int(1.0 / spacing) + 2
        
        for j in range(-range_val, range_val + 1):
            for i in range(-range_val, range_val + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3) / 2
                if np.hypot(x, y) <= max_radius:
                    points.append((x, y))
        
        points = np.array(points)
        
        if len(points) > n_points:
            indices = np.random.choice(len(points), n_points, replace=False)
            points = points[indices]
        elif len(points) < n_points:
            while len(points) < n_points:
                angle = np.random.uniform(0, 2*np.pi)
                r = np.random.uniform(0, max_radius)
                points = np.vstack([points, [r*np.cos(angle), r*np.sin(angle)]])
        
        return points

    # Initial point
    x0 = generate_hex_lattice(n).flatten()
    x0 += np.random.normal(0, 0.01, x0.shape)
    
    def residuals(x):
        centres = x.reshape(n, 2)
        
        # Calculate pairwise distances
        diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
        dists_sq = np.sum(diff**2, axis=2)
        
        # Mask diagonal
        np.fill_diagonal(dists_sq, 1.0)
        
        # Clip to avoid division by zero
        dists_sq = np.maximum(dists_sq, 1e-6)
        
        # Sum of inverse squared distances
        # We only need the upper triangle to avoid double counting, 
        # but summing all is fine for the gradient direction.
        pair_energy = np.sum(1.0 / dists_sq)
        
        # Boundary energy
        norms = np.linalg.norm(centres, axis=1)
        boundary_dists = 1.0 - norms
        boundary_dists = np.maximum(boundary_dists, 1e-6)
        boundary_energy = np.sum(1.0 / boundary_dists**2)
        
        # Return residuals as a vector
        # Scaling might be important. 
        # Pair energy is roughly n^2 / d^2. Boundary is n / b^2.
        # For n=300, pair energy is much larger.
        # Let's normalize or weight them.
        # Typical d ~ 0.1, b ~ 0.1.
        # 1/d^2 ~ 100. Sum ~ 90000.
        # 1/b^2 ~ 100. Sum ~ 30000.
        # They are of similar magnitude.
        
        return np.array([np.sqrt(pair_energy), np.sqrt(boundary_energy)])

    best_r = 0
    best_centres = None
    
    # Run optimization
    # We'll do a few restarts to escape local minima
    for attempt in range(5):
        if attempt > 0:
            if best_centres is not None:
                x0 = best_centres.flatten() + np.random.normal(0, 0.02, 2*n)
            else:
                x0 = generate_hex_lattice(n).flatten() + np.random.normal(0, 0.05, 2*n)
        
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=1000, 
                                   xtol=1e-8, ftol=1e-8, gtol=1e-8)
            
            centres = result.x.reshape(n, 2)
            
            # Calculate actual radius
            diff = centres[:, np.newaxis, :] - centres[np.newaxis, :, :]
            dists_sq = np.sum(diff**2, axis=2)
            np.fill_diagonal(dists_sq, 1.0)
            min_pair_dist_sq = np.min(dists_sq)
            min_pair_dist = np.sqrt(min_pair_dist_sq)
            
            norms = np.linalg.norm(centres, axis=1)
            min_boundary_dist = np.min(1.0 - norms)
            
            r = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if r > best_r:
                best_r = r
                best_centres = centres.copy()
                
        except Exception:
            continue

    if best_centres is None:
        return generate_hex_lattice(n)
        
    return best_centres
# EVOLVE-BLOCK-END
