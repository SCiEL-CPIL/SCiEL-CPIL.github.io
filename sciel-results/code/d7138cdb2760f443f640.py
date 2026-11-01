"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def construct():
    """
    Place 50 equal circles in a unit circle to maximize the common radius.
    Uses random initialization + Nelder-Mead optimization of the minimum clearance.
    """
    
    def objective(x):
        """
        Objective function to minimize: negative of the minimum clearance.
        x is a flat array of shape (2*N,) containing coordinates.
        Returns -min(min_pairwise_dist/2, min_boundary_dist)
        """
        pts = x.reshape(N, 2)
        
        # Boundary constraint: distance from each center to origin must be <= 1 - r
        # So boundary clearance for circle i is (1 - |c_i|)
        dists_to_origin = np.linalg.norm(pts, axis=1)
        boundary_clearances = 1.0 - dists_to_origin
        
        # Pairwise distances
        # Compute all pairwise distances efficiently
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dist_matrix = np.linalg.norm(diff, axis=2)
        # Set diagonal to inf so we don't consider self-distances
        np.fill_diagonal(dist_matrix, np.inf)
        min_pairwise_dist = np.min(dist_matrix)
        
        # The radius r is limited by min(min_pairwise_dist/2, min(boundary_clearances))
        # We want to maximize this, so we minimize its negative
        min_clearance = min(min_pairwise_dist / 2.0, np.min(boundary_clearances))
        
        return -min_clearance
    
    # Try multiple random initializations and keep the best
    best_x = None
    best_val = -np.inf
    
    n_trials = 10
    rng = np.random.default_rng(42)
    
    for trial in range(n_trials):
        # Random initialization within the unit disk
        # Use rejection sampling or polar coordinates
        angles = rng.uniform(0, 2 * np.pi, N)
        # Use sqrt for uniform distribution in disk
        radii = rng.uniform(0, 1, N) ** 0.5
        # Scale down a bit to avoid boundary issues initially
        radii *= 0.9
        
        x0 = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)]).flatten()
        
        # Run Nelder-Mead optimization
        try:
            result = minimize(
                objective,
                x0,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'maxfev': 10000,
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'adaptive': True
                }
            )
            
            val = -result.fun  # Convert back to positive clearance
            if val > best_val:
                best_val = val
                best_x = result.x
        except Exception:
            continue
    
    if best_x is None:
        # Fallback: return a simple valid configuration
        # Hexagonal grid fallback
        pts = []
        for i in range(10):
            for j in range(10):
                x = (i - 4.5) * 0.2
                y = (j - 4.5) * 0.2 + (0.1 if i % 2 else 0)
                if x*x + y*y < 0.81:
                    pts.append([x, y])
                if len(pts) >= N:
                    break
            if len(pts) >= N:
                break
        return np.array(pts[:N])
    
    return best_x.reshape(N, 2)
# EVOLVE-BLOCK-END
