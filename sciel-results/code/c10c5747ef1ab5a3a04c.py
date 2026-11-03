"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist
import time


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses a smooth least-squares objective with r as a variable to maximize packing radius.
    """
    n = N
    rng = np.random.default_rng(42)
    
    # Generate initial hexagonal lattice configuration
    best_centres = None
    best_r = 0.0
    
    for attempt in range(5):
        # Create hexagonal lattice with slight perturbation
        spacing = 2.0 / (np.sqrt(n) + 1.0) * (0.95 + 0.05 * attempt)
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 0.95
        points = points[mask]
        
        if len(points) < n:
            # Fallback: use random points in disk
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0, 0.9, n)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Add small random perturbation to break symmetry
            points = points[:n] + rng.normal(0, spacing * 0.1, size=(n, 2))
            # Project back inside unit disk
            dists = np.linalg.norm(points, axis=1)
            too_far = dists > 0.95
            if np.any(too_far):
                points[too_far] *= 0.95 / dists[too_far]
        
        # Initial r estimate
        dists = pdist(points)
        min_pair_dist = np.min(dists) if len(dists) > 0 else 1.0
        boundary_dists = 1.0 - np.linalg.norm(points, axis=1)
        min_boundary_dist = np.min(boundary_dists)
        r0 = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if r0 <= 0:
            continue
        
        # Variables: [x_1, y_1, ..., x_n, y_n, r]
        x0 = np.concatenate([points.flatten(), [r0]])
        
        def residuals(x):
            centres = x[:2*n].reshape(n, 2)
            r = x[2*n]
            
            # Pairwise distances
            dists = pdist(centres)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            # Residuals: want d_ij >= 2r and boundary_dist >= r
            # Penalize violations with smooth max
            pair_res = np.maximum(0, 2 * r - dists)
            boundary_res = np.maximum(0, r - boundary_dists)
            
            return np.concatenate([pair_res, boundary_res])
        
        # Run optimization
        try:
            result = least_squares(residuals, x0, method='trf', max_nfev=20000,
                                  xtol=1e-10, ftol=1e-10, gtol=1e-10,
                                  bounds=([-1.0]*2*n + [0.0], [1.0]*2*n + [0.5]))
            
            centres = result.x[:2*n].reshape(n, 2)
            r_opt = result.x[2*n]
            
            # Compute actual radius
            dists = pdist(centres)
            min_pair_dist = np.min(dists)
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            min_boundary_dist = np.min(boundary_dists)
            r_actual = min(min_pair_dist / 2.0, min_boundary_dist)
            
            # Check validity
            if r_actual > 0 and np.all(np.isfinite(centres)):
                if r_actual > best_r:
                    best_r = r_actual
                    best_centres = centres.copy()
        except Exception:
            continue
    
    # Fallback if optimization failed
    if best_centres is None or best_r <= 0:
        spacing = 2.0 / (np.sqrt(n) + 1.0)
        while True:
            k = int(1.0 / spacing) + 2
            rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                    for i in range(-k, k + 1) for j in range(-k, k + 1)]
            points = np.array(rows)
            radius = np.hypot(points[:, 0], points[:, 1])
            points = points[radius <= 1.0 - spacing / 2.0]
            if len(points) >= n:
                best_centres = points[:n]
                break
            spacing *= 0.98
    
    return best_centres
# EVOLVE-BLOCK-END
