"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.distance import pdist


def construct():
    """
    Construct centres of 300 equal circles inside the unit circle.
    Uses scipy.optimize.least_squares to maximize the minimum radius.
    """
    n = N
    
    # Estimate initial radius based on hexagonal packing density
    # Density ~ 0.9069
    # N * pi * r^2 ~ 0.9069 * pi * 1^2
    # r ~ sqrt(0.9069 / N)
    r_est = np.sqrt(0.9069 / n)
    s = 2.0 * r_est
    
    # Generate hexagonal lattice points
    k = int(1.0 / s) + 2
    points = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = i * s + (0.5 * s if j % 2 != 0 else 0)
            y = j * s * np.sqrt(3.0) / 2.0
            # Keep points well inside to allow for optimization
            if x*x + y*y <= (1.0 - r_est)**2:
                points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        # Fallback: reduce spacing
        s *= 0.9
        k = int(1.0 / s) + 2
        points = []
        for j in range(-k, k + 1):
            for i in range(-k, k + 1):
                x = i * s + (0.5 * s if j % 2 != 0 else 0)
                y = j * s * np.sqrt(3.0) / 2.0
                if x*x + y*y <= (1.0 - r_est)**2:
                    points.append([x, y])
        points = np.array(points)
        
    if len(points) >= n:
        # Select n points closest to origin
        dists_to_origin = np.linalg.norm(points, axis=1)
        indices = np.argsort(dists_to_origin, kind='stable')[:n]
        init_points = points[indices]
    else:
        # Should not happen for N=300
        init_points = np.random.uniform(-0.9, 0.9, size=(n, 2))
        
    x0_points = init_points.flatten()
    x0_r = r_est
    x0 = np.concatenate([x0_points, [x0_r]])
    
    def residuals(x):
        centres = x[:2*n].reshape(n, 2)
        r = x[2*n]
        
        dists = pdist(centres)
        norms = np.linalg.norm(centres, axis=1)
        boundary_dists = 1.0 - norms
        
        pair_residuals = 2 * r - dists
        boundary_residuals = r - boundary_dists
        
        return np.concatenate([pair_residuals, boundary_residuals])
    
    lower = np.concatenate([np.full(2*n, -1.0), [0.0]])
    upper = np.concatenate([np.full(2*n, 1.0), [1.0]])
    
    result = least_squares(residuals, x0, bounds=(lower, upper), method='trf', 
                           max_nfev=50000, xtol=1e-9, ftol=1e-9, gtol=1e-9)
    
    centres = result.x[:2*n].reshape(n, 2)
    r_opt = result.x[2*n]
    
    dists = pdist(centres)
    min_pair_dist = np.min(dists)
    norms = np.linalg.norm(centres, axis=1)
    min_boundary_dist = np.min(1.0 - norms)
    
    r_actual = min(min_pair_dist / 2.0, min_boundary_dist)
    
    if not np.isfinite(r_actual) or r_actual <= 0:
        dists_init = pdist(init_points)
        min_pair_init = np.min(dists_init)
        norms_init = np.linalg.norm(init_points, axis=1)
        min_boundary_init = np.min(1.0 - norms_init)
        r_actual = min(min_pair_init / 2.0, min_boundary_init)
        centres = init_points
        
    return centres
# EVOLVE-BLOCK-END
