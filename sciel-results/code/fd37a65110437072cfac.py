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
    Uses a smooth non-linear least-squares approach to maximize the minimum radius.
    """
    n = N
    
    def hexagonal_initial(n, margin=0.05):
        """Generate hexagonal lattice points inside unit circle with margin."""
        # Estimate spacing from area
        # Area per circle ~ pi*r^2, total area ~ pi*1^2
        # For n circles: r ~ 1/sqrt(n) roughly
        r_est = 1.0 / np.sqrt(n)
        spacing = 2.0 * r_est * 0.95  # slight shrink for margin
        
        k = int(1.0 / spacing) + 3
        points = []
        for j in range(-k, k + 1):
            y = j * spacing * np.sqrt(3.0) / 2.0
            for i in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                if np.hypot(x, y) <= 1.0 - r_est - margin:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) < n:
            # Fallback: denser grid
            spacing *= 0.9
            k = int(1.0 / spacing) + 3
            points = []
            for j in range(-k, k + 1):
                y = j * spacing * np.sqrt(3.0) / 2.0
                for i in range(-k, k + 1):
                    x = (i + 0.5 * (j % 2)) * spacing
                    if np.hypot(x, y) <= 1.0 - r_est - margin:
                        points.append([x, y])
            points = np.array(points)
        
        if len(points) < n:
            # Final fallback: uniform random
            angles = np.random.uniform(0, 2*np.pi, n)
            radii = np.sqrt(np.random.uniform(0, 1, n)) * (1.0 - r_est)
            points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Sort by distance from origin and take closest n
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists, kind='stable')[:n]
            points = points[idx]
        
        return points
    
    def compute_radius(centres):
        """Compute the maximum common radius for given centres."""
        if len(centres) < 2:
            return 1.0 - np.linalg.norm(centres, axis=1).min()
        dists = pdist(centres)
        min_pair = np.min(dists) / 2.0
        min_boundary = np.min(1.0 - np.linalg.norm(centres, axis=1))
        return min(min_pair, min_boundary)
    
    def smooth_max(x, beta=50.0):
        """Smooth approximation of max(0, x) using softplus-like function."""
        # Use a smooth approximation: (x + sqrt(x^2 + eps)) / 2
        eps = 1e-6
        return np.where(x > 0, x, (x + np.sqrt(x**2 + eps)) / 2.0)
    
    def optimize_from_start(x0, r_target, max_nfev=2000):
        """Run least_squares optimization from initial point x0 with target radius r_target."""
        n = len(x0) // 2
        
        def residuals(x):
            centres = x.reshape(n, 2)
            
            # Compute pairwise distances
            dists = pdist(centres)
            
            # Boundary distances
            boundary_dists = 1.0 - np.linalg.norm(centres, axis=1)
            
            # Deficits from target radius
            pair_deficits = np.maximum(0, 2 * r_target - dists)
            boundary_deficits = np.maximum(0, r_target - boundary_dists)
            
            # Use smooth version for better gradients
            pair_smooth = smooth_max(2 * r_target - dists, beta=50.0)
            boundary_smooth = smooth_max(r_target - boundary_dists, beta=50.0)
            
            return np.concatenate([pair_smooth, boundary_smooth])
        
        bounds = (np.full(2*n, -1.0), np.full(2*n, 1.0))
        
        try:
            result = least_squares(residuals, x0, method='trf', bounds=bounds,
                                  max_nfev=max_nfev, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            return result.x[:2*n].reshape(n, 2), result.cost
        except Exception:
            return x0.reshape(n, 2), np.inf
    
    # Generate initial configuration
    x0 = hexagonal_initial(n)
    
    # Compute initial radius
    r_current = compute_radius(x0)
    
    # Try to improve with iterative target radius increase
    best_centres = x0.copy()
    best_r = r_current
    
    # Multi-start with different initial configurations
    starts = [x0]
    
    # Add a few perturbed versions
    for trial in range(3):
        np.random.seed(42 + trial)
        x_pert = x0 + np.random.normal(0, 0.01, x0.shape)
        # Ensure inside unit circle
        norms = np.linalg.norm(x_pert, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            x_pert[mask] *= 0.95 / norms[mask, None]
        starts.append(x_pert)
    
    # Iterate: try to increase target radius
    r_target = r_current * 0.95  # Start slightly below current
    max_iterations = 10
    iter_count = 0
    
    for start_idx, start_centres in enumerate(starts):
        r_t = r_current * 0.95
        for _ in range(max_iterations):
            iter_count += 1
            centres, cost = optimize_from_start(start_centres.flatten(), r_t, max_nfev=1500)
            
            r_new = compute_radius(centres)
            
            if r_new > best_r:
                best_r = r_new
                best_centres = centres.copy()
            
            # If we successfully met the target, increase it
            if cost < 1e-8:
                r_t = min(r_t * 1.02, 0.06)  # Cap at reasonable value
            else:
                # If we couldn't meet target, try slightly lower
                r_t = max(r_t * 0.98, r_current * 0.9)
                break
        
        # Update start for next iteration with best result
        if best_r > r_current:
            r_current = best_r
    
    return best_centres
# EVOLVE-BLOCK-END
