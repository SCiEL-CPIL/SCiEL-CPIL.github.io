"""Program for packing 200 equal circles in a unit square using force-directed relaxation."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def halton_sequence(n, base):
    """Generate Halton sequence in [0, 1)."""
    result = np.zeros(n)
    f = 1.0
    for i in range(n):
        f = 1.0 / base
        idx = i + 1
        while idx > 0:
            result[i] += f * (idx % base)
            idx //= base
            f /= base
    return result

def construct():
    # Initialize with Halton sequence for better space-filling properties
    # Halton bases 2 and 3 are common for 2D
    x_h = halton_sequence(N, 2)
    y_h = halton_sequence(N, 3)
    centers = np.column_stack((x_h, y_h))
    
    # Map to [-0.5, 0.5]
    centers = centers - 0.5
    
    # Clamp to valid range with small margin
    centers = np.clip(centers, -0.499, 0.499)
    
    def objective(x):
        """Negative of the minimum radius (to minimize)."""
        c = x.reshape(-1, 2)
        # Compute all pairwise distances
        diff = c[:, np.newaxis, :] - c[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff ** 2, axis=-1))
        np.fill_diagonal(dists, np.inf)
        min_pair_dist = np.min(dists)
        
        # Distance to boundary
        dx = 0.5 - np.abs(c[:, 0])
        dy = 0.5 - np.abs(c[:, 1])
        min_boundary = np.min(np.minimum(dx, dy))
        
        # Radius is half the minimum of pair distances and boundary distances
        r = 0.5 * min(min_pair_dist, min_boundary)
        return -r
    
    best_centers = centers.copy()
    best_score = objective(centers.flatten())
    
    # Multi-start optimization
    rng = np.random.default_rng(42)
    
    # Try a few different perturbations
    for i in range(5):
        # Perturb the Halton sequence
        perturbation_scale = 0.01 * (1.0 / (i + 1))
        current_centers = centers + rng.normal(0, perturbation_scale, size=centers.shape)
        current_centers = np.clip(current_centers, -0.499, 0.499)
        
        x0 = current_centers.flatten()
        bounds = [(-0.5, 0.5)] * (2 * N)
        
        try:
            result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10})
            
            if result.fun < best_score:
                best_score = result.fun
                best_centers = result.x.reshape(-1, 2)
        except Exception:
            continue
            
    # Final clamp to ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
# EVOLVE-BLOCK-END
