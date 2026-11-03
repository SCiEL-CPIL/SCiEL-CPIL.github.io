"""Circle packing in unit square - Delaunay-based approach."""
N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import Delaunay
from scipy.optimize import minimize
import itertools


def _compute_radii(centers):
    """
    Given center positions, compute radii such that circles are:
    - Disjoint (interiors don't overlap)
    - Inside [0,1]^2
    - As large as possible (tangent to neighbors or boundary)
    
    Strategy: For each circle, its radius is limited by:
    1. Distance to nearest neighbor / 2
    2. Distance to boundary (min(x, 1-x, y, 1-y))
    
    We use an iterative approach: start with small radii, then grow them
    while maintaining constraints.
    
    Actually, a better approach: for each pair of adjacent circles (in Delaunay),
    the sum of radii should be <= distance between centers.
    For boundary constraints, radius <= min(x, 1-x, y, 1-y).
    
    We solve this by setting radii as the minimum of all constraints.
    """
    n = len(centers)
    radii = np.zeros(n)
    
    # Compute pairwise distances
    dists = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dists[i, j] = np.sqrt(dx*dx + dy*dy)
    
    # Boundary distances
    for i in range(n):
        bd = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
        # Radius limited by boundary
        radii[i] = bd
    
    # Now enforce pairwise constraints: r_i + r_j <= dist(i,j)
    # This is a system of linear inequalities. We can solve iteratively.
    # Simple iterative approach:
    for iteration in range(100):
        updated = False
        for i in range(n):
            # Update based on boundary
            bd = min(centers[i][0], 1 - centers[i][0], centers[i][1], 1 - centers[i][1])
            if radii[i] > bd:
                radii[i] = bd
                updated = True
            
            # Update based on neighbors
            for j in range(n):
                if i == j:
                    continue
                # r_i + r_j <= dist(i,j) => r_i <= dist(i,j) - r_j
                max_r = dists[i, j] - radii[j]
                if radii[i] > max_r:
                    radii[i] = max(0, max_r)
                    updated = True
        if not updated:
            break
    
    # Ensure non-negative
    radii = np.maximum(radii, 0)
    
    return radii


def _objective(centers_flat):
    """
    Objective to minimize: negative sum of radii.
    centers_flat is a flat array of shape (2*n,)
    """
    centers = centers_flat.reshape(-1, 2)
    
    # Keep centers in [0,1]^2 with small margin
    centers = np.clip(centers, 0.01, 0.99)
    
    radii = _compute_radii(centers)
    
    # Penalize if any radius is negative (shouldn't happen)
    total = np.sum(radii)
    return -total


def _run_optimization(initial_centers, max_iter=50):
    """
    Run optimization starting from initial_centers.
    Returns optimized centers and radii.
    """
    n = len(initial_centers)
    initial_flat = initial_centers.flatten()
    
    best_centers = initial_centers.copy()
    best_radii = _compute_radii(initial_centers)
    best_value = np.sum(best_radii)
    
    for it in range(max_iter):
        # Try to optimize
        try:
            result = minimize(
                _objective,
                initial_flat,
                method='L-BFGS-B',
                bounds=[(0.01, 0.99)] * (2 * n),
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            new_centers = result.x.reshape(-1, 2)
            new_radii = _compute_radii(new_centers)
            new_value = np.sum(new_radii)
            
            if new_value > best_value:
                best_centers = new_centers.copy()
                best_radii = new_radii.copy()
                best_value = new_value
                initial_flat = result.x.copy()
        except Exception:
            pass
        
        # Perturbation to escape local minima
        if it % 5 == 4:
            noise = np.random.randn(2 * n) * 0.01
            initial_flat = initial_flat + noise
            initial_flat = np.clip(initial_flat, 0.01, 0.99)
    
    return best_centers, best_radii, best_value


def _generate_initial_config(n):
    """Generate an initial configuration of n points in the unit square."""
    np.random.seed(42)
    
    # Start with a grid-like arrangement
    side = int(np.ceil(np.sqrt(n)))
    centers = []
    for i in range(side):
        for j in range(side):
            if len(centers) >= n:
                break
            x = (i + 0.5) / side + np.random.randn() * 0.05
            y = (j + 0.5) / side + np.random.randn() * 0.05
            x = np.clip(x, 0.1, 0.9)
            y = np.clip(y, 0.1, 0.9)
            centers.append([x, y])
    
    centers = np.array(centers[:n])
    return centers


def construct():
    """
    Pack N circles with disjoint interiors inside [0,1]^2 and maximize sum of radii.
    Returns array of shape (N, 3): rows (x, y, radius)
    """
    n = N
    
    # Generate initial configuration
    initial_centers = _generate_initial_config(n)
    
    # Run optimization
    best_centers, best_radii, best_value = _run_optimization(initial_centers, max_iter=30)
    
    # Verify validity and adjust if needed
    # Check all circles are inside [0,1]^2 and disjoint
    valid = True
    for i in range(n):
        x, y, r = best_centers[i][0], best_centers[i][1], best_radii[i]
        if x - r < -1e-6 or x + r > 1 + 1e-6 or y - r < -1e-6 or y + r > 1 + 1e-6:
            valid = False
            break
    
    if valid:
        for i in range(n):
            for j in range(i + 1, n):
                dx = best_centers[i][0] - best_centers[j][0]
                dy = best_centers[i][1] - best_centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_radii[i] + best_radii[j] - 1e-6:
                    valid = False
                    break
            if not valid:
                break
    
    if not valid:
        # Fallback: shrink all radii slightly
        best_radii = best_radii * 0.95
        # Re-check
        for i in range(n):
            for j in range(i + 1, n):
                dx = best_centers[i][0] - best_centers[j][0]
                dy = best_centers[i][1] - best_centers[j][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < best_radii[i] + best_radii[j] - 1e-6:
                    # Further shrink
                    scale = dist / (best_radii[i] + best_radii[j]) if (best_radii[i] + best_radii[j]) > 0 else 1.0
                    best_radii[i] *= scale * 0.99
                    best_radii[j] *= scale * 0.99
    
    # Clamp to ensure validity
    for i in range(n):
        x, y = best_centers[i]
        # Ensure inside square
        best_radii[i] = min(best_radii[i], x, 1 - x, y, 1 - y)
        best_radii[i] = max(best_radii[i], 0)
    
    # Final pairwise check with tiny epsilon shrink
    for i in range(n):
        for j in range(i + 1, n):
            dx = best_centers[i][0] - best_centers[j][0]
            dy = best_centers[i][1] - best_centers[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            required = best_radii[i] + best_radii[j]
            if required > dist:
                # Scale down both
                if required > 0:
                    scale = (dist - 1e-9) / required
                    best_radii[i] *= scale
                    best_radii[j] *= scale
    
    # Build result
    result = np.zeros((n, 3))
    for i in range(n):
        result[i, 0] = best_centers[i][0]
        result[i, 1] = best_centers[i][1]
        result[i, 2] = best_radii[i]
    
    return result
# EVOLVE-BLOCK-END
