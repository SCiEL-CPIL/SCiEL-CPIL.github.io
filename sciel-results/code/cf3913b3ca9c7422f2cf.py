"""Improved packing for 200 equal circles in a unit square."""
import time
import numpy as np
from scipy.optimize import minimize

N = 200
DEADLINE = 150.0  # seconds
SAFETY_MARGIN = 5.0  # seconds to leave before deadline


def _compute_radius(centers):
    """Compute the maximum common radius for given centers.
    
    Returns (radius, min_pair_dist, min_boundary_dist)
    """
    # Pairwise distances
    # Use a more efficient approach: compute all pairwise distances
    # For 200 points, that's ~20000 pairs, which is fine
    n = len(centers)
    
    # Distance matrix (upper triangle)
    # centers has shape (n, 2)
    # Compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances: min over each point of (0.5 - |x|) and (0.5 - |y|)
    # For each point, the distance to the nearest boundary
    bx = 0.5 - np.abs(centers[:, 0])
    by = 0.5 - np.abs(centers[:, 1])
    boundary_dists = np.minimum(bx, by)
    min_boundary = np.min(boundary_dists)
    
    radius = min(min_pair / 2.0, min_boundary)
    return radius, min_pair, min_boundary


def _neg_radius(centers_flat):
    """Negative radius objective for minimization.
    
    Args:
        centers_flat: flat array of length 2*N with [x0, y0, x1, y1, ...]
    Returns:
        Negative of the maximum common radius
    """
    centers = centers_flat.reshape(N, 2)
    r, _, _ = _compute_radius(centers)
    return -r


def _hexagonal_init(n_points):
    """Generate a hexagonal (close-packed) initialization for n_points circles.
    
    This provides a good starting point where points are roughly evenly spaced.
    """
    # Hexagonal lattice spacing
    # We want to fit n_points in a unit square [-0.5, 0.5]^2
    # Hexagonal packing: rows offset by half spacing
    # Try to find a good spacing
    
    # Estimate number of points per row
    # For a hexagonal lattice with spacing s:
    # Each row has ~1/s points, rows are spaced s*sqrt(3)/2 apart
    # Total points ~ (1/s) * (1/(s*sqrt(3)/2)) = 2/(s^2 * sqrt(3))
    # So s ~ sqrt(2/(n*sqrt(3)))
    
    s_est = np.sqrt(2.0 / (n_points * np.sqrt(3)))
    
    # Generate hexagonal lattice points
    points = []
    row = 0
    while len(points) < n_points:
        # y coordinate for this row
        y = -0.5 + s_est * np.sqrt(3) / 2 * row
        if y > 0.5:
            break
        # x offset for alternating rows
        offset = 0 if row % 2 == 0 else s_est / 2
        x = -0.5 + offset
        col = 0
        while x <= 0.5 and len(points) < n_points:
            points.append([x, y])
            x += s_est
            col += 1
        row += 1
    
    # If we don't have enough, add more with slightly different spacing
    if len(points) < n_points:
        # Fill remaining with a denser grid
        remaining = n_points - len(points)
        # Add points at the bottom
        for i in range(remaining):
            points.append([0.0, 0.0])  # placeholder, will be moved
    
    points = np.array(points[:n_points])
    
    # Center the points roughly in the square
    # Scale and translate to fit in [-0.5, 0.5]^2
    xmin, ymin = points.min(axis=0)
    xmax, ymax = points.max(axis=0)
    if xmax > xmin:
        points[:, 0] = (points[:, 0] - xmin) / (xmax - xmin) - 0.5
    if ymax > ymin:
        points[:, 1] = (points[:, 1] - ymin) / (ymax - ymin) - 0.5
    
    return points


def _grid_init(n_points):
    """Generate a grid initialization for n_points circles."""
    m = int(np.ceil(np.sqrt(n_points)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    return grid[:n_points]


def _local_search(centers, max_iters=50, step_size=0.01):
    """Simple local search to improve the packing.
    
    Args:
        centers: array of shape (N, 2)
        max_iters: maximum number of iterations
        step_size: initial step size
    Returns:
        Improved centers
    """
    best = centers.copy()
    best_r, _, _ = _compute_radius(best)
    
    for it in range(max_iters):
        improved = False
        for i in range(N):
            # Try moving point i in 8 directions
            for dx in [-step_size, 0, step_size]:
                for dy in [-step_size, 0, step_size]:
                    if dx == 0 and dy == 0:
                        continue
                    new_centers = best.copy()
                    new_centers[i, 0] += dx
                    new_centers[i, 1] += dy
                    # Check boundaries
                    if np.any(new_centers[:, 0] < -0.5) or np.any(new_centers[:, 0] > 0.5):
                        continue
                    if np.any(new_centers[:, 1] < -0.5) or np.any(new_centers[:, 1] > 0.5):
                        continue
                    r, _, _ = _compute_radius(new_centers)
                    if r > best_r + 1e-12:
                        best = new_centers
                        best_r = r
                        improved = True
                        break
                if improved:
                    break
        if not improved:
            step_size *= 0.5
            if step_size < 1e-6:
                break
    return best


def construct():
    """Construct the centers of 200 equal circles in the unit square.
    
    Returns:
        Array of shape (N, 2) with the centers
    """
    start_time = time.time()
    
    # Try multiple initializations and optimization strategies
    best_centers = _grid_init(N)
    best_r, _, _ = _compute_radius(best_centers)
    
    # Try hexagonal initialization
    hex_centers = _hexagonal_init(N)
    hex_r, _, _ = _compute_radius(hex_centers)
    if hex_r > best_r:
        best_centers = hex_centers
        best_r = hex_r
    
    # Use L-BFGS-B for optimization
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            result = minimize(
                _neg_radius,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 100, 'maxls': 50, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            if result.success or result.fun < -best_r:
                new_centers = result.x.reshape(N, 2)
                # Clip to square
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r, _, _ = _compute_radius(new_centers)
                if new_r > best_r:
                    best_centers = new_centers
                    best_r = new_r
        except Exception:
            pass
    
    # Local search refinement
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            refined = _local_search(best_centers, max_iters=30, step_size=0.005)
            refined_r, _, _ = _compute_radius(refined)
            if refined_r > best_r:
                best_centers = refined
                best_r = refined_r
        except Exception:
            pass
    
    # Another L-BFGS-B pass from the refined solution
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            result = minimize(
                _neg_radius,
                best_centers.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 50, 'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if result.success or result.fun < -best_r:
                new_centers = result.x.reshape(N, 2)
                new_centers = np.clip(new_centers, -0.5, 0.5)
                new_r, _, _ = _compute_radius(new_centers)
                if new_r > best_r:
                    best_centers = new_centers
                    best_r = new_r
        except Exception:
            pass
    
    # Final local search
    elapsed = time.time() - start_time
    if elapsed < DEADLINE - SAFETY_MARGIN:
        try:
            refined = _local_search(best_centers, max_iters=20, step_size=0.002)
            refined_r, _, _ = _compute_radius(refined)
            if refined_r > best_r:
                best_centers = refined
                best_r = refined_r
        except Exception:
            pass
    
    # Ensure validity: clip to square
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
