"""
Improved packing of 50 equal spheres in a unit cube.

Strategy:
1. Start with BCC lattice or grid initial configurations.
2. Use smooth surrogate objective (log-sum-exp soft-min) with gradient-based optimization.
3. Use scipy L-BFGS-B for efficient convergence.
4. Multiple restarts with perturbation to escape local minima.
"""

import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE_SECONDS = 115  # Safety margin before 120s limit


def _score(points):
    """Compute the maximum valid radius for the given sphere centers."""
    if points.shape != (N, 3):
        return 0.0
    if not np.all(np.isfinite(points)):
        return 0.0
    
    # Distance between all pairs
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    
    # Distance to walls: cube is [-0.5, 0.5]^3
    wall_dists = 0.5 - np.abs(points)
    min_wall = np.min(wall_dists)
    
    return min(min_pair, min_wall)


def _soft_min_energy(x, beta=50.0):
    """
    Smooth surrogate: negative log-sum-exp of pair distances and wall distances.
    Minimizing this maximizes the soft-min of all constraints.
    x is flattened (N*3,).
    """
    pts = x.reshape(N, 3)
    
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)  # (N, 3)
    
    # Combine all constraint values
    pair_vals = dists[np.triu_indices(N, k=1)]
    wall_vals = wall_dists.ravel()
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft min: -log-sum-exp(-beta * vals) / beta
    max_val = np.max(all_vals)
    exp_vals = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_vals))
    soft_min = max_val - np.log(len(all_vals)) / beta - np.log(exp_vals.sum()) / beta
    
    # We want to maximize soft_min, so minimize negative
    return -soft_min


def _soft_min_grad(x, beta=50.0):
    """Gradient of the soft-min energy."""
    pts = x.reshape(N, 3)
    
    # Pair distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, 1e10)
    dists = np.sqrt(dists_sq)
    
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    
    # Combine all constraint values
    pair_vals = dists[np.triu_indices(N, k=1)]
    wall_vals = wall_dists.ravel()
    all_vals = np.concatenate([pair_vals, wall_vals])
    
    # Soft min gradient
    max_val = np.max(all_vals)
    exp_vals = np.exp(-beta * (all_vals - max_val))
    weights = exp_vals / np.sum(exp_vals)
    
    grad = np.zeros(N * 3)
    
    # Pair distance gradients
    i_idx, j_idx = np.triu_indices(N, k=1)
    for k in range(len(i_idx)):
        i, j = i_idx[k], j_idx[k]
        d = pts[i] - pts[j]
        dist = dists[i, j]
        if dist > 1e-12:
            # d(dist)/d(pts[i]) = d/dist, d(dist)/d(pts[j]) = -d/dist
            grad[i*3:(i+1)*3] += weights[k] * (-beta) * d / dist
            grad[j*3:(j+1)*3] += weights[k] * (-beta) * (-d) / dist
    
    # Wall distance gradients
    wall_flat_idx = 0
    for i in range(N):
        for dim in range(3):
            # wall_dist = 0.5 - |pts[i,dim]|
            # d(wall_dist)/d(pts[i,dim]) = -sign(pts[i,dim])
            sign_val = np.sign(pts[i, dim])
            if sign_val == 0:
                sign_val = 1.0
            grad[i*3+dim] += weights[wall_flat_idx] * (-beta) * (-sign_val)
            wall_flat_idx += 1
    
    return grad


def _initial_bcc(n):
    """Create initial configuration from BCC lattice."""
    # BCC lattice points in a unit cell: corners + center
    # Generate a 3D grid of BCC unit cells
    m = int(np.ceil(n ** (1.0 / 3.0)))
    
    # Generate BCC points
    points = []
    for ix in range(m):
        for iy in range(m):
            for iz in range(m):
                # Corner point
                cx, cy, cz = ix, iy, iz
                points.append([cx, cy, cz])
                # Center point (offset by 0.5 in each direction)
                if ix < m - 1 or iy < m - 1 or iz < m - 1:
                    points.append([cx + 0.5, cy + 0.5, cz + 0.5])
    
    points = np.array(points)
    
    # Scale and center to fit in [-0.5, 0.5]^3
    # The BCC grid spans [0, m-1] in each dimension (approximately)
    # We want to fit n points
    if len(points) > n:
        # Randomly select n points
        idx = np.random.permutation(len(points))
        points = points[idx[:n]]
    elif len(points) < n:
        # Add random points
        extra = np.random.randn(n - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    
    # Scale to fit in [-0.5, 0.5]^3 with some margin
    # Find bounding box
    min_coords = np.min(points, axis=0)
    max_coords = np.max(points, axis=0)
    span = max_coords - min_coords
    max_span = np.max(span)
    
    if max_span > 0:
        scale = 0.8 / max_span  # 80% of cube for margin
        points = (points - (min_coords + max_coords) / 2) * scale
    
    # Clip to ensure inside cube
    points = np.clip(points, -0.45, 0.45)
    
    return points


def _initial_grid(n):
    """Create an initial grid-like configuration."""
    m = int(np.ceil(n ** (1.0 / 3.0)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    
    if len(grid) >= n:
        idx = np.random.permutation(len(grid))
        return grid[idx[:n]].copy()
    else:
        # Pad with random points
        extra = np.random.randn(n - len(grid), 3) * 0.3
        return np.vstack([grid, extra])


def _optimize_lbgfsb(points, beta=50.0, max_time=30.0):
    """Optimize using L-BFGS-B on the soft-min surrogate."""
    start_time = time.time()
    x0 = points.ravel()
    
    bounds = [(-0.5, 0.5)] * (N * 3)
    
    def objective(x):
        return _soft_min_energy(x, beta)
    
    def gradient(x):
        return _soft_min_grad(x, beta)
    
    result = minimize(
        objective,
        x0,
        jac=gradient,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'maxfun': 5000, 'ftol': 1e-12, 'gtol': 1e-8}
    )
    
    opt_points = result.x.reshape(N, 3)
    opt_points = np.clip(opt_points, -0.5, 0.5)
    
    return opt_points


def construct():
    """Construct the sphere centers."""
    start_time = time.time()
    
    best_points = None
    best_score = 0.0
    
    n_restarts = 0
    while time.time() - start_time < DEADLINE_SECONDS:
        n_restarts += 1
        elapsed = time.time() - start_time
        remaining = DEADLINE_SECONDS - elapsed
        
        if remaining < 5:
            break
        
        # Alternate between BCC and grid initialization
        if n_restarts % 2 == 1:
            points = _initial_bcc(N)
        else:
            points = _initial_grid(N)
        
        # Add small random perturbation to break symmetry
        points += np.random.randn(N, 3) * 0.01
        points = np.clip(points, -0.49, 0.49)
        
        # Optimize with L-BFGS-B
        time_per_opt = min(remaining - 2, 40)  # Leave 2s safety margin
        if time_per_opt < 5:
            break
            
        opt_points = _optimize_lbgfsb(points, beta=50.0, max_time=time_per_opt)
        
        # Polish with higher beta
        if time.time() - start_time < DEADLINE_SECONDS - 3:
            opt_points2 = _optimize_lbgfsb(opt_points, beta=100.0, max_time=10.0)
            opt_points = opt_points2
        
        opt_score = _score(opt_points)
        
        if opt_score > best_score:
            best_score = opt_score
            best_points = opt_points.copy()
        
        # If we've done enough restarts, break
        if n_restarts >= 10:
            break
    
    if best_points is None:
        best_points = _initial_grid(N)
    
    return best_points


# For quick testing
if __name__ == "__main__":
    centers = construct()
    print(f"Score: {_score(centers)}")
    print(f"Shape: {centers.shape}")
