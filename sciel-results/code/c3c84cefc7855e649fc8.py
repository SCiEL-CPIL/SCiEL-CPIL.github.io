"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import least_squares


def _hex_init(n, seed=0):
    """Generate n points from a hexagonal lattice inside the unit disk."""
    rng = np.random.RandomState(seed)
    # Try different spacings
    for attempt in range(20):
        spacing = 0.055 + rng.random() * 0.02
        k = int(2.0 / spacing) + 3
        xs, ys = np.meshgrid(
            np.arange(-k, k + 1) * spacing,
            np.arange(-k, k + 1) * spacing * np.sqrt(3.0) / 2.0
        )
        # Offset odd rows
        for j in range(2 * k + 1):
            if j % 2 == 1:
                xs[j, :] += spacing / 2.0
        pts = np.column_stack([xs.ravel(), ys.ravel()])
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r <= 1.0 - spacing / 2.0 - 0.001
        pts = pts[mask]
        if len(pts) >= n:
            # Select n points, preferring those closer to center
            idx = np.argsort(r[mask])[:n]
            return pts[idx].copy()
    # Fallback: random points
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = rng.uniform(0, 0.8, n)
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _compute_radius(pts):
    """Compute the maximum radius for given points."""
    n = len(pts)
    if n < 2:
        return min(1.0 - np.max(np.hypot(pts[:, 0], pts[:, 1])), 0.5)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to boundary
    boundary_dists = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(boundary_dists)
    
    return min(min_pair_dist / 2.0, min_boundary)


def _residuals(x, n):
    """
    Residuals for least_squares: we want to maximize the minimum of
    (pairwise distances / 2) and (distance to boundary).
    
    We minimize the negative of these, using a smooth approximation.
    The idea: for each pair (i,j), residual = -(d_ij/2 - target)
    where target is a lower bound. But simpler: use soft-min.
    
    Actually, let's use a direct approach:
    - For each pair, residual = -(d_ij - 2*r_target) where we want d_ij >= 2*r_target
    - For each point, residual = -(1 - |c_i| - r_target)
    
    But r_target is unknown. Instead, use the formulation where we
    minimize violations of the constraint that all distances are at least some value.
    
    Better approach: minimize the negative soft-min of all distances.
    soft_min = -logsumexp(-log(d_i)) for positive d_i
    But this is complex. Let's use a simpler penalty-based approach.
    
    We'll minimize: sum over pairs of max(0, 2*r - d_ij)^2 + sum over i of max(0, r - (1-|c_i|))^2
    where r is estimated as the current min radius.
    """
    pts = x.reshape(n, 2)
    n = len(pts)
    
    # Estimate current radius
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(boundary)
    r_current = min(min_pair / 2.0, min_boundary)
    
    # Target: slightly above current to push improvement
    r_target = r_current * 1.01 + 0.0001
    
    # Pairwise residuals: penalize if distance < 2*r_target
    upper = np.triu_indices(n, k=1)
    pair_dists = dists[upper]
    pair_resids = np.maximum(0, 2 * r_target - pair_dists)
    
    # Boundary residuals: penalize if 1 - |c_i| < r_target
    boundary_resids = np.maximum(0, r_target - boundary)
    
    return np.concatenate([pair_resids, boundary_resids])


def construct():
    n = N
    deadline = time.time() + 130  # 130s to leave margin before 150s kill
    
    # Get initial hexagonal packing
    x0 = _hex_init(n, seed=42)
    best_pts = x0.copy()
    best_r = _compute_radius(best_pts)
    
    # Multiple restarts with different initializations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 30:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = _hex_init(n, seed=restart * 100 + 7)
        
        # Run least_squares
        try:
            result = least_squares(
                _residuals,
                x_init.flatten(),
                args=(n,),
                method='trf',
                max_nfev=2000,
                xtol=1e-8,
                ftol=1e-8,
                gtol=1e-8
            )
            
            pts = result.x.reshape(n, 2)
            r_val = _compute_radius(pts)
            
            if r_val > best_r:
                best_r = r_val
                best_pts = pts.copy()
        except Exception:
            continue
        
        if time.time() > deadline - 15:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
