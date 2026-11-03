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


def _soft_min_residuals(x, n, r_target, beta=20.0):
    """
    Soft-min based residuals for least_squares.
    We want to maximize min(pairwise_dists/2, boundary_dists).
    
    Approach: minimize violations where distances are below 2*r_target.
    Use a smooth penalty: for each pair, if d_ij < 2*r_target, penalize.
    """
    pts = x.reshape(n, 2)
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    upper = np.triu_indices(n, k=1)
    pair_dists = dists[upper]
    
    # Boundary distances
    boundary = 1.0 - np.hypot(pts[:, 0], pts[:, 1])
    
    # Smooth penalty: penalize when distance < 2*r_target
    # Use: max(0, 2*r_target - d)^2 smoothed with softplus-like
    pair_viol = np.maximum(0, 2 * r_target - pair_dists)
    boundary_viol = np.maximum(0, r_target - boundary)
    
    return np.concatenate([pair_viol, boundary_viol])


def _polish_step(pts, n, r_current, deadline, max_iters=50):
    """Local polish: small perturbations and re-optimization."""
    best_pts = pts.copy()
    best_r = r_current
    
    for it in range(max_iters):
        if time.time() > deadline - 5:
            break
        
        # Add small noise
        scale = max(0.0005, 0.002 * (1 - it / max_iters))
        noise = np.random.randn(n, 2) * scale
        x_perturbed = (best_pts + noise).flatten()
        
        r_target = best_r * 1.001 + 0.00002
        
        try:
            result = least_squares(
                _soft_min_residuals,
                x_perturbed,
                args=(n, r_target),
                method='trf',
                max_nfev=500,
                xtol=1e-10,
                ftol=1e-10,
                gtol=1e-10
            )
            
            new_pts = result.x.reshape(n, 2)
            new_r = _compute_radius(new_pts)
            
            if new_r > best_r:
                best_r = new_r
                best_pts = new_pts.copy()
        except Exception:
            continue
    
    return best_pts, best_r


def construct():
    n = N
    deadline = time.time() + 130  # 130s to leave margin before 150s kill
    
    # Get initial hexagonal packing
    x0 = _hex_init(n, seed=42)
    best_pts = x0.copy()
    best_r = _compute_radius(best_pts)
    
    # Multiple restarts with different initializations
    n_restarts = 4
    for restart in range(n_restarts):
        if time.time() > deadline - 40:
            break
        
        if restart == 0:
            x_init = x0.copy()
        else:
            x_init = _hex_init(n, seed=restart * 100 + 7)
        
        # Run least_squares with adaptive target
        try:
            r_target = best_r * 1.01 + 0.0001
            
            result = least_squares(
                _soft_min_residuals,
                x_init.flatten(),
                args=(n, r_target),
                method='trf',
                max_nfev=3000,
                xtol=1e-9,
                ftol=1e-9,
                gtol=1e-9
            )
            
            pts = result.x.reshape(n, 2)
            r_val = _compute_radius(pts)
            
            if r_val > best_r:
                best_r = r_val
                best_pts = pts.copy()
        except Exception:
            continue
        
        if time.time() > deadline - 20:
            break
    
    # Final polish
    if time.time() < deadline - 10:
        best_pts, best_r = _polish_step(best_pts, n, best_r, deadline)
    
    return best_pts
# EVOLVE-BLOCK-END
