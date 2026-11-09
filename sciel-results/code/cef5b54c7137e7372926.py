"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time


def _pairwise_dists(pts):
    """Compute all pairwise distances for N points."""
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dists


def _compute_radius(pts):
    """Compute the radius for a given set of points."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    # Get minimum pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Distance to walls
    min_wall = np.min(np.minimum(0.5 - np.abs(pts), 0.5))
    # Actually need min over all coordinates
    min_wall = np.min(0.5 - np.abs(pts))
    
    return min(min_pair / 2.0, min_wall)


def _energy(pts, power=6, eps=1e-10):
    """
    Inverse-power-sum energy: penalize close pairs heavily.
    We want to maximize the minimum distance, so we minimize this energy.
    """
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    np.fill_diagonal(dists, np.inf)
    
    # Energy from pair interactions
    pair_energy = np.sum(1.0 / (dists + eps) ** power)
    
    # Energy from walls: penalize points too close to walls
    # We want points to stay away from walls
    wall_dist = 0.5 - np.abs(pts)
    wall_energy = np.sum(1.0 / (np.minimum(wall_dist, 1.0) + eps) ** power)
    
    return pair_energy + wall_energy


def _energy_grad(pts, power=6, eps=1e-10):
    """
    Gradient of the inverse-power-sum energy.
    """
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists_sq = np.sum(diff ** 2, axis=-1)
    dists = np.sqrt(dists_sq + eps)
    np.fill_diagonal(dists_sq, 1.0)
    np.fill_diagonal(dists, 1.0)
    
    # Gradient of pair energy: d/dx_i sum_j 1/|x_i - x_j|^p = -p * sum_j (x_i - x_j) / |x_i - x_j|^(p+2)
    # Using: d/dx_i (1/|x_i - x_j|^p) = -p * (x_i - x_j) / |x_i - x_j|^(p+2)
    
    # Compute weights: -p / dists^(p+2)
    weights = -power * diff / (dists_sq * dists)  # This is -p * (x_i - x_j) / |x_i-x_j|^(p+2)
    np.fill_diagonal(weights, 0.0)
    # Wait, let me recompute more carefully
    # weight[i,j] should be -p * (x_i - x_j) / |x_i - x_j|^(p+2)
    # But diff[i,j] = x_i - x_j, so weight = -p * diff[i,j] / dists_sq[i,j]^(1) * dists[i,j]^(-1)
    # = -p * diff[i,j] / (dists_sq[i,j] * dists[i,j])
    # That's what I have. Sum over j for each i.
    
    grad = np.sum(weights, axis=1)
    
    # Wall gradient: d/dx_i 1/(0.5 - |x_i|)^p
    # For x_i > 0: d/dx_i 1/(0.5 - x_i)^p = p / (0.5 - x_i)^(p+1)
    # For x_i < 0: d/dx_i 1/(0.5 + x_i)^p = -p / (0.5 + x_i)^(p+1)
    # Generally: sign(x_i) * p / (0.5 - |x_i|)^(p+1)
    
    wall_dist = 0.5 - np.abs(pts)
    wall_dist_clipped = np.maximum(wall_dist, eps)
    wall_grad = np.sign(pts) * power / wall_dist_clipped ** (power + 1)
    
    return grad + wall_grad


def _hexagonal_initial(n):
    """Generate a hexagonal packing initial configuration."""
    # For N points, use a hexagonal arrangement
    # Approximate number of rows needed
    # Hexagonal packing: each row has either k or k+1 points
    # Let's just create a grid and perturb slightly
    
    # Start with a square grid as base
    m = int(np.ceil(np.sqrt(n)))
    # Create m x m grid, take first n
    xs = np.linspace(-0.45, 0.45, m)
    ys = np.linspace(-0.45, 0.45, m)
    grid = np.array([(x, y) for y in ys for x in xs])
    pts = grid[:n].copy()
    
    # Add small random perturbation to break symmetry
    np.random.seed(42)
    pts += np.random.normal(0, 0.001, size=pts.shape)
    
    # Clip to square
    pts = np.clip(pts, -0.49, 0.49)
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 120  # Safety margin before 150s limit
    
    # Initial configuration
    pts = _hexagonal_initial(N)
    
    # Try multiple power values and pick the best
    best_pts = pts.copy()
    best_radius = _compute_radius(best_pts)
    
    power = 6
    
    # Run L-BFGS-B optimization
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    def objective(x):
        pts_local = x.reshape(-1, 2)
        return _energy(pts_local, power=power)
    
    def objective_grad(x):
        pts_local = x.reshape(-1, 2)
        return _energy_grad(pts_local, power=power)
    
    x0 = pts.flatten()
    
    # Optimize
    result = minimize(
        objective,
        x0,
        jac=objective_grad,
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 1000, 'maxls': 50, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 50000}
    )
    
    if result.success or result.fun < _energy(pts, power=power):
        new_pts = result.x.reshape(-1, 2)
        new_radius = _compute_radius(new_pts)
        if new_radius > best_radius:
            best_pts = new_pts
            best_radius = new_radius
    
    # Check time
    elapsed = time.time() - start_time
    if elapsed < deadline - 5:
        # Try a second round with higher power for more refinement
        power2 = 10
        x0 = best_pts.flatten()
        
        def objective2(x):
            pts_local = x.reshape(-1, 2)
            return _energy(pts_local, power=power2)
        
        def objective_grad2(x):
            pts_local = x.reshape(-1, 2)
            return _energy_grad(pts_local, power=power2)
        
        result2 = minimize(
            objective2,
            x0,
            jac=objective_grad2,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 500, 'maxls': 30, 'ftol': 1e-12, 'gtol': 1e-8, 'maxfun': 20000}
        )
        
        if result2.success or result2.fun < _energy(best_pts, power=power2):
            new_pts = result2.x.reshape(-1, 2)
            new_radius = _compute_radius(new_pts)
            if new_radius > best_radius:
                best_pts = new_pts
                best_radius = new_radius
    
    # Final check: ensure all points are strictly inside
    best_pts = np.clip(best_pts, -0.4999, 0.4999)
    
    return best_pts
# EVOLVE-BLOCK-END
