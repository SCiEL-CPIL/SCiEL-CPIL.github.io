"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _compute_radius(pts):
    """Compute the common radius for a set of points inside [-0.5, 0.5]^2."""
    if len(pts) == 0:
        return 0.0
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _initial_grid():
    """Generate a hexagonal-style initial grid."""
    pts = []
    # Hex grid parameters
    # For N=200, we want roughly sqrt(200) ~ 14 per side
    # Hex spacing: dx = 2*r, dy = r*sqrt(3)
    # Area per circle ~ r^2 * sqrt(3)
    # Total area ~ N * r^2 * sqrt(3) ~ 1 => r ~ 1/sqrt(N*sqrt(3))
    r_est = 1.0 / np.sqrt(N * np.sqrt(3))
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    n_cols = int(np.floor(1.0 / dx)) + 1
    n_rows = int(np.floor(1.0 / dy)) + 1
    
    x_start = -0.5 + r_est
    y_start = 0.5 - r_est
    
    for row in range(n_rows):
        y = y_start - row * dy
        if y < -0.5 + r_est:
            break
        offset = 0.0 if row % 2 == 0 else dx / 2
        for col in range(n_cols):
            x = x_start + col * dx + offset
            if x < 0.5 - r_est:
                pts.append([x, y])
    
    pts = np.array(pts)
    
    if len(pts) > N:
        pts = pts[:N]
    elif len(pts) < N:
        # Pad with random points
        rng = np.random.default_rng(42)
        extra = rng.uniform(-0.4, 0.4, size=(N - len(pts), 2))
        pts = np.vstack([pts, extra])
    
    return pts


def _optimize_repulsion(x0, deadline, max_iter=100):
    """
    Optimize using a repulsion-based energy function.
    Minimize sum of 1/d^4 for close pairs and walls.
    """
    def energy(x):
        pts = x.reshape(-1, 2)
        
        # Bounds check
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1e10
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        dists = np.maximum(dists, 1e-8)
        pair_energy = np.sum(1.0 / (dists ** 4))
        
        wall_dists = 0.5 - np.abs(pts)
        wall_dists = np.maximum(wall_dists, 1e-8)
        wall_energy = np.sum(1.0 / (wall_dists ** 4))
        
        return pair_energy + wall_energy
    
    bounds = [(-0.5, 0.5)] * (N * 2)
    
    try:
        res = minimize(
            energy,
            x0,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': max_iter, 'maxfun': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return res.x.reshape(-1, 2)
    except Exception:
        return x0.reshape(-1, 2)


def _local_refine(pts, deadline):
    """
    Local refinement: iteratively move each point to improve the minimum distance.
    """
    pts = pts.copy()
    improved = True
    while improved and time.time() < deadline - 2.0:
        improved = False
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            best_move = None
            best_score = _compute_radius(pts)
            
            # Try small perturbations
            for scale in [0.005, 0.002, 0.001]:
                for angle in np.linspace(0, 2 * np.pi, 8, endpoint=False):
                    dx = scale * np.cos(angle)
                    dy = scale * np.sin(angle)
                    new_pts = pts.copy()
                    new_pts[i, 0] += dx
                    new_pts[i, 1] += dy
                    
                    # Check bounds
                    if np.any(new_pts < -0.5) or np.any(new_pts > 0.5):
                        continue
                    
                    score = _compute_radius(new_pts)
                    if score > best_score + 1e-10:
                        best_score = score
                        best_move = (dx, dy)
            
            if best_move is not None:
                pts[i, 0] += best_move[0]
                pts[i, 1] += best_move[1]
                improved = True
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin before 120s limit
    
    # Initial guess: hex grid
    pts0 = _initial_grid()
    best_pts = pts0.copy()
    best_score = _compute_radius(best_pts)
    
    # Multiple restarts with optimization
    rng = np.random.default_rng(123)
    
    for restart in range(10):
        if time.time() > deadline - 30.0:
            break
        
        if restart == 0:
            x0 = pts0.flatten()
        else:
            # Perturb the best solution
            x0 = best_pts.flatten() + rng.normal(0, 0.02, size=N * 2)
            x0 = np.clip(x0, -0.49, 0.49)
        
        # Optimize
        pts_opt = _optimize_repulsion(x0, deadline, max_iter=80)
        score = _compute_radius(pts_opt)
        
        if score > best_score:
            best_score = score
            best_pts = pts_opt.copy()
    
    # Final local refinement
    if time.time() < deadline - 5.0:
        best_pts = _local_refine(best_pts, deadline)
        best_score = _compute_radius(best_pts)
    
    # Ensure valid output
    best_pts = np.clip(best_pts, -0.5, 0.5)
    
    return best_pts
# EVOLVE-BLOCK-END
