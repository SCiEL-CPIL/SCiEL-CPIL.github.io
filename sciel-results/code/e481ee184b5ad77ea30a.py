"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 200

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize, LinearConstraint
import scipy.spatial.distance as sd


def _score(pts):
    """Compute the radius (score) for a given set of points."""
    # pts shape (N, 2)
    # min distance between pairs
    if N == 1:
        return min(0.5 - np.abs(pts[0, 0]), 0.5 - np.abs(pts[0, 1]))
    
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _hex_grid_init():
    """Generate a hexagonal grid initialization."""
    # For N=200, a hex grid with appropriate spacing
    # Let's try to fit 200 points in a hex pattern
    # Approximate side length for hex grid: sqrt(N) ~ 14.1
    # Use rows and columns
    
    pts = []
    # Try different densities
    # Hex grid: row offset
    
    # Let's use a simple approach: generate a grid and then jitter
    # Or use a known good hex packing for ~200 circles
    
    # Side length of unit square is 1, centered at origin: [-0.5, 0.5]
    # Hex grid with radius r: horizontal spacing = 2r, vertical spacing = 2*r*sqrt(3)/2 = r*sqrt(3)
    # But we want to maximize r, so we want as many points as possible.
    
    # Let's estimate: if we use a hex grid, the number of points in a square of side 1
    # with hex spacing: 
    # Let dx = 2r, dy = r*sqrt(3)
    # Number of columns ~ 1/dx = 1/(2r)
    # Number of rows ~ 1/dy = 1/(r*sqrt(3))
    # Total ~ 1/(2r * r*sqrt(3)) = 1/(2*sqrt(3)*r^2)
    # Set to 200: 200 = 1/(2*sqrt(3)*r^2) => r^2 = 1/(400*sqrt(3)) => r = 1/(20*sqrt(3)) ~ 0.0288
    
    # But the known best is 0.0366, so hex grid alone won't reach it.
    # Let's just generate a reasonable hex grid and then optimize.
    
    r_est = 1.0 / (20 * np.sqrt(3))
    dx = 2 * r_est
    dy = r_est * np.sqrt(3)
    
    # Generate hex grid
    n_cols = int(np.floor(1.0 / dx)) + 1
    n_rows = int(np.floor(1.0 / dy)) + 1
    
    # Start from top-left, offset by r_est to be inside
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
    
    # If we have more than N, trim; if less, add more or adjust
    if len(pts) > N:
        # Keep the first N
        pts = pts[:N]
    elif len(pts) < N:
        # Add random points or adjust
        # For now, just return what we have and pad
        while len(pts) < N:
            pts = np.vstack([pts, np.random.uniform(-0.4, 0.4, size=2)])
    
    return pts


def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin
    
    # Initial guess: hex grid
    pts0 = _hex_grid_init()
    
    # Ensure pts0 is valid shape
    if len(pts0) != N:
        pts0 = pts0[:N] if len(pts0) > N else np.vstack([pts0, np.random.uniform(-0.4, 0.4, size=(N - len(pts0), 2))])
    
    best_pts = pts0.copy()
    best_score = _score(best_pts)
    
    # Objective: minimize negative radius (maximize radius)
    # But radius is min of many terms, which is non-smooth.
    # Instead, we can use a smooth approximation: minimize the sum of squared distances
    # or use the "repulsive" potential.
    
    # A common approach: minimize sum of 1/d^2 for close pairs + wall penalties
    # Or use the "soft" minimum: -log(sum(exp(-d/T))) for temperature T
    
    def objective(x):
        pts = x.reshape(-1, 2)
        
        # Check bounds
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1e6
        
        # Pairwise distances
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Soft minimum of distances
        T = 0.005
        # Only consider close pairs to avoid overflow
        mask = dists < 0.1
        if not np.any(mask):
            return 0.0
        
        min_d = np.min(dists[mask])
        
        # Wall distances
        wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
        min_wall = np.min(wall_dists)
        
        # The score is min(min_d/2, min_wall)
        # We want to maximize this, so minimize -score
        # But -min is non-smooth. Use: -min_d/2 - min_wall + penalty
        # Or simply: return -min(min_d/2, min_wall)
        
        # For optimization, let's use a barrier-like function
        # Minimize: sum of exp(-(d - r_target)/T) for all pairs and walls
        # where r_target is a target radius
        
        return -min(min_d / 2.0, min_wall)
    
    # This objective is non-smooth and may not work well with gradient methods.
    # Let's try a different approach: use scipy.optimize with a smooth surrogate.
    
    # Alternative: minimize the "energy" E = sum_{i<j} 1/d_{ij}^4 + sum_i 1/wall_dist_i^4
    # This will push points apart.
    
    def energy(x):
        pts = x.reshape(-1, 2)
        
        if np.any(pts < -0.5) or np.any(pts > 0.5):
            return 1e10
        
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, 1e10)
        
        # Avoid division by zero
        dists = np.maximum(dists, 1e-10)
        
        pair_energy = np.sum(1.0 / (dists ** 4))
        
        wall_dists = np.minimum(0.5 - np.abs(pts), 0.5)
        wall_dists = np.maximum(wall_dists, 1e-10)
        wall_energy = np.sum(1.0 / (wall_dists ** 4))
        
        return pair_energy + wall_energy
    
    # Try multiple restarts with different initializations
    np.random.seed(42)
    
    for restart in range(5):
        if time.time() > deadline - 20.0:
            break
        
        if restart == 0:
            x0 = pts0.flatten()
        else:
            # Perturb the best solution
            x0 = best_pts.flatten() + np.random.normal(0, 0.01, size=N * 2)
            x0 = np.clip(x0, -0.49, 0.49)
        
        # Use L-BFGS-B with bounds
        bounds = [(-0.5, 0.5)] * (N * 2)
        
        try:
            res = minimize(
                energy,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 500, 'maxfun': 2000, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            
            if res.success or res.fun < 1e8:
                pts_opt = res.x.reshape(-1, 2)
                score = _score(pts_opt)
                
                if score > best_score:
                    best_score = score
                    best_pts = pts_opt.copy()
                    
        except Exception as e:
            pass
        
        if time.time() > deadline - 10.0:
            break
    
    # Final refinement: local search on the best solution
    # Try a few iterations of moving each point slightly to improve the minimum distance
    
    pts = best_pts.copy()
    for iter in range(10):
        if time.time() > deadline - 5.0:
            break
        
        improved = False
        for i in range(N):
            if time.time() > deadline - 5.0:
                break
            
            # Try small moves in 4 directions
            for dx, dy in [(0.001, 0), (-0.001, 0), (0, 0.001), (0, -0.001)]:
                new_pts = pts.copy()
                new_pts[i, 0] += dx
                new_pts[i, 1] += dy
                
                if np.any(new_pts < -0.5) or np.any(new_pts > 0.5):
                    continue
                
                score = _score(new_pts)
                if score > best_score + 1e-12:
                    best_score = score
                    best_pts = new_pts
                    pts = new_pts
                    improved = True
                    break
        
        if not improved:
            break
    
    return best_pts
# EVOLVE-BLOCK-END
