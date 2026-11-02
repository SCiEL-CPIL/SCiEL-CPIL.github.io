"""Rewritten program: optimize 200 equal circles in unit square via gradient-free + gradient polish."""
import numpy as np
from scipy.optimize import minimize, LinearConstraint
import time

N = 200


def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    if pts.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    # Check bounds
    min_coord = pts.min(axis=0)
    max_coord = pts.max(axis=0)
    if min_coord[0] < -0.5 - 1e-12 or min_coord[1] < -0.5 - 1e-12:
        return 0.0
    if max_coord[0] > 0.5 + 1e-12 or max_coord[1] > 0.5 + 1e-12:
        return 0.0
    # Min distance to walls
    wall_dists = np.minimum(
        np.minimum(pts[:, 0] + 0.5, 0.5 - pts[:, 0]),
        np.minimum(pts[:, 1] + 0.5, 0.5 - pts[:, 1])
    )
    wall_min = wall_dists.min()
    # Min pairwise distance
    # Use efficient computation
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    # Set diagonal to inf
    np.fill_diagonal(dists, np.inf)
    pair_min = dists.min() / 2.0
    return min(wall_min, pair_min)


def _objective_neg_radius(pts):
    """Negative radius for minimization."""
    return -_compute_radius(pts)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # Initial guess: hexagonal packing pattern
    # For 200 circles, try a hexagonal arrangement
    # Start with a grid and perturb
    np.random.seed(42)
    
    # Create initial grid
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    pts = grid[:N].copy()
    
    best_pts = pts.copy()
    best_r = _compute_radius(best_pts)
    
    # Try multiple random initializations and local optimizations
    n_attempts = 0
    max_attempts = 50
    
    while time.time() < deadline - 5.0 and n_attempts < max_attempts:
        n_attempts += 1
        
        # Perturb the initial grid
        if n_attempts == 1:
            current_pts = pts.copy()
        else:
            # Add random perturbation to break symmetry
            scale = 0.05 / n_attempts
            current_pts = pts + np.random.randn(N, 2) * scale
            # Clip to valid region
            current_pts = np.clip(current_pts, -0.45, 0.45)
        
        # Use Nelder-Mead or similar derivative-free method
        # But for 400 dimensions, this is too slow. Instead, use a smarter approach.
        
        # Use gradient-based optimization with soft constraints
        # Objective: maximize min(pairwise_dist/2, wall_dist)
        # Equivalent to: maximize t such that all pairwise dist >= 2t and all wall dist >= t
        
        # Use scipy minimize with BFGS on a smooth surrogate
        # Soft min: -log(sum(exp(-x_i/eps))) / log(n) approximates min(x_i)
        
        def soft_min_surrogate(x, eps=0.01):
            # x is (2N,) flattened
            p = x.reshape(N, 2)
            # Pairwise distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists_sq = (diff ** 2).sum(axis=2)
            # Wall distances
            wall_dists = np.minimum(
                np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0]),
                np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
            )
            # Combine all distances
            # We want min of (pairwise_dist/2, wall_dist)
            # Approximate with soft min
            all_dists = np.concatenate([np.sqrt(dists_sq).min(axis=1) / 2, wall_dists])
            # Remove self-pairs (diagonal)
            # Actually, let's compute properly
            return -all_dists.min()
        
        # This approach is getting complex. Let me use a simpler but effective strategy:
        # 1. Start from hexagonal lattice
        # 2. Use iterative repulsion to find local optimum
        
        # Hexagonal lattice initialization
        if n_attempts <= 5:
            # Try different hexagonal spacings
            spacing = 0.072 * (1 + 0.02 * n_attempts)
            pts_hex = []
            row = 0
            while len(pts_hex) < N:
                y = (row * spacing * np.sqrt(3) / 2) - 0.5 + spacing * 0.5
                offset = (spacing / 2) if row % 2 else 0
                x = -0.5 + spacing / 2 + offset
                while x < 0.5 - spacing / 2 and len(pts_hex) < N:
                    pts_hex.append([x, y])
                    x += spacing
                row += 1
            if len(pts_hex) >= N:
                current_pts = np.array(pts_hex[:N])
        
        # Gradient-based polish using subgradient of the min function
        # The radius is min over all constraints. The subgradient points in the direction
        # that increases the most violated constraint.
        
        r = _compute_radius(current_pts)
        if r > best_r:
            best_r = r
            best_pts = current_pts.copy()
        
        # Try a few gradient steps
        for step in range(100):
            if time.time() > deadline - 3.0:
                break
            
            p = current_pts
            # Compute all pairwise distances and wall distances
            diff = p[:, np.newaxis, :] - p[np.newaxis, :, :]
            dists_sq = (diff ** 2).sum(axis=2)
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = np.minimum(
                np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0]),
                np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
            )
            
            # Find the minimum constraint
            pair_min_val = dists.min() / 2
            wall_min_val = wall_dists.min()
            
            if pair_min_val < wall_min_val:
                # Pairwise constraint is tight
                idx = np.unravel_index(dists.argmin(), dists.shape)
                i, j = idx
                if i > j:
                    i, j = j, i
                # Gradient: push i and j apart
                grad = np.zeros_like(p)
                dir_vec = p[i] - p[j]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-12:
                    dir_vec = dir_vec / norm
                grad[i] += dir_vec
                grad[j] -= dir_vec
            else:
                # Wall constraint is tight
                idx = wall_dists.argmin()
                i = idx
                # Determine which wall
                x, y = p[i]
                if x + 0.5 <= 0.5 - x and y + 0.5 <= 0.5 - y:
                    # Left wall
                    grad = np.zeros_like(p)
                    grad[i, 0] = 1.0
                elif 0.5 - x < x + 0.5 and y + 0.5 <= 0.5 - y:
                    # Right wall
                    grad = np.zeros_like(p)
                    grad[i, 0] = -1.0
                elif x + 0.5 <= 0.5 - x and 0.5 - y < y + 0.5:
                    # Bottom wall
                    grad = np.zeros_like(p)
                    grad[i, 1] = 1.0
                else:
                    # Top wall
                    grad = np.zeros_like(p)
                    grad[i, 1] = -1.0
            
            # Take a small step in the gradient direction
            lr = 0.001
            current_pts = p + lr * grad
            # Clip to valid region
            current_pts = np.clip(current_pts, -0.49, 0.49)
            
            new_r = _compute_radius(current_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = current_pts.copy()
        
        # Occasionally try a fresh random restart
        if n_attempts % 10 == 0:
            current_pts = np.random.uniform(-0.45, 0.45, size=(N, 2))
    
    # Final polish with smaller steps
    if time.time() < deadline - 2.0:
        current_pts = best_pts.copy()
        for step in range(200):
            if time.time() > deadline - 1.0:
                break
            
            p = current_pts
            diff = p[:, np.newaxis, :] - p[np.newaxis, :]
            dists_sq = (diff ** 2).sum(axis=2)
            dists = np.sqrt(dists_sq)
            np.fill_diagonal(dists, np.inf)
            
            wall_dists = np.minimum(
                np.minimum(p[:, 0] + 0.5, 0.5 - p[:, 0]),
                np.minimum(p[:, 1] + 0.5, 0.5 - p[:, 1])
            )
            
            pair_min_val = dists.min() / 2
            wall_min_val = wall_dists.min()
            
            if pair_min_val < wall_min_val:
                idx = np.unravel_index(dists.argmin(), dists.shape)
                i, j = idx
                if i > j:
                    i, j = j, i
                grad = np.zeros_like(p)
                dir_vec = p[i] - p[j]
                norm = np.linalg.norm(dir_vec)
                if norm > 1e-12:
                    dir_vec = dir_vec / norm
                grad[i] += dir_vec
                grad[j] -= dir_vec
            else:
                idx = wall_dists.argmin()
                i = idx
                x, y = p[i]
                if x + 0.5 <= 0.5 - x and y + 0.5 <= 0.5 - y:
                    grad = np.zeros_like(p)
                    grad[i, 0] = 1.0
                elif 0.5 - x < x + 0.5 and y + 0.5 <= 0.5 - y:
                    grad = np.zeros_like(p)
                    grad[i, 0] = -1.0
                elif x + 0.5 <= 0.5 - x and 0.5 - y < y + 0.5:
                    grad = np.zeros_like(p)
                    grad[i, 1] = 1.0
                else:
                    grad = np.zeros_like(p)
                    grad[i, 1] = -1.0
            
            lr = 0.0001
            current_pts = p + lr * grad
            current_pts = np.clip(current_pts, -0.49, 0.49)
            
            new_r = _compute_radius(current_pts)
            if new_r > best_r:
                best_r = new_r
                best_pts = current_pts.copy()
    
    return best_pts
