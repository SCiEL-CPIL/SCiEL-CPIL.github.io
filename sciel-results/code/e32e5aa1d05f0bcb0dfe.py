"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time


def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    # Pairwise distances
    n = len(pts)
    # Compute all pairwise distances efficiently
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    # Get minimum pairwise distance (excluding self)
    np.fill_diagonal(dists, np.inf)
    min_pair_dist = np.min(dists)
    
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)
    min_wall_dist = np.min(wall_dists)
    
    return min(min_pair_dist / 2.0, min_wall_dist)


def _smooth_objective(pts, beta=100.0):
    """Smooth surrogate for the minimum distance objective using log-sum-exp."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :]
    dists_sq = np.sum(diff ** 2, axis=2)
    np.fill_diagonal(dists_sq, np.inf)
    
    # Soft-min of pairwise distances via log-sum-exp on squared distances
    # We want to maximize the minimum distance, so minimize -log(sum(exp(-beta*d^2)))
    min_d_sq = np.min(dists_sq)
    # Use log-sum-exp for numerical stability
    shifted = -beta * dists_sq
    max_shifted = np.max(shifted)
    log_sum = max_shifted + np.log(np.sum(np.exp(shifted - max_shifted)))
    pair_term = -log_sum / beta
    
    # Wall constraints
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    wall_term = min_wall
    
    # Combined: we want both to be large
    # Use soft-min combination
    combined = np.logaddexp(np.log(pair_term + 1e-10), np.log(wall_term + 1e-10))
    return -combined  # Negative because we minimize


def _bcc_initial(n_spheres):
    """Generate BCC-like initial configuration for better starting point."""
    # Use a 3D grid with BCC structure
    # BCC has atoms at corners and center of each cube
    # For n spheres, find appropriate grid size
    
    # Start with a rough grid and add BCC centers
    m = int(np.ceil(n_spheres ** (1.0 / 3.0) + 0.5))
    
    # Generate grid points
    points = []
    for i in range(m):
        for j in range(m):
            for k in range(m):
                # Corner point
                x = (i + 0.5) / m - 0.5
                y = (j + 0.5) / m - 0.5
                z = (k + 0.5) / m - 0.5
                points.append([x, y, z])
                
                # BCC center (if within bounds)
                if i < m - 1 and j < m - 1 and k < m - 1:
                    bx = (i + 1.0) / m - 0.5
                    by = (j + 1.0) / m - 0.5
                    bz = (k + 1.0) / m - 0.5
                    points.append([bx, by, bz])
    
    points = np.array(points)
    
    # If we have more than needed, select best subset
    if len(points) > n_spheres:
        # Use greedy selection: pick points that maximize minimum distance
        selected = [points[0]]
        remaining = list(points[1:])
        
        while len(selected) < n_spheres and remaining:
            best_idx = 0
            best_min_dist = -1
            for idx, p in enumerate(remaining):
                min_d = min(np.linalg.norm(p - s) for s in selected)
                if min_d > best_min_dist:
                    best_min_dist = min_d
                    best_idx = idx
            selected.append(remaining.pop(best_idx))
        
        return np.array(selected[:n_spheres])
    else:
        return points[:n_spheres]


def construct():
    """Construct optimal sphere packing using simulated annealing."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Initial configuration: BCC-like
    pts = _bcc_initial(N).astype(np.float64)
    
    # Initial radius
    current_r = _compute_radius(pts)
    best_pts = pts.copy()
    best_r = current_r
    
    # Simulated annealing parameters
    T = 0.05  # Initial temperature
    T_min = 1e-7
    cooling_rate = 0.995
    n_iter = 50000
    
    rng = np.random.RandomState(42)
    
    for it in range(n_iter):
        # Check deadline
        if time.time() > deadline - 5.0:
            break
        
        # Generate perturbation
        if rng.random() < 0.7:
            # Move one random point
            idx = rng.randint(N)
            step = T * rng.randn(3)
            new_pts = pts.copy()
            new_pts[idx] += step
            # Clip to valid region (with small margin)
            new_pts[idx] = np.clip(new_pts[idx], -0.499, 0.499)
        else:
            # Move two random points
            idx1, idx2 = rng.choice(N, size=2, replace=False)
            step1 = T * rng.randn(3)
            step2 = T * rng.randn(3)
            new_pts = pts.copy()
            new_pts[idx1] += step1
            new_pts[idx2] += step2
            new_pts[idx1] = np.clip(new_pts[idx1], -0.499, 0.499)
            new_pts[idx2] = np.clip(new_pts[idx2], -0.499, 0.499)
        
        # Evaluate new configuration
        new_r = _compute_radius(new_pts)
        
        # Acceptance criterion
        delta = new_r - current_r
        if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
            pts = new_pts
            current_r = new_r
            
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        
        # Cool down
        T *= cooling_rate
        if T < T_min:
            T = T_min
    
    # Final polish: local optimization with smaller steps
    if time.time() < deadline - 10.0:
        T = 1e-5
        for _ in range(5000):
            if time.time() > deadline - 5.0:
                break
            
            idx = rng.randint(N)
            step = T * rng.randn(3)
            new_pts = pts.copy()
            new_pts[idx] += step
            new_pts[idx] = np.clip(new_pts[idx], -0.499, 0.499)
            
            new_r = _compute_radius(new_pts)
            delta = new_r - current_r
            if delta > 0 or rng.random() < np.exp(delta / max(T, 1e-10)):
                pts = new_pts
                current_r = new_r
                if current_r > best_r:
                    best_r = current_r
                    best_pts = pts.copy()
            T *= 0.99
    
    return best_pts
# EVOLVE-BLOCK-END
