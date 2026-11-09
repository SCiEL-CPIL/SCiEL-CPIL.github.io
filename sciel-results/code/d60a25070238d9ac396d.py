"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 100

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(pts):
    """Compute the maximum common radius for given centers."""
    n = len(pts)
    # Distance to walls
    wall_dists = 0.5 - np.abs(pts)  # shape (n, 3)
    min_wall = np.min(wall_dists, axis=1)  # shape (n,)
    wall_radius = np.min(min_wall)
    
    # Pairwise distances
    if n < 2:
        return wall_radius
    # Compute all pairwise squared distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    # Only consider upper triangle
    iu = np.triu_indices(n, k=1)
    pair_dists = np.sqrt(sq_dists[iu])
    min_pair = np.min(pair_dists)
    pair_radius = min_pair / 2.0
    
    return min(wall_radius, pair_radius)

def _soft_min_radius(pts, temperature=0.01):
    """
    Smooth surrogate for the minimum radius using log-sum-exp.
    r = min(wall_radius, pair_radius)
    We want to maximize this. Use soft-min: -logsumexp(-values/T)/log(N)
    """
    n = len(pts)
    # Wall distances (distance from each point to nearest wall)
    wall_dists = 0.5 - np.abs(pts)  # (n, 3)
    min_wall_per_point = np.min(wall_dists, axis=1)  # (n,)
    
    # Pairwise distances
    if n >= 2:
        diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        sq_dists = np.sum(diff ** 2, axis=2)
        iu = np.triu_indices(n, k=1)
        pair_dists = np.sqrt(sq_dists[iu])
        pair_radii = pair_dists / 2.0
    else:
        pair_radii = np.array([0.5])
    
    # Combine all "radius candidates"
    all_radii = np.concatenate([min_wall_per_point, pair_radii])
    
    # Soft-min: -logsumexp(-all_radii / T) * T
    neg = -all_radii / temperature
    # Shift for numerical stability
    max_neg = np.max(neg)
    logsumexp_val = max_neg + np.log(np.sum(np.exp(neg - max_neg)))
    soft_min = -logsumexp_val * temperature
    
    return soft_min

def _local_search(pts, max_iters=2000, time_limit=10.0, initial_step=0.02, min_step=1e-6):
    """
    Local search using smooth soft-min objective.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _soft_min_radius(best_pts, temperature=0.005)
    step = initial_step
    temperature = 0.01
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Generate random perturbation
        rng = np.random.default_rng()
        noise = rng.standard_normal(pts.shape) * step
        
        new_pts = best_pts + noise
        # Clip to valid range with small margin
        margin = 1e-9
        new_pts = np.clip(new_pts, -0.5 + margin, 0.5 - margin)
        
        new_score = _soft_min_radius(new_pts, temperature=temperature)
        
        if new_score > best_score:
            best_pts = new_pts
            best_score = new_score
            # Slightly increase step when improving
            step = min(step * 1.02, 0.05)
        else:
            # Decrease step when not improving
            step = max(step * 0.98, min_step)
        
        # Anneal temperature
        temperature = max(0.001, temperature * 0.999)
    
    return best_pts, best_score

def _high_precision_polish(pts, max_iters=5000, time_limit=15.0):
    """
    Final polish using exact radius computation and very small steps.
    """
    start_time = time.time()
    best_pts = pts.copy()
    best_score = _compute_radius(best_pts)
    step = 0.001
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Try small perturbations in each coordinate
        improved = False
        for i in range(len(pts)):
            for d in range(3):
                for direction in [1, -1]:
                    new_pts = best_pts.copy()
                    new_pts[i, d] += direction * step
                    new_pts[i, d] = np.clip(new_pts[i, d], -0.5 + 1e-9, 0.5 - 1e-9)
                    
                    new_score = _compute_radius(new_pts)
                    if new_score > best_score + 1e-12:
                        best_pts = new_pts
                        best_score = new_score
                        improved = True
        
        if not improved:
            step *= 0.5
            if step < 1e-8:
                break
    
    return best_pts, best_score

def construct():
    start_time = time.time()
    total_time_limit = 130.0  # Safety margin below 150s
    
    # Initial grid
    m = int(np.ceil(N ** (1.0 / 3.0) - 1e-9))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    initial_pts = grid[:N].copy()
    
    # Multi-start local search with different random perturbations
    best_overall = initial_pts.copy()
    best_score = _compute_radius(best_overall)
    
    np.random.seed(42)
    
    n_starts = 5
    for s in range(n_starts):
        if time.time() - start_time > total_time_limit - 30:
            break
        
        # Perturb the grid
        if s == 0:
            pts = initial_pts.copy()
        else:
            pts = initial_pts + np.random.standard_normal(initial_pts.shape) * 0.03
            pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Local search
        time_left = total_time_limit - (time.time() - start_time) - 20
        if time_left < 5:
            break
        
        pts, score = _local_search(pts, max_iters=3000, time_limit=min(10.0, time_left))
        
        # High precision polish
        time_left = total_time_limit - (time.time() - start_time) - 5
        if time_left > 3:
            pts, score = _high_precision_polish(pts, max_iters=3000, time_limit=min(10.0, time_left))
        
        exact_score = _compute_radius(pts)
        if exact_score > best_score:
            best_score = exact_score
            best_overall = pts.copy()
    
    # Final polish on the best result
    time_left = total_time_limit - (time.time() - start_time)
    if time_left > 5:
        best_overall, best_score = _high_precision_polish(best_overall, max_iters=5000, time_limit=min(12.0, time_left))
    
    return best_overall
# EVOLVE-BLOCK-END
