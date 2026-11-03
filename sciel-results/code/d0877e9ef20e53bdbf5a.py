import numpy as np
import time

N = 300

def _init_hex(n):
    """Generate hexagonal lattice points inside unit disk with proper spacing."""
    # Estimate spacing for n circles in unit disk
    # Area per circle ~ pi/n, so spacing ~ sqrt(pi/n)
    # For hex packing, spacing should be ~ 2*r where r is circle radius
    # Let's use a fine grid and pick n closest to center
    spacing = 0.04
    k = int(1.5 / spacing) + 1
    pts = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x*x + y*y <= 0.98:  # Leave margin from boundary
                pts.append([x, y])
    pts = np.array(pts)
    # Sort by distance to origin
    r = np.hypot(pts[:, 0], pts[:, 1])
    idx = np.argsort(r, kind="stable")[:n]
    return pts[idx]

def _compute_score(pts):
    """Compute the actual packing radius."""
    n = len(pts)
    if n < 2:
        return 0.0
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    np.fill_diagonal(dist, np.inf)
    min_dist = np.min(dist)
    
    r = np.hypot(pts[:, 0], pts[:, 1])
    min_boundary = np.min(1.0 - r)
    
    return min(min_dist / 2.0, min_boundary)

def _soft_min(values, temperature=0.01):
    """Compute soft minimum of values using log-sum-exp trick."""
    # soft_min = -log(sum(exp(-values/temperature))) * temperature
    # Use stable version: -temperature * log(sum(exp(-(values - min_val)/temperature))) - min_val
    min_val = np.min(values)
    exp_vals = np.exp(-(values - min_val) / temperature)
    return min_val - temperature * np.log(np.sum(exp_vals))

def _compute_forces(pts, r, temperature=0.01):
    """
    Compute smooth forces for optimization.
    Uses soft-min to create differentiable objective.
    """
    n = len(pts)
    # Compute pairwise distances
    dx = pts[:, 0:1] - pts[:, 0:1].T
    dy = pts[:, 1:2] - pts[:, 1:2].T
    dist = np.sqrt(dx * dx + dy * dy)
    
    # Avoid division by zero
    dist_safe = np.where(dist < 1e-8, 1e-8, dist)
    
    # We want to maximize min distance, so minimize -min distance
    # Use soft-min: soft_min(dist) approximates min(dist)
    # Gradient of soft_min w.r.t. dist_i = -exp(-dist_i/temp) / sum(exp(-dist_j/temp))
    
    # For each pair, compute weight for soft-min
    # Only consider pairs where dist < 2*r (potential conflicts)
    # But for smoothness, use all pairs with soft weighting
    
    # Compute pairwise overlaps (positive when circles would overlap)
    overlap = 2.0 * r - dist
    # Soft-weighted overlap: use sigmoid-like function
    # weight = 1 / (1 + exp(-overlap / temperature))
    weight = 1.0 / (1.0 + np.exp(-overlap / temperature))
    # Set diagonal to 0
    np.fill_diagonal(weight, 0.0)
    
    # Normalize direction
    dir_x = dx / dist_safe
    dir_y = dy / dist_safe
    
    # Force on i from j: weight * overlap * dir
    # This creates a smooth approximation of the repulsion force
    force_x = np.sum(weight * overlap * dir_x, axis=1)
    force_y = np.sum(weight * overlap * dir_y, axis=1)
    
    # Boundary forces
    r_pts = np.hypot(pts[:, 0], pts[:, 1])
    bound_overlap = r_pts - (1.0 - r)
    # Soft boundary force
    bound_weight = 1.0 / (1.0 + np.exp(-bound_overlap / temperature))
    
    # Direction is -c_i / |c_i|
    dir_bx = np.where(r_pts > 1e-8, -pts[:, 0] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    dir_by = np.where(r_pts > 1e-8, -pts[:, 1] / np.where(r_pts > 1e-8, r_pts, 1.0), 0.0)
    
    force_x += bound_weight * bound_overlap * dir_bx
    force_y += bound_weight * bound_overlap * dir_by
    
    return force_x, force_y

def _repulsion_step(pts, r, lr, temperature=0.01):
    """
    Perform one step of repulsion with smooth forces.
    """
    force_x, force_y = _compute_forces(pts, r, temperature)
    
    pts_new = pts.copy()
    pts_new[:, 0] += lr * force_x
    pts_new[:, 1] += lr * force_y
    
    # Project back inside unit circle (safety)
    r_new = np.hypot(pts_new[:, 0], pts_new[:, 1])
    mask = r_new > 1.0
    if np.any(mask):
        pts_new[mask, 0] *= (1.0 - 1e-6) / r_new[mask]
        pts_new[mask, 1] *= (1.0 - 1e-6) / r_new[mask]
        
    return pts_new

def _optimize_phase(pts, r_start, r_end, steps, lr_init, lr_end, temperature_init, temperature_end, time_limit, start_time):
    """
    Run optimization phase with annealed parameters.
    """
    best_pts = pts.copy()
    best_score = _compute_score(pts)
    
    for i in range(steps):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
            
        # Anneal parameters
        progress = i / max(steps, 1)
        lr = lr_init * (lr_end / lr_init) ** progress
        temperature = temperature_init * (temperature_end / temperature_init) ** progress
        
        # Anneal radius
        r = r_start + (r_end - r_start) * progress
        
        pts = _repulsion_step(pts, r, lr, temperature)
        
        # Check score periodically
        if i % 50 == 0:
            score = _compute_score(pts)
            if score > best_score:
                best_score = score
                best_pts = pts.copy()
                
    return best_pts, best_score

def construct():
    start_time = time.time()
    time_limit = 140.0  # 10s safety margin before 150s kill
    
    # Initialize
    pts = _init_hex(N)
    current_r = _compute_score(pts)
    
    best_pts = pts.copy()
    best_score = current_r
    
    # Phase 1: Coarse optimization with larger learning rate
    # Start with smaller r and increase
    r_start = max(0.03, current_r * 0.7)
    r_end = 0.054
    
    pts, score = _optimize_phase(
        pts, r_start, r_end, 
        steps=2000,
        lr_init=0.02, lr_end=0.001,
        temperature_init=0.05, temperature_end=0.005,
        time_limit=time_limit, start_time=start_time
    )
    if score > best_score:
        best_score = score
        best_pts = pts.copy()
    
    # Phase 2: Fine optimization around current best
    if time.time() - start_time < time_limit - 20:
        r_target = best_score
        pts, score = _optimize_phase(
            pts, r_target, r_target,
            steps=1500,
            lr_init=0.005, lr_end=0.0001,
            temperature_init=0.005, temperature_end=0.001,
            time_limit=time_limit, start_time=start_time
        )
        if score > best_score:
            best_score = score
            best_pts = pts.copy()
    
    # Phase 3: Final polish with very small steps
    if time.time() - start_time < time_limit - 5:
        r_target = best_score
        for i in range(500):
            if time.time() - start_time > time_limit:
                break
            lr = 0.0005 * (1.0 - i/500.0)
            temperature = 0.001
            pts = _repulsion_step(pts, r_target, lr, temperature)
            if i % 25 == 0:
                score = _compute_score(pts)
                if score > best_score:
                    best_score = score
                    best_pts = pts.copy()
    
    return best_pts
