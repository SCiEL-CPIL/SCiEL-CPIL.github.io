"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time

def _compute_radius(pts):
    """Compute the achievable radius for a set of points."""
    n = len(pts)
    # Min distance between pairs / 2
    min_pair = np.inf
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(pts[i] - pts[j])
            if d < min_pair:
                min_pair = d
    min_pair_r = min_pair / 2.0
    
    # Min distance to boundary
    min_boundary = np.min(1.0 - np.linalg.norm(pts, axis=1))
    
    return min(min_pair_r, min_boundary)

def _hexagonal_packing(n_circles):
    """Generate a hexagonal lattice packing as initial configuration."""
    # Use a hexagonal grid and select the best n_circles points
    spacing = 2.0 / (np.sqrt(n_circles) + 1.0)
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if np.hypot(x, y) <= 1.0 - spacing / 2.0:
                rows.append((x, y))
    points = np.array(rows)
    if len(points) >= n_circles:
        # Sort by distance from origin, take innermost
        dist = np.linalg.norm(points, axis=1)
        idx = np.argsort(dist, kind='stable')[:n_circles]
        return points[idx]
    # Fallback: uniform random in disk
    angles = np.random.uniform(0, 2*np.pi, n_circles)
    radii = np.sqrt(np.random.uniform(0, 1, n_circles))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])

def _repulsion_forces(pts, r_target, gamma=5.0):
    """Compute repulsion forces between circles and from boundary."""
    n = len(pts)
    forces = np.zeros_like(pts)
    r = r_target * 2.0  # diameter
    
    for i in range(n):
        for j in range(i+1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                # Add small random perturbation
                d = np.random.randn(2) * 1e-6
                dist = np.linalg.norm(d)
            if dist < r:
                # Strong repulsion when overlapping
                f = gamma * (r - dist) / dist
                forces[i] += f * d
                forces[j] -= f * d
        # Boundary repulsion
        dist_to_origin = np.linalg.norm(pts[i])
        if dist_to_origin > 1.0 - r_target:
            if dist_to_origin > 1e-10:
                f = gamma * (dist_to_origin - (1.0 - r_target)) / dist_to_origin
                forces[i] -= f * pts[i]
            else:
                forces[i] += np.random.randn(2) * 0.01
    return forces

def _anneal(pts, max_iters=5000, gamma=5.0, lr_init=0.05, lr_min=1e-6, cooldown=0.999):
    """Simulated annealing with directed repulsion forces."""
    n = len(pts)
    current_r = _compute_radius(pts)
    best_r = current_r
    best_pts = pts.copy()
    
    lr = lr_init
    for it in range(max_iters):
        forces = _repulsion_forces(pts, current_r, gamma)
        
        # Update positions with learning rate
        pts_new = pts + lr * forces
        # Keep points inside the unit disk
        dists = np.linalg.norm(pts_new, axis=1)
        mask = dists > 1.0
        if np.any(mask):
            pts_new[mask] = pts_new[mask] / dists[mask][:, None] * 0.999
        
        new_r = _compute_radius(pts_new)
        
        # Accept if better or with some probability
        if new_r >= current_r:
            pts = pts_new
            current_r = new_r
            if current_r > best_r:
                best_r = current_r
                best_pts = pts.copy()
        else:
            # Annealing acceptance
            if np.random.random() < np.exp((new_r - current_r) / max(lr * 0.1, 1e-8)):
                pts = pts_new
                current_r = new_r
        
        # Cool down
        lr *= cooldown
        if lr < lr_min:
            break
    
    return best_pts, best_r

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # Initial configuration
    pts = _hexagonal_packing(N)
    best_pts, best_r = _anneal(pts, max_iters=2000)
    
    elapsed = time.time() - start_time
    if elapsed > deadline:
        return best_pts
    
    # Multiple restarts with different initializations
    num_restarts = 5
    for restart in range(num_restarts):
        if time.time() > deadline:
            break
        
        # Generate new initial configuration with perturbation
        pts_init = _hexagonal_packing(N)
        # Add random perturbation
        pts_init += np.random.randn(N, 2) * 0.1
        # Project back to disk
        dists = np.linalg.norm(pts_init, axis=1)
        mask = dists > 0.95
        if np.any(mask):
            pts_init[mask] = pts_init[mask] / dists[mask][:, None] * 0.95
        
        # Anneal
        pts_candidate, r_candidate = _anneal(pts_init, max_iters=3000)
        
        if r_candidate > best_r:
            best_r = r_candidate
            best_pts = pts_candidate
    
    return best_pts
# EVOLVE-BLOCK-END
