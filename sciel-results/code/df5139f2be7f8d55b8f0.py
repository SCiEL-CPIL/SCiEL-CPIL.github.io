"""Rewritten packing program: multi-start with BCC/Halton inits, repulsion polishing, SA."""
import numpy as np
import time
from scipy.spatial import cKDTree

N = 100

def _get_deadline():
    return time.time() + 115.0

def _time_left(deadline):
    return max(0.0, deadline - time.time())

def _compute_radius(c):
    if not np.all(np.isfinite(c)):
        return 0.0
    if np.any(c > 0.5) or np.any(c < -0.5):
        return 0.0
    wall_dists = 0.5 - np.abs(c)
    r_wall = np.min(wall_dists)
    try:
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        r_pair = np.min(dists[:, 1]) / 2.0
    except Exception:
        r_pair = 0.0
    return min(r_wall, r_pair)

def _soft_min_objective(c, beta=50.0):
    """Approximate -min via soft min for gradient-like optimization."""
    wall_dists = 0.5 - np.abs(c)
    # Pair distances via KDTree
    try:
        tree = cKDTree(c)
        dists, _ = tree.query(c, k=2)
        pair_dists = dists[:, 1]
    except Exception:
        return 1e6
    
    all_dists = np.concatenate([wall_dists.flatten(), pair_dists])
    # Soft min: -log(sum(exp(-beta*x)))/beta
    m = np.min(all_dists)
    return -(np.log(np.sum(np.exp(-beta * (all_dists - m)))) + beta * m) / beta

def _build_grid_init():
    nx, ny, nz = 5, 5, 4
    dx, dy, dz = 1.0/nx, 1.0/ny, 1.0/nz
    x_coords = np.arange(nx) * dx - 0.5 + dx/2
    y_coords = np.arange(ny) * dy - 0.5 + dy/2
    z_coords = np.arange(nz) * dz - 0.5 + dz/2
    grid = np.array([(x, y, z) for z in z_coords for y in y_coords for x in x_coords])
    return grid[:N].copy()

def _build_bcc_init():
    """BCC-like initialization: simple cubic + body centers, keep best 100."""
    m = 5  # 5x5x5 = 125 points, take 100
    ticks = (np.arange(m) + 0.5) / m - 0.5
    # Simple cubic
    sc = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Body centers
    bc = sc + 0.5/m
    # Combine and clip
    all_pts = np.vstack([sc, bc])
    all_pts = np.clip(all_pts, -0.49, 0.49)
    # Keep first 100 or use a subset
    return all_pts[:N].copy()

def _build_halton_init():
    """Halton sequence initialization for quasi-random distribution."""
    def halton(index, base):
        f = 1.0
        r = 0.0
        while index > 0:
            f /= base
            r += f * (index % base)
            index //= base
        return r
    
    pts = np.zeros((N, 3))
    for i in range(1, N+1):
        pts[i-1, 0] = halton(i, 2) - 0.5
        pts[i-1, 1] = halton(i, 3) - 0.5
        pts[i-1, 2] = halton(i, 5) - 0.5
    return pts

def _polish_repulsion(centers, deadline, base_step=0.01, max_iters=5000):
    """Polish centers using repulsion forces to maximize minimum distance."""
    best_c = centers.copy()
    best_r = _compute_radius(best_c)
    stagnation = 0
    max_stag = 200
    step = base_step
    
    for it in range(max_iters):
        if time.time() > deadline:
            break
        
        wall_dists = 0.5 - np.abs(centers)
        r_wall = np.min(wall_dists)
        
        try:
            tree = cKDTree(centers)
            dists, _ = tree.query(centers, k=2)
            min_pair = np.min(dists[:, 1])
            r_pair = min_pair / 2.0
        except Exception:
            break
        
        curr_r = min(r_wall, r_pair)
        if curr_r > best_r + 1e-10:
            best_r = curr_r
            best_c = centers.copy()
            stagnation = 0
        else:
            stagnation += 1
            if stagnation > max_stag:
                step *= 0.9
                if step < 0.0005:
                    break
                stagnation = 0
        
        target_r = curr_r + 0.002
        force = np.zeros_like(centers)
        
        # Wall forces
        for k in range(3):
            d_wall = 0.5 - np.abs(centers[:, k])
            mask = d_wall < target_r
            if np.any(mask):
                viol = (target_r - d_wall[mask])
                dirs = -np.sign(centers[mask, k])
                dirs[dirs == 0] = 1.0
                force[mask, k] += viol * dirs
        
        # Pair forces
        target_dist = 2 * target_r
        try:
            pairs = tree.query_pairs(target_dist)
        except Exception:
            pairs = []
        
        for i, j in pairs:
            diff = centers[i] - centers[j]
            d = np.linalg.norm(diff)
            if d > 1e-10 and d < target_dist:
                direction = diff / d
                viol = target_dist - d
                force[i] += viol * direction
                force[j] -= viol * direction
        
        norm_f = np.linalg.norm(force)
        if norm_f > 1e-10:
            centers += step * (force / norm_f)
            centers = np.clip(centers, -0.499999, 0.499999)
    
    return best_c, best_r

def _optimize_from_initial(init, deadline):
    """Optimize from a given initial configuration."""
    centers = init.copy()
    # Phase 1: Repulsion polish
    c1, r1 = _polish_repulsion(centers, deadline, base_step=0.01, max_iters=3000)
    # Phase 2: Finer polish
    c2, r2 = _polish_repulsion(c1, deadline, base_step=0.003, max_iters=3000)
    return c2, r2

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    inits = [
        _build_grid_init(),
        _build_bcc_init(),
        _build_halton_init(),
    ]
    
    best_c = None
    best_r = 0.0
    
    np.random.seed(42)
    
    for init in inits:
        if time.time() > deadline - 10.0:
            break
        c, r = _optimize_from_initial(init, deadline)
        if r > best_r:
            best_r = r
            best_c = c
    
    # Additional random restarts if time permits
    attempts = 0
    while time.time() < deadline - 15.0 and attempts < 5:
        attempts += 1
        # Perturbed grid
        base = _build_grid_init()
        base += np.random.uniform(-0.05, 0.05, size=base.shape)
        base = np.clip(base, -0.49, 0.49)
        c, r = _optimize_from_initial(base, deadline)
        if r > best_r:
            best_r = r
            best_c = c
    
    if best_c is None:
        best_c = _build_grid_init()
    
    return best_c
