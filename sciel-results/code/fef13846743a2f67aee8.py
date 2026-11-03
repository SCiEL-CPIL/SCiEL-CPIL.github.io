"""Improved sphere packing for 50 equal spheres in a unit cube."""
import numpy as np
import time

N = 50

def _compute_radius(pts):
    """Compute the maximum common radius for given sphere centers."""
    n = len(pts)
    # Pairwise distances (vectorized)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_dist = np.min(dists)
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = np.min(wall_dists)
    return min(min_dist / 2.0, min_wall)

def _soft_min_radius(pts, beta=50.0):
    """Smooth surrogate: soft-min of pair distances and wall distances."""
    n = len(pts)
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    # Soft min of pair distances
    min_pair = -np.log(np.sum(np.exp(-beta * dists))) / beta
    # Wall distances
    wall_dists = 0.5 - np.abs(pts)
    min_wall = -np.log(np.sum(np.exp(-beta * wall_dists))) / beta
    return min(min_pair / 2.0, min_wall)

def _repulsion_step(pts, step_size, time_limit, start_time):
    """One repulsion-based optimization step."""
    n = len(pts)
    forces = np.zeros_like(pts)
    
    # Pairwise repulsion: push apart closest pairs
    for i in range(n):
        for j in range(i + 1, n):
            d = pts[i] - pts[j]
            dist = np.linalg.norm(d)
            if dist < 1e-10:
                d = np.random.randn(3) * 0.01
                dist = np.linalg.norm(d)
            # Force proportional to 1/dist^2 for close pairs
            f = d / (dist * dist)
            forces[i] += f
            forces[j] -= f
    
    # Wall repulsion: push inward if close to walls
    for i in range(n):
        for k in range(3):
            d_wall = 0.5 - abs(pts[i][k])
            if d_wall < 0.15:
                # Push toward center
                forces[i][k] -= np.sign(pts[i][k]) * (0.15 - d_wall) * 10.0
    
    # Normalize and apply
    norm_f = np.linalg.norm(forces, axis=1, keepdims=True)
    norm_f[norm_f == 0] = 1
    moves = forces / norm_f * step_size
    
    # Clip to stay in bounds
    new_pts = pts + moves
    new_pts = np.clip(new_pts, -0.4999999, 0.4999999)
    
    return new_pts

def _local_optimize(pts, time_limit):
    """Local optimization using repulsion + coordinate descent polish."""
    start_time = time.time()
    n = len(pts)
    pts = pts.copy()
    best_r = _compute_radius(pts)
    best_pts = pts.copy()
    
    # Phase 1: Repulsion-based optimization with adaptive step
    step = 0.05
    patience = 0
    max_patience = 200
    
    while time.time() - start_time < time_limit - 10.0:
        new_pts = _repulsion_step(pts, step, time_limit, start_time)
        new_r = _compute_radius(new_pts)
        
        if new_r > best_r + 1e-10:
            pts = new_pts
            best_r = new_r
            best_pts = pts.copy()
            patience = 0
            step = min(step * 1.2, 0.1)
        else:
            patience += 1
            step *= 0.95
            if patience > max_patience:
                break
    
    # Phase 2: Coordinate descent polish
    step_cd = 0.01
    no_improve = 0
    while time.time() - start_time < time_limit - 5.0:
        improved = False
        for i in range(n):
            if time.time() - start_time > time_limit - 5.0:
                break
            for dim in range(3):
                for direction in [1, -1]:
                    trial = pts.copy()
                    trial[i][dim] += direction * step_cd
                    if abs(trial[i][dim]) > 0.5:
                        continue
                    r = _compute_radius(trial)
                    if r > best_r + 1e-11:
                        pts = trial
                        best_r = r
                        best_pts = pts.copy()
                        improved = True
                        no_improve = 0
                        # Try bigger step
                        step_new = step_cd * 1.5
                        trial2 = pts.copy()
                        trial2[i][dim] += direction * step_new
                        if abs(trial2[i][dim]) <= 0.5:
                            r2 = _compute_radius(trial2)
                            if r2 > best_r + 1e-11:
                                pts = trial2
                                best_r = r2
                                best_pts = pts.copy()
                                step_cd = step_new
                            else:
                                step_cd = max(step_cd * 0.8, 0.002)
                        else:
                            step_cd = max(step_cd * 0.8, 0.002)
                    else:
                        step_cd = max(step_cd * 0.7, 0.002)
        
        if not improved:
            no_improve += 1
            if no_improve > 10:
                break
    
    return best_pts, best_r

def _generate_grid_start():
    """Generate a 50-point starting configuration from a jittered 4x4x4 grid."""
    m = 4
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y, z) for z in ticks for y in ticks for x in ticks])
    # Jitter to break symmetry
    np.random.seed(42)
    grid += np.random.randn(*grid.shape) * 0.02
    grid = np.clip(grid, -0.49, 0.49)
    return grid[:N].copy()

def _generate_fcc_start():
    """Generate starting configuration from FCC lattice points."""
    # FCC lattice: points at (0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0) mod 1
    # Scale to fit in [-0.5, 0.5]^3
    points = []
    for i in range(-1, 3):
        for j in range(-1, 3):
            for k in range(-1, 3):
                for offset in [(0,0,0), (0,0.5,0.5), (0.5,0,0.5), (0.5,0.5,0)]:
                    p = np.array([i + offset[0], j + offset[1], k + offset[2]])
                    if -0.5 <= p[0] <= 0.5 and -0.5 <= p[1] <= 0.5 and -0.5 <= p[2] <= 0.5:
                        points.append(p)
    points = np.array(points)
    # Keep only 50
    if len(points) > N:
        # Randomly select 50
        idx = np.random.choice(len(points), N, replace=False)
        points = points[idx]
    elif len(points) < N:
        # Pad with random points
        extra = np.random.randn(N - len(points), 3) * 0.3
        points = np.vstack([points, extra])
    return points[:N].copy()

def _generate_random_repulsive():
    """Generate random points with repulsion to avoid overlaps."""
    np.random.seed(123)
    pts = np.random.randn(N, 3) * 0.4
    # Simple repulsion iterations
    for _ in range(100):
        for i in range(N):
            for j in range(i + 1, N):
                d = pts[i] - pts[j]
                dist = np.linalg.norm(d)
                if dist < 0.15:
                    f = d / max(dist, 1e-10) * 0.01
                    pts[i] += f
                    pts[j] -= f
    pts = np.clip(pts, -0.49, 0.49)
    return pts

def construct():
    start_time = time.time()
    time_limit = 120.0  # Safety margin before 150s kill
    
    # Try multiple starting configurations
    candidates = []
    
    # 1. Jittered grid
    pts1 = _generate_grid_start()
    candidates.append(pts1)
    
    # 2. FCC-like
    pts2 = _generate_fcc_start()
    candidates.append(pts2)
    
    # 3. Random repulsive
    pts3 = _generate_random_repulsive()
    candidates.append(pts3)
    
    # Optimize each candidate with time budget
    best_pts = None
    best_r = 0
    
    for i, init_pts in enumerate(candidates):
        elapsed = time.time() - start_time
        remaining = time_limit - elapsed - 5.0  # 5s safety margin
        if remaining < 10.0:
            break
        
        # Allocate time: first candidate gets more, others get less
        if i == 0:
            t_budget = min(remaining * 0.5, 60.0)
        elif i == 1:
            t_budget = min(remaining * 0.3, 40.0)
        else:
            t_budget = min(remaining * 0.2, 30.0)
        
        pts, r = _local_optimize(init_pts, t_budget)
        
        if r > best_r:
            best_r = r
            best_pts = pts
    
    # Final polish on best result
    elapsed = time.time() - start_time
    remaining = time_limit - elapsed - 3.0
    if remaining > 5.0 and best_pts is not None:
        best_pts, best_r = _local_optimize(best_pts, remaining)
    
    # Ensure validity
    if best_pts is None:
        best_pts = _generate_grid_start()
    
    best_pts = np.clip(best_pts, -0.4999999, 0.4999999)
    
    return best_pts
