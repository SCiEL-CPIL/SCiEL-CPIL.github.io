"""Improved program: simulated annealing + local refinement for 100-sphere packing."""
import numpy as np
import time

N = 100
DEADLINE = 140.0  # safety margin before 150s kill

def _min_distance_and_radius(pts):
    """Return (min_pair_dist, min_wall_dist, overall_radius)."""
    # Pairwise distances
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    d2 = np.sum(diff * diff, axis=2)
    np.fill_diagonal(d2, np.inf)
    min_pair = np.sqrt(np.min(d2))
    # Wall distances
    wall = 0.5 - np.abs(pts)
    min_wall = np.min(wall)
    r = min(min_pair / 2.0, min_wall)
    return min_pair, min_wall, r

def _score(pts):
    """The actual score: largest common radius."""
    _, _, r = _min_distance_and_radius(pts)
    return r

def _local_optimize(pts, time_budget=2.0):
    """Local optimization: iteratively push apart close pairs and pull toward center."""
    best = pts.copy()
    best_r = _score(best)
    t0 = time.time()
    lr = 0.01
    for _ in range(2000):
        if time.time() - t0 > time_budget:
            break
        # Compute forces
        diff = best[:, np.newaxis, :] - best[np.newaxis, :]
        d2 = np.sum(diff * diff, axis=2)
        np.fill_diagonal(d2, 1e-30)
        dist = np.sqrt(d2)
        # Repulsion: push apart close pairs
        force = np.zeros_like(best)
        # Only act on pairs closer than some threshold
        mask = dist < 0.5
        for i in range(N):
            for j in range(N):
                if i != j and mask[i, j]:
                    d = dist[i, j]
                    if d < 1e-10:
                        # Random nudge
                        force[i] += 0.01 * np.random.randn(3)
                    else:
                        strength = (0.5 - d) / d
                        force[i] += strength * diff[i, j]
        # Wall repulsion: push inward if too close to wall
        for i in range(N):
            for k in range(3):
                if best[i, k] > 0.3:
                    force[i, k] -= (best[i, k] - 0.3) * 0.5
                elif best[i, k] < -0.3:
                    force[i, k] -= (best[i, k] + 0.3) * 0.5
        
        # Apply update
        new_pts = best + lr * force
        # Clamp to cube
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = _score(new_pts)
        if new_r > best_r:
            best = new_pts
            best_r = new_r
            lr = max(lr * 0.99, 0.001)
        else:
            lr = max(lr * 0.95, 0.001)
    return best, best_r

def _simulated_annealing(pts, time_budget=60.0):
    """Simulated annealing to maximize min distance."""
    best = pts.copy()
    best_r = _score(best)
    t0 = time.time()
    T = 0.01
    T_min = 1e-6
    cooling = 0.999
    
    while T > T_min:
        if time.time() - t0 > time_budget:
            break
        
        # Generate neighbor: perturb a random subset of points
        n_perturb = np.random.randint(1, 5)
        idx = np.random.choice(N, n_perturb, replace=False)
        new_pts = best.copy()
        new_pts[idx] += T * np.random.randn(n_perturb, 3)
        new_pts = np.clip(new_pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        new_r = _score(new_pts)
        delta = new_r - best_r
        
        if delta > 0 or np.random.random() < np.exp(delta / T):
            best = new_pts
            best_r = new_r
        
        T *= cooling
    
    return best, best_r

def _initial_grid():
    """Create initial configuration using a 5x5x4 grid."""
    # 5x5x4 = 100 points
    xs = np.linspace(-0.4, 0.4, 5)
    ys = np.linspace(-0.4, 0.4, 5)
    zs = np.linspace(-0.375, 0.375, 4)
    pts = np.array([(x, y, z) for z in zs for y in ys for x in xs])
    return pts

def construct():
    t_start = time.time()
    
    # Start from grid
    pts = _initial_grid()
    best_pts = pts.copy()
    best_r = _score(pts)
    
    # Phase 1: Local optimization
    if time.time() - t_start < DEADLINE - 30:
        pts, r = _local_optimize(pts, time_budget=10.0)
        if r > best_r:
            best_pts = pts
            best_r = r
    
    # Phase 2: Simulated annealing with restarts
    t0 = time.time()
    while time.time() - t_start < DEADLINE - 5:
        # Occasionally restart from perturbed best
        if np.random.random() < 0.1:
            pts = best_pts + 0.05 * np.random.randn(N, 3)
            pts = np.clip(pts, -0.5 + 1e-9, 0.5 - 1e-9)
        
        # Simulated annealing step
        remaining = DEADLINE - 5 - (time.time() - t_start)
        if remaining < 5:
            break
        pts, r = _simulated_annealing(pts, time_budget=min(remaining, 20.0))
        if r > best_r:
            best_pts = pts
            best_r = r
        
        # Local refine after SA
        if time.time() - t_start < DEADLINE - 10:
            pts, r = _local_optimize(pts, time_budget=5.0)
            if r > best_r:
                best_pts = pts
                best_r = r
    
    # Final local optimization
    if time.time() - t_start < DEADLINE - 2:
        best_pts, best_r = _local_optimize(best_pts, time_budget=3.0)
    
    # Ensure validity
    best_pts = np.clip(best_pts, -0.5 + 1e-9, 0.5 - 1e-9)
    
    return best_pts
