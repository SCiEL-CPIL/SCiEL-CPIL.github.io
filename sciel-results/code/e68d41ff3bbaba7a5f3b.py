"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time

N = 50

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize

def _compute_radius(centers):
    """Compute the maximum common radius for a set of circle centers."""
    # Distance to boundary
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    # Pairwise distances - only compute upper triangle for speed
    n = len(centers)
    min_pair = np.inf
    for i in range(n):
        # Vectorized: distance from center i to all centers j > i
        diffs = centers[i+1:] - centers[i]
        dists = np.sqrt(np.sum(diffs * diffs, axis=1))
        if len(dists) > 0:
            local_min = np.min(dists)
            if local_min < min_pair:
                min_pair = local_min
    
    min_pair /= 2.0
    return min(min_boundary, min_pair)


def _compute_radius_fast(centers):
    """Fast radius computation using cdist upper triangle."""
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    dists = cdist(centers, centers)
    # Only look at upper triangle
    iu = np.triu_indices(len(centers), k=1)
    min_pair = dists[iu[0], iu[1]].min() / 2.0
    
    return min(min_boundary, min_pair)


def _soft_min_radius(centers, beta=50.0):
    """Smooth approximation of min radius using log-sum-exp."""
    boundary = 0.5 - np.max(np.abs(centers), axis=1)
    min_boundary = np.min(boundary)
    
    if min_boundary <= 0:
        return 0.0
    
    n = len(centers)
    # Compute all pairwise distances
    diffs = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diffs * diffs, axis=2))
    np.fill_diagonal(dists, np.inf)
    
    # Log-sum-exp for soft minimum of distances/2
    half_dists = dists / 2.0
    min_d = half_dists.min()
    # Soft min: -log(sum(exp(-beta*x)))/beta
    exp_vals = np.exp(-beta * (half_dists - min_d))
    soft_min_dist = min_d - np.log(exp_vals.sum()) / beta
    
    return min(min_boundary, soft_min_dist)


def _hex_init(n):
    """Create a hexagonal grid initialization with multiple configurations."""
    best_centers = None
    best_score = -1
    
    for rows in range(4, 14):
        cols = int(np.ceil(n / rows))
        if cols < 3 or cols > 15:
            continue
        
        dx = 1.0 / (cols + 0.5)
        dy = dx * np.sqrt(3) / 2
        
        x_start = -0.5 + dx / 2
        y_start = -0.5 + dy / 2
        
        centers = []
        for i in range(rows):
            for j in range(cols):
                if len(centers) >= n:
                    break
                x = x_start + j * dx
                y = y_start + i * dy
                if i % 2 == 1:
                    x += dx / 2
                centers.append([x, y])
        
        if len(centers) >= n:
            centers = np.array(centers[:n])
            centers = np.clip(centers, -0.5, 0.5)
            s = _compute_radius(centers)
            if s > best_score:
                best_score = s
                best_centers = centers
    
    if best_centers is None:
        m = int(np.ceil(np.sqrt(n)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        best_centers = grid[:n]
    
    return best_centers


def _force_directed_relax(centers, max_iter=3000, time_budget=20.0):
    """Force-directed relaxation: push apart overlapping circles, pull toward center."""
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    # Target minimum distance based on current best
    target_d = max(current_score * 2.0, 0.05)
    
    for it in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        forces = np.zeros_like(current)
        
        # Pairwise repulsion for circles that are too close
        for i in range(N):
            for j in range(i + 1, N):
                diff = current[i] - current[j]
                d = np.sqrt(np.sum(diff * diff))
                if d < target_d and d > 1e-10:
                    # Force proportional to overlap
                    overlap = (target_d - d) / target_d
                    force_mag = overlap * 0.1
                    unit = diff / d
                    forces[i] += force_mag * unit
                    forces[j] -= force_mag * unit
                elif d <= 1e-10:
                    # Coincident: random small push
                    forces[i] += np.random.randn(2) * 0.01
                    forces[j] -= np.random.randn(2) * 0.01
        
        # Boundary forces: push inward if too close to wall
        for i in range(N):
            for k in range(2):
                margin = 0.5 - abs(current[i, k])
                if margin < current_score:
                    push = (current_score - margin) * 0.5
                    if current[i, k] > 0:
                        forces[i, k] -= push
                    else:
                        forces[i, k] += push
        
        # Apply forces with adaptive step
        step = 0.005 * (1.0 - it / max_iter) + 0.001
        current = current + forces * step
        current = np.clip(current, -0.5, 0.5)
        
        new_score = _compute_radius(current)
        if new_score > best_score:
            best = current.copy()
            best_score = new_score
    
    return best, best_score


def _local_search(centers, max_iter=2000, step_init=0.002, time_budget=20.0):
    """Efficient local search using coordinate descent with adaptive steps."""
    start_time = time.time()
    current = centers.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    step = step_init
    no_improve = 0
    
    for iteration in range(max_iter):
        if time.time() - start_time > time_budget:
            break
        
        improved = False
        
        for i in range(N):
            for dx in [step, -step]:
                for dy in [step, -step, 0]:
                    new_c = current.copy()
                    new_c[i] = current[i] + np.array([dx, dy])
                    
                    if np.any(np.abs(new_c[i]) > 0.5):
                        continue
                    
                    new_score = _compute_radius(new_c)
                    
                    if new_score > current_score + 1e-12:
                        current = new_c
                        current_score = new_score
                        improved = True
                        
                        if current_score > best_score:
                            best = current.copy()
                            best_score = current_score
                        
                        break
                if improved:
                    break
            if improved:
                break
        
        if improved:
            no_improve = 0
        else:
            no_improve += 1
            step *= 0.7
            if step < 1e-7:
                break
    
    return best, best_score


def _smooth_optimize(centers, time_budget=30.0):
    """Optimize using smooth log-sum-exp surrogate with L-BFGS-B."""
    start_time = time.time()
    
    def objective(x_flat):
        if time.time() - start_time > time_budget:
            return 0.0
        centers_2d = x_flat.reshape(-1, 2)
        # Maximize soft radius = minimize negative soft radius
        r = _soft_min_radius(centers_2d, beta=80.0)
        return -r
    
    x0 = centers.flatten()
    
    # Bounds
    bounds = [(-0.5, 0.5)] * (2 * N)
    
    result = minimize(
        objective,
        x0,
        method='L-BFGS-B',
        bounds=bounds,
        options={
            'maxiter': 200,
            'ftol': 1e-12,
            'gtol': 1e-8,
            'maxls': 50
        }
    )
    
    polished = result.x.reshape(-1, 2)
    polished = np.clip(polished, -0.5, 0.5)
    return polished


def _sa_improve(centers, time_budget=40.0):
    """Simulated annealing with improved efficiency."""
    start_time = time.time()
    deadline = start_time + time_budget
    
    current = centers.copy()
    current_score = _compute_radius(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.004
    step_size = 0.003
    cooling = 0.999
    
    while T > 1e-7:
        if time.time() > deadline - 2:
            break
        
        for _ in range(300):
            if time.time() > deadline - 2:
                break
            
            idx = np.random.randint(N)
            dx = np.random.uniform(-step_size, step_size)
            dy = np.random.uniform(-step_size, step_size)
            
            new_center = current.copy()
            new_center[idx] += [dx, dy]
            new_center[idx] = np.clip(new_center[idx], -0.5, 0.5)
            
            new_score = _compute_radius(new_center)
            
            if new_score >= current_score:
                current = new_center
                current_score = new_score
                if current_score > best_score:
                    best = current.copy()
                    best_score = current_score
            else:
                delta = new_score - current_score
                if np.random.random() < np.exp(delta / T):
                    current = new_center
                    current_score = new_score
        
        T *= cooling
        if T < 0.0005:
            step_size = 0.0008
        elif T < 0.002:
            step_size = 0.0015
    
    return best, best_score


def construct():
    start_time = time.time()
    total_time_limit = 140.0  # Leave 10s margin for safety
    
    np.random.seed(12345)
    
    best_centers = None
    best_score = -1
    
    # Hex init
    centers = _hex_init(N)
    score = _compute_radius(centers)
    if score > best_score:
        best_centers = centers.copy()
        best_score = score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Force-directed relaxation (fast, parallel-friendly)
    if remaining > 30:
        fd_best, fd_score = _force_directed_relax(best_centers, max_iter=2000, time_budget=min(15, remaining - 5))
        if fd_score > best_score:
            best_centers = fd_best
            best_score = fd_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Smooth surrogate optimization (L-BFGS-B on soft min)
    if remaining > 35:
        smooth_result = _smooth_optimize(best_centers, time_budget=min(20, remaining - 5))
        smooth_score = _compute_radius(smooth_result)
        if smooth_score > best_score:
            best_centers = smooth_result
            best_score = smooth_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Local search polish
    if remaining > 15:
        polished, pol_score = _local_search(best_centers, max_iter=1000, step_init=0.001, time_budget=min(10, remaining - 3))
        if pol_score > best_score:
            best_centers = polished
            best_score = pol_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # SA improvement
    if remaining > 25:
        sa_best, sa_score = _sa_improve(best_centers, time_budget=min(35, remaining - 5))
        if sa_score > best_score:
            best_centers = sa_best
            best_score = sa_score
    
    elapsed = time.time() - start_time
    remaining = total_time_limit - elapsed
    
    # Final local search polish
    if remaining > 10:
        final_polished, final_score = _local_search(best_centers, max_iter=500, step_init=0.0005, time_budget=min(8, remaining - 2))
        if final_score > best_score:
            best_centers = final_polished
            best_score = final_score
    
    # Ensure validity
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    # Final validation check
    final_r = _compute_radius(best_centers)
    if final_r <= 0 or not np.all(np.isfinite(best_centers)):
        best_centers = _hex_init(N)
    
    return best_centers
# EVOLVE-BLOCK-END
