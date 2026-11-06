"""Combined program: global repulsion + bottleneck-driven local optimization + polish."""
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.spatial import cKDTree

N = 200
START_TIME = time.time()
TIME_LIMIT = 140.0
SAFETY_MARGIN = 10.0


def _remaining_time():
    return TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)


def _compute_radius(centers):
    """Compute the maximum radius for given centers."""
    if len(centers) == 0:
        return 0.0
    dw = np.minimum(
        np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
        np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
    )
    wall_term = np.min(dw)
    n = len(centers)
    if n < 2:
        return float(wall_term)
    d = pdist(centers)
    min_pair = np.min(d)
    pair_term = min_pair / 2.0
    return min(float(wall_term), float(pair_term))


def _pdist_index_to_pair(k, n):
    """Convert pdist condensed index to (i, j) with i < j."""
    start = 0
    for i in range(n - 1):
        block_size = n - i - 1
        if k < start + block_size:
            j = i + 1 + (k - start)
            return i, j
        start += block_size
    return 0, 1


def _initial_grid():
    """Create initial grid arrangement with slight jitter."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers


def _optimize_repulsion(centers, time_limit=18.0):
    """Global repulsion-based optimization."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    learning_rate = 0.001
    decay = 0.999
    repulsion_power = 4
    
    for step in range(100000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        forces = np.zeros_like(centers)
        
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, 1e-10)
        
        repulsion = 1.0 / (dist ** repulsion_power)
        np.fill_diagonal(repulsion, 0)
        
        force_mag = repulsion * learning_rate
        forces += np.sum(force_mag[:, :, np.newaxis] * diff, axis=1)
        
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]
        
        wall_strength = 0.5
        forces[:, 0] += wall_strength * learning_rate / (dist_left ** 2 + 1e-10)
        forces[:, 0] -= wall_strength * learning_rate / (dist_right ** 2 + 1e-10)
        forces[:, 1] += wall_strength * learning_rate / (dist_bottom ** 2 + 1e-10)
        forces[:, 1] -= wall_strength * learning_rate / (dist_top ** 2 + 1e-10)
        
        centers += forces
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)
        
        current_r = _compute_radius(centers)
        if current_r > best_r:
            best_r = current_r
            best_centers = centers.copy()
        
        learning_rate *= decay
    
    return best_centers, best_r


def _bottleneck_local_opt(centers, time_limit=15.0):
    """Bottleneck-driven local optimization: push apart closest pairs and wall-limited points."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.005
    no_improve_count = 0
    
    for it in range(200000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        if min_pair_dist / 2.0 <= min_wall_dist:
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move
        
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 100:
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.005
                centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
                no_improve_count = 0
                step_size = max(0.0005, step_size * 0.9)
            elif no_improve_count > 50:
                step_size = max(0.0005, step_size * 0.95)
    
    return best_centers, best_r


def _polish(centers, time_limit=5.0):
    """High-precision polish: very small steps on bottleneck constraints."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.0001
    
    for it in range(500000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        if min_pair_dist / 2.0 <= min_wall_dist:
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                continue
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            idx = min_wall_idx
            x, y = centers[idx]
            dl = x + 0.5
            dr = 0.5 - x
            db = y + 0.5
            dt = 0.5 - y
            min_d = min(dl, dr, db, dt)
            if min_d == dl:
                move = np.array([step_size, 0])
            elif min_d == dr:
                move = np.array([-step_size, 0])
            elif min_d == db:
                move = np.array([0, step_size])
            else:
                move = np.array([0, -step_size])
            centers[idx] += move
        
        centers = np.clip(centers, -0.5 + 1e-12, 0.5 - 1e-12)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
    
    return best_centers, best_r


def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    np.random.seed(42)
    best_centres = _initial_grid()
    best_r = _compute_radius(best_centres)
    
    # Phase 1: Global repulsion to escape initial grid symmetry
    if _remaining_time() > 40.0:
        rep_centers, rep_r = _optimize_repulsion(best_centres, time_limit=20.0)
        if rep_r > best_r:
            best_r = rep_r
            best_centres = rep_centers.copy()
    
    # Phase 2: Bottleneck-driven local optimization with random restarts
    num_restarts = 8
    for restart in range(num_restarts):
        if _remaining_time() < 25.0:
            break
        
        if restart == 0:
            centres = best_centres.copy()
        else:
            centres = best_centres + np.random.randn(N, 2) * 0.015
            centres = np.clip(centres, -0.5 + 1e-6, 0.5 - 1e-6)
        
        # Bottleneck local optimization
        local_centers, local_r = _bottleneck_local_opt(centres, time_limit=12.0)
        
        # Polish
        if _remaining_time() > 10.0:
            polished_centers, polished_r = _polish(local_centers, time_limit=5.0)
            if polished_r > local_r:
                local_r = polished_r
                local_centers = polished_centers
        
        if local_r > best_r:
            best_r = local_r
            best_centres = local_centers.copy()
    
    # Final polish on the best result
    if _remaining_time() > 5.0:
        final_centers, final_r = _polish(best_centres, time_limit=5.0)
        if final_r > best_r:
            best_r = final_r
            best_centres = final_centers
    
    best_centres = np.clip(best_centres, -0.5, 0.5)
    return best_centres
