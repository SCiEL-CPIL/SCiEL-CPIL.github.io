"""
Combined program for packing 200 equal circles in a unit square.
Strategy: 
1. Initialize with a grid.
2. Global repulsion to spread out.
3. Bottleneck-driven local optimization (push apart the closest pair or wall-constrained point).
4. Simulated annealing style escapes to break local minima.
5. High-precision polish.
"""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 145.0  # Safety margin for 150s limit

def _remaining_time():
    return TIME_LIMIT - (time.time() - START_TIME)

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
    """Create initial grid arrangement."""
    m = int(np.ceil(np.sqrt(N)))
    # For N=200, m=15 (225 cells). We take the first 200.
    # A better grid might be 14x15 = 210. Let's use a rectangular grid.
    # 14 * 15 = 210.
    rows = 14
    cols = 15
    ticks_x = (np.arange(cols) + 0.5) / cols - 0.5
    ticks_y = (np.arange(rows) + 0.5) / rows - 0.5
    grid = np.array([(x, y) for y in ticks_y for x in ticks_x])
    centers = grid[:N].copy()
    # Add small jitter to break symmetry
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers

def _optimize_repulsion(centers, time_limit=10.0):
    """Global repulsion-based optimization to get a good starting point."""
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

def _bottleneck_local_opt(centers, time_limit=120.0):
    """Bottleneck-driven local optimization with annealing escapes."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.002
    no_improve_count = 0
    max_no_improve = 50
    
    for it in range(1000000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute current radius and identify bottleneck
        dw = np.minimum(
            np.minimum(0.5 + centers[:, 0], 0.5 - centers[:, 0]),
            np.minimum(0.5 + centers[:, 1], 0.5 - centers[:, 1])
        )
        min_wall_idx = np.argmin(dw)
        min_wall_dist = dw[min_wall_idx]
        
        d = pdist(centers)
        min_pair_idx = np.argmin(d)
        min_pair_dist = d[min_pair_idx]
        
        # Determine if bottleneck is pair or wall
        if min_pair_dist / 2.0 <= min_wall_dist:
            # Bottleneck is a pair
            i, j = _pdist_index_to_pair(min_pair_idx, N)
            vec = centers[i] - centers[j]
            norm = np.linalg.norm(vec)
            if norm < 1e-9:
                # Coincident points, random push
                vec = np.random.randn(2)
                norm = np.linalg.norm(vec)
            direction = vec / norm
            move = step_size * direction
            centers[i] += move
            centers[j] -= move
        else:
            # Bottleneck is a wall
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
            if no_improve_count > max_no_improve:
                # Escape local minimum: perturb a random point
                # Choose a point involved in the current bottleneck or random
                if np.random.rand() < 0.5:
                    # Perturb one of the bottleneck points
                    if min_pair_dist / 2.0 <= min_wall_dist:
                        idx = np.random.choice([i, j])
                    else:
                        idx = min_wall_idx
                else:
                    idx = np.random.randint(N)
                
                centers[idx] += np.random.randn(2) * 0.005
                centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
                
                no_improve_count = 0
                # Slightly reduce step size to help convergence after escape
                step_size = max(0.0005, step_size * 0.9)
        
        # Gradually reduce step size for finer tuning
        if it % 100 == 0:
            step_size = max(0.0001, step_size * 0.995)
            
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
        
        centers = np.clip(centers, -0.5 + 1e-10, 0.5 - 1e-10)
        
        new_r = _compute_radius(centers)
        if new_r > best_r:
            best_r = new_r
            best_centers = centers.copy()
            
    return best_centers, best_r

def construct():
    """Construct the optimal packing of 200 equal circles in a unit square."""
    start_time = time.time()
    
    # 1. Initial grid
    centers = _initial_grid()
    
    # 2. Global repulsion to spread out (quick phase)
    centers, r = _optimize_repulsion(centers, time_limit=5.0)
    
    # 3. Bottleneck local optimization (main workhorse)
    # Use most of the time budget here
    elapsed = time.time() - start_time
    time_left = TIME_LIMIT - elapsed - 10.0  # Reserve 10s for polish and safety
    if time_left > 10.0:
        centers, r = _bottleneck_local_opt(centers, time_limit=time_left)
    
    # 4. Polish
    elapsed = time.time() - start_time
    time_left = TIME_LIMIT - elapsed - 2.0  # Reserve 2s for return
    if time_left > 1.0:
        centers, r = _polish(centers, time_limit=time_left)
    
    return centers
