"""Combined program: hexagonal initialization + energy minimization + bottleneck refinement."""
import numpy as np
import time
from scipy.spatial.distance import pdist

N = 200
START_TIME = time.time()
TIME_LIMIT = 150.0
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


def _hexagonal_initial():
    """Create hexagonal packing initial configuration."""
    # Hexagonal packing with spacing s
    # For N points in unit square, estimate spacing
    # Area per point ~ 1/N, hexagon area per point = s^2 * sqrt(3)/2
    # So s ~ sqrt(2/(N*sqrt(3)))
    s = np.sqrt(2.0 / (N * np.sqrt(3)))
    
    # Hexagonal grid
    dx = s
    dy = s * np.sqrt(3) / 2
    
    centers = []
    row = 0
    while len(centers) < N:
        x_offset = 0 if row % 2 == 0 else dx / 2
        col = 0
        while len(centers) < N:
            x = x_offset + col * dx - 0.5 + dx / 2
            y = row * dy - 0.5 + dy / 2
            if -0.5 + 1e-6 <= x <= 0.5 - 1e-6 and -0.5 + 1e-6 <= y <= 0.5 - 1e-6:
                centers.append([x, y])
            col += 1
        row += 1
    
    centers = np.array(centers[:N])
    # Add small jitter to break symmetry
    centers += np.random.normal(0, 0.001, centers.shape)
    centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
    return centers


def _optimize_energy(centers, time_limit=20.0):
    """Global energy minimization with repulsion forces."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    learning_rate = 0.002
    decay = 0.9995
    repulsion_power = 4
    
    for step in range(200000):
        elapsed = time.time() - start_time
        if elapsed > time_limit:
            break
        
        # Compute repulsion forces
        diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
        dist_sq = np.sum(diff**2, axis=2)
        np.fill_diagonal(dist_sq, 1e-10)
        dist = np.sqrt(dist_sq)
        
        repulsion = 1.0 / (dist_sq ** (repulsion_power / 2.0))
        np.fill_diagonal(repulsion, 0)
        
        forces = np.sum(repulsion[:, :, np.newaxis] * diff, axis=1)
        forces *= learning_rate
        
        # Wall repulsion
        dist_left = centers[:, 0] + 0.5
        dist_right = 0.5 - centers[:, 0]
        dist_bottom = centers[:, 1] + 0.5
        dist_top = 0.5 - centers[:, 1]
        
        wall_strength = 0.3
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


def _bottleneck_refine(centers, time_limit=20.0):
    """Bottleneck-driven local optimization."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.003
    no_improve_count = 0
    
    for it in range(300000):
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
            if no_improve_count > 150:
                idx = np.random.randint(N)
                centers[idx] += np.random.randn(2) * 0.003
                centers[idx] = np.clip(centers[idx], -0.5 + 1e-10, 0.5 - 1e-10)
                no_improve_count = 0
                step_size = max(0.0003, step_size * 0.9)
            elif no_improve_count > 75:
                step_size = max(0.0003, step_size * 0.95)
    
    return best_centers, best_r


def _polish(centers, time_limit=10.0):
    """High-precision polish with very small steps."""
    start_time = time.time()
    centers = centers.copy().astype(np.float64)
    best_r = _compute_radius(centers)
    best_centers = centers.copy()
    
    step_size = 0.00005
    
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
    """Main function: combine multiple strategies to maximize circle packing radius."""
    if _remaining_time() < 5.0:
        # Not enough time, return basic hexagonal
        return _hexagonal_initial()
    
    # Try multiple initializations and keep the best
    best_centers = None
    best_r = 0.0
    
    # Strategy 1: Hexagonal initialization
    if _remaining_time() > 50.0:
        centers = _hexagonal_initial()
        centers, r = _optimize_energy(centers, time_limit=min(20.0, _remaining_time() - 40.0))
        centers, r = _bottleneck_refine(centers, time_limit=min(20.0, _remaining_time() - 20.0))
        centers, r = _polish(centers, time_limit=min(10.0, _remaining_time() - 5.0))
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Strategy 2: Grid initialization with different parameters
    if _remaining_time() > 50.0 and best_r < 0.0365:
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        centers = grid[:N].copy()
        centers += np.random.normal(0, 0.001, centers.shape)
        centers = np.clip(centers, -0.5 + 1e-6, 0.5 - 1e-6)
        
        centers, r = _optimize_energy(centers, time_limit=min(20.0, _remaining_time() - 40.0))
        centers, r = _bottleneck_refine(centers, time_limit=min(20.0, _remaining_time() - 20.0))
        centers, r = _polish(centers, time_limit=min(10.0, _remaining_time() - 5.0))
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    # Strategy 3: Random initialization
    if _remaining_time() > 50.0 and best_r < 0.0365:
        centers = np.random.uniform(-0.5 + 0.02, 0.5 - 0.02, (N, 2))
        
        centers, r = _optimize_energy(centers, time_limit=min(20.0, _remaining_time() - 40.0))
        centers, r = _bottleneck_refine(centers, time_limit=min(20.0, _remaining_time() - 20.0))
        centers, r = _polish(centers, time_limit=min(10.0, _remaining_time() - 5.0))
        if r > best_r:
            best_r = r
            best_centers = centers.copy()
    
    if best_centers is None:
        return _hexagonal_initial()
    
    return best_centers
