"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import numpy as np
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize


def _is_valid(x, y, r, tol=1e-9):
    """Check validity: circles inside [0,1]^2 and disjoint interiors."""
    n = len(x)
    if np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    # Vectorized pairwise check
    dx = x[:, None] - x[None, :]
    dy = y[:, None] - y[None, :]
    dist_sq = dx * dx + dy * dy
    min_dist = r[:, None] + r[None, :]
    # Only check upper triangle
    iu = np.triu_indices(n, k=1)
    if np.any(dist_sq[iu] < min_dist[iu] ** 2 - 1e-10):
        return False
    return True


def _score(x, y, r):
    """Sum of radii."""
    return float(np.sum(r))


def _vectorized_relax(pos, max_iters=2000, tol=1e-10, time_limit=10.0, start_time=None):
    """
    Vectorized force-based relaxation using NumPy broadcasting.
    Much faster than O(n^2) Python loops.
    """
    n = len(pos)
    pos = pos.copy().astype(np.float64)
    if start_time is None:
        start_time = time.time()
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        x = pos[:, 0]
        y = pos[:, 1]
        r = pos[:, 2]
        
        # Vectorized pairwise distances
        dx = x[:, None] - x[None, :]  # (n, n)
        dy = y[:, None] - y[None, :]  # (n, n)
        dist = np.sqrt(dx * dx + dy * dy + 1e-30)  # avoid div by zero
        min_dist = r[:, None] + r[None, :]  # (n, n)
        
        # Overlap forces: positive if overlapping
        overlap = np.maximum(min_dist - dist, 0.0)
        # Normalize direction (avoid div by zero)
        safe_dist = np.where(dist > 1e-12, dist, 1.0)
        fx = dx / safe_dist * overlap
        fy = dy / safe_dist * overlap
        
        # Sum forces on each circle
        force_x = np.sum(fx, axis=1)
        force_y = np.sum(fy, axis=1)
        
        # Wall constraints
        # Left/right walls
        wall_fx = np.zeros(n)
        wall_fy = np.zeros(n)
        # Left wall: x < r
        left_pen = np.maximum(r - x, 0.0)
        wall_fx += left_pen
        # Right wall: x > 1 - r
        right_pen = np.maximum(x - (1 - r), 0.0)
        wall_fx -= right_pen
        # Bottom wall: y < r
        bottom_pen = np.maximum(r - y, 0.0)
        wall_fy += bottom_pen
        # Top wall: y > 1 - r
        top_pen = np.maximum(y - (1 - r), 0.0)
        wall_fy -= top_pen
        
        # Apply forces with damping
        damping = 0.5
        pos[:, 0] += damping * (force_x + wall_fx)
        pos[:, 1] += damping * (force_y + wall_fy)
        
        # Clamp to valid region
        pos[:, 0] = np.clip(pos[:, 0], pos[:, 2], 1 - pos[:, 2])
        pos[:, 1] = np.clip(pos[:, 1], pos[:, 2], 1 - pos[:, 2])
        
        # Check convergence
        max_force = np.max(np.abs(force_x) + np.abs(force_y)) + np.max(np.abs(wall_fx) + np.abs(wall_fy))
        if max_force < tol:
            break
    
    # Grow radii iteratively
    for grow_iter in range(100):
        if time.time() - start_time > time_limit:
            break
        x = pos[:, 0]
        y = pos[:, 1]
        r = pos[:, 2]
        
        # Compute max possible radius for each circle
        # Wall constraints
        wall_r = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
        
        # Pairwise constraints
        dx = x[:, None] - x[None, :]
        dy = y[:, None] - y[None, :]
        dist = np.sqrt(dx * dx + dy * dy + 1e-30)
        # available[i,j] = dist[i,j] - r[j]
        available = dist - r[None, :]
        np.fill_diagonal(available, np.inf)
        pair_r = np.min(available, axis=1)
        
        max_r = np.minimum(wall_r, pair_r)
        max_r = np.maximum(max_r, 0.0)
        
        # Grow toward max
        growth = (max_r - r) * 0.3
        if np.max(growth) < 1e-10:
            break
        r = r + growth
        pos[:, 2] = r
        
        # Re-clamp positions
        pos[:, 0] = np.clip(pos[:, 0], pos[:, 2], 1 - pos[:, 2])
        pos[:, 1] = np.clip(pos[:, 1], pos[:, 2], 1 - pos[:, 2])
    
    return pos


def _simulated_annealing(pos, max_iters=5000, t_start=0.05, t_end=0.001, time_limit=20.0, start_time=None):
    """
    Simulated annealing to escape local optima.
    Moves random circles and accepts worse solutions with probability exp(-delta/T).
    """
    n = len(pos)
    if start_time is None:
        start_time = time.time()
    best_pos = pos.copy()
    best_score = _score(best_pos[:, 0], best_pos[:, 1], best_pos[:, 2])
    curr_pos = pos.copy()
    curr_score = best_score
    
    rng = np.random.RandomState(42)
    
    for it in range(max_iters):
        if time.time() - start_time > time_limit:
            break
        
        # Annealing schedule
        T = t_start * (t_end / t_start) ** (it / max_iters)
        
        # Pick random circle to move
        i = rng.randint(n)
        angle = rng.uniform(0, 2 * np.pi)
        step = rng.uniform(0.005, 0.05)
        
        new_pos = curr_pos.copy()
        new_pos[i, 0] += step * np.cos(angle)
        new_pos[i, 1] += step * np.sin(angle)
        
        # Clamp
        new_pos[i, 0] = np.clip(new_pos[i, 0], new_pos[i, 2], 1 - new_pos[i, 2])
        new_pos[i, 1] = np.clip(new_pos[i, 1], new_pos[i, 2], 1 - new_pos[i, 2])
        
        # Check validity
        if _is_valid(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2]):
            new_score = _score(new_pos[:, 0], new_pos[:, 1], new_pos[:, 2])
            delta = new_score - curr_score
            if delta > 0 or rng.random() < np.exp(-delta / T):
                curr_pos = new_pos
                curr_score = new_score
                if curr_score > best_score:
                    best_pos = new_pos.copy()
                    best_score = curr_score
    
    return best_pos


def _make_grid_initial(n):
    """Create a grid initial placement."""
    side = int(np.ceil(np.sqrt(n)))
    positions = []
    for i in range(side):
        for j in range(side):
            if len(positions) < n:
                x = (i + 0.5) / side
                y = (j + 0.5) / side
                r = 0.5 / side - 1e-4
                positions.append([x, y, r])
    return np.array(positions[:n])


def _make_hex_initial(n):
    """Create a hexagonal packing initial placement."""
    positions = []
    row_height = 1.0 / (n + 2)
    col_width = row_height * 2 / np.sqrt(3)
    y = row_height * 0.5
    row = 0
    while len(positions) < n:
        x_offset = 0.0
        if row % 2 == 1:
            x_offset = col_width * 0.5
        x = x_offset + col_width * 0.5
        col = 0
        while len(positions) < n and x <= 1.0:
            r = row_height * 0.5 - 1e-4
            positions.append([x, y, r])
            x += col_width
            col += 1
        y += row_height
        row += 1
    return np.array(positions[:n])


def _make_corner_heavy(n, seed=0):
    """Place circles near corners and edges for better packing."""
    rng = np.random.RandomState(seed)
    positions = []
    # Place some near corners
    corners = [(0.1, 0.1), (0.9, 0.1), (0.1, 0.9), (0.9, 0.9)]
    for cx, cy in corners:
        if len(positions) < n:
            positions.append([cx + rng.uniform(-0.02, 0.02), cy + rng.uniform(-0.02, 0.02), 0.05])
    # Fill rest randomly
    while len(positions) < n:
        x = rng.uniform(0.08, 0.92)
        y = rng.uniform(0.08, 0.92)
        r = 0.03 + rng.random() * 0.04
        positions.append([x, y, r])
    return np.array(positions[:n])


def _make_random_initial(n, seed):
    """Create a random initial placement."""
    rng = np.random.RandomState(seed)
    positions = []
    for _ in range(n):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        r = 0.01 + rng.random() * 0.05
        positions.append([x, y, r])
    return np.array(positions[:n])


def construct():
    start_time = time.time()
    deadline = start_time + 130  # 20s safety margin
    
    # Generate multiple initial configurations
    inits = []
    inits.append(_make_grid_initial(N))
    inits.append(_make_hex_initial(N))
    inits.append(_make_corner_heavy(N, seed=0))
    inits.append(_make_corner_heavy(N, seed=1))
    
    # Add random inits
    for seed in range(8):
        if time.time() > deadline - 30:
            break
        inits.append(_make_random_initial(N, seed))
    
    best_result = None
    best_score = -np.inf
    
    for init in inits:
        if time.time() > deadline - 30:
            break
        
        # Vectorized relaxation (fast)
        relaxed = _vectorized_relax(init, max_iters=3000, time_limit=15.0, start_time=start_time)
        
        # Simulated annealing polish
        sa_result = _simulated_annealing(relaxed, max_iters=3000, time_limit=15.0, start_time=start_time)
        
        # Final vectorized relax after SA
        final = _vectorized_relax(sa_result, max_iters=2000, time_limit=10.0, start_time=start_time)
        
        score = _score(final[:, 0], final[:, 1], final[:, 2])
        
        if _is_valid(final[:, 0], final[:, 1], final[:, 2]):
            if score > best_score:
                best_score = score
                best_result = final
    
    if best_result is None:
        best_result = _make_grid_initial(N)
    
    return best_result
# EVOLVE-BLOCK-END
