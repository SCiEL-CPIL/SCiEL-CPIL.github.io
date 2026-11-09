import numpy as np
import time

N = 50
DEADLINE = 145.0  # safety margin before 150s

def _compute_radius(centers):
    """Compute the radius score for a given set of centers."""
    if len(centers) != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5) or np.any(centers > 0.5):
        return 0.0
    
    # Boundary distances
    d_left = centers[:, 0] + 0.5
    d_right = 0.5 - centers[:, 0]
    d_bottom = centers[:, 1] + 0.5
    d_top = 0.5 - centers[:, 1]
    
    min_boundary = np.min(np.min(np.stack([d_left, d_right, d_bottom, d_top]), axis=0))
    
    # Pairwise distances
    min_pairwise = np.inf
    for i in range(N):
        for j in range(i + 1, N):
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < min_pairwise:
                min_pairwise = dist
                
    min_pairwise_radius = min_pairwise / 2.0
    
    return min(min_boundary, min_pairwise_radius)

def _grid_init(m, n, rng):
    """Generate a grid initialization with random jitter."""
    pts = []
    for i in range(n):
        for j in range(m):
            if len(pts) >= N:
                break
            x = (j + 0.5) / m - 0.5
            y = (i + 0.5) / n - 0.5
            pts.append([x, y])
    pts = np.array(pts[:N])
    # Add small random jitter to break symmetry
    jitter = rng.normal(0, 0.02, size=(N, 2))
    pts = pts + jitter
    return np.clip(pts, -0.49, 0.49)

def _random_init(rng):
    """Generate a random initialization."""
    return rng.uniform(-0.45, 0.45, size=(N, 2))

def _hex_init(rng):
    """Generate a hexagonal-ish initialization."""
    # Try to fit 50 circles in a hex pattern
    # 8 columns, 7 rows = 56, take 50
    # Or 7 columns, 8 rows = 56
    pts = []
    # Hex grid: alternate rows offset
    spacing_x = 1.0 / 8.0
    spacing_y = spacing_x * np.sqrt(3) / 2
    
    for row in range(8):
        offset = spacing_x / 2 if row % 2 == 1 else 0
        for col in range(8):
            if len(pts) >= N:
                break
            x = (col + 0.5) * spacing_x - 0.5 + offset
            y = (row + 0.5) * spacing_y - 0.5
            if -0.5 <= x <= 0.5 and -0.5 <= y <= 0.5:
                pts.append([x, y])
    
    pts = np.array(pts[:N])
    if len(pts) < N:
        # Fill remaining with random
        while len(pts) < N:
            pts = np.vstack([pts, rng.uniform(-0.45, 0.45, size=(1, 2))])
    pts = pts[:N]
    
    # Add small random jitter
    jitter = rng.normal(0, 0.01, size=(N, 2))
    pts = pts + jitter
    return np.clip(pts, -0.49, 0.49)

def _local_optimize(centers, time_limit=5.0):
    """
    Local optimization using gradient-based method on the radius objective.
    We maximize the minimum of:
    - pairwise distances / 2
    - boundary distances
    
    Use a simple approach: iteratively push the most constrained circles apart.
    """
    start_time = time.time()
    centers = centers.copy()
    
    while time.time() - start_time < time_limit:
        # Compute current radius
        r = _compute_radius(centers)
        
        # Find the most constraining pairs
        # Compute all pairwise distances
        min_dist = np.inf
        min_i, min_j = -1, -1
        min_boundary = np.inf
        min_boundary_idx = -1
        min_boundary_side = -1
        
        # Pairwise
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i, 0] - centers[j, 0]
                dy = centers[i, 1] - centers[j, 1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < min_dist:
                    min_dist = dist
                    min_i, min_j = i, j
        
        # Boundary
        for i in range(N):
            d_left = centers[i, 0] + 0.5
            d_right = 0.5 - centers[i, 0]
            d_bottom = centers[i, 1] + 0.5
            d_top = 0.5 - centers[i, 1]
            
            if d_left < min_boundary:
                min_boundary = d_left
                min_boundary_idx = i
                min_boundary_side = 'left'
            if d_right < min_boundary:
                min_boundary = d_right
                min_boundary_idx = i
                min_boundary_side = 'right'
            if d_bottom < min_boundary:
                min_boundary = d_bottom
                min_boundary_idx = i
                min_boundary_side = 'bottom'
            if d_top < min_boundary:
                min_boundary = d_top
                min_boundary_idx = i
                min_boundary_side = 'top'
        
        current_r = min(min_dist / 2.0, min_boundary)
        
        # Determine which constraint is most binding
        if min_dist / 2.0 <= min_boundary:
            # Pairwise constraint is binding
            i, j = min_i, min_j
            dx = centers[i, 0] - centers[j, 0]
            dy = centers[i, 1] - centers[j, 1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < 1e-10:
                # Random perturbation
                centers[i, 0] += 0.001
                centers[j, 0] -= 0.001
            else:
                # Push apart
                ux = dx / dist
                uy = dy / dist
                # Move each by a small amount
                step = 0.001
                centers[i, 0] += ux * step
                centers[i, 1] += uy * step
                centers[j, 0] -= ux * step
                centers[j, 1] -= uy * step
        else:
            # Boundary constraint is binding
            i = min_boundary_idx
            step = 0.001
            if min_boundary_side == 'left':
                centers[i, 0] += step
            elif min_boundary_side == 'right':
                centers[i, 0] -= step
            elif min_boundary_side == 'bottom':
                centers[i, 1] += step
            elif min_boundary_side == 'top':
                centers[i, 1] -= step
        
        # Clip to bounds
        centers = np.clip(centers, -0.5, 0.5)
        
        # Check for improvement
        new_r = _compute_radius(centers)
        if new_r <= current_r - 1e-12:
            # No improvement, break
            break
    
    return centers

def _simulated_annealing(centers, time_limit=20.0, rng=None):
    """
    Simulated annealing to explore the landscape.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    start_time = time.time()
    centers = centers.copy()
    current_r = _compute_radius(centers)
    best_r = current_r
    best_centers = centers.copy()
    
    T = 0.01  # Initial temperature
    cooling_rate = 0.999
    
    while time.time() - start_time < time_limit:
        # Generate a random move
        # Pick a random circle and move it by a small random amount
        i = rng.integers(0, N)
        dx = rng.normal(0, T * 0.5)
        dy = rng.normal(0, T * 0.5)
        
        old_x, old_y = centers[i, 0], centers[i, 1]
        centers[i, 0] = np.clip(old_x + dx, -0.5, 0.5)
        centers[i, 1] = np.clip(old_y + dy, -0.5, 0.5)
        
        new_r = _compute_radius(centers)
        delta = new_r - current_r
        
        # Accept if better, or with probability exp(delta / T)
        if delta > 0 or rng.random() < np.exp(delta / T):
            current_r = new_r
            if new_r > best_r:
                best_r = new_r
                best_centers = centers.copy()
        else:
            # Revert
            centers[i, 0] = old_x
            centers[i, 1] = old_y
        
        # Cool down
        T *= cooling_rate
        if T < 0.001:
            T = 0.001
    
    return best_centers, best_r

def _construct():
    start_time = time.time()
    deadline = DEADLINE
    
    rng = np.random.default_rng(42)
    
    best_centers = None
    best_radius = -np.inf
    
    # Generate multiple initializations
    inits = []
    
    # 1. Grid 7x8
    inits.append(_grid_init(7, 8, rng))
    # 2. Grid 8x7
    inits.append(_grid_init(8, 7, rng))
    # 3. Hexagonal
    inits.append(_hex_init(rng))
    
    # 4. Random initializations
    for _ in range(5):
        inits.append(_random_init(rng))
    
    # Run SA + local optimization on each init
    for i, init in enumerate(inits):
        if time.time() > deadline:
            break
        
        remaining = deadline - time.time()
        
        # Allocate time for SA and local optimization
        sa_time = min(10.0, remaining * 0.7)
        local_time = min(5.0, remaining * 0.3)
        
        try:
            # SA
            sa_centers, sa_r = _simulated_annealing(init, time_limit=sa_time, rng=rng)
            
            # Local optimization
            local_centers = _local_optimize(sa_centers, time_limit=local_time)
            local_r = _compute_radius(local_centers)
            
            if local_r > best_radius:
                best_radius = local_r
                best_centers = local_centers.copy()
        except Exception:
            pass
    
    # If we have time, do more SA rounds on the best result
    if best_centers is not None and time.time() < deadline - 10.0:
        try:
            remaining = deadline - time.time()
            sa_time = min(15.0, remaining * 0.7)
            local_time = min(5.0, remaining * 0.3)
            
            # SA on best
            sa_centers, sa_r = _simulated_annealing(best_centers, time_limit=sa_time, rng=rng)
            local_centers = _local_optimize(sa_centers, time_limit=local_time)
            local_r = _compute_radius(local_centers)
            
            if local_r > best_radius:
                best_radius = local_r
                best_centers = local_centers.copy()
        except Exception:
            pass
    
    # Fallback
    if best_centers is None:
        best_centers = _grid_init(7, 8, rng)
        best_radius = _compute_radius(best_centers)
    
    # Final validity check and clipping
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers

def construct():
    return _construct()
