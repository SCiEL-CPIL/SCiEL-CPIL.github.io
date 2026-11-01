"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 32

# EVOLVE-BLOCK-START
import numpy as np
import time

def _validity_check(arr):
    """Check if circles are valid: inside unit square, disjoint interiors."""
    n = len(arr)
    for i in range(n):
        x, y, r = arr[i]
        if r <= 0:
            return False
        if x - r < -1e-9 or x + r > 1 + 1e-9 or y - r < -1e-9 or y + r > 1 + 1e-9:
            return False
    for i in range(n):
        for j in range(i+1, n):
            xi, yi, ri = arr[i]
            xj, yj, rj = arr[j]
            dist = np.sqrt((xi-xj)**2 + (yi-yj)**2)
            if dist < ri + rj - 1e-9:
                return False
    return True

def _sum_radii(arr):
    return np.sum(arr[:, 2])

def _physics_preprocess(points, max_iters=2000, dt=0.01, repulsion=10.0):
    """Physics-based push-apart to get a better starting configuration."""
    points = points.copy()
    n = len(points)
    
    for _ in range(max_iters):
        forces = np.zeros((n, 2))
        for i in range(n):
            for j in range(i+1, n):
                dx = points[j][0] - points[i][0]
                dy = points[j][1] - points[i][1]
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 1e-10:
                    dx, dy = 0.01, 0.01
                    dist = np.sqrt(2)*0.01
                # Repulsive force
                f = repulsion / (dist * dist)
                fx = f * dx / dist
                fy = f * dy / dist
                forces[i] += [-fx, -fy]
                forces[j] += [fx, fy]
        
        # Boundary forces
        for i in range(n):
            r = 0.02  # initial radius estimate
            if points[i][0] < r:
                forces[i][0] += repulsion
            if points[i][0] > 1 - r:
                forces[i][0] -= repulsion
            if points[i][1] < r:
                forces[i][1] += repulsion
            if points[i][1] > 1 - r:
                forces[i][1] -= repulsion
        
        points[:, :2] += dt * forces
        points[:, 0] = np.clip(points[:, 0], 0.01, 0.99)
        points[:, 1] = np.clip(points[:, 1], 0.01, 0.99)
    
    return points

def _initial_grid():
    """Grid initialization."""
    side = int(np.ceil(np.sqrt(N)))
    cells = []
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            r = 0.5 / side - 0.001
            cells.append([x, y, r])
    return np.array(cells[:N])

def _initial_hex():
    """Hexagonal packing initialization."""
    # Use hexagonal close packing pattern
    rows = []
    r_est = 0.05  # estimate radius
    dx = 2 * r_est
    dy = np.sqrt(3) * r_est
    
    y = r_est
    row_idx = 0
    while len(rows) < N:
        x = r_est + (row_idx % 2) * dx/2
        while x < 1 - r_est:
            if len(rows) < N:
                rows.append([x, y, r_est])
            x += dx
        y += dy
        row_idx += 1
    
    return np.array(rows[:N])

def _initial_random(seed=42):
    """Random initialization with physics preprocessing."""
    rng = np.random.default_rng(seed)
    points = rng.random((N, 2)) * 0.9 + 0.05
    radii = np.full(N, 0.03)
    points = np.column_stack([points, radii])
    points = _physics_preprocess(points, max_iters=500)
    return points

def _compute_radii(points):
    """Compute maximum possible radius for each circle given positions."""
    n = len(points)
    radii = np.zeros(n)
    for i in range(n):
        x, y = points[i]
        # Distance to boundaries
        r_bound = min(x, 1-x, y, 1-y)
        # Distance to other circles
        r_other = np.inf
        for j in range(n):
            if i == j:
                continue
            dx = points[j][0] - x
            dy = points[j][1] - y
            dist = np.sqrt(dx*dx + dy*dy)
            r_other = min(r_other, dist/2)
        radii[i] = min(r_bound, r_other)
    return radii

def _optimize(points, max_time=100):
    """Optimize positions to maximize sum of radii."""
    start_time = time.time()
    points = points.copy()
    n = len(points)
    
    # Compute initial radii
    radii = _compute_radii(points)
    best_sum = np.sum(radii)
    best_points = points.copy()
    
    lr = 0.01
    momentum = 0.9
    vel = np.zeros_like(points)
    
    for iteration in range(50000):
        if time.time() - start_time > max_time:
            break
        
        # Compute radii and gradients
        radii = _compute_radii(points)
        grad = np.zeros_like(points)
        
        for i in range(n):
            x, y = points[i]
            r_bound = min(x, 1-x, y, 1-y)
            r_other = np.inf
            active_j = -1
            
            for j in range(n):
                if i == j:
                    continue
                dx = points[j][0] - x
                dy = points[j][1] - y
                dist = np.sqrt(dx*dx + dy*dy)
                if dist/2 < r_other:
                    r_other = dist/2
                    active_j = j
            
            r_i = min(r_bound, r_other)
            
            # Gradient: move to increase r_i
            if r_bound < r_other:
                # Constrained by boundary
                if x - r_bound < 1-x - r_bound and x - r_bound < y - r_bound and x - r_bound < 1-y - r_bound:
                    grad[i][0] = -1
                elif 1-x - r_bound < x - r_bound and 1-x - r_bound < y - r_bound and 1-x - r_bound < 1-y - r_bound:
                    grad[i][0] = 1
                elif y - r_bound < x - r_bound and y - r_bound < 1-x - r_bound and y - r_bound < 1-y - r_bound:
                    grad[i][1] = -1
                elif 1-y - r_bound < x - r_bound and 1-y - r_bound < 1-x - r_bound and 1-y - r_bound < y - r_bound:
                    grad[i][1] = 1
            else:
                # Constrained by another circle
                if active_j >= 0:
                    dx = x - points[active_j][0]
                    dy = y - points[active_j][1]
                    dist = np.sqrt(dx*dx + dy*dy)
                    if dist > 1e-10:
                        grad[i][0] = dx / dist
                        grad[i][1] = dy / dist
        
        # Update with momentum
        vel = momentum * vel + lr * grad
        points += vel
        points[:, 0] = np.clip(points[:, 0], 0.001, 0.999)
        points[:, 1] = np.clip(points[:, 1], 0.001, 0.999)
        
        # Check improvement
        new_radii = _compute_radii(points)
        new_sum = np.sum(new_radii)
        if new_sum > best_sum:
            best_sum = new_sum
            best_points = points.copy()
        
        # Adaptive learning rate
        if iteration % 1000 == 0:
            lr *= 0.95
    
    return best_points, best_sum

def construct():
    start_time = time.time()
    deadline = start_time + 110  # Safety margin
    
    best_points = None
    best_sum = -np.inf
    
    # Try multiple initializations
    initializations = [
        ("grid", _initial_grid()),
        ("hex", _initial_hex()),
        ("rand1", _initial_random(42)),
        ("rand2", _initial_random(123)),
    ]
    
    for name, init_points in initializations:
        if time.time() > deadline:
            break
        
        # Optimize from this initialization
        opt_points, opt_sum = _optimize(init_points, max_time=min(25, deadline - time.time()))
        
        if opt_sum > best_sum:
            best_sum = opt_sum
            best_points = opt_points.copy()
    
    if best_points is None:
        # Fallback to grid
        best_points = _initial_grid()
    
    # Compute final radii
    radii = _compute_radii(best_points)
    
    # Final result
    result = np.column_stack([best_points[:, 0], best_points[:, 1], radii])
    
    return result
# EVOLVE-BLOCK-END
