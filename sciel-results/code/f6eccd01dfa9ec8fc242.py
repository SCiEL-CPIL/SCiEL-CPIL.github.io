import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 26
PAIRS = [(i, j) for i in range(N) for j in range(i + 1, N)]
PAIRS_ARR = np.array(PAIRS)
N_PAIRS = len(PAIRS_ARR)

def _is_valid(circles, tol=1e-8):
    """Check if circles are within [0,1]^2 and non-overlapping."""
    x = circles[:, 0]
    y = circles[:, 1]
    r = circles[:, 2]
    
    # Boundary check: circle must be inside [0,1]^2
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    if np.any(r < -tol):
        return False
        
    # Non-overlap check
    i_idx, j_idx = PAIRS_ARR[:, 0], PAIRS_ARR[:, 1]
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx**2 + dy**2)
    r_sum = r[i_idx] + r[j_idx]
    
    if np.any(dist < r_sum - tol):
        return False
    return True

def _objective(params):
    """
    Objective function: -sum(r) + soft penalty for violations.
    Uses log-sum-exp for smooth, focused gradients on worst violations.
    """
    x = params[0:N]
    y = params[N:2*N]
    r = params[2*N:3*N]
    
    # Target: maximize sum(r), so minimize -sum(r)
    obj = -np.sum(r)
    
    # Boundary violations: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    # Violation magnitude = max(0, r - x), max(0, x + r - 1), etc.
    v1 = np.maximum(0, r - x)
    v2 = np.maximum(0, x + r - 1)
    v3 = np.maximum(0, r - y)
    v4 = np.maximum(0, y + r - 1)
    
    # Overlap violations: dist_ij - (r_i + r_j) >= 0
    i_idx, j_idx = PAIRS_ARR[:, 0], PAIRS_ARR[:, 1]
    dx = x[i_idx] - x[j_idx]
    dy = y[i_idx] - y[j_idx]
    dist = np.sqrt(dx**2 + dy**2 + 1e-12)
    r_sum = r[i_idx] + r[j_idx]
    v5 = np.maximum(0, r_sum - dist)
    
    # Combine all violations into a single log-sum-exp penalty
    # This focuses the gradient on the worst violations
    all_v = np.concatenate([v1, v2, v3, v4, v5])
    # Clip to avoid overflow in exp
    all_v = np.clip(all_v, 0, 100)
    
    # Log-sum-exp with temperature
    beta = 50.0  # Higher = sharper focus on worst violations
    m = np.max(all_v)
    if m > 1e-10:
        lse = m + np.log(np.sum(np.exp(beta * (all_v - m)))) / beta
    else:
        lse = 0.0
    
    # Penalty weight
    penalty_weight = 100.0
    return obj + penalty_weight * lse

def _make_grid():
    """Create a grid initial configuration."""
    side = int(np.ceil(np.sqrt(N)))
    circles = np.zeros((N, 3))
    for i in range(N):
        row = i // side
        col = i % side
        circles[i, 0] = (col + 0.5) / side
        circles[i, 1] = (row + 0.5) / side
        circles[i, 2] = 0.5 / side - 0.01
    return circles

def _make_hexagonal():
    """Create a hexagonal packing initial configuration."""
    circles = np.zeros((N, 3))
    # Hexagonal packing with varying radii
    spacing = 0.2
    positions = []
    for row in range(5):
        for col in range(5):
            x = col * spacing * 1.5 + 0.15
            y = row * spacing * np.sqrt(3) + 0.15
            if row % 2 == 1:
                x += spacing * 0.75
            if x < 0.95 and y < 0.95:
                positions.append((x, y))
    
    positions = positions[:N]
    while len(positions) < N:
        positions.append((0.5, 0.5))
    
    for i in range(N):
        circles[i, 0] = positions[i][0]
        circles[i, 1] = positions[i][1]
        circles[i, 2] = 0.1
    return circles

def _make_random(seed=42):
    """Create a random initial configuration."""
    np.random.seed(seed)
    circles = np.zeros((N, 3))
    circles[:, 0] = np.random.uniform(0.1, 0.9, N)
    circles[:, 1] = np.random.uniform(0.1, 0.9, N)
    circles[:, 2] = np.random.uniform(0.05, 0.15, N)
    return circles

def _force_directed_anneal(circles, time_limit=5.0):
    """
    Force-directed annealing: treat overlaps as springs, boundaries as walls.
    This is a physics-based approach that can escape local minima.
    """
    current = circles.copy()
    best = circles.copy()
    best_score = -np.inf
    
    T = 0.1
    T_min = 0.001
    alpha = 0.999
    
    start_time = time.time()
    
    while T > T_min and (time.time() - start_time) < time_limit:
        # Compute forces
        forces = np.zeros_like(current)
        
        # Boundary forces
        for i in range(N):
            x, y, r = current[i]
            if x < r:
                forces[i, 0] += (r - x) * 10
            if x > 1 - r:
                forces[i, 0] -= (x - (1 - r)) * 10
            if y < r:
                forces[i, 1] += (r - y) * 10
            if y > 1 - r:
                forces[i, 1] -= (y - (1 - r)) * 10
        
        # Overlap forces
        for i_idx, j_idx in PAIRS:
            dx = current[i_idx, 0] - current[j_idx, 0]
            dy = current[i_idx, 1] - current[j_idx, 1]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            r_sum = current[i_idx, 2] + current[j_idx, 2]
            
            if dist < r_sum:
                overlap = r_sum - dist
                # Force proportional to overlap, along the line connecting centers
                fx = overlap * dx / dist * 5
                fy = overlap * dy / dist * 5
                forces[i_idx, 0] += fx
                forces[i_idx, 1] += fy
                forces[j_idx, 0] -= fx
                forces[j_idx, 1] -= fy
        
        # Apply forces with annealing
        for i in range(N):
            current[i, 0] += forces[i, 0] * T + np.random.normal(0, T * 0.01)
            current[i, 1] += forces[i, 1] * T + np.random.normal(0, T * 0.01)
            # Try to grow radius slightly
            current[i, 2] += T * 0.001 + np.random.normal(0, T * 0.005)
            current[i, 2] = max(current[i, 2], 0.01)
            current[i, 0] = np.clip(current[i, 0], 0, 1)
            current[i, 1] = np.clip(current[i, 1], 0, 1)
        
        # Evaluate
        if _is_valid(current):
            score = np.sum(current[:, 2])
            if score > best_score:
                best_score = score
                best = current.copy()
        
        T *= alpha
    
    return best

def _slsqp_polish(circles, time_limit=10.0):
    """
    Use SLSQP with explicit constraints for a high-precision polish.
    """
    def obj_func(p):
        return -np.sum(p[2*N:3*N])
    
    # Constraints: x - r >= 0, x + r <= 1, y - r >= 0, y + r <= 1
    constraints = []
    for i in range(N):
        # x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[i] - p[2*N+i]})
        # 1 - x_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[i] - p[2*N+i]})
        # y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: p[N+i] - p[2*N+i]})
        # 1 - y_i - r_i >= 0
        constraints.append({'type': 'ineq', 'fun': lambda p, i=i: 1 - p[N+i] - p[2*N+i]})
    
    # Overlap constraints: dist_ij - (r_i + r_j) >= 0
    for i_idx, j_idx in PAIRS:
        def overlap_con(p, i=i_idx, j=j_idx):
            dx = p[i] - p[j]
            dy = p[N+i] - p[N+j]
            dist = np.sqrt(dx**2 + dy**2 + 1e-12)
            return dist - p[2*N+i] - p[2*N+j]
        constraints.append({'type': 'ineq', 'fun': overlap_con})
    
    # Bounds
    bounds = [(0, 1)] * N + [(0, 1)] * N + [(0.01, 0.5)] * N
    
    try:
        res = minimize(
            obj_func,
            circles.flatten(),
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 500, 'ftol': 1e-12, 'time_limit': time_limit}
        )
        if res.success or _is_valid(res.x.reshape(N, 3)):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return circles

def _lbfgsb_polish(circles, time_limit=10.0):
    """
    Use L-BFGS-B with penalty method for polishing.
    """
    def objective(p):
        return _objective(p)
    
    try:
        res = minimize(
            objective,
            circles.flatten(),
            method='L-BFGS-B',
            options={'maxiter': 2000, 'ftol': 1e-15, 'gtol': 1e-12, 'maxfun': 50000}
        )
        if _is_valid(res.x.reshape(N, 3)):
            return res.x.reshape(N, 3)
    except Exception:
        pass
    
    return circles

def _post_process(circles):
    """
    Ensure circles are strictly valid by shrinking radii if necessary.
    """
    x = circles[:, 0].copy()
    y = circles[:, 1].copy()
    r = circles[:, 2].copy()
    
    # Adjust for boundaries
    for i in range(N):
        min_r = min(x[i], y[i], 1 - x[i], 1 - y[i])
        if r[i] > min_r:
            r[i] = min_r * (1 - 1e-8)
    
    # Adjust for overlaps iteratively
    for _ in range(50):
        i_idx, j_idx = PAIRS_ARR[:, 0], PAIRS_ARR[:, 1]
        dx = x[i_idx] - x[j_idx]
        dy = y[i_idx] - y[j_idx]
        dist = np.sqrt(dx**2 + dy**2 + 1e-12)
        r_sum = r[i_idx] + r[j_idx]
        
        overlaps = r_sum - dist
        max_overlap = np.max(overlaps)
        
        if max_overlap < 1e-9:
            break
        
        # Shrink radii of overlapping pairs
        for k in range(N_PAIRS):
            if overlaps[k] > 1e-12:
                i, j = PAIRS[k]
                # Shrink both radii proportionally to their current values
                total_r = r[i] + r[j]
                if total_r > 0:
                    shrink_i = overlaps[k] * 0.5 * r[i] / total_r
                    shrink_j = overlaps[k] * 0.5 * r[j] / total_r
                    r[i] = max(r[i] - shrink_i, 0.01)
                    r[j] = max(r[j] - shrink_j, 0.01)
    
    return np.column_stack((x, y, r))

def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s kill
    
    best_solution = None
    best_score = -np.inf
    
    def try_solution(circles, label=""):
        nonlocal best_solution, best_score
        valid_circles = _post_process(circles)
        if _is_valid(valid_circles):
            score = np.sum(valid_circles[:, 2])
            if score > best_score:
                best_score = score
                best_solution = valid_circles.copy()
                print(f"  {label}: score={score:.6f}")
    
    # Generate initial configurations
    inits = [
        (_make_grid(), "grid"),
        (_make_hexagonal(), "hex"),
        (_make_random(42), "rand42"),
        (_make_random(123), "rand123"),
        (_make_random(999), "rand999"),
    ]
    
    # Phase 1: Force-directed annealing from each init
    for init, label in inits:
        if time.time() > deadline - 30:
            break
        remaining = deadline - time.time() - 20  # Leave time for polish
        fd_time = min(8.0, remaining / 2.0)
        if fd_time < 1.0:
            break
        
        fd_result = _force_directed_anneal(init, time_limit=fd_time)
        try_solution(fd_result, f"FD-{label}")
    
    # Phase 2: L-BFGS-B polish from best results
    if best_solution is not None and time.time() < deadline - 15:
        polished = _lbfgsb_polish(best_solution, time_limit=min(10.0, deadline - time.time() - 10))
        try_solution(polished, "L-BFGS-B")
    
    # Phase 3: SLSQP with hard constraints for final polish
    if best_solution is not None and time.time() < deadline - 5:
        slsqp_result = _slsqp_polish(best_solution, time_limit=min(10.0, deadline - time.time() - 3))
        try_solution(slsqp_result, "SLSQP")
    
    # Fallback if no valid solution found
    if best_solution is None:
        grid = _make_grid()
        best_solution = _post_process(grid)
        best_score = np.sum(best_solution[:, 2])
    
    return best_solution
