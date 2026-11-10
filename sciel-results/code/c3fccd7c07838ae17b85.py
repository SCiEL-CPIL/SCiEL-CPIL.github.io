# EVOLVE-BLOCK-START
import time
import numpy as np

# EVOLVE-BLOCK-END

# Precompute sofa points once at module level for speed
_SOFA_U = np.arange(-3.0, 1.5, 0.01)
_SOFA_V = np.arange(-1.5, 1.5, 0.01)
_SOFA_X, _SOFA_Y = np.meshgrid(_SOFA_U, _SOFA_V, indexing='ij')
_SOFA_PTS = np.stack([_SOFA_X.ravel(), _SOFA_Y.ravel()], axis=1)  # shape (N, 2)
_N_SOFA = len(_SOFA_PTS)

def _generate_placements(waypoints):
    """
    Generate the full list of placements (including start (0,0,0) and end (-2,0,90)).
    Each placement is (x, y, a_degrees).
    """
    P0 = np.array([0.0, 0.0, 0.0])
    PN = np.array([-2.0, 0.0, 90.0])
    # Ensure waypoints are in a list of arrays
    if len(waypoints.shape) == 1:
        waypoints = waypoints.reshape(1, -1)
    full = [P0] + [np.array(p, dtype=np.float64) for p in waypoints] + [PN]
    placements = [full[0]]
    for i in range(len(full) - 1):
        p1 = full[i]
        p2 = full[i + 1]
        da = abs(p2[2] - p1[2])
        m = max(1, int(np.ceil(da - 1e-9)))
        for k in range(1, m + 1):
            t = k / m
            placements.append((1 - t) * p1 + t * p2)
    return np.array(placements, dtype=np.float64)

def _compute_area(waypoints):
    """
    Compute the area of the sofa (intersection of all corridor placements).
    Vectorized for speed.
    """
    if len(waypoints.shape) == 1:
        waypoints = waypoints.reshape(1, -1)
        
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0
    
    x = placements[:, 0]
    y = placements[:, 1]
    a_rad = np.radians(placements[:, 2])
    
    # Broadcast: 
    # x: (M, 1), y: (M, 1), a_rad: (M, 1)
    # sofa_x: (1, N), sofa_y: (1, N)
    
    sx = _SOFA_PTS[:, 0].reshape(1, -1)
    sy = _SOFA_PTS[:, 1].reshape(1, -1)
    
    dx = sx - x.reshape(-1, 1)
    dy = sy - y.reshape(-1, 1)
    
    ca = np.cos(a_rad).reshape(-1, 1)
    sa = np.sin(a_rad).reshape(-1, 1)
    
    # u = ca * dx + sa * dy
    # v = -sa * dx + ca * dy
    # Note: R(-a) is [cos(a), sin(a); -sin(a), cos(a)]
    
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    
    # Inside condition:
    # Part 1: 0 <= u <= 1, -3 <= v <= 1
    # Part 2: -3 <= u <= 1, 0 <= v <= 1
    
    inside1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    inside2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    
    inside = inside1 | inside2
    
    # A sofa point is valid if it is inside ALL placements.
    mask = np.all(inside, axis=0)
    
    return mask.sum() * 0.0001

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Simulated Annealing on waypoint coordinates and angles.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 10.0

    # Initial best path (from current program's seed)
    seed_wp = np.array([
        [-0.1, 0.1, 15.0],
        [-0.2, 0.2, 20.0],
        [-0.3, 0.3, 34.0],
        [-0.3, 0.3, 40.0],
        [-0.4, 0.3, 50.0],
        [-0.5, 0.2, 55.0],
        [-0.6, 0.2, 75.0],
        [-0.7, 0.3, 86.0],
    ], dtype=np.float64)

    best_wp = seed_wp.copy()
    best_area = _compute_area(best_wp)
    
    # Simulated Annealing Parameters
    n_wp = len(best_wp)
    current_wp = best_wp.copy()
    current_area = best_area
    
    # Initial temperature
    T = 0.05
    # Cooling rate
    alpha = 0.999
    
    # We will try to optimize a fixed number of waypoints (e.g., 10)
    # to allow the path to be more flexible than the 8-point seed.
    # Let's start with 10 waypoints.
    n_wp_target = 10
    if n_wp != n_wp_target:
        # Resample current best to n_wp_target
        angles = np.linspace(15, 86, n_wp_target)
        # Simple interpolation for x, y
        x_interp = np.interp(angles, best_wp[:, 2], best_wp[:, 0])
        y_interp = np.interp(angles, best_wp[:, 2], best_wp[:, 1])
        current_wp = np.column_stack((x_interp, y_interp, angles))
        current_area = _compute_area(current_wp)
        if current_area > best_area:
            best_area = current_area
            best_wp = current_wp.copy()
        n_wp = n_wp_target

    # SA Loop
    # We estimate ~1000-5000 evaluations are possible in 600s depending on M.
    # M is roughly 90-100.
    # _compute_area takes ~10-20ms.
    # So ~30000 evaluations in 600s is possible.
    
    while time.time() < safety_deadline:
        # Generate a perturbation
        # We perturb x, y, and a for a random subset of waypoints
        # Or all waypoints with small noise
        
        # Choose a random waypoint index
        idx = np.random.randint(n_wp)
        
        # Perturbation sizes
        # x, y: ~0.05
        # a: ~5 degrees
        dx = np.random.normal(0, 0.05)
        dy = np.random.normal(0, 0.05)
        da = np.random.normal(0, 5.0)
        
        # Ensure angles stay within reasonable bounds (0 to 90)
        new_a = current_wp[idx, 2] + da
        if new_a < 5.0:
            new_a = 5.0
        if new_a > 85.0:
            new_a = 85.0
            
        # Update the waypoint
        new_wp = current_wp.copy()
        new_wp[idx, 0] += dx
        new_wp[idx, 1] += dy
        new_wp[idx, 2] = new_a
        
        # Ensure monotonicity of angles? 
        # The evaluator interpolates between consecutive rows.
        # If angles are not monotonic, the path might fold back, which is allowed but usually suboptimal.
        # However, the problem statement says "rotated by a degrees". 
        # The interpolation is linear in (x,y,a).
        # It's generally better to keep angles roughly monotonic increasing.
        # Let's enforce that the new angle is between neighbors.
        if idx > 0:
            prev_a = current_wp[idx-1, 2]
            if new_a <= prev_a:
                new_a = prev_a + 1.0
        if idx < n_wp - 1:
            next_a = current_wp[idx+1, 2]
            if new_a >= next_a:
                new_a = next_a - 1.0
        
        new_wp[idx, 2] = new_a
        
        # Compute new area
        new_area = _compute_area(new_wp)
        
        # Acceptance criterion
        delta = new_area - current_area
        if delta > 0:
            # Always accept improvement
            current_wp = new_wp
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_wp = current_wp.copy()
        else:
            # Accept with probability exp(-delta / T)
            # Note: delta is negative, so -delta is positive.
            # Area is small (~0.1), so delta is small (~0.001).
            # T starts at 0.05.
            # exp(-0.001 / 0.05) = exp(-0.02) ~ 0.98. High acceptance.
            # As T decreases, acceptance of worse solutions drops.
            if np.random.rand() < np.exp(-delta / T):
                current_wp = new_wp
                current_area = new_area
        
        # Cool down
        T *= alpha
        
        # If T is too low, we might get stuck. 
        # We can add a "reheat" or just rely on the fact that we start near a good solution.
        # Given the time budget, we just let it cool.
        
        # Optional: Occasionally perturb more strongly to escape local minima
        if np.random.rand() < 0.01:
            # Large jump
            jump_idx = np.random.randint(n_wp)
            current_wp[jump_idx, 0] += np.random.normal(0, 0.2)
            current_wp[jump_idx, 1] += np.random.normal(0, 0.2)
            current_wp[jump_idx, 2] += np.random.normal(0, 10.0)
            current_wp[jump_idx, 2] = np.clip(current_wp[jump_idx, 2], 5.0, 85.0)
            current_area = _compute_area(current_wp)
            if current_area > best_area:
                best_area = current_area
                best_wp = current_wp.copy()

    # Final refinement: Local search around best_wp
    # Use a smaller step size
    T_local = 0.001
    current_wp = best_wp.copy()
    current_area = best_area
    
    local_deadline = safety_deadline - 5.0
    while time.time() < local_deadline:
        idx = np.random.randint(n_wp)
        dx = np.random.normal(0, 0.01)
        dy = np.random.normal(0, 0.01)
        da = np.random.normal(0, 1.0)
        
        new_wp = current_wp.copy()
        new_wp[idx, 0] += dx
        new_wp[idx, 1] += dy
        new_wp[idx, 2] = np.clip(new_wp[idx, 2] + da, 5.0, 85.0)
        
        # Enforce monotonicity
        if idx > 0 and new_wp[idx, 2] <= new_wp[idx-1, 2]:
            new_wp[idx, 2] = new_wp[idx-1, 2] + 0.1
        if idx < n_wp - 1 and new_wp[idx, 2] >= new_wp[idx+1, 2]:
            new_wp[idx, 2] = new_wp[idx+1, 2] - 0.1
            
        new_area = _compute_area(new_wp)
        
        if new_area > current_area:
            current_wp = new_wp
            current_area = new_area
            if current_area > best_area:
                best_area = current_area
                best_wp = current_wp.copy()
        else:
            if np.random.rand() < np.exp(-(current_area - new_area) / T_local):
                current_wp = new_wp
                current_area = new_area

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        return seed_wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
