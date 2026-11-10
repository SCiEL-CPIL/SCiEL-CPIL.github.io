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
    """
    placements = _generate_placements(waypoints)
    total = len(placements)
    if total == 0:
        return 0.0
    
    # Optimization: Check if any placement eliminates all points early?
    # No, we need the intersection.
    
    # We can vectorize the check for all placements against all sofa points?
    # placements: (M, 3), sofa_pts: (N, 2)
    # This creates an (M, N) boolean matrix.
    # M is small (number of steps), N is ~45000.
    # M * N * 8 bytes is manageable if M is small.
    # M is roughly sum(ceil(da)). If we have 10 waypoints with 10 deg steps, M ~ 90.
    # 90 * 45000 * 8 = ~32 MB. This is fine.
    
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
    # So u = cos(a)*dx + sin(a)*dy
    # v = -sin(a)*dx + cos(a)*dy
    
    u = ca * dx + sa * dy
    v = -sa * dx + ca * dy
    
    # Inside condition:
    # Part 1: 0 <= u <= 1, -3 <= v <= 1
    # Part 2: -3 <= u <= 1, 0 <= v <= 1
    
    inside1 = (u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)
    inside2 = (u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0)
    
    inside = inside1 | inside2
    
    # A sofa point is valid if it is inside ALL placements.
    # inside shape: (M, N)
    # We want: for each j (sofa point), all i (placements) are True.
    # mask = np.all(inside, axis=0)
    
    # To save memory/time, we can do:
    # valid_count = np.sum(inside, axis=0)
    # mask = valid_count == total
    
    # But np.all is usually optimized.
    mask = np.all(inside, axis=0)
    
    return mask.sum() * 0.0001

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses a grid search strategy focusing on smooth arcs.
    Returns: np.ndarray of shape (n, 3), n >= 1, all finite.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 10.0

    best_area = -1.0
    best_wp = None

    # The seed path is a good baseline
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

    seed_area = _compute_area(seed_wp)
    if seed_area > best_area:
        best_area = seed_area
        best_wp = seed_wp.copy()

    # Strategy: Grid Search on "Arc" parameters
    # We assume the optimal path is close to a circular arc or a sequence of straight lines.
    # Let's try to optimize the "turning radius" and "position" of the turn.
    
    # Parameterization:
    # We define waypoints by their angle.
    # Let's fix the number of waypoints to 10 for consistent search.
    # We will sweep over possible "centers" of the turn and "radii".
    
    # Alternatively, a direct grid search on the waypoint coordinates.
    # Since the problem is 3D (x, y, a) for each waypoint, and we have ~8-10 waypoints,
    # full grid search is impossible.
    
    # Instead, we will refine the seed using a local grid search around the best found so far.
    # But the directive asks for a MATERIALLY DIFFERENT strategy.
    # Grid search is different from Nelder-Mead.
    
    # Let's try a structured grid search on a reduced set of parameters:
    # We assume the path is symmetric or follows a specific curve.
    # Let's try paths defined by a polynomial in angle.
    # x(a) = c0 + c1*a + c2*a^2
    # y(a) = d0 + d1*a + d2*a^2
    
    # We will search over coefficients c0, c1, c2, d0, d1, d2.
    # This reduces the dimensionality from 3*N_wp to 6 parameters.
    
    n_wp = 10
    angles = np.linspace(10, 80, n_wp)
    
    # Coefficients ranges:
    # x is roughly -1 to 0. a is 0 to 90.
    # c0 ~ -0.5, c1 ~ 0.01, c2 ~ 0.001
    # y is roughly 0 to 0.5.
    # d0 ~ 0.1, d1 ~ 0.005, d2 ~ 0.001
    
    # We will do a coarse grid search first, then a fine grid search.
    
    # Coarse Grid
    c0_vals = np.linspace(-0.8, -0.2, 5)
    c1_vals = np.linspace(-0.01, 0.01, 5)
    c2_vals = np.linspace(-0.002, 0.002, 5)
    d0_vals = np.linspace(-0.1, 0.3, 5)
    d1_vals = np.linspace(-0.01, 0.01, 5)
    d2_vals = np.linspace(-0.002, 0.002, 5)
    
    # This is 5^6 = 15625 evaluations.
    # Each evaluation takes ~50ms? 15625 * 0.05 = 780s. Too slow.
    # We need to reduce the search space or speed up evaluation.
    
    # Let's reduce the grid to 3^6 = 729 evaluations.
    c0_vals = np.linspace(-0.7, -0.3, 3)
    c1_vals = np.linspace(-0.005, 0.005, 3)
    c2_vals = np.linspace(-0.001, 0.001, 3)
    d0_vals = np.linspace(0.0, 0.2, 3)
    d1_vals = np.linspace(-0.005, 0.005, 3)
    d2_vals = np.linspace(-0.001, 0.001, 3)
    
    # Total 729 evaluations.
    # Let's estimate time.
    # _compute_area is vectorized. 
    # M (placements) depends on angles. 
    # Angles are fixed in this parameterization.
    # M is constant for all candidates.
    # M ~ 90.
    # Vectorized check is fast.
    
    best_coeffs = None
    best_area_grid = -1.0
    
    for c0 in c0_vals:
        for c1 in c1_vals:
            for c2 in c2_vals:
                for d0 in d0_vals:
                    for d1 in d1_vals:
                        for d2 in d2_vals:
                            if time.time() > safety_deadline:
                                break
                            
                            # Generate waypoints
                            wp = np.zeros((n_wp, 3))
                            wp[:, 0] = c0 + c1 * angles + c2 * angles**2
                            wp[:, 1] = d0 + d1 * angles + d2 * angles**2
                            wp[:, 2] = angles
                            
                            area = _compute_area(wp)
                            if area > best_area_grid:
                                best_area_grid = area
                                best_coeffs = (c0, c1, c2, d0, d1, d2)
                                best_wp = wp.copy()
                                best_area = area

    # Now refine around the best found coefficients
    if best_coeffs is not None and time.time() < safety_deadline - 60:
        c0, c1, c2, d0, d1, d2 = best_coeffs
        step = 0.02
        
        # Fine grid search
        c0_range = np.linspace(c0 - step, c0 + step, 5)
        c1_range = np.linspace(c1 - step/10, c1 + step/10, 5)
        c2_range = np.linspace(c2 - step/100, c2 + step/100, 5)
        d0_range = np.linspace(d0 - step, d0 + step, 5)
        d1_range = np.linspace(d1 - step/10, d1 + step/10, 5)
        d2_range = np.linspace(d2 - step/100, d2 + step/100, 5)
        
        for c0 in c0_range:
            for c1 in c1_range:
                for c2 in c2_range:
                    for d0 in d0_range:
                        for d1 in d1_range:
                            for d2 in d2_range:
                                if time.time() > safety_deadline:
                                    break
                                
                                wp = np.zeros((n_wp, 3))
                                wp[:, 0] = c0 + c1 * angles + c2 * angles**2
                                wp[:, 1] = d0 + d1 * angles + d2 * angles**2
                                wp[:, 2] = angles
                                
                                area = _compute_area(wp)
                                if area > best_area:
                                    best_area = area
                                    best_wp = wp.copy()

    # Ensure output is valid
    if best_wp is None:
        return seed_wp

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        return seed_wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
