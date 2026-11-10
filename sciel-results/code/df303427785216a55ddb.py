# EVOLVE-BLOCK-START
import time
import numpy as np
import random
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
    
    mask = np.ones(_N_SOFA, dtype=bool)
    for p in placements:
        xx, yy, aa = p[0], p[1], np.radians(p[2])
        ca, sa = np.cos(aa), np.sin(aa)
        dx = _SOFA_PTS[:, 0] - xx
        dy = _SOFA_PTS[:, 1] - yy
        u = ca * dx + sa * dy
        v = -sa * dx + ca * dy
        inside = ((u >= 0.0) & (u <= 1.0) & (v >= -3.0) & (v <= 1.0)) | \
                 ((u >= -3.0) & (u <= 1.0) & (v >= 0.0) & (v <= 1.0))
        mask &= inside
        if not mask.any():
            return 0.0  # Early exit if no points remain
    area = mask.sum() * 0.0001
    return area

def _bezier_points(control_pts, num_samples):
    """
    Evaluate a cubic Bezier curve defined by control_pts (4x3 array) at num_samples points.
    Returns array of shape (num_samples, 3).
    """
    # Control points: P0, P1, P2, P3
    P0, P1, P2, P3 = control_pts
    t = np.linspace(0, 1, num_samples)
    t2 = t**2
    t3 = t**3
    
    # Bezier basis functions
    B0 = 1 - 3*t2 + 3*t3
    B1 = 3*t - 6*t2 + 3*t3
    B2 = 3*t2 - 3*t3
    B3 = t3
    
    B0 = B0[:, np.newaxis]
    B1 = B1[:, np.newaxis]
    B2 = B2[:, np.newaxis]
    B3 = B3[:, np.newaxis]
    
    # Curve = B0*P0 + B1*P1 + B2*P2 + B3*P3
    curve = B0 * P0 + B1 * P1 + B2 * P2 + B3 * P3
    return curve

def _evaluate_control_points(ctrl_pts, num_samples=10):
    """
    Evaluate the area for a given set of Bezier control points.
    ctrl_pts: shape (4, 3)
    """
    # The Bezier curve generates the intermediate waypoints.
    # We exclude the first and last points of the Bezier curve because they are fixed by the problem (start and end).
    # Actually, the problem says "The evaluator puts (0, 0, 0) before your rows and (-2, 0, 90) after them."
    # So our returned array should NOT include (0,0,0) or (-2,0,90).
    # The Bezier curve should go from near (0,0,0) to near (-2,0,90) but we can just use the interior points.
    
    # Let's generate points along the Bezier curve.
    # We want the curve to connect the start and end.
    # P0 = (0,0,0), P3 = (-2,0,90)
    # P1 and P2 are free parameters.
    # We will generate num_samples-1 interior points to return.
    
    # To ensure smoothness and validity, we fix P0 and P3.
    P0 = np.array([0.0, 0.0, 0.0])
    P3 = np.array([-2.0, 0.0, 90.0])
    P1 = ctrl_pts[0]
    P2 = ctrl_pts[1]
    
    full_ctrl = np.array([P0, P1, P2, P3])
    
    # Sample points
    pts = _bezier_points(full_ctrl, num_samples)
    
    # Exclude first and last
    waypoints = pts[1:-1]
    
    # Validate angles? The Bezier interpolation of angle might go out of [0, 90] if P1/P2 angles are extreme.
    # We should clip angles to [0, 90] to be safe? Or just let the optimizer handle it?
    # The problem allows any angle, but physically the corridor rotates.
    # If angle goes negative, it rotates the other way.
    # Let's just use the generated points.
    
    return _compute_area(waypoints)

def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Simulated Annealing on Bezier control points.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 5.0

    # Initial guess:
    # A reasonable path for the sofa problem involves moving up and then left.
    # Start (0,0,0). End (-2,0,90).
    # P1 should be somewhat up and left. P2 should be further left.
    # Let's try:
    # P1 = (-0.5, 1.0, 30)
    # P2 = (-1.5, 1.0, 60)
    
    best_ctrl = np.array([
        [-0.5, 1.0, 30.0],
        [-1.5, 1.0, 60.0]
    ])
    
    # Evaluate initial
    best_area = _evaluate_control_points(best_ctrl)
    
    current_ctrl = best_ctrl.copy()
    current_area = best_area
    
    # Simulated Annealing parameters
    T = 1.0
    cooling_rate = 0.995
    min_T = 1e-5
    max_iter = 5000
    
    rng = np.random.default_rng(42)
    
    for i in range(max_iter):
        if time.time() > safety_deadline:
            break
            
        # Generate neighbor
        # Perturb P1 and P2
        # Step size proportional to temperature? Or fixed?
        # Let's use fixed step size but scaled by T for acceptance.
        step_size = 0.1
        
        neighbor = current_ctrl + rng.normal(0, step_size, size=current_ctrl.shape)
        
        # Optional: Bound checking to keep parameters in reasonable range
        # Angles should probably stay within -90 to 180 or so to avoid crazy loops
        # Positions should stay within -5 to 5
        
        # Evaluate neighbor
        neighbor_area = _evaluate_control_points(neighbor)
        
        # Acceptance probability
        delta = neighbor_area - current_area
        if delta > 0:
            # Always accept improvement
            current_ctrl = neighbor
            current_area = neighbor_area
            if current_area > best_area:
                best_area = current_area
                best_ctrl = current_ctrl.copy()
        else:
            # Accept with probability exp(delta / T)
            if T > 0:
                prob = np.exp(delta / T)
                if rng.random() < prob:
                    current_ctrl = neighbor
                    current_area = neighbor_area
        
        # Cool down
        T *= cooling_rate
        if T < min_T:
            # Reheat? Or just stop?
            # Let's try a few restarts if we have time
            # For now, break if T is too low
            pass
            
    # Final polish with coordinate descent on the Bezier control points?
    # Or just return the best Bezier path.
    
    # Generate final waypoints from best_ctrl
    P0 = np.array([0.0, 0.0, 0.0])
    P3 = np.array([-2.0, 0.0, 90.0])
    full_ctrl = np.array([P0, best_ctrl[0], best_ctrl[1], P3])
    
    # Use a higher number of samples for the final output to ensure smoothness
    # The evaluator interpolates between our rows.
    # If we provide fewer rows, the evaluator does more interpolation.
    # If we provide more rows, we have finer control.
    # Let's provide 20 intermediate points.
    num_samples = 22 # 20 intermediate + 2 endpoints
    pts = _bezier_points(full_ctrl, num_samples)
    waypoints = pts[1:-1]
    
    # Ensure finite
    if not np.all(np.isfinite(waypoints)):
        # Fallback to a simple linear path
        waypoints = np.array([
            [-0.5, 0.5, 22.5],
            [-1.0, 1.0, 45.0],
            [-1.5, 0.5, 67.5]
        ])
        
    return waypoints.astype(np.float64)

# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
