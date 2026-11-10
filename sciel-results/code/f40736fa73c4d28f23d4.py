# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize
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


def _arc_path_params(params):
    """
    Generate waypoints from a parametric arc-turn model.
    params: [radius, center_x, center_y]
    Path: Horizontal approach -> Arc -> Vertical departure
    """
    r, cx, cy = params
    r = max(r, 0.1)
    
    waypoints = []
    
    # 1. Horizontal Approach (a=0)
    # Start at x=0, y=0. Move left to tangent point.
    # Tangent point for horizontal approach to circle (cx, cy) radius r.
    # The approach is along y=0. The tangent point must have y=0.
    # Wait, the circle is (cx, cy). A horizontal line y=cy is tangent if cy=0.
    # But the start is (0,0) and end is (-2,0).
    # Let's assume the turn happens in the region x in [-2, 0], y in [0, 2] or similar.
    # Actually, the start corridor is horizontal (y in [-3, 1], x in [0, 1] relative to corner).
    # Corner at (0,0). So the corridor occupies x in [0,1], y in [-3,1] and x in [-3,1], y in [0,1].
    # The "inner corner" is (0,0).
    # The sofa must fit inside.
    # The path starts at (0,0,0). The corridor is L-shaped.
    # To move from horizontal to vertical (90 deg), we rotate the corridor.
    
    # Let's use a simpler parametric model:
    # Waypoints are defined by angles 0, 15, 30, 45, 60, 75, 90.
    # The positions (x, y) follow a circular arc.
    # Center of arc (cx, cy), radius r.
    # Angle of the "corner" of the corridor relative to the arc center?
    # Let's define the path of the corner (x, y) as a function of the rotation angle a.
    # x(a) = cx + r * cos(theta(a))
    # y(a) = cy + r * sin(theta(a))
    # theta(a) = theta0 + k * a
    
    # For a 90 degree turn, if we map a in [0, 90] to theta in [0, 90] (or similar).
    # Let's try:
    # theta(a) = pi/2 - a * pi/180 ? No.
    # If a=0, corridor is horizontal. Corner at (0,0).
    # If a=90, corridor is vertical. Corner at (-2,0).
    
    # Let's define the center of the turn.
    # A common solution for the "sofa problem" involves a circular arc.
    # The inner corner traces a curve.
    
    # Let's parameterize the corner position (x(a), y(a)) by a circle.
    # x(a) = cx + r * cos(omega * a + phi)
    # y(a) = cy + r * sin(omega * a + phi)
    
    # We have 3 params: r, cx, cy. We can fix omega and phi to satisfy boundary conditions?
    # At a=0, (x,y) = (0,0).
    # At a=90, (x,y) = (-2,0).
    
    # Let's try a specific family:
    # The corner moves along a quarter circle.
    # Start (0,0). End (-2,0).
    # If it's a quarter circle, the center must be (0, 2) or (-2, 2) or (0, -2) or (-2, -2).
    # If center is (-1, 1), radius sqrt(2).
    # x(a) = -1 + sqrt(2) * cos(theta(a))
    # y(a) = 1 + sqrt(2) * sin(theta(a))
    # At a=0, we want (0,0).
    # 0 = -1 + sqrt(2) cos(theta0) => cos(theta0) = 1/sqrt(2) => theta0 = pi/4.
    # 0 = 1 + sqrt(2) sin(theta0) => sin(theta0) = -1/sqrt(2) => theta0 = -pi/4. Contradiction.
    
    # So a simple quarter circle centered at (-1,1) doesn't work with constant speed in angle.
    
    # Let's allow the center and radius to vary, and define the path as:
    # x(a) = cx + r * cos(alpha * a + beta)
    # y(a) = cy + r * sin(alpha * a + beta)
    # We have 4 params: r, cx, cy, beta. (alpha fixed to map 90 deg to some angle range).
    # Let's fix alpha such that the total angle change is 90 degrees (pi/2).
    # theta(a) = beta + (pi/2) * (a/90).
    # At a=0, theta=beta. At a=90, theta=beta+pi/2.
    # Constraints:
    # x(0) = cx + r cos(beta) = 0
    # y(0) = cy + r sin(beta) = 0
    # x(90) = cx + r cos(beta+pi/2) = cx - r sin(beta) = -2
    # y(90) = cy + r sin(beta+pi/2) = cy + r cos(beta) = 0
    
    # From y(0)=0 and y(90)=0:
    # cy + r sin(beta) = 0
    # cy + r cos(beta) = 0
    # => r sin(beta) = r cos(beta) => tan(beta) = 1 => beta = pi/4 (or 5pi/4).
    # If beta = pi/4:
    # cy + r/sqrt(2) = 0 => cy = -r/sqrt(2)
    # cx + r/sqrt(2) = 0 => cx = -r/sqrt(2)
    # Check x(90):
    # cx - r sin(pi/4) = -r/sqrt(2) - r/sqrt(2) = -2r/sqrt(2) = -r*sqrt(2).
    # We need x(90) = -2.
    # -r*sqrt(2) = -2 => r = 2/sqrt(2) = sqrt(2).
    # Then cx = -1, cy = -1.
    
    # So the ONLY circular arc that connects (0,0) to (-2,0) with a 90 degree turn in the "corner"
    # (where the corner rotates by 90 degrees relative to the path tangent? No, the corridor rotates by 90)
    # is fixed to r=sqrt(2), cx=-1, cy=-1.
    
    # This suggests the "circle" model is too constrained if we require the corner to follow the circle exactly with linear angle mapping.
    
    # Alternative: The "inner corner" of the L-shape is the point (x,y).
    # The L-shape is defined relative to (x,y).
    # Maybe the path is not a circle for the corner, but the *walls* follow circles?
    
    # Let's go back to free waypoints but use a better initialization and optimizer.
    # The previous attempt used random walk.
    # Let's use Nelder-Mead on 8 waypoints (24 params).
    
    # Initial guess: The fixed circle solution (r=sqrt(2), cx=-1, cy=-1) gives specific waypoints.
    # Let's generate waypoints from this circle.
    
    def _circle_waypoints():
        r = np.sqrt(2.0)
        cx = -1.0
        cy = -1.0
        beta = np.pi / 4.0
        angles = np.linspace(0, 90, 8)
        wps = []
        for a in angles:
            theta = beta + (np.pi / 2) * (a / 90.0)
            x = cx + r * np.cos(theta)
            y = cy + r * np.sin(theta)
            wps.append([x, y, a])
        return np.array(wps)
    
    return _circle_waypoints()


def moving_sofa_2d() -> np.ndarray:
    """
    Find a path of waypoints to maximize the sofa area.
    Uses Nelder-Mead on a parametric arc model, then polishes with coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + 600.0
    safety_deadline = deadline - 10.0

    best_area = -1.0
    best_wp = None
    
    # 1. Get initial guess from parametric model
    init_wp = _arc_path_params([np.sqrt(2), -1.0, -1.0])
    init_area = _compute_area(init_wp)
    best_area = init_area
    best_wp = init_wp.copy()
    
    # 2. Nelder-Mead optimization on free waypoints
    # Flatten waypoints to a vector
    x0 = init_wp.ravel()
    n_wp = len(init_wp)
    
    def objective(x):
        wp = x.reshape(n_wp, 3)
        # Ensure angles are sorted? Nelder-Mead might break this.
        # If angles are not sorted, the path might fold back, which is allowed?
        # The problem says "linear interpolations of (x, y, a)".
        # It doesn't explicitly forbid non-monotonic angles, but physically it's weird.
        # However, the evaluator just interpolates.
        # Let's allow non-monotonic angles to give the optimizer more freedom.
        # But wait, the "sofa" is the intersection.
        # If the path folds back, it might create a smaller intersection.
        # Let's try without constraints first.
        
        # Bounds check? Nelder-Mead doesn't handle bounds well.
        # We can penalize large deviations.
        
        area = _compute_area(wp)
        return -area
    
    # Run Nelder-Mead
    try:
        res = minimize(objective, x0, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-6, 'disp': False})
        
        if res.success or res.fun < objective(x0):
            new_wp = res.x.reshape(n_wp, 3)
            new_area = -res.fun
            if new_area > best_area:
                best_area = new_area
                best_wp = new_wp.copy()
    except Exception:
        pass
    
    # 3. Coordinate Descent Polish
    # Perturb each waypoint slightly to fine-tune.
    curr_wp = best_wp.copy()
    curr_area = best_area
    
    step_size = 0.05
    no_improve = 0
    max_no_improve = 50
    
    for i in range(n_wp):
        if time.time() > safety_deadline:
            break
            
        for _ in range(100): # Iterate per waypoint
            if time.time() > safety_deadline:
                break
            if no_improve > max_no_improve:
                break
                
            # Perturb x, y, a for waypoint i
            for dim in range(3):
                for sign in [1, -1]:
                    trial = curr_wp.copy()
                    trial[i, dim] += sign * step_size
                    # Basic validity: keep angles in [0, 90]?
                    # If we allow non-monotonic, we don't need to check sorting.
                    # But keep them in reasonable bounds.
                    if trial[i, 2] < -10 or trial[i, 2] > 100:
                        continue
                    if trial[i, 0] < -3 or trial[i, 0] > 1:
                        continue
                    if trial[i, 1] < -2 or trial[i, 1] > 2:
                        continue
                        
                    new_area = _compute_area(trial)
                    if new_area > curr_area:
                        curr_wp = trial
                        curr_area = new_area
                        no_improve = 0
                        if curr_area > best_area:
                            best_area = curr_area
                            best_wp = curr_wp.copy()
                        break # Try next dim or step
                    else:
                        no_improve += 1
                else:
                    continue
                break # If improved, break inner loop to restart direction search?
            # Decay step size
            step_size *= 0.95
            
    # Return best
    if best_wp is None:
        return init_wp

    best_wp = np.asarray(best_wp, dtype=np.float64)
    if not np.all(np.isfinite(best_wp)):
        best_wp = init_wp

    return best_wp
# EVOLVE-BLOCK-START
# (code already defined above)
# EVOLVE-BLOCK-END
