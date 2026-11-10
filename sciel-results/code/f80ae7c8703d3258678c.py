import numpy as np
import time
import scipy.optimize as so

def moving_sofa_2d() -> np.ndarray:
    """
    Optimizes the path of the moving sofa to maximize the area of the intersection.
    """
    start_time = time.time()
    time_limit = 600.0
    safety_margin = 30.0

    # Sofa points definition
    i_vals = np.arange(-3.0, 1.5, 0.01)
    j_vals = np.arange(-1.5, 1.5, 0.01)
    P_x, P_y = np.meshgrid(i_vals, j_vals, indexing='xy')
    P_x = P_x.flatten()
    P_y = P_y.flatten()
    N_pts = len(P_x)

    # Parameterization
    angles = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 85.0, 88.0])
    n_angles = len(angles)
    n_params = 2 * n_angles

    # Initial guess
    base_x = [-0.1, -0.2, -0.3, -0.3, -0.4, -0.5, -0.6, -0.7]
    base_y = [0.1, 0.2, 0.3, 0.3, 0.3, 0.2, 0.2, 0.3]
    base_a = [15.0, 20.0, 34.0, 40.0, 50.0, 55.0, 75.0, 86.0]

    init_x = np.interp(angles, base_a, base_x)
    init_y = np.interp(angles, base_a, base_y)

    p0 = np.concatenate([init_x, init_y])

    # Pre-compute dense path structure
    # Anchors: 0=Start(0,0,0), 1..N=Waypoints, N+1=End(-2,0,90)
    # For each dense point, store which two anchors it interpolates between and the t value
    
    dense_info = []  # List of (anchor1_idx, anchor2_idx, t)
    
    # Segment 0: Start (0,0,0) to Waypoint 0
    a1, a2 = 0.0, angles[0]
    m = max(1, int(np.ceil(abs(a2 - a1))))
    for k in range(1, m):
        t = k / m
        dense_info.append((0, 1, t))
    # Append Waypoint 0
    dense_info.append((1, 1, 1.0))  # Represents waypoint itself
    
    # Segments between Waypoints
    for i in range(n_angles - 1):
        a1, a2 = angles[i], angles[i+1]
        m = max(1, int(np.ceil(abs(a2 - a1))))
        for k in range(1, m):
            t = k / m
            dense_info.append((i + 1, i + 2, t))
        # Append Waypoint i+1
        dense_info.append((i + 2, i + 2, 1.0))
    
    # Segment N: Waypoint N-1 to End (-2,0,90)
    a1, a2 = angles[n_angles-1], 90.0
    m = max(1, int(np.ceil(abs(a2 - a1))))
    for k in range(1, m):
        t = k / m
        dense_info.append((n_angles, n_angles + 1, t))
    # Append End
    dense_info.append((n_angles + 1, n_angles + 1, 1.0))
    
    n_dense = len(dense_info)
    
    # Pre-compute cos/sin for each dense point's angle
    # Angle at each dense point:
    # For anchor i (1..N): angle = angles[i-1]
    # For anchor 0: 0, for anchor N+1: 90
    anchor_angles = np.zeros(n_angles + 2)
    anchor_angles[0] = 0.0
    anchor_angles[1:n_angles+1] = angles
    anchor_angles[n_angles+1] = 90.0
    
    dense_angles = np.zeros(n_dense)
    for idx, (a1_idx, a2_idx, t) in enumerate(dense_info):
        if a1_idx == a2_idx:
            dense_angles[idx] = anchor_angles[a1_idx]
        else:
            dense_angles[idx] = anchor_angles[a1_idx] + t * (anchor_angles[a2_idx] - anchor_angles[a1_idx])
    
    dense_cos = np.cos(np.deg2rad(dense_angles))
    dense_sin = np.sin(np.deg2rad(dense_angles))
    
    # Pre-compute weight matrices for x and y interpolation
    # For each dense point, x = (1-t)*x_anchor1 + t*x_anchor2
    # Anchor 0: x=0, y=0 (fixed)
    # Anchor 1..N: x=x_vals[i-1], y=y_vals[i-1]
    # Anchor N+1: x=-2, y=0 (fixed)
    
    # We'll store for each dense point:
    # x_weight_0: coefficient for x_vals[0] (if anchor1 or anchor2 is 1)
    # x_weight_1: coefficient for x_vals[1]
    # ...
    # Actually, let's use a sparse representation: for each dense point, store the two anchor indices and t
    
    # For efficient evaluation, pre-compute:
    # For each dense point j, we need:
    # x_j = (1-t_j) * x_anchor1_j + t_j * x_anchor2_j
    # where x_anchor_i depends on whether i is 0, 1..N, or N+1
    
    # Let's create arrays for fast lookup
    dense_a1 = np.array([d[0] for d in dense_info])
    dense_a2 = np.array([d[1] for d in dense_info])
    dense_t = np.array([d[2] for d in dense_info])
    
    # For x coordinates:
    # If anchor is 0: x = 0
    # If anchor is 1..N: x = x_vals[anchor-1]
    # If anchor is N+1: x = -2
    
    def get_anchor_x(anchor_idx, x_vals):
        """Get x coordinate for an anchor given x_vals."""
        result = np.zeros(len(anchor_idx))
        mask0 = anchor_idx == 0
        maskN = anchor_idx == n_angles + 1
        maskmid = ~(mask0 | maskN)
        result[mask0] = 0.0
        result[maskN] = -2.0
        if np.any(maskmid):
            idxs = anchor_idx[maskmid] - 1
            result[maskmid] = x_vals[idxs]
        return result
    
    def get_anchor_y(anchor_idx, y_vals):
        """Get y coordinate for an anchor given y_vals."""
        result = np.zeros(len(anchor_idx))
        mask0 = anchor_idx == 0
        maskN = anchor_idx == n_angles + 1
        maskmid = ~(mask0 | maskN)
        result[mask0] = 0.0
        result[maskN] = 0.0
        if np.any(maskmid):
            idxs = anchor_idx[maskmid] - 1
            result[maskmid] = y_vals[idxs]
        return result

    def evaluate_path(x_vals, y_vals):
        """Evaluate the area for given x, y waypoints."""
        # Compute dense path coordinates
        x_a1 = get_anchor_x(dense_a1, x_vals)
        x_a2 = get_anchor_x(dense_a2, x_vals)
        y_a1 = get_anchor_y(dense_a1, y_vals)
        y_a2 = get_anchor_y(dense_a2, y_vals)
        
        dense_x = (1 - dense_t) * x_a1 + dense_t * x_a2
        dense_y = (1 - dense_t) * y_a1 + dense_t * y_a2
        
        # Evaluate membership
        in_sofa = np.ones(N_pts, dtype=bool)
        
        for idx in range(n_dense):
            x = dense_x[idx]
            y = dense_y[idx]
            cos_t = dense_cos[idx]
            sin_t = dense_sin[idx]
            
            dx = P_x - x
            dy = P_y - y
            
            u = dx * cos_t + dy * sin_t
            v = -dx * sin_t + dy * cos_t
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            mask = cond1 | cond2
            
            in_sofa &= mask
            
            if not np.any(in_sofa):
                return 0.0
        
        return np.sum(in_sofa) * 0.0001

    # Objective function
    def objective(params):
        x_vals = params[:n_angles]
        y_vals = params[n_angles:]
        area = evaluate_path(x_vals, y_vals)
        return -area

    # Optimize
    best_area = -objective(p0)
    best_params = p0.copy()
    
    def time_guard():
        return time.time() - start_time > (time_limit - safety_margin)

    try:
        res = so.minimize(objective, p0, method='Powell', 
                          options={'maxiter': 3000, 'maxfev': 3000, 'disp': False})
        
        if not time_guard():
            if -res.fun > best_area:
                best_area = -res.fun
                best_params = res.x.copy()
    except Exception as e:
        pass
        
    if not time_guard():
        try:
            res2 = so.minimize(objective, best_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'maxfev': 2000, 'disp': False, 'xatol': 1e-8, 'fatol': 1e-10})
            if not time_guard():
                if -res2.fun > best_area:
                    best_area = -res2.fun
                    best_params = res2.x.copy()
        except Exception:
            pass

    # If still time, try a second Powell run with perturbation
    if not time_guard():
        try:
            np.random.seed(42)
            noise = np.random.normal(0, 0.02, size=n_params)
            p0_restart = best_params + noise
            res3 = so.minimize(objective, p0_restart, method='Powell', 
                               options={'maxiter': 1000, 'maxfev': 1000, 'disp': False})
            if not time_guard():
                if -res3.fun > best_area:
                    best_area = -res3.fun
                    best_params = res3.x.copy()
        except Exception:
            pass

    # Construct final path
    x_vals = best_params[:n_angles]
    y_vals = best_params[n_angles:]
    
    final_path = []
    for i in range(n_angles):
        final_path.append([x_vals[i], y_vals[i], angles[i]])
        
    path_arr = np.array(final_path, dtype=np.float64)
    
    return path_arr
