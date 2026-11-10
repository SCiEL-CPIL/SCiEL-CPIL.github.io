import numpy as np
from scipy.optimize import minimize
import time

def moving_sofa_2d() -> np.ndarray:
    # Grid definition
    xs = np.arange(-3, 1.5, 0.01)
    ys = np.arange(-1.5, 1.5, 0.01)
    X, Y = np.meshgrid(xs, ys)
    W = np.stack([X.flatten(), Y.flatten()], axis=1)
    N = W.shape[0]
    
    # Base L-shape check function
    def check_base(W_rot):
        u = W_rot[:, 0]
        v = W_rot[:, 1]
        # Union of two rectangles
        rect1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
        rect2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
        return rect1 | rect2

    def get_path_linear(p0, p1, n_steps=91):
        # p0, p1 are (x, y, a)
        # Linear interpolation
        ts = np.linspace(0, 1, n_steps)
        x = p0[0] * (1 - ts) + p1[0] * ts
        y = p0[1] * (1 - ts) + p1[1] * ts
        a = p0[2] * (1 - ts) + p1[2] * ts
        return np.column_stack([x, y, a])

    def evaluate_area(path):
        # path is (n, 3)
        # Initialize mask as all True
        mask = np.ones(N, dtype=bool)
        
        # Loop over placements
        for i in range(path.shape[0]):
            x_c, y_c, alpha_deg = path[i]
            alpha_rad = np.deg2rad(alpha_deg)
            
            # Translate
            W_trans = W - np.array([x_c, y_c])
            
            # Rotate by -alpha
            c = np.cos(-alpha_rad)
            s = np.sin(-alpha_rad)
            # R.T is the inverse rotation
            W_rot = W_trans * np.array([c, s])
            # Wait, rotation matrix R(-a) = [[c, s], [-s, c]]
            # W_rot = R(-a) (W - c)
            # Let's do it explicitly to avoid confusion
            u = c * W_trans[:, 0] + s * W_trans[:, 1]
            v = -s * W_trans[:, 0] + c * W_trans[:, 1]
            
            # Check base
            m = check_base(np.column_stack([u, v]))
            mask &= m
            
            # Early exit if empty
            if not np.any(mask):
                return 0.0
                
        return np.sum(mask) * 0.0001

    # Time limit guard
    start_time = time.time()
    time_limit = 590 # Leave margin
    
    # Define start and end
    start_pt = np.array([0.0, 0.0, 0.0])
    end_pt = np.array([-2.0, 0.0, 90.0])
    
    # 1. Try linear path
    path_lin = get_path_linear(start_pt, end_pt)
    score_lin = evaluate_area(path_lin)
    
    best_path = path_lin
    best_score = score_lin
    
    # 2. Try a few heuristic paths if linear score is low or 0
    # Heuristic: Move to center of overlap, rotate, move to end
    # Overlap of start and end is roughly x in [-2, 1], y in [0, 1]
    # Let's try a path that goes to (-1, 0.5, 45)
    mid_pt = np.array([-1.0, 0.5, 45.0])
    path_mid1 = get_path_linear(start_pt, mid_pt, 46)
    path_mid2 = get_path_linear(mid_pt, end_pt, 45)
    path_mid = np.vstack([path_mid1, path_mid2[1:]]) # Remove duplicate mid
    score_mid = evaluate_area(path_mid)
    
    if score_mid > best_score:
        best_path = path_mid
        best_score = score_mid
        
    # If best score is still 0, try random paths
    if best_score == 0.0:
        for _ in range(10):
            if time.time() - start_time > time_limit:
                break
            # Random intermediate point
            rx = np.random.uniform(-2, 0)
            ry = np.random.uniform(-0.5, 0.5)
            ra = 45.0
            rmid = np.array([rx, ry, ra])
            p1 = get_path_linear(start_pt, rmid, 46)
            p2 = get_path_linear(rmid, end_pt, 45)
            p = np.vstack([p1, p2[1:]])
            s = evaluate_area(p)
            if s > best_score:
                best_path = p
                best_score = s

    # 3. Optimization using B-spline control points
    # Only if we have time and a non-zero baseline
    if best_score > 0.0 and time.time() - start_time < time_limit - 60:
        ctrl_angles = np.array([0, 18, 36, 54, 72, 90])
        
        def get_path_spline(ctrl_pts):
            from scipy.interpolate import CubicSpline
            try:
                cs_x = CubicSpline(ctrl_angles, ctrl_pts[:, 0], bc_type='natural')
                cs_y = CubicSpline(ctrl_angles, ctrl_pts[:, 1], bc_type='natural')
            except Exception:
                cs_x = CubicSpline(ctrl_angles, ctrl_pts[:, 0], bc_type='not-a-knot')
                cs_y = CubicSpline(ctrl_angles, ctrl_pts[:, 1], bc_type='not-a-knot')
            eval_angles = np.arange(0, 91, 1.0)
            xs_path = cs_x(eval_angles)
            ys_path = cs_y(eval_angles)
            return np.column_stack([xs_path, ys_path, eval_angles])

        def objective(params):
            ctrl_pts = params.reshape(-1, 2)
            # Enforce endpoints
            ctrl_pts[0] = [start_pt[0], start_pt[1]]
            ctrl_pts[-1] = [end_pt[0], end_pt[1]]
            path = get_path_spline(ctrl_pts)
            return -evaluate_area(path)

        # Initial guess from best_path if it has similar structure, or linear
        # Extract x, y from best_path at control angles
        init_x = np.interp(ctrl_angles, best_path[:, 2], best_path[:, 0])
        init_y = np.interp(ctrl_angles, best_path[:, 2], best_path[:, 1])
        x0 = np.array(init_x + init_y).flatten()
        
        try:
            res = minimize(objective, x0, method='SLSQP', 
                           options={'maxiter': 50, 'ftol': 1e-6})
            
            # Calculate new score
            new_path = get_path_spline(res.x.reshape(-1, 2))
            new_score = evaluate_area(new_path)
            
            if new_score > best_score:
                best_path = new_path
                best_score = new_score
                
            # Local search
            current_params = res.x
            while time.time() - start_time < time_limit:
                x0_try = current_params + np.random.normal(0, 0.05, size=current_params.shape)
                # Enforce endpoints in the try vector too? The objective does it, but let's be safe
                res2 = minimize(objective, x0_try, method='SLSQP', 
                                options={'maxiter': 20, 'ftol': 1e-6})
                
                new_path2 = get_path_spline(res2.x.reshape(-1, 2))
                new_score2 = evaluate_area(new_path2)
                
                if new_score2 > best_score:
                    best_path = new_path2
                    best_score = new_score2
                    current_params = res2.x
                else:
                    # If no improvement, stop to save time
                    break
                    
        except Exception:
            pass

    return best_path
