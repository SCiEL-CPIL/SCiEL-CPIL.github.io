import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def moving_sofa_2d() -> np.ndarray:
    """
    Finds a path for the moving sofa problem using Differential Evolution 
    followed by Nelder-Mead refinement.
    """
    start_time = time.time()
    time_limit = 550  # Leave 50s margin
    
    # Grid definition
    # Full grid: 450x300
    # Subsampled grid: 225x150 (step 2)
    sub_step = 2 
    x_sub = np.arange(-3, 1.5, 0.01 * sub_step)
    y_sub = np.arange(-1.5, 1.5, 0.01 * sub_step)
    X, Y = np.meshgrid(x_sub, y_sub, indexing='ij')
    grid_points = np.column_stack((X.ravel(), Y.ravel()))
    n_points = len(grid_points)
    
    # Area factor for subsampled grid
    area_factor_sub = (0.01 * sub_step) ** 2

    # Start and end are fixed
    start_point = np.array([0.0, 0.0, 0.0])
    end_point = np.array([-2.0, 0.0, 90.0])
    
    # Number of intermediate waypoints
    n_waypoints = 8
    
    # Bounds for intermediate waypoints
    bounds = [(-3.0, 1.0), (-3.0, 1.0), (0.0, 90.0)] * n_waypoints

    def evaluate_area(path, grid_pts, area_factor):
        """
        Calculate the area of the sofa intersection for a given path.
        """
        # Generate interpolated steps
        steps = [path[0]]
        for i in range(len(path) - 1):
            p1 = path[i]
            p2 = path[i + 1]
            da = abs(p2[2] - p1[2])
            m = max(1, int(np.ceil(da)))
            if m > 1:
                t = np.linspace(0, 1, m + 1)[1:]
                new_pts = p1[None, :] + t[:, None] * (p2 - p1)[None, :]
                steps.extend(new_pts.tolist())
        steps.append(path[-1])
        
        mask = np.ones(len(grid_pts), dtype=bool)
        
        for step in steps:
            x_c, y_c, angle_deg = step
            angle_rad = np.deg2rad(angle_deg)
            cos_a = np.cos(angle_rad)
            sin_a = np.sin(angle_rad)
            
            dx = grid_pts[:, 0] - x_c
            dy = grid_pts[:, 1] - y_c
            
            u = cos_a * dx + sin_a * dy
            v = -sin_a * dx + cos_a * dy
            
            cond1 = (u >= 0) & (u <= 1) & (v >= -3) & (v <= 1)
            cond2 = (u >= -3) & (u <= 1) & (v >= 0) & (v <= 1)
            
            inside = cond1 | cond2
            mask &= inside
            
            if not np.any(mask):
                return 0.0

        return np.sum(mask) * area_factor

    def objective(params):
        pts = params.reshape(-1, 3)
        path = np.vstack((start_point, pts, end_point))
        return -evaluate_area(path, grid_points, area_factor_sub)

    best_result = None
    
    # Run Differential Evolution
    def callback(xk, convergence):
        if time.time() - start_time > time_limit * 0.7:
            raise Exception("Time limit approaching")

    try:
        result = differential_evolution(
            objective, 
            bounds, 
            maxiter=30, 
            popsize=10,
            tol=0.01,
            mutation=(0.5, 1.5),
            recombination=0.7,
            seed=42,
            callback=callback,
            polish=False
        )
        if result.fun < 0:
            best_result = result.x
    except Exception:
        pass

    # If DE didn't find a result, use baseline
    if best_result is None:
        final_path = np.array(
            [
                [-0.1, 0.1, 15.0],
                [-0.2, 0.2, 20.0],
                [-0.3, 0.3, 34.0],
                [-0.3, 0.3, 40.0],
                [-0.4, 0.3, 50.0],
                [-0.5, 0.2, 55.0],
                [-0.6, 0.2, 75.0],
                [-0.7, 0.3, 86.0],
            ],
            dtype=np.float64,
        )
    else:
        final_path = best_result.reshape(-1, 3)

    # Refine with Nelder-Mead if time permits
    if time.time() - start_time < time_limit:
        def obj_nm(params):
            pts = params.reshape(-1, 3)
            path = np.vstack((start_point, pts, end_point))
            return -evaluate_area(path, grid_points, area_factor_sub)
        
        try:
            res_nm = minimize(
                obj_nm, 
                final_path.ravel(), 
                method='Nelder-Mead', 
                options={'maxiter': 100, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            if res_nm.fun < 0:
                final_path = res_nm.x.reshape(-1, 3)
        except Exception:
            pass

    if not np.all(np.isfinite(final_path)):
        final_path = np.array(
            [
                [-0.1, 0.1, 15.0],
                [-0.2, 0.2, 20.0],
                [-0.3, 0.3, 34.0],
                [-0.3, 0.3, 40.0],
                [-0.4, 0.3, 50.0],
                [-0.5, 0.2, 55.0],
                [-0.6, 0.2, 75.0],
                [-0.7, 0.3, 86.0],
            ],
            dtype=np.float64,
        )
        
    return final_path
