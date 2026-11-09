import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 14

# Precompute triple indices for N=14
_i_list, _j_list, _k_list = [], [], []
for _i in range(N):
    for _j in range(_i + 1, N):
        for _k in range(_j + 1, N):
            _i_list.append(_i)
            _j_list.append(_j)
            _k_list.append(_k)
_TRIPLE_I = np.array(_i_list)
_TRIPLE_J = np.array(_j_list)
_TRIPLE_K = np.array(_k_list)

def get_score(points):
    """
    Calculate the exact score: min triangle area / convex hull area.
    points: np.array of shape (N, 2)
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Min Triangle Area (Vectorized)
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)

    min_area = np.min(areas)
    if min_area < 1e-10:
        return 0.0

    return min_area / hull_area

def smooth_score(points, beta):
    """
    Calculate the smooth score using soft-min approximation.
    points: np.array of shape (N, 2)
    beta: float, sharpness of the soft-min
    """
    # 1. Convex Hull Area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-10:
            return 0.0
    except Exception:
        return 0.0

    # 2. Soft-Min Triangle Area
    p1 = points[_TRIPLE_I]
    p2 = points[_TRIPLE_J]
    p3 = points[_TRIPLE_K]

    v1 = p2 - p1
    v2 = p3 - p1
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we subtract the max
    max_area = np.max(areas)
    shifted_areas = areas - max_area
    # exp(-beta * shifted_areas) is safe as shifted_areas <= 0
    sum_exp = np.sum(np.exp(-beta * shifted_areas))
    soft_min_area = max_area - (1.0 / beta) * np.log(sum_exp)
    
    if soft_min_area < 1e-10:
        return 0.0

    return soft_min_area / hull_area

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We stop searching at 145s to be safe.
    deadline = start_time + 145.0

    best_points = None
    best_score = 0.0
    rng = np.random.default_rng(42)

    # Initializations
    initializations = []
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)
    reg_poly = np.column_stack((x, y))
    initializations.append(reg_poly)
    
    # 2. Perturbed Regular Polygons
    # Small perturbations to break symmetry and explore local landscape
    for i in range(20):
        # Random rotation
        theta = rng.uniform(0, 2*np.pi)
        c, s = np.cos(theta), np.sin(theta)
        R = np.array([[c, -s], [s, c]])
        pts = reg_poly @ R.T
        
        # Small angular and radial perturbations
        ang_noise = rng.normal(0, 0.1, N)
        rad_noise = rng.uniform(0.9, 1.0, N)
        
        new_angles = angles + ang_noise
        pts[:, 0] = rad_noise * np.cos(new_angles)
        pts[:, 1] = rad_noise * np.sin(new_angles)
        
        initializations.append(pts)

    # 3. Random points in unit disk, filtered for "goodness"
    for i in range(20):
        pts = rng.uniform(-1, 1, (N, 2))
        try:
            hull = ConvexHull(pts)
            if hull.volume > 0.5:
                initializations.append(pts)
        except:
            pass

    # Annealing schedule for beta
    # Start with low beta (smooth) and increase to high beta (sharp)
    beta_schedule = [10.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    for init_idx, init_pts in enumerate(initializations):
        if time.time() > deadline - 5.0:
            break
            
        current_pts = init_pts.copy()
        
        # Annealing loop
        for beta in beta_schedule:
            if time.time() > deadline - 5.0:
                break
                
            def objective_smooth(flat_points):
                pts = flat_points.reshape(N, 2)
                score = smooth_score(pts, beta)
                return -score

            try:
                res = minimize(
                    objective_smooth, 
                    current_pts.flatten(), 
                    method='L-BFGS-B', 
                    options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-8, 'maxls': 20}
                )
                if res.fun < -best_score or best_points is None:
                    # Check exact score to update best
                    exact_sc = get_score(res.x.reshape(N, 2))
                    if exact_sc > best_score:
                        best_score = exact_sc
                        best_points = res.x.reshape(N, 2)
                        
                current_pts = res.x.reshape(N, 2)
            except Exception:
                pass

        # Final polish with exact objective using Nelder-Mead or Powell
        # Nelder-Mead is often better for final polish on nonsmooth functions
        if time.time() > deadline - 10.0:
            continue
            
        def objective_exact(flat_points):
            pts = flat_points.reshape(N, 2)
            return -get_score(pts)

        try:
            res_nm = minimize(
                objective_exact, 
                current_pts.flatten(), 
                method='Nelder-Mead', 
                options={'xatol': 1e-10, 'fatol': 1e-12, 'maxiter': 2000, 'maxfev': 10000}
            )
            exact_sc = get_score(res_nm.x.reshape(N, 2))
            if exact_sc > best_score:
                best_score = exact_sc
                best_points = res_nm.x.reshape(N, 2)
        except Exception:
            pass

    if best_points is None:
        # Fallback to regular polygon
        best_points = reg_poly

    return best_points
