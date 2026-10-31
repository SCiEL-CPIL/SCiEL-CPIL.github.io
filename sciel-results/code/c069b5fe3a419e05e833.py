import numpy as np
import time
import scipy.spatial as spat
from scipy.optimize import minimize
from itertools import combinations

N = 13
# Pre-compute triangle indices
_triple_idx = list(combinations(range(N), 3))
_i_arr = np.array([t[0] for t in _triple_idx])
_j_arr = np.array([t[1] for t in _triple_idx])
_k_arr = np.array([t[2] for t in _triple_idx])

def _triangle_areas(points):
    """Compute areas of all C(N,3) triangles."""
    pts = points
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(det)

def _hull_area(points):
    """Compute convex hull area."""
    try:
        hull = spat.ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-9 # Avoid division by zero, effectively score 0

def calculate_score(points):
    """
    Calculate the minimum triangle area divided by the convex hull area.
    """
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 0.0
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    return min_area / hull_area

def _smooth_min_objective(params, temperature):
    """
    Objective function for optimization.
    Minimizes the smooth approximation of the negative min triangle area relative to hull area.
    We maximize min_area / hull_area, so we minimize -log(sum(exp(-area/T))) / hull_area approximation?
    Actually, let's minimize: - (smooth_min(areas) / hull_area)
    smooth_min(areas) = -T * log( sum(exp(-area_i / T)) )
    This is a lower bound on the min area that converges to min area as T->0.
    """
    # Unpack parameters: [r_in, r_out, angle_offsets...]
    # For simplicity in this implementation, we will use a specific parameterization:
    # 13 points defined by 2 rings:
    # Ring 1 (outer): 13 points? No, usually inner points exist.
    # Let's use a generic parameterization:
    # First 13*2 floats are (x, y) for each point? That's 26 dims.
    # Or, use polar coordinates for a regular polygon perturbation?
    
    # Let's stick to a simpler, robust parameterization for the optimizer:
    # We will optimize the raw (x,y) coordinates of the 13 points.
    # This is 26 dimensions.
    
    pts = params.reshape(N, 2)
    
    # Calculate areas
    v1 = pts[_j_arr] - pts[_i_arr]
    v2 = pts[_k_arr] - pts[_i_arr]
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(det)
    
    # Hull area
    hull_area = _hull_area(pts)
    if hull_area < 1e-6:
        return 1e6 # Penalty for degenerate hull
    
    # Smooth min of areas
    # To avoid overflow in exp, shift by max area
    max_area = np.max(areas)
    shifted_areas = areas / temperature
    # sum_exp = sum(exp(-shifted_areas))
    # smooth_min = -T * log(sum_exp)
    # We want to maximize smooth_min / hull_area
    # So we minimize: - smooth_min / hull_area
    
    # Numerical stability
    # log(sum(exp(-x))) = -x_max + log(sum(exp(-(x - x_max)))) where x_max is min(areas)?
    # Let min_a = min(areas).
    # exp(-a/T) is largest for smallest a.
    # log(sum(exp(-a/T))) = -min_a/T + log(sum(exp(-(a-min_a)/T)))
    # smooth_min = -T * [ -min_a/T + log(...) ] = min_a - T * log(sum(exp(-(a-min_a)/T)))
    
    min_a = np.min(areas)
    exp_terms = np.exp(-(areas - min_a) / temperature)
    sum_exp = np.sum(exp_terms)
    smooth_min_val = min_a - temperature * np.log(sum_exp)
    
    # Objective to minimize
    return -smooth_min_val / hull_area

def _optimize_from_initial(init_pts, time_budget):
    """
    Run L-BFGS-B optimization starting from init_pts.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    # Initial point flattened
    x0 = init_pts.flatten()
    
    best_x = x0.copy()
    best_score = calculate_score(init_pts)
    
    # Annealing schedule for temperature
    # Start high to explore, end low to exploit
    temps = [1.0, 0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
    
    # We don't know how much time each temp will take, so we allocate time proportionally
    # or just iterate through them if time permits.
    # Let's give each temperature a specific time slice.
    n_temps = len(temps)
    time_per_temp = time_budget / n_temps
    
    for T in temps:
        if time.time() > deadline:
            break
            
        # Define objective with current T
        def obj(x):
            return _smooth_min_objective(x, T)
        
        try:
            # Use L-BFGS-B
            res = minimize(obj, x0, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            
            x_opt = res.x
            score_opt = calculate_score(x_opt.reshape(N, 2))
            
            if score_opt > best_score:
                best_score = score_opt
                best_x = x_opt.copy()
            
            # Update starting point for next temperature
            x0 = x_opt
        except Exception:
            # If optimization fails, keep current x0
            pass

    return best_x.reshape(N, 2), best_score

def _generate_initial_configs():
    """
    Generate diverse initial configurations.
    """
    configs = []
    
    # 1. Regular 13-gon
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    configs.append(np.column_stack((np.cos(angles), np.sin(angles))))
    
    # 2. Two-ring: 1 center, 12 outer
    center = np.array([[0, 0]])
    angles12 = np.linspace(0, 2*np.pi, 12, endpoint=False)
    outer12 = np.column_stack((np.cos(angles12), np.sin(angles12)))
    configs.append(np.vstack([center, outer12]))
    
    # 3. Two-ring: 2 inner, 11 outer
    angles2 = np.linspace(0, 2*np.pi, 2, endpoint=False)
    inner2 = 0.5 * np.column_stack((np.cos(angles2), np.sin(angles2)))
    angles11 = np.linspace(0, 2*np.pi, 11, endpoint=False)
    outer11 = np.column_stack((np.cos(angles11), np.sin(angles11)))
    configs.append(np.vstack([inner2, outer11]))
    
    # 4. Two-ring: 3 inner, 10 outer
    angles3 = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner3 = 0.6 * np.column_stack((np.cos(angles3), np.sin(angles3)))
    angles10 = np.linspace(0, 2*np.pi, 10, endpoint=False)
    outer10 = np.column_stack((np.cos(angles10), np.sin(angles10)))
    configs.append(np.vstack([inner3, outer10]))
    
    # 5. Random perturbations of regular polygon
    for _ in range(3):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + 0.1 * np.random.randn(N)
        angles += 0.05 * np.random.randn(N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(pts)
        
    # 6. Random points in disk
    for _ in range(3):
        r = np.sqrt(np.random.rand(N))
        theta = np.random.rand(N) * 2 * np.pi
        pts = np.column_stack((r * np.cos(theta), r * np.sin(theta)))
        configs.append(pts)
        
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    configs = _generate_initial_configs()
    
    # Sort configs by initial score to prioritize good starts?
    # Or just iterate. Let's iterate and keep track of time.
    
    n_configs = len(configs)
    # Allocate time per config. If we have 12 configs, ~12s each.
    # But some might fail fast.
    
    for i, init_pts in enumerate(configs):
        # Check time
        remaining = deadline - time.time()
        if remaining < 5.0:
            break
            
        # Allocate time for this config
        # If it's the last few, give less time
        time_for_config = min(15.0, remaining - 2.0)
        
        try:
            pts, score = _optimize_from_initial(init_pts, time_for_config)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue
            
    # If we have time left, do a final polish on the best point found
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 5.0:
            try:
                # Use a very low temperature for final polish
                def obj_final(x):
                    return _smooth_min_objective(x, 0.005)
                
                res = minimize(obj_final, best_points.flatten(), method='L-BFGS-B',
                               options={'maxiter': 100, 'ftol': 1e-14, 'gtol': 1e-10})
                pts_final = res.x.reshape(N, 2)
                score_final = calculate_score(pts_final)
                
                if score_final > best_score:
                    best_score = score_final
                    best_points = pts_final
            except Exception:
                pass

    # Fallback
    if best_points is None:
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))
        
    return best_points

if __name__ == "__main__":
    pts = construct()
    print(calculate_score(pts))
