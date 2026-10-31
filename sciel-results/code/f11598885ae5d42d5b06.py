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
        return 1e-9

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

def _soft_min(areas, temperature):
    """
    Smooth approximation of min using log-sum-exp.
    As temperature -> 0, this approaches the true minimum.
    """
    # Use the log-sum-exp trick for numerical stability
    min_val = np.min(areas)
    if temperature < 1e-10:
        return min_val
    # soft_min = -T * log(sum(exp(-areas/T)))
    # For stability: soft_min = min_val - T * log(sum(exp(-(areas - min_val)/T)))
    shifted = (areas - min_val) / temperature
    # Clip to avoid overflow
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(-shifted)))
    return min_val - temperature * log_sum

def _smooth_objective(params, temperature):
    """
    Smooth objective: minimize -soft_min(areas) / hull_area
    params: flat array of shape (2*N,) representing N points
    """
    points = params.reshape(N, 2)
    
    # Check if points are degenerate
    if np.any(np.isinf(points)) or np.any(np.isnan(points)):
        return 1e10
    
    areas = _triangle_areas(points)
    hull_area = _hull_area(points)
    
    if hull_area < 1e-9:
        return 1e10
    
    soft_min_area = _soft_min(areas, temperature)
    score = soft_min_area / hull_area
    return -score  # We minimize, so negate

def _coordinate_descent_polish(points, time_budget, initial_step=0.02):
    """
    Coordinate descent: optimize each point's x and y coordinates independently.
    Uses smooth surrogate with adaptive temperature.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Adaptive temperature: start high, decay over time
    base_temp = 0.05
    decay_rate = 0.95
    no_improve_count = 0
    
    while time.time() < deadline - 2.0:
        # Decay temperature
        base_temp *= decay_rate
        if base_temp < 1e-6:
            base_temp = 1e-6
        
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 2.0:
                break
            
            # Optimize x-coordinate of point i
            for coord in range(2):
                orig_val = current_points[i, coord]
                best_val = orig_val
                best_local_score = current_score
                
                # Try multiple step sizes
                for step_mult in [0.5, 1.0, 2.0, 4.0]:
                    step = initial_step * step_mult
                    for sign in [-1, 1]:
                        new_points = current_points.copy()
                        new_points[i, coord] = orig_val + sign * step
                        
                        # Quick check with smooth objective
                        temp = max(base_temp, 0.01)
                        neg_score = _smooth_objective(new_points.flatten(), temp)
                        score = -neg_score
                        
                        if score > best_local_score:
                            best_local_score = score
                            best_val = orig_val + sign * step
                
                # Apply best
                if best_local_score > current_score + 1e-12:
                    current_points[i, coord] = best_val
                    current_score = calculate_score(current_points)
                    improved = True
                    
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
        
        if not improved:
            no_improve_count += 1
            initial_step *= 0.8
            if initial_step < 1e-5 or no_improve_count > 15:
                break
        else:
            no_improve_count = 0
            initial_step = min(initial_step * 1.1, 0.05)
    
    return best_points, best_score

def _local_search(points, time_budget):
    """
    Combined local search: smooth surrogate + coordinate descent.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_points = points.copy()
    current_score = calculate_score(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    # Phase 1: Gradient-based optimization using smooth surrogate
    time_phase1 = time_budget * 0.4
    if time.time() < deadline - time_phase1:
        try:
            # Use scipy minimize with smooth objective
            x0 = current_points.flatten()
            result = minimize(
                _smooth_objective,
                x0,
                args=(0.03,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'maxfun': 500}
            )
            if result.success or result.fun < 0:
                new_points = result.x.reshape(N, 2)
                new_score = calculate_score(new_points)
                if new_score > current_score:
                    current_points = new_points
                    current_score = new_score
                    if current_score > best_score:
                        best_score = current_score
                        best_points = current_points.copy()
        except Exception:
            pass
    
    # Phase 2: Coordinate descent polishing
    remaining = deadline - time.time()
    if remaining > 3.0:
        try:
            pts, score = _coordinate_descent_polish(current_points, remaining - 2.0)
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            pass
    
    return best_points, best_score

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
        
    # 7. Three-ring configuration
    angles_inner = np.linspace(0, 2*np.pi, 3, endpoint=False)
    inner_ring = 0.3 * np.column_stack((np.cos(angles_inner), np.sin(angles_inner)))
    angles_mid = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.2
    mid_ring = 0.65 * np.column_stack((np.cos(angles_mid), np.sin(angles_mid)))
    angles_outer = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.4
    outer_ring = np.column_stack((np.cos(angles_outer), np.sin(angles_outer)))
    configs.append(np.vstack([inner_ring, mid_ring, outer_ring]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin
    
    best_score = -1.0
    best_points = None
    
    configs = _generate_initial_configs()
    
    n_configs = len(configs)
    
    for i, init_pts in enumerate(configs):
        # Check time
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
            
        # Allocate time for this config
        time_for_config = min(12.0, remaining - 5.0)
        
        try:
            pts, score = _local_search(init_pts, time_for_config)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue
            
    # If we have time left, do a final polish on the best point found
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 8.0:
            try:
                pts_final, score_final = _local_search(best_points, remaining - 5.0)
                
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
