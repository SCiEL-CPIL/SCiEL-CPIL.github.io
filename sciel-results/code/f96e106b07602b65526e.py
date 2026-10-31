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

def _fast_min_area(points, hull_area=None):
    """
    Fast computation of minimum triangle area, optionally with precomputed hull area.
    Returns (min_area, hull_area) for efficiency.
    """
    areas = _triangle_areas(points)
    min_area = np.min(areas)
    if hull_area is None:
        hull_area = _hull_area(points)
    return min_area, hull_area

def _smooth_min_objective(points, temperature=100.0):
    """
    Smooth approximation of negative min triangle area / hull area.
    Uses log-sum-exp for smoothness.
    """
    areas = _triangle_areas(points)
    hull_area = _hull_area(points)
    if hull_area <= 1e-9:
        return 1e6
    
    # Use log-sum-exp to approximate min
    # min(x) = -lim_{T->0} T * log(sum(exp(-x/T)))
    inv_areas = -areas / temperature
    max_inv = np.max(inv_areas)
    log_sum = max_inv + np.log(np.sum(np.exp(inv_areas - max_inv)))
    smooth_min = -temperature * log_sum
    
    return -smooth_min / hull_area

def _coordinate_descent_polish(points, time_budget, temperature_start=50.0):
    """
    Coordinate descent polishing with adaptive temperature.
    """
    start_time = time.time()
    deadline = start_time + time_budget
    
    current_points = points.copy()
    current_min_area, current_hull_area = _fast_min_area(current_points)
    current_score = current_min_area / current_hull_area if current_hull_area > 1e-9 else 0.0
    
    best_points = current_points.copy()
    best_score = current_score
    
    step_size = 0.02
    temperature = temperature_start
    no_improve = 0
    
    while time.time() < deadline - 0.5:
        improved = False
        
        for i in range(N):
            if time.time() > deadline - 0.5:
                break
            
            # Try moving point i in multiple directions
            best_local_score = current_score
            best_local_point = current_points[i].copy()
            
            # Generate directions: towards centroid, random, towards neighbors
            others = np.delete(current_points, i, axis=0)
            centroid = np.mean(others, axis=0)
            
            directions = []
            
            # Centroid direction
            dir_c = centroid - current_points[i]
            norm_c = np.linalg.norm(dir_c)
            if norm_c > 1e-8:
                directions.append(dir_c / norm_c)
            
            # Random directions
            for _ in range(6):
                rd = np.random.randn(2)
                rd /= np.linalg.norm(rd)
                directions.append(rd)
            
            # Directions towards nearest neighbors
            dists = np.linalg.norm(others - current_points[i], axis=1)
            nearest_idx = np.argsort(dists)[:3]
            for ni in nearest_idx:
                dir_n = others[ni] - current_points[i]
                norm_n = np.linalg.norm(dir_n)
                if norm_n > 1e-8:
                    directions.append(dir_n / norm_n)
            
            # Try each direction with multiple step sizes
            for d in directions:
                for mult in [0.3, 0.7, 1.0, 1.5]:
                    step = step_size * mult
                    new_point = current_points[i] + step * d
                    new_points = current_points.copy()
                    new_points[i] = new_point
                    
                    # Fast score: reuse hull area (changes slowly)
                    new_min_area, new_hull_area = _fast_min_area(new_points, current_hull_area)
                    new_score = new_min_area / new_hull_area if new_hull_area > 1e-9 else 0.0
                    
                    if new_score > best_local_score + 1e-12:
                        best_local_score = new_score
                        best_local_point = new_point
                        # Update hull area for next iterations
                        current_hull_area = new_hull_area
            
            # Apply best improvement
            if best_local_score > current_score + 1e-12:
                current_points[i] = best_local_point
                current_score = best_local_score
                improved = True
                
                if current_score > best_score:
                    best_score = current_score
                    best_points = current_points.copy()
        
        if improved:
            no_improve = 0
            step_size = min(step_size * 1.05, 0.05)
            temperature *= 0.95
        else:
            no_improve += 1
            step_size *= 0.8
            temperature = max(temperature * 0.9, 5.0)
            
            if step_size < 1e-5 or no_improve > 30:
                break
    
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
    for _ in range(4):
        angles = np.linspace(0, 2*np.pi, N, endpoint=False)
        radii = 1.0 + 0.15 * np.random.randn(N)
        angles += 0.08 * np.random.randn(N)
        pts = np.column_stack((radii * np.cos(angles), radii * np.sin(angles)))
        configs.append(pts)
        
    # 6. Random points in disk
    for _ in range(4):
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
    
    # 8. Golden angle spiral
    golden_angle = np.pi * (3 - np.sqrt(5))
    radii = np.sqrt(np.linspace(0.1, 1.0, N))
    thetas = np.arange(N) * golden_angle
    configs.append(np.column_stack((radii * np.cos(thetas), radii * np.sin(thetas))))
    
    # 9. Quasi-regular: regular polygon with slight radial perturbation
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = 1.0 + 0.05 * np.sin(2 * angles)
    configs.append(np.column_stack((radii * np.cos(angles), radii * np.sin(angles))))
    
    # 10. Another three-ring with different proportions
    angles_inner2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + 0.3
    inner2_ring = 0.25 * np.column_stack((np.cos(angles_inner2), np.sin(angles_inner2)))
    angles_mid2 = np.linspace(0, 2*np.pi, 5, endpoint=False) + 0.1
    mid2_ring = 0.55 * np.column_stack((np.cos(angles_mid2), np.sin(angles_mid2)))
    angles_outer2 = np.linspace(0, 2*np.pi, 4, endpoint=False) + 0.5
    outer2_ring = 0.9 * np.column_stack((np.cos(angles_outer2), np.sin(angles_outer2)))
    configs.append(np.vstack([inner2_ring, mid2_ring, outer2_ring]))
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + 142.0  # 8s safety margin
    
    best_score = -1.0
    best_points = None
    
    configs = _generate_initial_configs()
    
    n_configs = len(configs)
    
    # First pass: quick evaluation and coordinate descent polish
    for i, init_pts in enumerate(configs):
        # Check time
        remaining = deadline - time.time()
        if remaining < 8.0:
            break
            
        # Allocate time for this config
        time_for_config = min(12.0, remaining - 5.0)
        
        try:
            pts, score = _coordinate_descent_polish(init_pts, time_for_config)
            
            if score > best_score:
                best_score = score
                best_points = pts
        except Exception:
            continue
            
    # Final polish on the best point found with more time
    if best_points is not None:
        remaining = deadline - time.time()
        if remaining > 15.0:
            try:
                pts_final, score_final = _coordinate_descent_polish(best_points, remaining - 5.0, temperature_start=20.0)
                
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
