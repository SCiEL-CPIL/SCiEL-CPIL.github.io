import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

N = 13
# Pre-compute triangle indices for vectorized area calculation
_TRIPLE_IDX = np.array([
    (i, j, k)
    for i in range(N)
    for j in range(i + 1, N)
    for k in range(j + 1, N)
], dtype=np.int32)

def _triangle_areas_vectorized(points):
    """Compute areas of all triangles formed by pairs of points."""
    pi = points[_TRIPLE_IDX[:, 0]]
    pj = points[_TRIPLE_IDX[:, 1]]
    pk = points[_TRIPLE_IDX[:, 2]]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross)

def _hull_area(points):
    """Compute area of the convex hull of the points."""
    try:
        if len(points) < 3:
            return 0.0
        # Check if points are collinear or identical
        if np.allclose(points, points[0]):
            return 0.0
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _score(points):
    """Compute the score: min triangle area / hull area."""
    if len(points) < 3:
        return 0.0
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 0.0
    areas = _triangle_areas_vectorized(points)
    min_area = np.min(areas)
    if min_area < 1e-15:
        return 0.0
    return min_area / hull_area

def _objective_full(x_flat):
    """Objective function for optimization: minimize negative score."""
    points = x_flat.reshape(N, 2)
    score = _score(points)
    if score <= 0:
        return 1e10
    return -score

def _soft_min_objective(x_flat, beta=100):
    """
    Smooth surrogate using log-sum-exp soft min.
    Minimizing the soft min of areas is equivalent to maximizing the soft max of 1/areas.
    We use -soft_min(areas) / hull_area.
    """
    points = x_flat.reshape(N, 2)
    # Center and scale for stability
    centroid = np.mean(points, axis=0)
    points = points - centroid
    
    if np.allclose(points, 0):
        return 1e10
    
    hull_area = _hull_area(points)
    if hull_area < 1e-12:
        return 1e10
    
    areas = _triangle_areas_vectorized(points)
    areas = np.maximum(areas, 1e-15)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    # To avoid overflow, we can use the log-sum-exp trick, but for small arrays (286) direct is often fine if beta is not too large.
    # Let's use a robust log-sum-exp implementation.
    min_area = np.min(areas)
    exp_terms = np.exp(-beta * (areas - min_area))
    log_sum_exp = -beta * min_area + np.log(np.sum(exp_terms))
    soft_min_val = -log_sum_exp / beta
    
    score = soft_min_val / hull_area
    if score <= 0:
        return 1e10
    return -score

def _make_ring_config(n_outer, n_inner, r_inner, angle_offset):
    """Create a two-ring configuration."""
    angles_outer = np.linspace(0, 2 * np.pi, n_outer, endpoint=False)
    angles_inner = np.linspace(angle_offset, 2 * np.pi + angle_offset, n_inner, endpoint=False)
    
    outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
    inner_pts = np.column_stack([r_inner * np.cos(angles_inner), r_inner * np.sin(angles_inner)])
    
    return np.vstack([outer_pts, inner_pts])

def _coordinate_descent_polish(points, max_iter=200, step_size=0.05, time_limit=10.0):
    """
    Local polish using coordinate descent with random perturbations.
    This helps escape local minima of the non-smooth objective.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    
    for i in range(max_iter):
        if time.time() - start_time > time_limit:
            break
        if current_score <= 0:
            break
            
        improved = False
        # Try perturbing each coordinate of each point
        for idx in range(N):
            for dim in range(2):
                # Try several step sizes
                for step in [step_size, step_size/2, step_size*2]:
                    # Positive and negative directions
                    for direction in [1, -1]:
                        trial = current_points.copy()
                        trial[idx, dim] += direction * step
                        
                        trial_score = _score(trial)
                        if trial_score > current_score:
                            current_points = trial
                            current_score = trial_score
                            improved = True
                            # If improved, try to refine with smaller steps in same direction
                            # (handled by next iteration or smaller steps in loop)
        
        if not improved:
            # If no improvement with current step size, reduce it
            step_size *= 0.7
            if step_size < 1e-5:
                break
        else:
            # If improved, keep step size or increase slightly? Keep it for now.
            pass
            
    return current_points, current_score

def _local_polish(points, time_budget=5.0):
    """
    Multi-stage local polish:
    1. Smooth objective (Powell) to get close.
    2. Coordinate descent to escape local minima.
    3. Nelder-Mead for final precision.
    """
    start_time = time.time()
    current_points = points.copy()
    current_score = _score(current_points)
    
    # Stage 1: Smooth optimization
    if time.time() - start_time < time_budget * 0.4:
        try:
            res = minimize(
                lambda x: _soft_min_objective(x, beta=50),
                current_points.flatten(),
                method='Powell',
                options={'maxiter': 100, 'xatol': 1e-8, 'fatol': 1e-8}
            )
            new_score = _score(res.x.reshape(N, 2))
            if new_score > current_score:
                current_points = res.x.reshape(N, 2)
                current_score = new_score
        except Exception:
            pass

    # Stage 2: Coordinate Descent
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 1.0:
        current_points, current_score = _coordinate_descent_polish(
            current_points, 
            max_iter=100, 
            step_size=0.05, 
            time_limit=remaining_time * 0.5
        )

    # Stage 3: Nelder-Mead on exact objective
    remaining_time = time_budget - (time.time() - start_time)
    if remaining_time > 0.5:
        try:
            res_nm = minimize(
                _objective_full,
                current_points.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-10, 'adaptive': True}
            )
            new_score = _score(res_nm.x.reshape(N, 2))
            if new_score > current_score:
                current_points = res_nm.x.reshape(N, 2)
                current_score = new_score
        except Exception:
            pass

    return current_points, current_score

def construct():
    start_time = time.time()
    # Safety margin: evaluator kills at 150s. We aim to finish by 140s.
    deadline = start_time + 140.0
    
    best_points = None
    best_score = 0.0
    
    def _update_best(points):
        nonlocal best_points, best_score
        if points is None or len(points) != N:
            return
        s = _score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()

    # --- Phase 1: Global Search with Differential Evolution ---
    # DE is robust to the non-smoothness of the min-triangle-area objective.
    # We run DE for a limited time to find a good global starting point.
    if time.time() < deadline - 60:
        try:
            bounds = [(-2.0, 2.0)] * (2 * N)
            de_params = {
                'maxiter': 50, # Reduced to allow more local polish
                'popsize': 15,
                'mutation': (0.5, 1.0),
                'recombination': 0.7,
                'tol': 1e-6,
                'seed': 42,
                'polish': False,
                'init': 'sobol',
            }
            
            result = differential_evolution(
                _objective_full,
                bounds,
                **de_params
            )
            
            if result.success or result.fun < 1e9:
                de_points = result.x.reshape(N, 2)
                _update_best(de_points)
                
        except Exception as e:
            pass

    # --- Phase 2: Local Refinement from Multiple Starts ---
    start_points = []
    
    # 1. Add DE result if we have it
    if best_points is not None:
        start_points.append(best_points.flatten())
    
    # 2. Regular 13-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    regular_13 = np.column_stack([np.cos(angles), np.sin(angles)])
    start_points.append(regular_13.flatten())
    
    # 3. Two-ring configurations
    for n_out in [5, 6, 7, 8]:
        n_in = N - n_out
        if n_in < 1:
            continue
        for r_in in [0.3, 0.5, 0.7]:
            for offset in [0, np.pi / n_out]:
                pts = _make_ring_config(n_out, n_in, r_in, offset)
                start_points.append(pts.flatten())
    
    # 4. Random starts
    for _ in range(3):
        start_points.append(np.random.uniform(-1, 1, (N, 2)).flatten())

    # Optimization Pipeline for each start
    for x0_flat in start_points:
        if time.time() > deadline - 15:
            break
            
        current_x = x0_flat.copy()
        current_score = _score(current_x.reshape(N, 2))
        
        if current_score > best_score:
            best_score = current_score
            best_points = current_x.reshape(N, 2).copy()
        
        # Run local polish with a time budget
        time_budget = min(10.0, deadline - time.time() - 5)
        if time_budget > 2.0:
            polished_points, polished_score = _local_polish(current_x.reshape(N, 2), time_budget=time_budget)
            _update_best(polished_points)
            
            # If we found a new best, we might want to polish it again with more time?
            # _local_polish already does multi-stage.
            
    # Final global check: if we have time, do one more DE run from the best point neighborhood?
    # Or just return best.
    
    if best_points is None:
        return np.random.rand(N, 2)
        
    return best_points
