import numpy as np
from scipy.optimize import differential_evolution, minimize
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)

# Precompute indices for vectorized area calculation
idx_i, idx_j, idx_k = [], [], []
for i in range(N):
    for j in range(i + 1, N):
        for k in range(j + 1, N):
            idx_i.append(i)
            idx_j.append(j)
            idx_k.append(k)

idx_i = np.array(idx_i)
idx_j = np.array(idx_j)
idx_k = np.array(idx_k)

def _all_areas(points):
    """
    Compute all triangle areas. points: (N, 2).
    Returns array of shape (165,).
    """
    p1 = points[idx_i]
    p2 = points[idx_j]
    p3 = points[idx_k]
    
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    areas = 0.5 * np.abs(cross)
    return areas

def min_triangle_area_vec(points):
    """
    points: (N, 2) array
    Returns the minimum area of any triangle formed by 3 points.
    """
    return np.min(_all_areas(points))

def _all_inside(points):
    """
    Check if all points are inside or on the boundary of the equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, sqrt(3)/2)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # y >= 0
    if np.any(y < -1e-10):
        return False
    
    # y <= sqrt(3) * x (left side)
    if np.any(y > SQRT3 * x + 1e-10):
        return False
        
    # y <= sqrt(3) * (1 - x) (right side)
    if np.any(y > SQRT3 * (1 - x) + 1e-10):
        return False
        
    # x >= 0 and x <= 1 (implied by above for y>=0, but good to check)
    if np.any(x < -1e-10) or np.any(x > 1 + 1e-10):
        return False
        
    return True

def _params_to_points_symmetric(params):
    """
    Map symmetric parameters to 11 points.
    Structure:
    - Point 0: on the symmetry axis (x=0.5), parameterized by y only.
    - Points 1-10: 5 pairs of mirrored points.
      params[1+2i], params[2+2i] = x, y for left point of pair i.
      Right point is (1-x, y).
    
    params: array of shape (11,)
    """
    pts = np.zeros((N, 2))
    # Axis point
    y0 = params[0]
    y0 = np.clip(y0, 0.0, H)
    pts[0] = [0.5, y0]
    
    for i in range(5):
        x = params[1 + 2*i]
        y = params[2 + 2*i]
        
        # Clip x to [0, 0.5]
        x = np.clip(x, 0.0, 0.5)
        
        # Clip y to be inside the triangle
        max_y_left = SQRT3 * x
        max_y_right = SQRT3 * (1 - x)
        max_y = min(max_y_left, max_y_right, H)
        
        y = np.clip(y, 0.0, max_y)
        
        pts[1 + 2*i] = [x, y]
        pts[2 + 2*i] = [1.0 - x, y]
        
    return pts

def _points_to_params_symmetric(points):
    """
    Extract symmetric parameters from a full point set for initialization.
    Assumes the input points are roughly symmetric.
    """
    pts = points.copy()
    params = np.zeros(11)
    
    # Find the point closest to the axis x=0.5 to be the axis point
    dists = np.abs(pts[:, 0] - 0.5)
    axis_idx = np.argmin(dists)
    params[0] = pts[axis_idx, 1]
    
    # For the remaining 10 points, sort by x to pair them
    remaining_indices = [i for i in range(N) if i != axis_idx]
    remaining_points = pts[remaining_indices]
    
    # Sort remaining points by x coordinate
    sort_idx = np.argsort(remaining_points[:, 0])
    sorted_remaining = remaining_points[sort_idx]
    
    for i in range(5):
        left_pt = sorted_remaining[2*i]
        right_pt = sorted_remaining[2*i + 1]
        
        # Ensure left_pt is actually on the left
        if left_pt[0] > 0.5:
             left_pt, right_pt = right_pt, left_pt
             
        x_val = left_pt[0]
        y_val = (left_pt[1] + right_pt[1]) / 2.0
        
        params[1 + 2*i] = x_val
        params[2 + 2*i] = y_val
        
    return params

def _inside_penalty(points):
    """Calculate penalty for points outside the triangle."""
    x = points[:, 0]
    y = points[:, 1]
    # Distances to boundaries (negative means outside)
    d1 = y
    d2 = SQRT3 * x - y
    d3 = SQRT3 * (1 - x) - y
    
    # Sum of squared violations
    violations = np.maximum(0, -d1)**2 + np.maximum(0, -d2)**2 + np.maximum(0, -d3)**2
    return np.sum(violations)

def objective_symmetric(params):
    """
    Objective in symmetric parameter space: -min_area + penalty.
    """
    pts = _params_to_points_symmetric(params)
    pen = _inside_penalty(pts)
    if pen > 1e-4:
        return 1e6 + pen
    area = min_triangle_area_vec(pts)
    return -area + 10.0 * pen

def soft_min_area(points, beta=100.0):
    """
    Compute a smooth approximation of the minimum triangle area using log-sum-exp.
    """
    areas = _all_areas(points)
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_area = np.max(areas)
    exp_terms = np.exp(-beta * (areas - max_area))
    soft_min = max_area - np.log(np.sum(exp_terms)) / beta
    
    return soft_min

def gradient_descent_smooth(points, start_time, deadline, 
                            max_iter=2000, lr_init=0.01, lr_decay=0.995,
                            beta_init=20.0, beta_final=200.0):
    """
    Gradient descent on the smooth soft-min objective.
    """
    current_points = points.copy()
    
    best_points = current_points.copy()
    best_min_area = min_triangle_area_vec(best_points)
    
    rng = np.random.default_rng(42)
    lr = lr_init
    beta = beta_init
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Animate beta
        progress = iteration / max_iter
        beta = beta_init + (beta_final - beta_init) * progress
        
        # Compute gradient numerically
        eps = 1e-6
        grad_points = np.zeros_like(current_points)
        
        for i in range(N):
            for j in range(2):
                old_val = current_points[i, j]
                current_points[i, j] = old_val + eps
                obj_p = -soft_min_area(current_points, beta=beta)
                current_points[i, j] = old_val - eps
                obj_m = -soft_min_area(current_points, beta=beta)
                current_points[i, j] = old_val
                grad_points[i, j] = (obj_p - obj_m) / (2 * eps)
        
        # Gradient descent step
        new_points = current_points - lr * grad_points
        
        # Project points back into triangle if needed
        if not _all_inside(new_points):
            # Simple projection: clamp to boundaries
            for i in range(N):
                x, y = new_points[i]
                if y < 0: y = 0
                if y > SQRT3 * x: y = SQRT3 * x
                if y > SQRT3 * (1 - x): y = SQRT3 * (1 - x)
                new_points[i] = [x, y]
            
            if not _all_inside(new_points):
                # If still invalid, skip or reduce step
                lr *= 0.9
                continue
        
        # Evaluate actual min area
        new_min_area = min_triangle_area_vec(new_points)
        
        # Accept if better
        if new_min_area > best_min_area:
            best_min_area = new_min_area
            best_points = new_points.copy()
        
        current_points = new_points
        lr *= lr_decay
        
        if lr < 1e-10:
            break
    
    return best_points

def simulate_annealing(points, start_time, deadline, initial_temp=0.1, cooling_rate=0.995, max_iter=5000):
    """
    Simulated annealing on the smooth soft-min objective.
    """
    current_points = points.copy()
    current_score = min_triangle_area_vec(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    rng = np.random.default_rng(42)
    temperature = initial_temp
    
    for iteration in range(max_iter):
        if time.time() > deadline - 5:
            break
        
        # Generate a random perturbation
        perturbation = rng.normal(0, temperature, size=current_points.shape)
        new_points = current_points + perturbation
        
        # Check if new points are inside the triangle
        if not _all_inside(new_points):
            continue
        
        new_score = min_triangle_area_vec(new_points)
        
        # Accept if better or with probability exp(-delta/T)
        # Note: we are maximizing score, so delta = current - new
        if new_score > current_score:
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
        else:
            # delta = current_score - new_score > 0
            prob = np.exp((new_score - current_score) / temperature)
            if rng.random() < prob:
                current_points = new_points
                current_score = new_score
        
        # Cool down
        temperature *= cooling_rate
        
        if temperature < 1e-8:
            break
    
    return best_points

def refine_local(points, start_time, deadline):
    """
    Perform local refinement using gradient descent and simulated annealing.
    """
    if time.time() > deadline:
        return points
    
    best_points = points.copy()
    best_score = min_triangle_area_vec(best_points)
    
    # Phase 1: Gradient Descent
    if time.time() < deadline - 20:
        gd_deadline = deadline - 20
        gd_points = gradient_descent_smooth(best_points, start_time, gd_deadline)
        gd_score = min_triangle_area_vec(gd_points)
        if gd_score > best_score:
            best_score = gd_score
            best_points = gd_points.copy()
    
    # Phase 2: Simulated Annealing
    if time.time() < deadline - 10:
        sa_deadline = deadline - 10
        sa_points = simulate_annealing(best_points, start_time, sa_deadline, 
                                       initial_temp=0.05, cooling_rate=0.998, max_iter=2000)
        sa_score = min_triangle_area_vec(sa_points)
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_points.copy()
    
    # Phase 3: Nelder-Mead Polish on Reduced Space
    if time.time() < deadline - 2:
        # Convert to symmetric params for NM
        params = _points_to_params_symmetric(best_points)
        
        rng = np.random.default_rng(42)
        for attempt in range(3):
            if time.time() > deadline - 2:
                break
            
            if attempt == 0:
                x0 = params.copy()
            else:
                perturbation = rng.normal(0, 0.005, size=params.shape)
                x0 = params + perturbation
                # Ensure x0 is valid
                pts_check = _params_to_points_symmetric(x0)
                if not _all_inside(pts_check):
                    continue
            
            try:
                res = minimize(
                    objective_symmetric,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                
                if time.time() > deadline - 2:
                    break
                    
                new_params = res.x
                new_points = _params_to_points_symmetric(new_params)
                
                if _all_inside(new_points):
                    new_score = min_triangle_area_vec(new_points)
                    if new_score > best_score:
                        best_score = new_score
                        best_points = new_points.copy()
                        
            except Exception:
                continue
    
    return best_points

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin before 150s limit
    
    # Bounds for symmetric parameters
    bounds_sym = [(0, H)]
    for i in range(5):
        bounds_sym.append((0, 0.5))
        bounds_sym.append((0, H))
    
    best_params_sym = None
    best_score = -np.inf
    
    def update_best_sym(params_sym):
        nonlocal best_params_sym, best_score
        if params_sym is not None:
            pts = _params_to_points_symmetric(params_sym)
            if _inside_penalty(pts) > 1e-6:
                return
            score = min_triangle_area_vec(pts)
            if score > best_score:
                best_score = score
                best_params_sym = params_sym.copy()
                # Also update best_points for refinement
                # We'll store points separately or just recompute
    # We need to track points too
    best_points = None
    
    def update_best_full(points):
        nonlocal best_params_sym, best_score, best_points
        if points is not None:
            # Clamp to ensure validity
            pts = points.copy()
            for i in range(N):
                x, y = pts[i]
                x = np.clip(x, 0.0, 1.0)
                y = np.clip(y, 0.0, H)
                max_y_left = SQRT3 * x
                if y > max_y_left:
                    y = max_y_left
                max_y_right = SQRT3 * (1 - x)
                if y > max_y_right:
                    y = max_y_right
                pts[i] = [x, y]
            
            if _all_inside(pts):
                score = min_triangle_area_vec(pts)
                if score > best_score:
                    best_score = score
                    best_points = pts.copy()
                    best_params_sym = _points_to_params_symmetric(pts)

    rng = np.random.default_rng(42)
    
    # Phase 1: Global Optimization with Differential Evolution
    if time.time() < deadline - 30:
        try:
            res_de = differential_evolution(
                objective_symmetric,
                bounds_sym,
                maxiter=200,
                popsize=15,
                tol=0.01,
                seed=42,
                polish=True,
                init='sobol'
            )
            
            candidate_params = res_de.x
            candidate_points = _params_to_points_symmetric(candidate_params)
            update_best_full(candidate_points)
            
        except Exception:
            pass
        
    if best_points is None:
        # Fallback: random valid points
        for _ in range(100):
            params = np.zeros(11)
            params[0] = rng.uniform(0, H)
            for i in range(5):
                x = rng.uniform(0, 0.5)
                max_y = min(SQRT3 * x, SQRT3 * (1 - x), H)
                y = rng.uniform(0, max_y)
                params[1 + 2*i] = x
                params[2 + 2*i] = y
            pts = _params_to_points_symmetric(params)
            if _all_inside(pts):
                update_best_full(pts)
                if best_points is not None:
                    break
    
    if best_points is None:
        # Last resort: center point + 5 pairs
        best_points = np.zeros((N, 2))
        best_points[0] = [0.5, H/3]
        for i in range(5):
            x = 0.1 + i * 0.08
            y = H/3
            best_points[1 + 2*i] = [x, y]
            best_points[2 + 2*i] = [1 - x, y]
        update_best_full(best_points)

    # Phase 2: Local Refinement
    if time.time() < deadline - 5:
        best_points = refine_local(best_points, start_time, deadline)
    
    # Final validation
    if not _all_inside(best_points):
        # Project to valid region
        for i in range(N):
            x, y = best_points[i]
            if y < 0: y = 0
            if y > SQRT3 * x: y = SQRT3 * x
            if y > SQRT3 * (1 - x): y = SQRT3 * (1 - x)
            best_points[i] = [x, y]
            
    return best_points
