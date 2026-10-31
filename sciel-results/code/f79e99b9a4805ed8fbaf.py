"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize

def _min_triangle_area(points):
    """Compute minimum area among all C(11,3)=165 triangles."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _inside_triangle(p, tol=1e-6):
    """Check if point is inside or on boundary of equilateral triangle."""
    h = np.sqrt(3) / 2
    x, y = p[0], p[1]
    if x < -tol or x > 1+tol or y < -tol or y > h+tol:
        return False
    if y < -tol:
        return False
    if y > 2*h*x + tol:
        return False
    if y > 2*h*(1-x) + tol:
        return False
    return True

def _all_inside(points, tol=1e-6):
    """Check if all points are inside the triangle."""
    return all(_inside_triangle(p, tol) for p in points)

def _barycentric_to_cartesian(params):
    """
    Convert barycentric coordinates (u, v) for each point to Cartesian.
    params shape (N, 2).
    u in [0,1], v in [0,1].
    Mapping: u = r1, v = r2*(1-r1) ensures u>=0, v>=0, u+v<=1.
    A=(0,0), B=(1,0), C=(0.5, sqrt(3)/2)
    P = u*B + v*C + (1-u-v)*A
    """
    h = np.sqrt(3) / 2
    u = params[:, 0]
    v_raw = params[:, 1]
    # Apply transformation to ensure feasibility
    v = v_raw * (1 - u)
    
    x = u * 1.0 + v * 0.5
    y = u * 0.0 + v * h
    
    return np.column_stack((x, y))

def _smooth_min_area(params, temperature=0.01):
    """
    Smooth approximation of the negative minimum triangle area.
    Uses log-sum-exp trick for soft min.
    Objective to minimize: -soft_min(areas)
    soft_min(x) = -T * log(sum(exp(-x_i/T)))
    """
    points = _barycentric_to_cartesian(params)
    
    # Compute all triangle areas
    areas = []
    n = N
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                areas.append(area)
    
    areas = np.array(areas)
    
    # Soft min
    # We want to maximize the min area, so minimize -min area.
    # soft_min = -T * log(sum(exp(-areas/T)))
    # Objective = -soft_min = T * log(sum(exp(-areas/T)))
    
    # Numerical stability
    max_val = np.max(areas)
    if max_val < 1e-9:
        return 0.0
        
    # exp(-areas/T) can overflow if T is small and areas are large? No, areas are small.
    # But if T is very small, exp(-area/T) underflows for large areas.
    # We can shift by max area? No, standard log-sum-exp is:
    # log(sum(exp(x_i))) = max_x + log(sum(exp(x_i - max_x)))
    # Here we have sum(exp(-areas/T)). Let y_i = -areas/T.
    # max_y = -min(areas)/T.
    # This is getting complicated. Let's just use the direct formula with clipping.
    
    # If T is small, the soft min approaches the hard min.
    # Let's use a slightly larger T for stability in DE, and smaller for polish.
    
    try:
        # To avoid underflow/overflow:
        # log_sum_exp(y) where y = -areas/T
        # max_y = np.max(y)
        # result = max_y + log(sum(exp(y - max_y)))
        
        y = -areas / temperature
        max_y = np.max(y)
        # If max_y is very large (areas near 0), exp(y-max_y) is fine.
        # If areas are large, y is very negative, max_y is very negative.
        
        sum_exp = np.sum(np.exp(y - max_y))
        log_sum_exp = max_y + np.log(sum_exp)
        
        # soft_min = -T * log_sum_exp
        # obj = -soft_min = T * log_sum_exp
        obj = temperature * log_sum_exp
        
        return obj
        
    except (OverflowError, FloatingPointError):
        return 100.0

def _negative_min_area(params):
    """Objective for differential evolution: negative of min triangle area (hard)."""
    points = _barycentric_to_cartesian(params)
    return -_min_triangle_area(points)

def _local_polish(points, deadline):
    """
    Local refinement using SLSQP on the smooth objective in barycentric space.
    Then a final Nelder-Mead on Cartesian for hard min.
    """
    # Convert to barycentric parameters for optimization
    # We need to invert _barycentric_to_cartesian?
    # Actually, we can just optimize in Cartesian with constraints, or
    # re-parameterize.
    # Let's stick to Cartesian for the final polish as it's more direct for hard min.
    
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    h = np.sqrt(3) / 2
    
    # 1. SLSQP with smooth objective in barycentric space?
    # We don't have the barycentric params easily from Cartesian without solving.
    # Let's just use Nelder-Mead on Cartesian with a smooth penalty for boundaries.
    
    def objective_cartesian(params):
        pts = params.reshape(-1, 2)
        # Penalty for being outside
        penalty = 0.0
        for p in pts:
            x, y = p
            if x < 0: penalty += (x)**2 * 100
            if x > 1: penalty += (x-1)**2 * 100
            if y < 0: penalty += (y)**2 * 100
            if y > h: penalty += (y-h)**2 * 100
            if y > 2*h*x: penalty += (y - 2*h*x)**2 * 100
            if y > 2*h*(1-x): penalty += (y - 2*h*(1-x))**2 * 100
        
        if penalty > 1e-4:
            return 10.0 + penalty
            
        areas = []
        for i in range(N):
            for j in range(i+1, N):
                for k in range(j+1, N):
                    p1, p2, p3 = pts[i], pts[j], pts[k]
                    area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                    areas.append(area)
        areas = np.array(areas)
        
        # Use smooth min for objective to help gradient-based or NM converge better
        # But NM doesn't use gradients.
        # Let's just return -min_area.
        return -np.min(areas)

    # 2. Nelder-Mead Polish
    try:
        result = minimize(objective_cartesian, points.flatten(), method='Nelder-Mead',
                         options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12})
        if result.fun < -current_min_area:
            new_points = result.x.reshape(-1, 2)
            if _all_inside(new_points):
                new_min_area = _min_triangle_area(new_points)
                if new_min_area > current_min_area:
                    points = new_points
                    current_min_area = new_min_area
    except Exception:
        pass

    # 3. Coordinate Descent for fine-tuning
    for _ in range(50):
        if time.time() > deadline - 2:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                for step in [0.001, 0.0001, 0.00001]:
                    old_val = points[i, coord]
                    points[i, coord] = old_val + step
                    if _all_inside(points):
                        new_area = _min_triangle_area(points)
                        if new_area > current_min_area:
                            current_min_area = new_area
                            improved = True
                            continue
                    points[i, coord] = old_val - step
                    if _all_inside(points):
                        new_area = _min_triangle_area(points)
                        if new_area > current_min_area:
                            current_min_area = new_area
                            improved = True
                            continue
                    points[i, coord] = old_val
        if not improved:
            break

    return points, current_min_area

def construct():
    deadline = time.time() + 135
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: Differential Evolution in Barycentric Space
    # Bounds for u, v are [0, 1]
    bounds = [(0, 1)] * 2 * N
    
    # We use the smooth objective for DE to navigate the landscape better
    # DE doesn't need gradients, but a smoother landscape helps it avoid getting stuck in sharp valleys
    # caused by the non-differentiable min function.
    
    # We will run DE with a slightly higher temperature for the soft min
    # to make the landscape smoother, then polish with hard min.
    
    def de_objective(params):
        # Use a moderate temperature
        return _smooth_min_area(params, temperature=0.05)
        
    if time.time() < deadline - 30:
        try:
            de_result = differential_evolution(
                de_objective, bounds,
                seed=42, maxiter=200, popsize=20,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=False, # We will do our own polish
                init='sobol'
            )
            de_params = de_result.x
            de_points = _barycentric_to_cartesian(de_params)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
            
    # Strategy 2: Multi-start with Local Polish
    if best_points is None or time.time() < deadline - 20:
        rng = np.random.RandomState(123)
        
        for seed in range(5):
            if time.time() > deadline - 10:
                break
            
            # Generate random points in barycentric coordinates
            params = rng.random((N, 2))
            points = _barycentric_to_cartesian(params)
            
            polished, min_area = _local_polish(points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Strategy 3: Final Polish on the best result
    if best_points is not None and time.time() < deadline - 5:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    return best_points
# EVOLVE-BLOCK-END
