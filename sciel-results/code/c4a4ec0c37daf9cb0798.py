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

def _nlp_objective(params):
    """
    NLP objective: maximize alpha (minimum triangle area).
    params: [x0,y0, x1,y1, ..., x10,y10, alpha]
    Minimize -alpha subject to:
      - area(i,j,k) >= alpha for all triples
      - all points inside triangle
    """
    n = 11
    points = params[:2*n].reshape(n, 2)
    alpha = params[2*n]
    
    # Check if all points are inside
    h = np.sqrt(3) / 2
    for p in points:
        x, y = p[0], p[1]
        if x < -1e-4 or x > 1+1e-4 or y < -1e-4 or y > h+1e-4:
            return 100.0
        if y > 2*h*x + 1e-4:
            return 100.0
        if y > 2*h*(1-x) + 1e-4:
            return 100.0
    
    # Compute min area
    min_area = _min_triangle_area(points)
    
    # If any triangle has area less than alpha, penalty
    if min_area < alpha - 1e-8:
        return -alpha + (alpha - min_area) * 100
    
    return -alpha

def _nlp_constraints(params):
    """
    Constraints for SLSQP:
    - Each triangle area >= alpha
    - Points inside triangle
    Returns list of constraint dicts.
    """
    n = 11
    points = params[:2*n].reshape(n, 2)
    alpha = params[2*n]
    h = np.sqrt(3) / 2
    
    constraints = []
    
    # Triangle area constraints
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                p1, p2, p3 = points[i], points[j], points[k]
                area = 0.5 * abs((p2[0]-p1[0])*(p3[1]-p1[1]) - (p3[0]-p1[0])*(p2[1]-p1[1]))
                # area >= alpha
                constraints.append({
                    'type': 'ineq',
                    'fun': lambda a=area, al=alpha: a - al
                })
    
    # Boundary constraints for each point
    for i in range(n):
        x, y = points[i, 0], points[i, 1]
        # x >= 0
        constraints.append({'type': 'ineq', 'fun': lambda x=x: x})
        # x <= 1
        constraints.append({'type': 'ineq', 'fun': lambda x=x: 1 - x})
        # y >= 0
        constraints.append({'type': 'ineq', 'fun': lambda y=y: y})
        # y <= h
        constraints.append({'type': 'ineq', 'fun': lambda y=y, hh=h: hh - y})
        # y <= 2*h*x
        constraints.append({'type': 'ineq', 'fun': lambda y=y, x=x, hh=h: 2*hh*x - y})
        # y <= 2*h*(1-x)
        constraints.append({'type': 'ineq', 'fun': lambda y=y, x=x, hh=h: 2*hh*(1-x) - y})
    
    return constraints

def _nlp_bounds():
    """Bounds for NLP parameters."""
    h = np.sqrt(3) / 2
    bounds = [(0, 1), (0, h)] * 11
    bounds.append((0.001, 0.1))  # alpha
    return bounds

def _solve_nlp(x0, deadline):
    """
    Solve NLP problem from initial guess x0.
    Returns (points, min_area) or (None, 0) on failure.
    """
    try:
        result = minimize(
            _nlp_objective, x0,
            method='SLSQP',
            bounds=_nlp_bounds(),
            constraints=_nlp_constraints(x0),
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False}
        )
        
        if result.success or result.fun < -0.005:
            points = result.x[:22].reshape(11, 2)
            if _all_inside(points):
                min_area = _min_triangle_area(points)
                return points, min_area
    except Exception:
        pass
    return None, 0.0

def _local_polish(points, deadline):
    """Local refinement using coordinate descent."""
    points = points.copy()
    current_min_area = _min_triangle_area(points)
    
    # Coordinate descent refinement
    for _ in range(50):
        if time.time() > deadline - 3:
            break
        improved = False
        for i in range(N):
            for coord in range(2):
                for step in [0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001]:
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

def _get_good_initial_config():
    """Generate a good initial configuration for NLP."""
    h = np.sqrt(3) / 2
    # Symmetric arrangement: 3 corners, 3 edge midpoints, 5 interior
    points = np.array([
        [0.05, 0.05],           # bottom-left corner
        [0.95, 0.05],           # bottom-right corner
        [0.5, h - 0.05],        # top corner
        [0.5, 0.08],            # bottom edge midpoint
        [0.08, 0.08 + 2*h*0.08],  # left edge
        [0.92, 0.08 + 2*h*0.08],  # right edge
        [0.5, h/3],             # center
        [0.35, h/3 - 0.05],     # interior
        [0.65, h/3 - 0.05],     # interior
        [0.4, h/3 + 0.1],       # interior
        [0.6, h/3 + 0.1]        # interior
    ])
    return points

def construct():
    deadline = time.time() + 140  # Safety margin before 150s limit
    h = np.sqrt(3) / 2
    
    best_points = None
    best_min_area = 0.0
    
    # Strategy 1: NLP with multiple initializations
    initial_configs = []
    
    # Config 1: symmetric arrangement
    initial_configs.append(_get_good_initial_config())
    
    # Config 2: hexagonal-like
    cx, cy = 0.5, h/3
    r = 0.12
    hex_points = np.array([
        [0.05, 0.05],
        [0.95, 0.05],
        [0.5, h - 0.05],
        [0.5, 0.08],
        [0.08, 0.08 + 2*h*0.08],
        [0.92, 0.08 + 2*h*0.08],
        [cx, cy],
        [cx + r, cy],
        [cx - r, cy],
        [cx, cy + r*0.866],
        [cx, cy - r*0.866]
    ])
    initial_configs.append(hex_points)
    
    # Config 3: grid-like
    grid_points = np.array([
        [0.1, 0.1], [0.5, 0.1], [0.9, 0.1],
        [0.2, 0.3], [0.5, 0.3], [0.8, 0.3],
        [0.3, 0.5], [0.5, 0.5], [0.7, 0.5],
        [0.4, 0.65], [0.6, 0.65]
    ])
    initial_configs.append(grid_points)
    
    for init_pts in initial_configs:
        if time.time() > deadline - 20:
            break
        
        # Build initial params: [x0,y0,...,x10,y10, alpha]
        alpha_init = _min_triangle_area(init_pts) * 0.8
        x0 = np.concatenate([init_pts.flatten(), [alpha_init]])
        
        nlp_points, nlp_area = _solve_nlp(x0, deadline)
        
        if nlp_points is not None and nlp_area > best_min_area:
            best_min_area = nlp_area
            best_points = nlp_points.copy()
    
    # Strategy 2: Differential evolution as backup
    if best_points is None or time.time() < deadline - 30:
        def _negative_min_area(params):
            points = params.reshape(-1, 2)
            if not _all_inside(points):
                penalty = 0.0
                for p in points:
                    x, y = p[0], p[1]
                    if x < 0 or x > 1 or y < 0 or y > h:
                        penalty += 10.0
                    if y > 2*h*x + 1e-4:
                        penalty += 10.0
                    if y > 2*h*(1-x) + 1e-4:
                        penalty += 10.0
                return -0.001 + penalty
            return -_min_triangle_area(points)
        
        bounds = [(0, 1)] * 2 * N
        try:
            de_result = differential_evolution(
                _negative_min_area, bounds,
                seed=42, maxiter=80, popsize=15,
                tol=1e-8, mutation=(0.5, 1.5), recombination=0.7,
                polish=True, init='sobol'
            )
            de_points = de_result.x.reshape(-1, 2)
            if _all_inside(de_points):
                de_min_area = _min_triangle_area(de_points)
                if de_min_area > best_min_area:
                    best_min_area = de_min_area
                    best_points = de_points.copy()
        except Exception:
            pass
    
    # Strategy 3: Random initializations with local polish
    if best_points is None or time.time() < deadline - 15:
        rng = np.random.RandomState(123)
        
        for seed in range(6):
            if time.time() > deadline - 10:
                break
            
            points = []
            for _ in range(N):
                while True:
                    u, v = rng.random(), rng.random()
                    if u + v < 0.95 and u > 0.02 and v > 0.02:
                        px = u + 0.5 * v
                        py = h * v
                        points.append([px, py])
                        break
            
            points = np.array(points)
            polished, min_area = _local_polish(points, deadline)
            
            if min_area > best_min_area:
                best_min_area = min_area
                best_points = polished
    
    # Final polish on the best result
    if best_points is not None and time.time() < deadline - 3:
        best_points, best_min_area = _local_polish(best_points, deadline)
    
    if best_points is None:
        # Fallback: simple grid
        best_points = _get_good_initial_config()
    
    return best_points
# EVOLVE-BLOCK-END
