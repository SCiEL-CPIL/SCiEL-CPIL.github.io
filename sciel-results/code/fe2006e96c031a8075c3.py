import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

N = 11
H = np.sqrt(3) / 2
S3 = np.sqrt(3)

# Pre-compute indices for all 165 triangles
tri_indices = list(combinations(range(N), 3))
tri_indices = np.array(tri_indices)
I = tri_indices[:, 0]
J = tri_indices[:, 1]
K = tri_indices[:, 2]

def _min_area(points):
    """
    points: (N, 2)
    Returns the minimum area of any triangle formed by 3 points.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    # Cross product z-component
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _penalty(points):
    """
    Calculate penalty for points outside the unit equilateral triangle.
    Triangle vertices: (0,0), (1,0), (0.5, H)
    """
    x = points[:, 0]
    y = points[:, 1]
    
    # Distance to boundary violations
    v1 = np.maximum(0, -y)
    v2 = np.maximum(0, y - S3 * x)
    v3 = np.maximum(0, y - S3 * (1 - x))
    
    return np.sum(v1**2 + v2**2 + v3**2)

def _objective(points):
    """
    Objective to maximize: min_area - penalty
    """
    area = _min_area(points)
    pen = _penalty(points)
    return area - 1000.0 * pen

def _smooth_objective(points, beta=50.0):
    """
    Smooth approximation of min area using log-sum-exp.
    """
    Pi = points[I]
    Pj = points[J]
    Pk = points[K]
    
    v1 = Pj - Pi
    v2 = Pk - Pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Log-sum-exp approximation of min
    min_approx = -np.log(np.sum(np.exp(-beta * areas))) / beta
    
    pen = _penalty(points)
    return min_approx - 1000.0 * pen

def _random_valid_point(rng):
    """
    Generate a random point inside the triangle.
    """
    while True:
        u = rng.random()
        v = rng.random()
        if u + v < 1:
            x = u + 0.5 * v
            y = H * v
            return np.array([x, y])

def _initial_population(rng, count=1):
    """
    Generate initial valid configurations.
    """
    configs = []
    for _ in range(count):
        pts = []
        for _ in range(N):
            pts.append(_random_valid_point(rng))
        configs.append(np.array(pts))
    return configs

def _grid_search_initialization():
    """
    Generate a structured initial configuration using a triangular lattice pattern
    within the triangle. Uses a hexagonal packing inspired layout.
    """
    points = []
    
    # Use a triangular lattice with spacing optimized for 11 points
    # Place points in rows with alternating offsets
    # Row 0 (bottom): 4 points
    # Row 1: 3 points
    # Row 2: 2 points  
    # Row 3: 1 point
    # Row 4: 1 point (top)
    
    # Define lattice parameters
    dx = 0.22  # horizontal spacing
    dy = dx * S3 / 2  # vertical spacing for equilateral triangle lattice
    
    # Row 0: y=0, x from 0.1 to 0.9
    for i in range(4):
        x = 0.1 + i * dx
        y = 0.0
        points.append(np.array([x, y]))
    
    # Row 1: y=dy, x offset by dx/2
    for i in range(3):
        x = 0.21 + i * dx
        y = dy
        points.append(np.array([x, y]))
    
    # Row 2: y=2*dy
    for i in range(2):
        x = 0.32 + i * dx
        y = 2 * dy
        points.append(np.array([x, y]))
    
    # Row 3: y=3*dy
    x = 0.43
    y = 3 * dy
    points.append(np.array([x, y]))
    
    # Row 4: top point
    x = 0.5
    y = 4 * dy
    points.append(np.array([x, y]))
    
    points = np.array(points)
    
    # Ensure all points are inside the triangle with proper clipping
    for i in range(len(points)):
        x, y = points[i]
        # Clip to triangle boundaries
        if y < 0:
            y = 0
        if y > S3 * x:
            x = y / S3
        if y > S3 * (1 - x):
            x = 1 - y / S3
        points[i] = np.array([x, y])
    
    return points

def _simulated_annealing(initial_points, time_limit, rng):
    """
    Run Simulated Annealing with adaptive step size and multiple restarts.
    """
    current_points = initial_points.copy()
    current_score = _objective(current_points)
    
    best_points = current_points.copy()
    best_score = current_score
    
    T = 0.1  # Higher initial temperature
    cooling_rate = 0.9999  # Slower cooling
    min_T = 1e-8
    step_scale = 0.15  # Larger initial step
    
    start_time = time.time()
    no_improve_count = 0
    
    while T > min_T:
        if time.time() - start_time > time_limit:
            break
        
        # Propose a move
        idx = rng.integers(N)
        noise = rng.normal(0, step_scale, size=2)
        
        new_points = current_points.copy()
        new_points[idx] += noise
        
        # Project back to triangle if needed
        x, y = new_points[idx]
        if y < 0:
            y = 0
        if y > S3 * x:
            x = y / S3
        if y > S3 * (1 - x):
            x = 1 - y / S3
        new_points[idx] = np.array([x, y])
        
        new_score = _objective(new_points)
        
        delta = new_score - current_score
        
        if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
            current_points = new_points
            current_score = new_score
            
            if current_score > best_score:
                best_score = current_score
                best_points = current_points.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
        else:
            no_improve_count += 1
        
        # Adaptive step size
        if no_improve_count > 200:
            step_scale = max(0.005, step_scale * 0.85)
            no_improve_count = 0
        elif no_improve_count == 0:
            step_scale = min(0.2, step_scale * 1.03)
        
        T *= cooling_rate
        
        # Occasionally, do a larger jump to escape local optima
        if rng.random() < 0.01:
             idx = rng.integers(N)
             new_points = current_points.copy()
             new_points[idx] = _random_valid_point(rng)
             new_score = _objective(new_points)
             delta = new_score - current_score
             if delta > 0 or (T > min_T and rng.random() < np.exp(delta / T)):
                 current_points = new_points
                 current_score = new_score
                 if current_score > best_score:
                     best_score = current_score
                     best_points = current_points.copy()
    
    return best_points, best_score

def _nlp_refine(points, time_limit, rng):
    """
    Refine points using SLSQP to maximize the minimum triangle area directly.
    """
    current_points = points.copy()
    current_score = _objective(current_points)
    best_points = current_points.copy()
    best_score = current_score
    
    start_time = time.time()
    
    # We want to maximize min_area. 
    # SLSQP minimizes. So we minimize -min_area.
    # Use the smooth approximation for the objective in SLSQP.
    
    bounds = [(0.0, 1.0), (0.0, H)] * N
    
    def _min_smooth_obj(flat_pts):
        pts = flat_pts.reshape(-1, 2)
        return -_smooth_objective(pts, beta=200.0)  # Higher beta for better approximation
    
    # Try SLSQP first
    try:
        res = minimize(
            _min_smooth_obj, 
            current_points.flatten(), 
            method='SLSQP',
            bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10}
        )
        
        if res.success or res.fun < -best_score + 1e-6:
            polished_pts = res.x.reshape(-1, 2)
            # Clip to triangle
            x = polished_pts[:, 0]
            y = polished_pts[:, 1]
            y = np.maximum(y, 0)
            viol1 = y > S3 * x
            if np.any(viol1):
                x[viol1] = y[viol1] / S3
            viol2 = y > S3 * (1 - x)
            if np.any(viol2):
                x[viol2] = 1 - y[viol2] / S3
            polished_pts[:, 0] = x
            polished_pts[:, 1] = y
            
            polished_score = _objective(polished_pts)
            if polished_score > best_score:
                best_score = polished_score
                best_points = polished_pts.copy()
                
    except Exception:
        pass
    
    # Try L-BFGS-B if time remains
    if time.time() - start_time < time_limit - 5:
        try:
            res2 = minimize(
                _min_smooth_obj, 
                best_points.flatten(), 
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'gtol': 1e-12}
            )
            polished_pts2 = res2.x.reshape(-1, 2)
            x = polished_pts2[:, 0]
            y = polished_pts2[:, 1]
            y = np.maximum(y, 0)
            viol1 = y > S3 * x
            if np.any(viol1):
                x[viol1] = y[viol1] / S3
            viol2 = y > S3 * (1 - x)
            if np.any(viol2):
                x[viol2] = 1 - y[viol2] / S3
            polished_pts2[:, 0] = x
            polished_pts2[:, 1] = y
            
            polished_score2 = _objective(polished_pts2)
            if polished_score2 > best_score:
                best_score = polished_score2
                best_points = polished_pts2.copy()
        except Exception:
            pass
    
    # Final high-precision polish: directly optimize with tighter tolerance
    if time.time() - start_time < time_limit - 2:
        try:
            def _final_obj(flat_pts):
                pts = flat_pts.reshape(-1, 2)
                return -_min_area(pts) + 1000.0 * _penalty(pts)
            
            res3 = minimize(
                _final_obj,
                best_points.flatten(),
                method='SLSQP',
                bounds=bounds,
                options={'maxiter': 50, 'ftol': 1e-12}
            )
            polished_pts3 = res3.x.reshape(-1, 2)
            x = polished_pts3[:, 0]
            y = polished_pts3[:, 1]
            y = np.maximum(y, 0)
            viol1 = y > S3 * x
            if np.any(viol1):
                x[viol1] = y[viol1] / S3
            viol2 = y > S3 * (1 - x)
            if np.any(viol2):
                x[viol2] = 1 - y[viol2] / S3
            polished_pts3[:, 0] = x
            polished_pts3[:, 1] = y
            
            polished_score3 = _objective(polished_pts3)
            if polished_score3 > best_score:
                best_score = polished_score3
                best_points = polished_pts3.copy()
        except Exception:
            pass
    
    return best_points, best_score

def construct():
    start_time = time.time()
    timeout = 145.0  # Safety margin from 150s kill
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -np.inf
    
    # Phase 1: Multi-start SA with diverse initializations
    num_starts = 6
    time_per_start = (timeout - 40) / num_starts  # Leave 40s for NLP refinement
    
    # Use different initialization strategies
    init_strategies = [
        _grid_search_initialization,
        lambda: _initial_population(rng, 1)[0],
        lambda: _initial_population(rng, 1)[0],
        lambda: _initial_population(rng, 1)[0],
        lambda: _initial_population(rng, 1)[0],
        lambda: _initial_population(rng, 1)[0],
    ]
    
    for i in range(num_starts):
        if time.time() - start_time > timeout - 40:
            break
        
        init_pts = init_strategies[i]()
        
        # Run SA
        sa_pts, sa_score = _simulated_annealing(init_pts, time_per_start, rng)
        
        if sa_score > best_score:
            best_score = sa_score
            best_points = sa_pts
    
    # Phase 2: NLP-based refinement (SLSQP + L-BFGS-B + final polish)
    if best_points is not None and time.time() - start_time < timeout - 15:
        nlp_time = timeout - 15 - (time.time() - start_time)
        nlp_pts, nlp_score = _nlp_refine(best_points, nlp_time, rng)
        
        if nlp_score > best_score:
            best_score = nlp_score
            best_points = nlp_pts
    
    # Final validity check
    if best_points is None:
        best_points = _grid_search_initialization()
    
    x = best_points[:, 0]
    y = best_points[:, 1]
    
    # Clip y >= 0
    y = np.maximum(y, 0)
    
    # Clip y <= sqrt(3)*x
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    
    # Clip y <= sqrt(3)*(1-x)
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    # Re-check after clipping
    y = np.maximum(y, 0)
    viol1 = y > S3 * x
    if np.any(viol1):
        x[viol1] = y[viol1] / S3
    viol2 = y > S3 * (1 - x)
    if np.any(viol2):
        x[viol2] = 1 - y[viol2] / S3
    
    best_points[:, 0] = x
    best_points[:, 1] = y
    
    return best_points
