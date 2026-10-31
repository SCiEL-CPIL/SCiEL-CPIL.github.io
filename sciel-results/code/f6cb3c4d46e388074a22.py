"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 11
import numpy as np
import time
from scipy.optimize import minimize
from itertools import combinations

# EVOLVE-BLOCK-START

# The equilateral triangle vertices
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, np.sqrt(3) / 2])
SQRT3 = np.sqrt(3)

def _triangle_area_vec(pts):
    """Vectorized computation of all triangle areas. Returns array of shape (C(n,3),)."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    return np.array(areas)

def _min_area(pts):
    """Compute the minimum area of any triangle formed by three of the points."""
    n = len(pts)
    min_a = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                if area < min_a:
                    min_a = area
    return min_a

def _min_area_smooth(pts, beta=50.0):
    """Smooth approximation of min area using log-sum-exp."""
    n = len(pts)
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                ax, ay = pts[i]
                bx, by = pts[j]
                cx, cy = pts[k]
                area = 0.5 * abs((bx - ax) * (cy - ay) - (by - ay) * (cx - ax))
                areas.append(area)
    areas = np.array(areas)
    if len(areas) == 0:
        return 0.0
    # log-sum-exp for min: -logsumexp(-areas/beta) * beta
    max_neg = np.max(-areas / beta)
    lse = max_neg + np.log(np.sum(np.exp(-areas / beta - max_neg)))
    return -lse * beta

def _point_in_triangle(p, tol=1e-9):
    """Check if point p is inside or on the boundary of the reference triangle."""
    x, y = p
    if y < -tol:
        return False
    if y > SQRT3 * (1 - x) + tol:
        return False
    if y > SQRT3 * x + tol:
        return False
    return True

def _clamp_to_triangle(p):
    """Clamp a point to be inside the triangle."""
    x, y = p
    if x < 0: x = 0
    if x > 1: x = 1
    if y < 0: y = 0
    if y > SQRT3/2: y = SQRT3/2
    
    val1 = SQRT3*x + y - SQRT3
    if val1 > 0:
        x -= (val1 / 4) * SQRT3
        y -= (val1 / 4) * 1
        
    val2 = y - SQRT3*x
    if val2 > 0:
        x += (val2 / 2) * SQRT3
        y -= (val2 / 2) * 1
    
    if x < 0: x = 0
    if x > 1: x = 1
    if y < 0: y = 0
    if y > SQRT3/2: y = SQRT3/2
    
    return np.array([x, y])

def _clamp_all(pts):
    """Clamp all points to be inside the triangle."""
    return np.array([_clamp_to_triangle(p) for p in pts])

def _penalty(pts):
    """Penalty for points outside the triangle."""
    pen = 0
    for p in pts:
        if not _point_in_triangle(p):
            x, y = p
            d1 = max(0, -y)
            d2 = max(0, y - SQRT3*(1-x))
            d3 = max(0, y - SQRT3*x)
            pen += max(d1, d2, d3)
    return pen

def _objective_flat(x, use_smooth=False, beta=50.0):
    """Objective function for optimization: minimize negative min area + penalty."""
    p = x.reshape(N, 2)
    if use_smooth:
        min_a = _min_area_smooth(p, beta)
    else:
        min_a = _min_area(p)
    pen = _penalty(p)
    return -min_a + 100 * pen

def _initial_placement(rng):
    """Generate initial points using structured + random approach."""
    pts = []
    # Place some points at vertices and edge midpoints for structure
    # Vertices
    pts.append([0.0, 0.0])
    pts.append([1.0, 0.0])
    pts.append([0.5, SQRT3/2])
    # Edge midpoints
    pts.append([0.5, 0.0])
    pts.append([0.75, SQRT3/4])
    pts.append([0.25, SQRT3/4])
    # Center
    pts.append([0.5, SQRT3/6])
    # Remaining 4 points randomly
    for _ in range(N - 7):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u = 1 - u
            v = 1 - v
        x = u * 1.0 + v * 0.5
        y = v * SQRT3 / 2
        pts.append([x, y])
    return np.array(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 145  # Safety margin before 150s
    
    rng = np.random.default_rng(42)
    
    best_points = None
    best_score = -1
    
    # Strategy 1: Multiple random restarts with smooth objective and L-BFGS-B
    n_starts = 20
    
    for start_idx in range(n_starts):
        if time.time() > deadline - 15:
            break
            
        # Generate initial configuration
        if start_idx == 0:
            pts = _initial_placement(rng)
        else:
            pts = []
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u = 1 - u
                    v = 1 - v
                x = u * 1.0 + v * 0.5
                y = v * SQRT3 / 2
                pts.append([x, y])
            pts = np.array(pts)
        
        # Phase 1: Smooth objective with L-BFGS-B
        def objective_smooth(x):
            return _objective_flat(x, use_smooth=True, beta=30.0)
        
        try:
            res1 = minimize(objective_smooth, pts.flatten(), method='L-BFGS-B',
                           options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8})
            p1 = res1.x.reshape(N, 2)
            p1 = _clamp_all(p1)
            score1 = _min_area(p1)
        except Exception:
            p1 = pts
            score1 = _min_area(pts)
        
        # Phase 2: Nelder-Mead on non-smooth objective
        def objective_nm(x):
            return _objective_flat(x, use_smooth=False)
        
        try:
            res2 = minimize(objective_nm, p1.flatten(), method='Nelder-Mead',
                           options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-7})
            p2 = res2.x.reshape(N, 2)
            p2 = _clamp_all(p2)
            score2 = _min_area(p2)
        except Exception:
            p2 = p1
            score2 = score1
        
        # Phase 3: Tight polish with Nelder-Mead
        if score2 > best_score:
            try:
                res3 = minimize(objective_nm, p2.flatten(), method='Nelder-Mead',
                               options={'maxiter': 3000, 'xatol': 1e-9, 'fatol': 1e-9})
                p3 = res3.x.reshape(N, 2)
                p3 = _clamp_all(p3)
                score3 = _min_area(p3)
                if _penalty(p3) < 1e-6 and score3 > best_score:
                    best_score = score3
                    best_points = p3.copy()
            except Exception:
                pass
        
        if _penalty(p2) < 1e-6 and score2 > best_score:
            best_score = score2
            best_points = p2.copy()
    
    if best_points is None:
        # Fallback: structured placement
        pts = _initial_placement(rng)
        return pts
    
    current_best = best_points.copy()
    current_score = best_score
    
    # Strategy 2: Local search refinement with adaptive step size
    remaining_time = deadline - time.time()
    if remaining_time > 10:
        step_size = 0.03
        no_improve_count = 0
        max_no_improve = 150
        
        while time.time() < deadline - 3 and no_improve_count < max_no_improve:
            improved = False
            for i in range(N):
                if time.time() > deadline - 3:
                    break
                # Try several random perturbations
                for _ in range(8):
                    delta = rng.normal(0, step_size, 2)
                    new_pts = current_best.copy()
                    new_pts[i] = current_best[i] + delta
                    new_pts[i] = _clamp_to_triangle(new_pts[i])
                    
                    new_score = _min_area(new_pts)
                    if new_score > current_score + 1e-10:
                        current_best = new_pts
                        current_score = new_score
                        improved = True
                        no_improve_count = 0
                        break
            
            if not improved:
                no_improve_count += 1
                step_size *= 0.85
                if step_size < 1e-5:
                    step_size = 1e-5
    
    return current_best

# EVOLVE-BLOCK-END
