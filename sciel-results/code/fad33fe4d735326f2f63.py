import numpy as np
import time
import itertools
from scipy.optimize import minimize

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4.0

# Precompute indices for all 165 triangles
_triplets = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _triplets])
_T_J = np.array([t[1] for t in _triplets])
_T_K = np.array([t[2] for t in _triplets])

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points using vectorized operations."""
    pi = points[_T_I]
    pj = points[_T_J]
    pk = points[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = 0.5 * np.abs(cross)
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _bary_to_cart(bary):
    """Convert barycentric coordinates (a, b, c) to cartesian (x, y)."""
    x = bary[:, 1] * 1.0 + bary[:, 2] * 0.5
    y = bary[:, 2] * S3H
    return np.column_stack([x, y])

def _cart_to_bary(cart):
    """Convert cartesian (x, y) to barycentric coordinates (a, b, c)."""
    x = cart[:, 0]
    y = cart[:, 1]
    c = y / S3H
    b = x - 0.5 * c
    a = 1.0 - b - c
    return np.column_stack([a, b, c])

def _project_bary_to_valid(bary):
    """Project barycentric coordinates to be valid (non-negative, sum to 1)."""
    bary = bary.copy()
    for i in range(N):
        a, b, c = bary[i]
        if a < 0:
            a = 0
            s = b + c
            if s > 0:
                b /= s
                c /= s
            else:
                b = c = 0.5
        if b < 0:
            b = 0
            s = a + c
            if s > 0:
                a /= s
                c /= s
            else:
                a = c = 0.5
        if c < 0:
            c = 0
            s = a + b
            if s > 0:
                a /= s
                b /= s
            else:
                a = b = 0.5
        bary[i] = [a, b, c]
    return bary

def _soft_min_area(points, beta=50.0):
    """Smooth approximation of the minimum area using log-sum-exp."""
    areas = _compute_areas(points)
    min_a = np.min(areas)
    log_sum = np.log(np.sum(np.exp(-beta * (areas - min_a))))
    return min_a - log_sum / beta

def _slp_optimize(points, time_limit=15.0, max_iter=100):
    """
    Successive Linear Programming approach:
    1. Find the minimum area triangle
    2. Solve a linear program to maximize that specific area
    3. Repeat
    """
    start_t = time.time()
    current_points = points.copy()
    best_points = current_points.copy()
    best_min = _min_area(current_points)
    
    for iteration in range(max_iter):
        if time.time() - start_t > time_limit:
            break
        
        # Find the current minimum area triangle
        areas = _compute_areas(current_points)
        min_idx = np.argmin(areas)
        min_area = areas[min_idx]
        
        # Update best
        if min_area > best_min:
            best_min = min_area
            best_points = current_points.copy()
        
        # Get the three points involved
        i, j, k = _triplets[min_idx]
        
        # Set up LP: maximize area of triangle (i, j, k)
        # Area = 0.5 * |(xi - xj)(yi - yk) - (yi - yj)(xi - xk)|
        # For simplicity, assume orientation and maximize the signed area
        # We'll use scipy's linprog
        
        # Variables: x0, y0, x1, y1, ..., x10, y10 (22 variables)
        # Objective: maximize area of triangle (i, j, k)
        # Area = 0.5 * (xi(yj - yk) + xj(yk - yi) + xk(yi - yj))
        # = 0.5 * [yi*(xj - xk) + yj*(xk - xi) + yk*(xi - xj)]
        
        # Let's define the objective coefficients for the signed area
        # c vector of length 22
        c = np.zeros(22)
        
        # Coefficients for xi, yi, xj, yj, xk, yk
        # Area = 0.5 * (xi*yj - xi*yk + xj*yk - xj*yi + xk*yi - xk*yj)
        # = 0.5 * [yi*(xj - xk) + yj*(xk - xi) + yk*(xi - xj)]
        
        # This is bilinear, so LP won't work directly.
        # Instead, let's use a different approach:
        # Fix the other 8 points, and optimize the 3 points of the min triangle
        # to maximize their area subject to constraints.
        
        # Actually, let's just do a local search approach instead.
        # Move the 3 points of the min triangle to increase its area.
        
        # Compute gradients for the 3 points
        # d(area)/d(xi) = 0.5 * (yj - yk)
        # d(area)/d(yi) = 0.5 * (xk - xj)
        # d(area)/d(xj) = 0.5 * (yk - yi)
        # d(area)/d(yj) = 0.5 * (xi - xk)
        # d(area)/d(xk) = 0.5 * (yi - yj)
        # d(area)/d(yk) = 0.5 * (xj - xi)
        
        grad_i = np.array([0.5 * (current_points[j, 1] - current_points[k, 1]),
                           0.5 * (current_points[k, 0] - current_points[j, 0])])
        grad_j = np.array([0.5 * (current_points[k, 1] - current_points[i, 1]),
                           0.5 * (current_points[i, 0] - current_points[k, 0])])
        grad_k = np.array([0.5 * (current_points[i, 1] - current_points[j, 1]),
                           0.5 * (current_points[j, 0] - current_points[i, 0])])
        
        # Move in the direction of the gradient
        step = 0.01
        new_points = current_points.copy()
        new_points[i] += step * grad_i
        new_points[j] += step * grad_j
        new_points[k] += step * grad_k
        
        # Project to triangle
        bary = _cart_to_bary(new_points)
        bary = _project_bary_to_valid(bary)
        new_points = _bary_to_cart(bary)
        
        # Check if this improves the min area
        new_min = _min_area(new_points)
        
        if new_min > min_area + 1e-12:
            current_points = new_points
        else:
            # Try smaller step
            step = 0.005
            new_points = current_points.copy()
            new_points[i] += step * grad_i
            new_points[j] += step * grad_j
            new_points[k] += step * grad_k
            bary = _cart_to_bary(new_points)
            bary = _project_bary_to_valid(bary)
            new_points = _bary_to_cart(bary)
            new_min = _min_area(new_points)
            
            if new_min > min_area + 1e-12:
                current_points = new_points
            else:
                # No improvement, break
                break
    
    return best_points, best_min

def _local_nudge_search(points, max_iter=300, step_sizes=None, time_limit=None):
    """
    Local search that only moves points involved in the smallest triangles.
    """
    if step_sizes is None:
        step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002]
    
    current_points = points.copy()
    current_min = _min_area(current_points)
    best_points = current_points.copy()
    best_min = current_min
    
    if time_limit is None:
        time_limit = 30.0
    start_t = time.time()
    
    for step_size in step_sizes:
        if time.time() - start_t > time_limit:
            break
        
        for _ in range(max_iter):
            if time.time() - start_t > time_limit:
                break
            
            areas = _compute_areas(current_points)
            min_area = float(np.min(areas))
            threshold = min_area + 1e-10
            min_indices = np.where(areas <= threshold)[0]
            involved = set()
            for idx in min_indices:
                involved.add(int(_T_I[idx]))
                involved.add(int(_T_J[idx]))
                involved.add(int(_T_K[idx]))
            
            if min_area > best_min:
                best_min = min_area
                best_points = current_points.copy()
            
            improved = False
            for idx in involved:
                if time.time() - start_t > time_limit:
                    break
                
                for _ in range(15):
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    
                    new_x = current_points[idx, 0] + dx
                    new_y = current_points[idx, 1] + dy
                    
                    test_points = current_points.copy()
                    test_points[idx, 0] = new_x
                    test_points[idx, 1] = new_y
                    bary = _cart_to_bary(test_points)
                    bary = _project_bary_to_valid(bary)
                    test_points = _bary_to_cart(bary)
                    test_min = _min_area(test_points)
                    
                    if test_min > min_area + 1e-12:
                        current_points = test_points
                        improved = True
                        break
                
                if improved:
                    break
            
            if not improved:
                break
    
    return best_points, best_min

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    np.random.seed(42)
    rng = np.random.default_rng(12345)
    
    best_points = None
    best_score = -np.inf
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (deadline - margin)
    
    def _update_best(pts):
        nonlocal best_points, best_score
        area = _min_area(pts)
        if area > best_score:
            best_score = area
            best_points = pts.copy()
    
    # Initial symmetric configuration
    init = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.5, 0.0],
        [0.75, S3H / 2],
        [0.25, S3H / 2],
        [1/3, S3H/3],
        [0.333, 0.0],
        [0.667, 0.0],
        [0.5, S3H/3],
        [0.2, 0.1],
    ], dtype=float)
    bary = _cart_to_bary(init)
    bary = _project_bary_to_valid(bary)
    init = _bary_to_cart(bary)
    _update_best(init)
    
    # Phase A: Multi-start with SLP
    bases = [init]
    for _ in range(10):
        pts = init + rng.normal(0, 0.2, size=(N, 2))
        bary = _cart_to_bary(pts)
        bary = _project_bary_to_valid(bary)
        pts = _bary_to_cart(bary)
        bases.append(pts)
    
    n_starts = 18
    for s in range(n_starts):
        if not _elapsed_ok(30):
            break
        
        base = bases[s % len(bases)]
        pts = base + rng.normal(0, 0.05, size=(N, 2))
        bary = _cart_to_bary(pts)
        bary = _project_bary_to_valid(bary)
        pts = _bary_to_cart(bary)
        _update_best(pts)
        
        # SLP optimization
        if _elapsed_ok(10):
            pts, _ = _slp_optimize(pts, time_limit=min(10.0, deadline - time.time() - 8))
            _update_best(pts)
    
    # Phase B: Local nudge search on the best
    if best_points is not None and _elapsed_ok(15):
        refined_points, refined_score = _local_nudge_search(
            best_points,
            max_iter=300,
            step_sizes=[0.01, 0.005, 0.002, 0.001, 0.0005, 0.0002, 0.0001],
            time_limit=min(12.0, deadline - time.time() - 5)
        )
        _update_best(refined_points)
    
    # Final fine polish
    if best_points is not None and _elapsed_ok(5):
        def nm_obj(x):
            p = x.reshape(-1, 2)
            bary = _cart_to_bary(p)
            bary = _project_bary_to_valid(bary)
            p = _bary_to_cart(bary)
            return -_min_area(p)
        
        try:
            res = minimize(nm_obj, best_points.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-12})
            p = res.x.reshape(-1, 2)
            bary = _cart_to_bary(p)
            bary = _project_bary_to_valid(bary)
            p = _bary_to_cart(bary)
            _update_best(p)
        except Exception:
            pass
    
    if best_points is None:
        best_points = init
    
    bary = _cart_to_bary(best_points)
    bary = _project_bary_to_valid(bary)
    best_points = _bary_to_cart(bary)
    
    return best_points
