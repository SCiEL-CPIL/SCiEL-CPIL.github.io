import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Safety margin from 150s limit.
DEADLINE = 140.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

# Triangle vertices in Cartesian
V0 = np.array([0.0, 0.0])
V1 = np.array([1.0, 0.0])
V2 = np.array([0.5, S3H])

def _vectorized_min_area_cart(pts):
    """Vectorized computation of min triangle area from Cartesian points."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    return np.min(areas)

def _soft_min_area_cart(pts, beta):
    """
    Soft minimum of triangle areas using log-sum-exp.
    Returns the approximate minimum area.
    """
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Handle potential overflow in exp by ensuring shifted <= 0
    # Since max_a is the max, areas - max_a <= 0, so -beta*(...) >= 0? No.
    # areas <= max_a => areas - max_a <= 0 => -beta*(areas - max_a) >= 0.
    # Wait, if areas is small, -beta*areas is large positive.
    # Let's re-derive.
    # min(x) approx -1/beta log(sum(exp(-beta x)))
    # To avoid overflow in exp(-beta x), we shift by max(x).
    # exp(-beta x) = exp(-beta (x - max_x) - beta max_x) = exp(-beta(x-max_x)) * exp(-beta max_x)
    # sum(exp(-beta x)) = exp(-beta max_x) * sum(exp(-beta(x-max_x)))
    # log(sum(exp(-beta x))) = -beta max_x + log(sum(exp(-beta(x-max_x))))
    # min approx -1/beta (-beta max_x + log(sum(exp(-beta(x-max_x)))))
    # min approx max_x - 1/beta log(sum(exp(-beta(x-max_x))))
    
    # Here x = areas.
    # x - max_a <= 0.
    # -beta * (x - max_a) >= 0.
    # So exp(-beta * (x - max_a)) can be large if x is much smaller than max_a?
    # No, if x is small, x-max_a is large negative, -beta*(x-max_a) is large positive.
    # This will overflow.
    
    # Correct shift for MINIMUM:
    # We want to emphasize the SMALL values.
    # Standard log-sum-exp for MAX: max(x) approx 1/beta log(sum(exp(beta x)))
    # Shift by max(x): exp(beta x) = exp(beta(x-max)) * exp(beta max)
    # sum = exp(beta max) sum(exp(beta(x-max)))
    # log(sum) = beta max + log(sum(exp(beta(x-max))))
    # max approx max + 1/beta log(sum(exp(beta(x-max))))
    # Note: x-max <= 0, so beta(x-max) <= 0, exp is <= 1. Safe.
    
    # For MINIMUM:
    # min(x) approx -1/beta log(sum(exp(-beta x)))
    # Shift by MIN(x)? No, shift by MAX(x) is bad for min because exp(-beta x) is huge for small x.
    # We should shift by MIN(x) to make exponents negative?
    # Let m = min(x). x - m >= 0.
    # exp(-beta x) = exp(-beta(x-m) - beta m) = exp(-beta(x-m)) * exp(-beta m)
    # sum = exp(-beta m) sum(exp(-beta(x-m)))
    # log(sum) = -beta m + log(sum(exp(-beta(x-m))))
    # min approx -1/beta (-beta m + log(sum(exp(-beta(x-m)))))
    # min approx m - 1/beta log(sum(exp(-beta(x-m))))
    
    # Here x-m >= 0, so -beta(x-m) <= 0. exp is <= 1. Safe.
    
    min_a = np.min(areas)
    shifted = -beta * (areas - min_a)
    # shifted <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = min_a - log_sum / beta
    return soft_min

def _project_cartesian(pts):
    """Project points into the triangle if they are slightly outside."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts

def _is_valid_cart(pts, tol=1e-6):
    """Check if all points are inside the triangle."""
    for i in range(len(pts)):
        x, y = pts[i]
        if y < -tol or y > S3H + tol:
            return False
        x_min = y / S3
        x_max = 1 - y / S3
        if x < x_min - tol or x > x_max + tol:
            return False
    return True

def _constraints_cart(pts_flat):
    """
    Define constraints for SLSQP.
    Points are in Cartesian (x, y).
    Triangle: (0,0), (1,0), (0.5, S3H).
    Inequalities:
    1. y >= 0
    2. y <= S3H
    3. x >= y/S3  (Left edge)
    4. x <= 1 - y/S3 (Right edge)
    """
    pts = pts_flat.reshape(-1, 2)
    cons = []
    # y >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=pts: x[i, 1]})
    # y <= S3H
    cons.append({'type': 'ineq', 'fun': lambda x, i=pts: S3H - x[i, 1]})
    # x - y/S3 >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=pts: x[i, 0] - x[i, 1]/S3})
    # 1 - y/S3 - x >= 0
    cons.append({'type': 'ineq', 'fun': lambda x, i=pts: 1 - x[i, 1]/S3 - x[i, 0]})
    return cons

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # Initial configuration: High symmetry guess
    # Corners, mid-edges, centroid, and points on medians
    init_cart = np.array([
        [0.0, 0.0],          # Corner 0
        [1.0, 0.0],          # Corner 1
        [0.5, S3H],          # Corner 2
        [0.5, 0.0],          # Mid bottom
        [0.75, S3H / 2],     # Mid right
        [0.25, S3H / 2],     # Mid left
        [1/3, S3H/3],        # Centroid
        [0.333, 0.0],        # Bottom edge
        [0.667, 0.0],        # Bottom edge
        [0.5, S3H/3],        # Median
        [0.2, 0.1],          # Interior
    ], dtype=float)
    
    # Ensure initial points are valid
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    # Define the objective function for SLSQP
    def _objective_cart(pts_flat, beta):
        pts = pts_flat.reshape(-1, 2)
        # Soft min area
        ma = _soft_min_area_cart(pts, beta)
        return -ma

    # Multi-start optimization
    # We will use a set of starting points.
    # 1. The symmetric initial guess.
    # 2. Random points.
    # 3. Points derived from known Heilbronn configurations (if we had them, but we don't).
    
    starts = [init_cart.flatten()]
    
    # Generate random starts
    # We need to ensure random points are inside the triangle.
    # Use barycentric coordinates for random generation.
    for _ in range(10):
        u = rng.random(N)
        v = rng.random(N)
        # Normalize to sum to 1
        s = u + v
        u /= s
        v /= s
        w = 1 - u - v
        # P = u*V0 + v*V1 + w*V2
        x = v * 1.0 + w * 0.5
        y = w * S3H
        rand_pts = np.column_stack((x, y))
        starts.append(rand_pts.flatten())

    # Beta schedule: Annealing from smooth to sharp
    # Low beta -> smooth landscape, easier to find basin.
    # High beta -> sharp landscape, precise minimum.
    beta_schedule = [5.0, 20.0, 100.0, 500.0]
    
    # To save time, we will not run the full beta schedule for every start.
    # We will do a "coarse" pass with low beta for all starts,
    # then a "fine" pass with high beta for the top few candidates.
    
    candidates = []
    
    # Phase 1: Coarse optimization with low beta
    beta_coarse = 10.0
    for i, start_flat in enumerate(starts):
        if not _elapsed_ok(30):
            break
            
        # Project start to valid
        start_pts = _project_cartesian(start_flat.reshape(-1, 2))
        start_flat = start_pts.flatten()
        
        try:
            result = minimize(
                lambda x: _objective_cart(x, beta_coarse),
                start_flat,
                method='SLSQP',
                constraints=_constraints_cart(start_flat),
                options={'maxiter': 100, 'ftol': 1e-8}
            )
            opt_pts = _project_cartesian(result.x.reshape(-1, 2))
            area = _vectorized_min_area_cart(opt_pts)
            candidates.append((area, opt_pts))
        except Exception:
            continue

    # Sort candidates by area
    candidates.sort(key=lambda x: x[0], reverse=True)
    
    # Phase 2: Fine optimization with high beta for top candidates
    # Take top 5 candidates
    top_candidates = candidates[:5]
    
    for area, pts in top_candidates:
        if not _elapsed_ok(20):
            break
            
        # Use the result from coarse as start for fine
        start_flat = pts.flatten()
        
        # Run annealing
        current_pts = pts
        for beta in [50.0, 200.0, 1000.0]:
            if not _elapsed_ok(10):
                break
            try:
                result = minimize(
                    lambda x, b=beta: _objective_cart(x, b),
                    current_pts.flatten(),
                    method='SLSQP',
                    constraints=_constraints_cart(current_pts),
                    options={'maxiter': 200, 'ftol': 1e-10}
                )
                new_pts = _project_cartesian(result.x.reshape(-1, 2))
                # Check if it's an improvement
                new_area = _vectorized_min_area_cart(new_pts)
                if new_area > _vectorized_min_area_cart(current_pts):
                    current_pts = new_pts
            except Exception:
                break
        
        _update_best(current_pts)

    # Phase 3: Local Polish using Active Set
    # The minimum area is determined by a small number of triangles.
    # We can try to maximize this minimum area by moving the points involved.
    
    if best_pts is not None and _elapsed_ok(30):
        current_pts = best_pts.copy()
        current_area = best_area
        
        # Identify the triangles that define the minimum
        pi = current_pts[_T_I]
        pj = current_pts[_T_J]
        pk = current_pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        # Find indices where area is close to min
        min_a = np.min(areas)
        tol = 1e-4
        active_idx = np.where(areas < min_a + tol)[0]
        
        # Get unique point indices involved in these triangles
        active_points = set()
        for idx in active_idx:
            active_points.add(_T_I[idx])
            active_points.add(_T_J[idx])
            active_points.add(_T_K[idx])
        
        active_points = list(active_points)
        
        # If we have a small set of active points, we can try to optimize them specifically
        # using a higher beta and more iterations.
        if len(active_points) < N:
             # Create a sub-problem? SLSQP doesn't support sub-problems easily.
             # Instead, we just run a very high beta optimization on the full set,
             # but starting from the current best.
             try:
                 result = minimize(
                     lambda x: _objective_cart(x, 5000.0),
                     current_pts.flatten(),
                     method='SLSQP',
                     constraints=_constraints_cart(current_pts),
                     options={'maxiter': 500, 'ftol': 1e-12}
                 )
                 new_pts = _project_cartesian(result.x.reshape(-1, 2))
                 new_area = _vectorized_min_area_cart(new_pts)
                 if new_area > current_area:
                     _update_best(new_pts)
             except Exception:
                 pass

    # Final check
    if best_pts is None:
        return init_cart
        
    return best_pts

# EVOLVE-BLOCK-END
