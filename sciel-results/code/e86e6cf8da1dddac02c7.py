import numpy as np
from scipy.optimize import minimize, differential_evolution
import time
import itertools

N = 11
H = np.sqrt(3) / 2
SQRT3 = np.sqrt(3)
TRI_AREA = SQRT3 / 4.0

# Precompute triangle area coefficients: area(i,j,k) = 0.5 * |a*x + b*y + c|
# where a, b, c depend on the triangle vertices
# For points p_i, p_j, p_k: area = 0.5 * |x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j)|
# In terms of all 22 coordinates: area = 0.5 * |sum of linear terms|

# Precompute the 165 combinations and their coefficient vectors
_indices = list(itertools.combinations(range(N), 3))
# For each triple (i,j,k), the area = 0.5 * |det([p_i, p_j, p_k])|
# = 0.5 * |(x_i(y_j - y_k) + x_j(y_k - y_i) + x_k(y_i - y_j))|
# This is linear in the 22 point coordinates.
# Coefficient vector for coordinate 2*m (x_m) and 2*m+1 (y_m):
# For triple (i,j,k):
#   coeff_x[i] = y_j - y_k
#   coeff_y[i] = x_k - x_j
#   coeff_x[j] = y_k - y_i
#   coeff_y[j] = x_i - x_k
#   coeff_x[k] = y_i - y_j
#   coeff_y[k] = x_j - x_i
# But this depends on the actual point coordinates, so we can't precompute fixed coefficients.
# Instead, we compute areas directly from points.

def _compute_areas(points):
    """Compute all 165 triangle areas from 11 points."""
    x = points[:, 0]
    y = points[:, 1]
    areas = np.empty(len(_indices))
    for idx, (i, j, k) in enumerate(_indices):
        areas[idx] = 0.5 * abs(x[i]*(y[j]-y[k]) + x[j]*(y[k]-y[i]) + x[k]*(y[i]-y[j]))
    return areas

def _min_area(points):
    """Return the minimum triangle area."""
    return float(np.min(_compute_areas(points)))

def _soft_min_area(points, temperature=0.005):
    """
    Smooth approximation of min area using log-sum-exp.
    soft_min(a) = -T * log(sum(exp(-a_i / T)))
    This is differentiable and approximates min(a) as T -> 0.
    """
    areas = _compute_areas(points)
    # Clip very small areas to avoid numerical issues
    areas = np.maximum(areas, 1e-15)
    scaled = -areas / temperature
    # Numerically stable log-sum-exp
    max_scaled = np.max(scaled)
    log_sum = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -temperature * log_sum

def _objective(params, temperature=0.005):
    """
    Objective to minimize: negative soft-min area.
    params: 22 values in [0,1] representing barycentric coordinates.
    For each point m: (u_m, v_m) with u_m >= 0, v_m >= 0, u_m + v_m <= 1.
    Point = u_m * A0 + v_m * A1 + (1-u_m-v_m) * A2
    where A0=(0,0), A1=(1,0), A2=(0.5, H).
    """
    pts = np.empty((N, 2))
    for m in range(N):
        u = params[2*m]
        v = params[2*m + 1]
        # Barycentric to Cartesian
        pts[m, 0] = v * 1.0 + (1.0 - u - v) * 0.5
        pts[m, 1] = (1.0 - u - v) * H
    
    # Check validity (should always be valid with barycentric coords in [0,1] sum <= 1)
    # But due to numerical issues, clip
    if np.any(pts[:, 0] < -1e-9) or np.any(pts[:, 0] > 1.0 + 1e-9):
        return 1e10
    if np.any(pts[:, 1] < -1e-9) or np.any(pts[:, 1] > H + 1e-9):
        return 1e10
    
    soft_min = _soft_min_area(pts, temperature)
    return -soft_min

def _bary_to_cart(bary):
    """Convert barycentric coords (u,v) array to Cartesian points."""
    pts = np.empty((N, 2))
    for m in range(N):
        u = bary[m, 0]
        v = bary[m, 1]
        pts[m, 0] = v * 1.0 + (1.0 - u - v) * 0.5
        pts[m, 1] = (1.0 - u - v) * H
    return pts

def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v)."""
    bary = np.empty((N, 2))
    for m in range(N):
        x = pts[m, 0]
        y = pts[m, 1]
        # Solve: x = v + 0.5*(1-u-v), y = H*(1-u-v)
        # y = H - H*u - H*v => u + v = 1 - y/H
        # x = v + 0.5 - 0.5*u - 0.5*v = 0.5 + 0.5*v - 0.5*u
        # So: u + v = 1 - y/H
        #     0.5*v - 0.5*u = x - 0.5
        #     v - u = 2*x - 1
        # From these: v = (1 - y/H + 2*x - 1)/2 = (2*x - y/H)/2 = x - y/(2H)
        #             u = (1 - y/H - 2*x + 1)/2 = (2 - y/H - 2*x)/2 = 1 - x - y/(2H)
        # Check: u + v = 1 - x - y/(2H) + x - y/(2H) = 1 - y/H ✓
        bary[m, 0] = 1.0 - x - y / (2.0 * H)
        bary[m, 1] = x - y / (2.0 * H)
    return bary

def _random_valid_barycentric(rng):
    """Generate random valid barycentric coordinates."""
    bary = np.empty((N, 2))
    for m in range(N):
        while True:
            u = rng.random()
            v = rng.random()
            if u + v <= 1.0:
                bary[m, 0] = u
                bary[m, 1] = v
                break
    return bary

def _initial_guess_grid():
    """Generate a structured initial guess based on a grid pattern."""
    # Use a 3x3 grid plus center, mapped into the triangle
    # This gives a reasonable starting configuration
    bary = np.empty((N, 2))
    # Place points in a roughly uniform pattern
    # Row 0 (bottom): 3 points
    # Row 1 (middle): 4 points  
    # Row 2 (top): 3 points
    # Plus 1 center? That's 11.
    
    # Let's use a simpler approach: place points at specific barycentric positions
    # that form a reasonable grid
    positions = [
        (0.1, 0.1), (0.4, 0.1), (0.7, 0.1),  # bottom row
        (0.2, 0.3), (0.5, 0.3), (0.8, 0.3),  # middle row (but 0.8+0.3=1.1 > 1, invalid)
        (0.15, 0.5), (0.45, 0.5), (0.75, 0.5),  # upper middle
        (0.3, 0.7), (0.6, 0.7),  # top
    ]
    # Fix: ensure u + v <= 1
    for m, (u, v) in enumerate(positions):
        if u + v > 1.0:
            # Scale down
            s = 1.0 / (u + v)
            u *= s
            v *= s
        bary[m, 0] = u
        bary[m, 1] = v
    return bary

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Bounds for barycentric coordinates: each in [0, 1]
    bounds = [(0.0, 1.0)] * (2 * N)
    
    best_points = None
    best_score = -np.inf
    
    def _try_optimize(initial_params, temperature, maxiter, method='L-BFGS-B'):
        """Try to optimize from initial params, return (points, score) or (None, -inf)."""
        try:
            result = minimize(
                _objective,
                initial_params,
                args=(temperature,),
                method=method,
                bounds=bounds,
                options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-8}
            )
            if result.success or result.fun < -np.inf + 1:
                pts = _bary_to_cart(result.x.reshape(N, 2))
                # Verify validity
                if np.all(pts[:, 0] >= -1e-8) and np.all(pts[:, 0] <= 1.0 + 1e-8) and \
                   np.all(pts[:, 1] >= -1e-8) and np.all(pts[:, 1] <= H + 1e-8):
                    # Check triangle containment more carefully
                    x = pts[:, 0]
                    y = pts[:, 1]
                    if np.all(y >= -1e-8) and np.all(y <= SQRT3 * x + 1e-8) and \
                       np.all(y <= -SQRT3 * x + SQRT3 + 1e-8):
                        score = _min_area(pts)
                        return pts, score
        except Exception:
            pass
        return None, -np.inf
    
    # Phase 1: Multiple restarts with L-BFGS-B using smooth objective
    # Use different temperatures and initial points
    
    # Start with a good structured guess
    initial_bary = _initial_guess_grid()
    initial_params = initial_bary.flatten()
    
    pts, score = _try_optimize(initial_params, temperature=0.01, maxiter=300)
    if pts is not None and score > best_score:
        best_points = pts
        best_score = score
    
    # Additional restarts from random initial points
    num_restarts = 8
    for i in range(num_restarts):
        if time.time() > deadline - 20:
            break
        
        # Generate random initial barycentric coords
        random_bary = _random_valid_barycentric(rng)
        random_params = random_bary.flatten()
        
        # Use slightly different temperatures
        temp = 0.01 + 0.005 * i
        
        pts, score = _try_optimize(random_params, temperature=temp, maxiter=200)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
    
    # Phase 2: Refine best result with lower temperature for higher precision
    if best_points is not None:
        if time.time() > deadline - 15:
            return best_points
        
        # Convert best points back to barycentric for refinement
        best_bary = _cart_to_bary(best_points)
        # Clip to valid range
        best_bary = np.clip(best_bary, 0.0, 1.0)
        # Ensure u + v <= 1
        for m in range(N):
            if best_bary[m, 0] + best_bary[m, 1] > 1.0:
                s = 1.0 / (best_bary[m, 0] + best_bary[m, 1])
                best_bary[m, 0] *= s
                best_bary[m, 1] *= s
        
        best_params = best_bary.flatten()
        
        # Refine with lower temperature
        pts, score = _try_optimize(best_params, temperature=0.003, maxiter=500)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
        
        # Further refinement with even lower temperature
        if time.time() > deadline - 10:
            return best_points
        
        best_bary2 = _cart_to_bary(best_points)
        best_bary2 = np.clip(best_bary2, 0.0, 1.0)
        for m in range(N):
            if best_bary2[m, 0] + best_bary2[m, 1] > 1.0:
                s = 1.0 / (best_bary2[m, 0] + best_bary2[m, 1])
                best_bary2[m, 0] *= s
                best_bary2[m, 1] *= s
        
        pts, score = _try_optimize(best_bary2.flatten(), temperature=0.001, maxiter=1000)
        if pts is not None and score > best_score:
            best_points = pts
            best_score = score
    
    # Phase 3: If we still have time, try differential evolution for global search
    if best_points is None or time.time() < deadline - 30:
        if time.time() > deadline - 25:
            if best_points is not None:
                return best_points
            # Fallback: return a valid configuration
            bary = _random_valid_barycentric(rng)
            return _bary_to_cart(bary)
        
        def de_objective(params):
            return _objective(params, temperature=0.01)
        
        try:
            de_result = differential_evolution(
                de_objective,
                bounds,
                maxiter=50,
                popsize=10,
                tol=1e-6,
                seed=42,
                mutation=(0.5, 1.5),
                recombination=0.9,
                polish=False,
                init='sobol'
            )
            
            if de_result.fun < -np.inf + 1:
                de_bary = de_result.x.reshape(N, 2)
                de_bary = np.clip(de_bary, 0.0, 1.0)
                for m in range(N):
                    if de_bary[m, 0] + de_bary[m, 1] > 1.0:
                        s = 1.0 / (de_bary[m, 0] + de_bary[m, 1])
                        de_bary[m, 0] *= s
                        de_bary[m, 1] *= s
                
                de_pts = _bary_to_cart(de_bary)
                de_score = _min_area(de_pts)
                
                if de_score > best_score:
                    best_points = de_pts
                    best_score = de_score
                
                # Refine DE result
                if time.time() < deadline - 10:
                    pts, score = _try_optimize(de_bary.flatten(), temperature=0.005, maxiter=300)
                    if pts is not None and score > best_score:
                        best_points = pts
                        best_score = score
        except Exception:
            pass
    
    # Final check
    if best_points is None:
        # Fallback: return a valid configuration
        bary = _random_valid_barycentric(rng)
        return _bary_to_cart(bary)
    
    return best_points
