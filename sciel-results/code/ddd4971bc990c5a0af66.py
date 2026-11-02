"""Combined program for Heilbronn-11: Barycentric optimization with soft-min annealing."""
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
# The evaluator kills after 150s. We aim to return well before.
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


def _bary_to_cart(uv_flat):
    """
    Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2).
    w = 1 - u - v.
    P = u*V0 + v*V1 + w*V2.
    Since V0=(0,0), P = v*(1,0) + w*(0.5, S3H) = (v + 0.5w, w*S3H).
    """
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """
    Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v.
    Points must be inside the triangle.
    """
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        # w = y / S3H
        w = y / S3H
        # x = v + 0.5 * w  => v = x - 0.5 * w
        v = x - 0.5 * w
        # u = 1 - v - w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv


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
    Soft minimum of triangle areas using log-sum-exp for numerical stability.
    This provides a smooth, differentiable approximation to the hard minimum.
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
    # To avoid underflow/overflow, shift by max(areas)
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # np.exp(shifted) will be <= 1, so sum is safe
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
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
    # This helps the optimizer start in a good basin
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
    
    # Convert initial to barycentric for optimization
    init_uv = _cart_to_bary(init_cart)
    
    def _objective_bary(uv_flat, beta):
        """
        Objective function for L-BFGS-B in barycentric coordinates.
        Minimizes negative soft-min area.
        """
        pts = _bary_to_cart(uv_flat)
        # Note: L-BFGS-B with bounds (0,1) for u and v ensures w>=0 implicitly?
        # No, bounds are on u and v independently. We need u>=0, v>=0, u+v<=1.
        # Standard L-BFGS-B handles box constraints. The simplex constraint u+v<=1 is not a box.
        # However, if we optimize over (u, v) with bounds [0,1]x[0,1], we might get u+v > 1.
        # To handle this robustly, we can use a penalty or transform.
        # Alternatively, we can just use bounds [0,1] and add a penalty for u+v > 1.
        # Or, better, we can optimize over (u, v) where we implicitly assume w=1-u-v >= 0.
        # Let's add a smooth penalty for u+v > 1 to keep points inside.
        
        uv = uv_flat.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Penalty for w < 0 (i.e., u+v > 1)
        # Use a quadratic penalty for violation
        viol = np.maximum(0, -w)
        penalty = 1000.0 * np.sum(viol ** 2)
        
        ma = _soft_min_area_cart(pts, beta)
        return -ma + penalty

    def _optimize_bary(uv_start, beta_schedule):
        """
        Run L-BFGS-B with annealing beta.
        """
        uv_curr = uv_start.copy()
        for beta in beta_schedule:
            if not _elapsed_ok(10):
                break
            try:
                # Bounds: u in [0, 1], v in [0, 1]. 
                # The constraint u+v<=1 is handled by penalty.
                bounds = [(0, 1), (0, 1)] * N
                result = minimize(
                    lambda x: _objective_bary(x, beta),
                    uv_curr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                uv_curr = result.x
                # Project to simplex to ensure validity for next step
                uv_curr = _project_to_simplex(uv_curr)
            except Exception:
                break
        return uv_curr

    def _project_to_simplex(uv_flat):
        """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
        uv = uv_flat.reshape(-1, 2).copy()
        for i in range(len(uv)):
            u, v = uv[i]
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))
            if u + v > 1.0:
                scale = 1.0 / (u + v)
                u *= scale
                v *= scale
            uv[i] = [u, v]
        return uv.flatten()

    # Phase 1: Multi-start optimization in barycentric coordinates
    # We use a set of beta values to anneal from smooth to sharp
    beta_schedule = [10.0, 30.0, 80.0, 200.0, 500.0]
    
    # Generate starting points: perturb the initial symmetric config
    n_starts = 40
    for s in range(n_starts):
        if not _elapsed_ok(20):
            break
        
        # Create a perturbed start
        if s == 0:
            uv_start = init_uv.copy()
        else:
            # Random perturbation in barycentric space
            uv_start = init_uv + rng.normal(0, 0.1, size=(N, 2))
            uv_start = _project_to_simplex(uv_start)
        
        # Optimize
        uv_opt = _optimize_bary(uv_start, beta_schedule)
        
        # Convert to Cartesian and check
        pts_opt = _bary_to_cart(uv_opt)
        pts_opt = _project_cartesian(pts_opt)
        _update_best(pts_opt)
        
        # If we found a significantly better solution, we might want to restart from it
        # But the loop continues with independent starts to explore global space.

    # Phase 2: Local polish on the best configuration found
    # The soft-min with high beta is very close to the hard min.
    # However, to ensure maximum precision, we can do a final Nelder-Mead step
    # on the hard min area objective in Cartesian coordinates.
    # Nelder-Mead is derivative-free and handles non-smooth functions well.
    
    if best_pts is not None and _elapsed_ok(10):
        def _hard_min_obj(x):
            pts = x.reshape(-1, 2)
            # Project to ensure we stay in triangle during optimization steps
            pts = _project_cartesian(pts)
            return -_vectorized_min_area_cart(pts)
        
        try:
            result = minimize(
                _hard_min_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished_pts = result.x.reshape(-1, 2)
            polished_pts = _project_cartesian(polished_pts)
            _update_best(polished_pts)
        except Exception:
            pass

    # Final sanity check
    if best_pts is None:
        # Fallback to initial
        best_pts = init_cart
        best_area = _vectorized_min_area_cart(best_pts)
    else:
        # Ensure the returned points are strictly inside
        best_pts = _project_cartesian(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
