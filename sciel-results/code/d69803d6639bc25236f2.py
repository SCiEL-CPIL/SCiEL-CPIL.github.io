"""Combined program for Heilbronn-11: Barycentric optimization with SLSQP t-variable and soft-min polish."""
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
DEADLINE = 135.0 

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])
N_TRIP = len(_TRIPLETS)

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
    
    # --- Phase 1: SLSQP with auxiliary variable t ---
    # Maximize t subject to:
    #   area(i,j,k) >= t for all triplets
    #   u_i >= 0, v_i >= 0, u_i + v_i <= 1 for all i
    # We minimize -t.
    
    def _slsqp_objective(x):
        """Objective: minimize -t. x = [u_1, v_1, ..., u_N, v_N, t]"""
        return -x[-1]
    
    def _slsqp_constraints(x):
        """
        Constraints:
        1. For each triplet (i,j,k): area(i,j,k) - t >= 0
        2. For each point i: u_i >= 0, v_i >= 0, u_i + v_i <= 1
        """
        uv_flat = x[:-1]
        t = x[-1]
        pts = _bary_to_cart(uv_flat)
        
        # Compute all areas
        pi = pts[_T_I]
        pj = pts[_T_J]
        pk = pts[_T_K]
        dx1 = pj[:, 0] - pi[:, 0]
        dy1 = pj[:, 1] - pi[:, 1]
        dx2 = pk[:, 0] - pi[:, 0]
        dy2 = pk[:, 1] - pi[:, 1]
        cross = dx1 * dy2 - dy1 * dx2
        areas = np.abs(cross) * 0.5
        
        # Area constraints: area - t >= 0
        area_cons = areas - t
        
        # Barycentric constraints:
        uv = uv_flat.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # u >= 0
        cons_u = u
        # v >= 0
        cons_v = v
        # w >= 0  =>  1 - u - v >= 0
        cons_w = w
        
        # Combine all constraints into a single array for SLSQP
        # SLSQP expects a list of dicts or a single array with 'type'
        # It's easier to return a single array and specify types.
        # However, scipy SLSQP accepts a list of constraints.
        # Let's create the list.
        
        constraints = []
        
        # Area constraints (165 of them)
        # This might be slow to build a list of 165 dicts.
        # Alternative: use a single vector constraint with 'ineq' type.
        # scipy.optimize.minimize with method='SLSQP' accepts:
        # constraints = [{'type': 'ineq', 'fun': lambda x: ...}]
        # If we have many, we can combine them into one vector function.
        
        # Let's combine area constraints into one vector
        # And barycentric constraints into another vector
        # But SLSQP handles vector constraints well.
        
        # Constraint 1: areas - t >= 0 (vector of length 165)
        # Constraint 2: u >= 0 (vector of length 11)
        # Constraint 3: v >= 0 (vector of length 11)
        # Constraint 4: w >= 0 (vector of length 11)
        
        # We can return a single big vector of all ineq constraints.
        all_cons = np.concatenate([area_cons, cons_u, cons_v, cons_w])
        return all_cons

    # Define bounds for SLSQP
    # u_i in [0, 1], v_i in [0, 1]
    # t in [0, TRI_AREA] (max possible area is the triangle area itself, but min area will be much smaller)
    slsqp_bounds = [(0, 1), (0, 1)] * N + [(0, TRI_AREA)]
    
    # Initial guess for SLSQP
    # Start from the best point found so far (or init)
    # t initial guess: current min area
    init_t = _vectorized_min_area_cart(init_cart) if best_pts is None else best_area
    if init_t == 0:
        init_t = 1e-4
    
    x0_slsqp = np.concatenate([init_uv.flatten(), [init_t]])
    
    # Run SLSQP
    # SLSQP can be slow with many constraints. We limit iterations.
    # We will run this in a loop with restarts if needed, but SLSQP is sensitive to start.
    # Let's try one robust run first.
    
    if _elapsed_ok(20):
        try:
            # Define the constraint function properly for SLSQP
            # SLSQP 'ineq' constraint means fun(x) >= 0
            
            # We need to pass the constraint function that returns the vector
            # and tell SLSQP that it's an 'ineq' constraint.
            
            # Note: SLSQP with many constraints can be tricky.
            # Let's use the standard interface.
            
            res_slsqp = minimize(
                _slsqp_objective,
                x0_slsqp,
                method='SLSQP',
                bounds=slsqp_bounds,
                constraints=[{'type': 'ineq', 'fun': _slsqp_constraints}],
                options={'maxiter': 100, 'ftol': 1e-6}
            )
            
            if res_slsqp.success or res_slsqp.fun < 0: # -t is the objective, so lower is better
                x_opt = res_slsqp.x
                uv_opt = x_opt[:-1]
                # Project to ensure validity
                uv_opt = _project_to_simplex(uv_opt)
                pts_opt = _bary_to_cart(uv_opt)
                _update_best(pts_opt)
        except Exception as e:
            pass # Fall back to other methods if SLSQP fails

    # --- Phase 2: L-BFGS-B with soft-min annealing (Fallback/Refinement) ---
    # This is the original robust approach. We use it to polish the result from SLSQP
    # or if SLSQP didn't improve things.
    
    def _objective_bary(uv_flat, beta):
        """
        Objective function for L-BFGS-B in barycentric coordinates.
        Minimizes negative soft-min area.
        """
        pts = _bary_to_cart(uv_flat)
        
        uv = uv_flat.reshape(-1, 2)
        u = uv[:, 0]
        v = uv[:, 1]
        w = 1.0 - u - v
        
        # Penalty for w < 0 (i.e., u+v > 1)
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
                bounds = [(0, 1), (0, 1)] * N
                result = minimize(
                    lambda x: _objective_bary(x, beta),
                    uv_curr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                uv_curr = result.x
                uv_curr = _project_to_simplex(uv_curr)
            except Exception:
                break
        return uv_curr

    # If we have a good best_pts, polish it with L-BFGS-B
    if best_pts is not None and _elapsed_ok(20):
        best_uv = _cart_to_bary(best_pts)
        # Use a high beta schedule for polishing
        beta_schedule = [50.0, 100.0, 200.0, 500.0, 1000.0]
        polished_uv = _optimize_bary(best_uv, beta_schedule)
        polished_pts = _bary_to_cart(polished_uv)
        _update_best(polished_pts)

    # Multi-start L-BFGS-B if time permits and we haven't found a great solution
    # Threshold: if best_area is low, try more starts
    if _elapsed_ok(30):
        # Determine if we need more exploration
        # Heilbronn number for n=11 is approx 1/11^2 * 4.5 ? No, it's ~ 1/121 * 4.5?
        # Actually, the known best is around 0.035-0.04 for area fraction.
        # Raw area is fraction * TRI_AREA.
        # Let's just try a few random starts if we have time.
        
        n_starts = 5
        for s in range(n_starts):
            if not _elapsed_ok(15):
                break
            
            # Random start
            uv_start = np.zeros((N, 2))
            for i in range(N):
                u = rng.random()
                v = rng.random()
                if u + v > 1:
                    u, v = 1 - u, 1 - v
                uv_start[i] = [u, v]
            
            uv_opt = _optimize_bary(uv_start.flatten(), [10.0, 50.0, 200.0])
            pts_opt = _bary_to_cart(uv_opt)
            _update_best(pts_opt)
            
            # Polish the new best
            if _elapsed_ok(10):
                best_uv = _cart_to_bary(best_pts)
                polished_uv = _optimize_bary(best_uv, [200.0, 500.0])
                polished_pts = _bary_to_cart(polished_uv)
                _update_best(polished_pts)

    # --- Phase 3: Final Nelder-Mead polish on hard min ---
    # Nelder-Mead is derivative-free and can handle the non-smooth min directly.
    if best_pts is not None and _elapsed_ok(10):
        def _hard_min_obj(x):
            pts = x.reshape(-1, 2)
            pts = _project_cartesian(pts)
            return -_vectorized_min_area_cart(pts)
        
        try:
            result = minimize(
                _hard_min_obj,
                best_pts.flatten(),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12}
            )
            polished_pts = result.x.reshape(-1, 2)
            polished_pts = _project_cartesian(polished_pts)
            _update_best(polished_pts)
        except Exception:
            pass

    # Final sanity check
    if best_pts is None:
        best_pts = init_cart
        best_area = _vectorized_min_area_cart(best_pts)
    else:
        best_pts = _project_cartesian(best_pts)
        
    return best_pts
# EVOLVE-BLOCK-END
