"""Heilbronn-11: Hybrid Simulated Annealing and L-BFGS-B optimization."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

# Precompute all triplets for N=11 points
_TRIPLETS = list(itertools.combinations(range(N), 3))
_T_I = np.array([t[0] for t in _TRIPLETS])
_T_J = np.array([t[1] for t in _TRIPLETS])
_T_K = np.array([t[2] for t in _TRIPLETS])

DEADLINE = 115.0  # Safety margin from 120s limit

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
        w = y / S3H
        v = x - 0.5 * w
        u = 1.0 - v - w
        uv[i] = [u, v]
    return uv

def _min_area_cart(pts):
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
    Returns a differentiable approximation of the min area.
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
    
    # Handle case where areas are 0 or very small to avoid log(0)
    # We want soft min. 
    # soft_min(x) = -1/beta * log(sum(exp(-beta*x)))
    # To avoid overflow, subtract max
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    # Clip to avoid underflow in exp, though -beta*(a-max) <= 0
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min

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
    
    # Penalty for leaving the triangle (should be 0 if projected)
    viol = np.maximum(0, -w)
    penalty = 1000.0 * np.sum(viol ** 2)
    
    ma = _soft_min_area_cart(pts, beta)
    return -ma + penalty

def _sa_step(uv_curr, temp, rng, beta):
    """
    Perform one step of Simulated Annealing.
    Returns new uv and energy.
    """
    # Perturb: Gaussian noise
    noise_scale = 0.05 * temp  # Scale noise with temperature
    uv_new = uv_curr + rng.normal(0, noise_scale, size=uv_curr.shape)
    uv_new = _project_to_simplex(uv_new)
    
    # Evaluate new energy
    pts_new = _bary_to_cart(uv_new)
    uv_arr = uv_new.reshape(-1, 2)
    w = 1.0 - uv_arr[:, 0] - uv_arr[:, 1]
    viol = np.maximum(0, -w)
    penalty = 1000.0 * np.sum(viol ** 2)
    e_new = -_soft_min_area_cart(pts_new, beta) + penalty
    
    return uv_new, e_new

def construct():
    start_time = time.time()
    rng = np.random.default_rng(42)
    
    best_pts = None
    best_area = 0.0
    best_uv = None
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area, best_uv
        # Check validity
        valid = True
        for i in range(len(pts)):
            x, y = pts[i]
            if y < -1e-6 or y > S3H + 1e-6:
                valid = False
                break
            x_min = y / S3
            x_max = 1 - y / S3
            if x < x_min - 1e-6 or x > x_max + 1e-6:
                valid = False
                break
        if not valid:
            return
        
        area = _min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
            best_uv = _cart_to_bary(pts)
    
    # Initial configuration: High symmetry guess
    # Vertices, midpoints, centroid, etc.
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
    
    init_cart = _project_to_simplex(_cart_to_bary(init_cart).flatten())
    init_uv = _project_to_simplex(_cart_to_bary(init_cart).flatten())
    
    _update_best(_bary_to_cart(init_uv))
    
    # Phase 1: Simulated Annealing
    # We use a relatively high beta for SA to keep the landscape smooth enough for exploration
    # but low enough to distinguish good from bad.
    beta_sa = 50.0 
    t_start = time.time()
    
    # SA Parameters
    T_init = 0.1
    T_final = 0.001
    alpha = 0.95 # Cooling rate
    
    uv_curr = init_uv.copy()
    pts_curr = _bary_to_cart(uv_curr)
    uv_arr = uv_curr.reshape(-1, 2)
    w = 1.0 - uv_arr[:, 0] - uv_arr[:, 1]
    viol = np.maximum(0, -w)
    penalty = 1000.0 * np.sum(viol ** 2)
    E_curr = -_soft_min_area_cart(pts_curr, beta_sa) + penalty
    
    T = T_init
    steps = 0
    
    while _elapsed_ok(20): # Reserve 20s for polishing
        # Perturb
        uv_new, E_new = _sa_step(uv_curr, T, rng, beta_sa)
        
        delta_E = E_new - E_curr
        if delta_E < 0 or rng.random() < np.exp(-delta_E / T):
            uv_curr = uv_new
            E_curr = E_new
            pts_curr = _bary_to_cart(uv_curr)
            
            # Update best if this step is good
            # We check hard min area periodically to update best_pts
            if steps % 10 == 0:
                _update_best(pts_curr)
        
        # Cooling
        T *= alpha
        if T < T_final:
            T = T_final
            
        steps += 1
        
        # If we've been stuck for a while, reheat slightly?
        # For simplicity, we just let it cool.
        
    # Phase 2: Local Polish with L-BFGS-B
    if best_uv is not None and _elapsed_ok(5):
        # Try to polish the best solution found
        # Use a higher beta for polishing to approximate hard min more closely
        beta_polish = 500.0
        
        try:
            result = minimize(
                lambda x: _objective_bary(x, beta_polish),
                best_uv,
                method='L-BFGS-B',
                bounds=[(0, 1), (0, 1)] * N,
                options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
            )
            uv_polished = _project_to_simplex(result.x)
            pts_polished = _bary_to_cart(uv_polished)
            _update_best(pts_polished)
        except Exception:
            pass

    # Phase 3: Multi-start Polish
    # If we have time, try to polish a few random restarts or other good candidates
    # We don't have a list of candidates, so we will just do a few random restarts of SA + Polish
    # But SA is expensive. Instead, let's do a few random perturbations of the best solution and polish them.
    
    if best_uv is not None:
        for _ in range(5):
            if not _elapsed_ok(5):
                break
            
            # Perturb best solution
            uv_pert = best_uv + rng.normal(0, 0.05, size=best_uv.shape)
            uv_pert = _project_to_simplex(uv_pert)
            
            try:
                result = minimize(
                    lambda x: _objective_bary(x, 200.0), # Lower beta for broader basin
                    uv_pert,
                    method='L-BFGS-B',
                    bounds=[(0, 1), (0, 1)] * N,
                    options={'maxiter': 50, 'ftol': 1e-8, 'gtol': 1e-6}
                )
                uv_polished = _project_to_simplex(result.x)
                pts_polished = _bary_to_cart(uv_polished)
                _update_best(pts_polished)
            except Exception:
                pass

    if best_pts is None:
        return _bary_to_cart(init_uv)

    # Final projection to ensure validity
    best_pts = _bary_to_cart(_project_to_simplex(best_uv))
    return best_pts

# EVOLVE-BLOCK-END
