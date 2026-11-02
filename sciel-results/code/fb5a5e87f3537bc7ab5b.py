"""Combined program for Heilbronn-11: Structured Multi-Start + Dual-Representation Annealing."""
N = 11

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize
import itertools

S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # area of the equilateral triangle with side 1

DEADLINE = 140  # safety margin from 150s limit

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
    """Convert flat array of barycentric coords (u, v) for N points to Cartesian (N, 2)."""
    uv = uv_flat.reshape(-1, 2)
    u = uv[:, 0]
    v = uv[:, 1]
    w = 1.0 - u - v
    x = v * 1.0 + w * 0.5
    y = w * S3H
    return np.column_stack((x, y))


def _cart_to_bary(pts):
    """Convert Cartesian points to barycentric coords (u, v) where w = 1-u-v."""
    uv = np.zeros((len(pts), 2))
    for i in range(len(pts)):
        x, y = pts[i]
        w = y / S3H if S3H != 0 else 0
        v = x - 0.5 * w
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
    """Soft minimum of triangle areas using log-sum-exp."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    dx1 = pj[:, 0] - pi[:, 0]
    dy1 = pj[:, 1] - pi[:, 1]
    dx2 = pk[:, 0] - pi[:, 0]
    dy2 = pk[:, 1] - pi[:, 1]
    cross = dx1 * dy2 - dy1 * dx2
    areas = np.abs(cross) * 0.5
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
    log_sum = np.log(np.sum(np.exp(shifted)))
    soft_min = max_a - log_sum / beta
    return soft_min


def _project_cartesian(pts):
    """Project points into the triangle."""
    pts = pts.copy()
    for i in range(len(pts)):
        x, y = pts[i]
        y = max(0.0, min(S3H, y))
        x_min = y / S3
        x_max = 1 - y / S3
        x = max(x_min, min(x_max, x))
        pts[i] = [x, y]
    return pts


def _project_to_simplex(uv_flat):
    """Project barycentric coords to the simplex u>=0, v>=0, u+v<=1."""
    uv = uv_flat.reshape(-1, 2).copy()
    for i in range(len(uv)):
        u, v = uv[i]
        u = np.clip(u, 0, 1)
        v = np.clip(v, 0, 1)
        if u + v > 1:
            scale = 1.0 / (u + v)
            u *= scale
            v *= scale
        uv[i] = [u, v]
    return uv.flatten()


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
    
    def _elapsed_ok(margin):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _vectorized_min_area_cart(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()

    # --- Phase 0: Generate Diverse Initial Configurations ---
    
    # 1. High Symmetry "Hexagonal" Packing Approximation
    # Try to place points in a grid that fits the triangle
    init_hex = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.25, S3H/4],
        [0.75, S3H/4],
        [0.5, S3H/2],
        [0.125, S3H/2],
        [0.875, S3H/2],
        [0.375, 3*S3H/4],
        [0.625, 3*S3H/4],
        [0.5, S3H*0.75],
    ], dtype=float)
    
    # 2. Grid-based
    init_grid = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.2, 0.0],
        [0.4, 0.0],
        [0.6, 0.0],
        [0.8, 0.0],
        [0.3, S3H/3],
        [0.5, S3H/3],
        [0.7, S3H/3],
        [0.5, 2*S3H/3],
    ], dtype=float)
    
    # 3. Random perturbation of centroid
    init_centroid = np.array([
        [0.0, 0.0],
        [1.0, 0.0],
        [0.5, S3H],
        [0.3, 0.1],
        [0.7, 0.1],
        [0.5, 0.2],
        [0.2, 0.3],
        [0.8, 0.3],
        [0.4, 0.4],
        [0.6, 0.4],
        [0.5, 0.5],
    ], dtype=float)
    
    bases = [init_hex, init_grid, init_centroid]
    # Add some random starts
    for _ in range(5):
        pts = np.random.rand(N, 2)
        # Map to triangle
        u = np.sqrt(pts[:, 0])
        v = pts[:, 1] * (1 - u)
        x = v + 0.5 * (1 - u - v)
        y = (1 - u - v) * S3H
        bases.append(np.column_stack((x, y)))
        
    # Ensure all bases are valid
    valid_bases = []
    for b in bases:
        b = _project_cartesian(b)
        if _is_valid_cart(b):
            valid_bases.append(b)
    
    if not valid_bases:
         # Fallback
         pts = np.array([[0,0],[1,0],[0.5,S3H]] + [np.random.rand(2) for _ in range(8)])
         return _project_cartesian(pts)

    # Initialize best with the best base
    for b in valid_bases:
        _update_best(b)
        
    # --- Phase 1: Multi-Start Optimization with Annealing ---
    
    # Beta schedule: Smooth to Sharp
    beta_schedule = [10.0, 50.0, 200.0, 1000.0]
    
    # We will perform a fixed number of "rounds" of optimization on each base
    # but dynamically adjust how many based on time.
    
    def _optimize_bary(uv_start, beta):
        """Optimize in barycentric coordinates."""
        def objective(x):
            pts = _bary_to_cart(x)
            uv = x.reshape(-1, 2)
            u = uv[:, 0]
            v = uv[:, 1]
            w = 1.0 - u - v
            viol_w = np.maximum(0, -w)
            penalty = 1000.0 * np.sum(viol_w ** 2)
            ma = _soft_min_area_cart(pts, beta)
            return -ma + penalty
        
        bounds = [(0, 1), (0, 1)] * N
        try:
            result = minimize(
                objective,
                uv_start,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            return result.x
        except Exception:
            return uv_start

    def _optimize_cart(pts_start, beta):
        """Optimize in Cartesian coordinates."""
        def objective(x):
            pts = x.reshape(-1, 2)
            pts = _project_cartesian(pts)
            ma = _soft_min_area_cart(pts, beta)
            return -ma
        
        try:
            result = minimize(
                objective,
                pts_start.flatten(),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            pts = result.x.reshape(-1, 2)
            pts = _project_cartesian(pts)
            return pts
        except Exception:
            return pts_start

    # Iterate over bases. We don't know how much time each takes, so we check time inside.
    # We will do a "full annealing pass" for each base if time permits.
    
    for base_idx, base in enumerate(valid_bases):
        if not _elapsed_ok(30):
            break
            
        # Start from this base
        pts_curr = base.copy()
        _update_best(pts_curr)
        
        # Alternate between Barycentric and Cartesian refinement
        # We do this for each beta in the schedule
        for beta in beta_schedule:
            if not _elapsed_ok(20):
                break
                
            # 1. Barycentric step
            uv_curr = _cart_to_bary(pts_curr)
            uv_opt = _optimize_bary(uv_curr, beta)
            uv_opt = _project_to_simplex(uv_opt)
            pts_curr = _bary_to_cart(uv_opt)
            _update_best(pts_curr)
            
            # 2. Cartesian step
            pts_opt = _optimize_cart(pts_curr, beta)
            _update_best(pts_opt)
            
            # Keep the better of the two for the next beta step? 
            # Or just use the Cartesian one as it's easier to perturb?
            # Let's use the one with better area.
            area_b = _vectorized_min_area_cart(pts_curr)
            area_c = _vectorized_min_area_cart(pts_opt)
            if area_c > area_b:
                pts_curr = pts_opt
            else:
                pts_curr = pts_curr # keep barycentric result if it was better/equal
                
    # --- Phase 2: Nelder-Mead Polish on the Best Found ---
    if best_pts is not None and _elapsed_ok(10):
        # Try to polish the global best with NM
        def nm_obj(x):
            p = x.reshape(-1, 2)
            p = _project_cartesian(p)
            return -_vectorized_min_area_cart(p)
        
        try:
            # Use a slightly perturbed version to avoid getting stuck in a local flat spot
            start_nm = best_pts + rng.normal(0, 0.01, size=(N, 2))
            start_nm = _project_cartesian(start_nm)
            
            res = minimize(nm_obj, start_nm.flatten(), method='Nelder-Mead',
                           options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            p = res.x.reshape(-1, 2)
            p = _project_cartesian(p)
            _update_best(p)
            
            # Also try polishing the exact best point
            res2 = minimize(nm_obj, best_pts.flatten(), method='Nelder-Mead',
                            options={'maxiter': 5000, 'xatol': 1e-9, 'fatol': 1e-11})
            p2 = res2.x.reshape(-1, 2)
            p2 = _project_cartesian(p2)
            _update_best(p2)
        except Exception:
            pass

    # Final fallback
    if best_pts is None:
        pts = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u = 1 - u
                v = 1 - v
            x = v + 0.5 * (1 - u - v)
            y = (1 - u - v) * S3H
            pts[i] = [x, y]
        return pts
    
    return _project_cartesian(best_pts)

# EVOLVE-BLOCK-END
