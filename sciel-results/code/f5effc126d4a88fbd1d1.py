"""Combined program for Heilbronn-11: SA global search + barycentric L-BFGS-B polish."""
import numpy as np
import time
from scipy.optimize import minimize
import itertools

N = 11

# EVOLVE-BLOCK-START
S3 = np.sqrt(3)
S3H = S3 / 2
TRI_AREA = S3 / 4  # Area of the equilateral triangle with side 1

DEADLINE = 145.0

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
        w = y / S3H
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
    """
    Soft minimum of triangle areas using log-sum-exp for numerical stability.
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
    
    max_a = np.max(areas)
    shifted = -beta * (areas - max_a)
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


def _min_area_fast(pts):
    """Fast min triangle area computation."""
    pi = pts[_T_I]
    pj = pts[_T_J]
    pk = pts[_T_K]
    cross = (pj[:, 0] - pi[:, 0]) * (pk[:, 1] - pi[:, 1]) - \
            (pj[:, 1] - pi[:, 1]) * (pk[:, 0] - pi[:, 0])
    return np.min(np.abs(cross)) * 0.5


def construct():
    start_time = time.time()
    rng = np.random.default_rng(12345)
    
    best_pts = None
    best_area = 0.0
    
    def _elapsed_ok(margin=0):
        return (time.time() - start_time) < (DEADLINE - margin)
    
    def _update_best(pts):
        nonlocal best_pts, best_area
        if not _is_valid_cart(pts):
            return
        area = _min_area_fast(pts)
        if area > best_area:
            best_area = area
            best_pts = pts.copy()
    
    # Better initial configuration: use known good structure for Heilbronn-11
    # Corners + edge midpoints + interior points arranged symmetrically
    init_cart = np.array([
        [0.0, 0.0],           # Corner 0
        [1.0, 0.0],           # Corner 1
        [0.5, S3H],           # Corner 2
        [0.5, 0.0],           # Mid bottom edge
        [0.75, S3H / 2],      # Mid right edge
        [0.25, S3H / 2],      # Mid left edge
        [1/3, S3H/3],         # Centroid
        [0.333, 0.0],         # Bottom edge 1/3
        [0.667, 0.0],         # Bottom edge 2/3
        [0.5, S3H/3],         # On median
        [0.5, 2*S3H/3],       # On median upper
    ], dtype=float)
    
    init_cart = _project_cartesian(init_cart)
    _update_best(init_cart)
    
    init_uv = _cart_to_bary(init_cart)
    
    def _objective_bary(uv_flat, beta):
        """
        Objective function for L-BFGS-B in barycentric coordinates.
        Minimizes negative soft-min area.
        """
        pts = _bary_to_cart(uv_flat)
        
        uv = uv_flat.reshape(-1, 2)
        w = 1.0 - uv[:, 0] - uv[:, 1]
        viol = np.maximum(0, -w)
        penalty = 500.0 * np.sum(viol ** 2)
        
        ma = _soft_min_area_cart(pts, beta)
        return -ma + penalty

    def _optimize_bary(uv_start, beta_schedule, maxiter=150):
        """
        Run L-BFGS-B with annealing beta.
        """
        uv_curr = uv_start.copy()
        bounds = [(0, 1), (0, 1)] * N
        for beta in beta_schedule:
            if not _elapsed_ok(5):
                break
            try:
                result = minimize(
                    lambda x, b=beta: _objective_bary(x, b),
                    uv_curr,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': maxiter, 'ftol': 1e-12, 'gtol': 1e-10}
                )
                uv_curr = result.x
                uv_curr = _project_to_simplex(uv_curr)
            except Exception:
                break
        return uv_curr

    # Phase 1: Simulated Annealing global search in Cartesian space
    # This explores the landscape broadly before local polish
    sa_start = time.time()
    sa_budget = 60.0  # seconds for SA
    
    # Start from a random-ish configuration
    sa_pts = np.zeros((N, 2))
    for i in range(N):
        u = rng.random()
        v = rng.random()
        if u + v > 1:
            u, v = 1 - u, 1 - v
        w = 1 - u - v
        sa_pts[i] = [v + 0.5*w, w*S3H]
    sa_pts = _project_cartesian(sa_pts)
    sa_curr_area = _min_area_fast(sa_pts)
    sa_best_pts = sa_pts.copy()
    sa_best_area = sa_curr_area
    
    T0 = 0.05  # initial temperature (area scale)
    T_end = 0.001
    cool_rate = (T_end / T0) ** (1.0 / 5000)
    
    for step in range(5000):
        if time.time() - sa_start > sa_budget:
            break
        if not _elapsed_ok(15):
            break
        
        T = T0 * (cool_rate ** step)
        
        # Random perturbation: move one or two points
        n_move = rng.integers(1, 3)
        idx = rng.choice(N, size=n_move, replace=False)
        for i in idx:
            dx = rng.normal(0, 0.05)
            dy = rng.normal(0, 0.05)
            new_pt = sa_pts[i].copy()
            new_pt[0] += dx
            new_pt[1] += dy
            # Project into triangle
            new_pt[1] = max(0.0, min(S3H, new_pt[1]))
            x_min = new_pt[1] / S3
            x_max = 1 - new_pt[1] / S3
            new_pt[0] = max(x_min, min(x_max, new_pt[0]))
            
            old_pt = sa_pts[i].copy()
            sa_pts[i] = new_pt
            new_area = _min_area_fast(sa_pts)
            
            delta = new_area - sa_curr_area
            if delta > 0 or rng.random() < np.exp(delta / T):
                sa_curr_area = new_area
                if new_area > sa_best_area:
                    sa_best_area = new_area
                    sa_best_pts = sa_pts.copy()
            else:
                sa_pts[i] = old_pt
        
        # Update global best
        if sa_best_area > best_area:
            best_area = sa_best_area
            best_pts = sa_best_pts.copy()
    
    # Phase 2: Local polish of SA result using L-BFGS-B with soft-min annealing
    if _elapsed_ok(20):
        sa_uv = _cart_to_bary(sa_best_pts)
        polished_uv = _optimize_bary(sa_uv.flatten(), [50.0, 150.0, 400.0, 800.0], maxiter=200)
        polished_cart = _bary_to_cart(polished_uv)
        _update_best(polished_cart)
    
    # Phase 3: Multi-start L-BFGS-B from diverse initial points
    beta_schedule = [20.0, 60.0, 150.0, 400.0, 800.0]
    
    # Use SA best + random starts
    starts = []
    if best_pts is not None:
        starts.append(_cart_to_bary(best_pts).flatten())
    
    # Add the initial symmetric config
    starts.append(init_uv.flatten())
    
    # Add random starts
    for s in range(8):
        if not _elapsed_ok(10):
            break
        uv_rand = np.zeros((N, 2))
        for i in range(N):
            u = rng.random()
            v = rng.random()
            if u + v > 1:
                u, v = 1 - u, 1 - v
            uv_rand[i] = [u, v]
        starts.append(uv_rand.flatten())
    
    for uv_start in starts:
        if not _elapsed_ok(8):
            break
        uv_opt = _optimize_bary(uv_start, beta_schedule, maxiter=100)
        cart_opt = _bary_to_cart(uv_opt)
        _update_best(cart_opt)
    
    # Phase 4: Final intensive polish of the best solution
    if best_pts is not None and _elapsed_ok(15):
        best_uv = _cart_to_bary(best_pts)
        
        # High-beta polish
        polished_uv = _optimize_bary(best_uv.flatten(), [300.0, 600.0, 1200.0, 2000.0], maxiter=200)
        polished_cart = _bary_to_cart(polished_uv)
        _update_best(polished_cart)
        
        # Try small perturbations around best
        for s in range(10):
            if not _elapsed_ok(3):
                break
            perturbed_uv = best_uv + rng.normal(0, 0.01, size=(N, 2))
            perturbed_uv = _project_to_simplex(perturbed_uv.flatten())
            perturbed_uv = _optimize_bary(perturbed_uv, [500.0, 1000.0, 2000.0], maxiter=100)
            perturbed_cart = _bary_to_cart(perturbed_uv)
            _update_best(perturbed_cart)
    
    # Phase 5: Use remaining time for one more SA round if time permits
    remaining = DEADLINE - (time.time() - start_time) - 5
    if remaining > 20 and best_pts is not None:
        # Brief SA refinement around current best
        sa2_pts = best_pts.copy()
        sa2_area = best_area
        T = 0.02
        cool_rate2 = (0.001 / T) ** (1.0 / 3000)
        
        for step in range(3000):
            if not _elapsed_ok(5):
                break
            T *= cool_rate2
            
            n_move = rng.integers(1, 3)
            idx = rng.choice(N, size=n_move, replace=False)
            for i in idx:
                dx = rng.normal(0, 0.03)
                dy = rng.normal(0, 0.03)
                new_pt = sa2_pts[i].copy()
                new_pt[0] += dx
                new_pt[1] += dy
                new_pt[1] = max(0.0, min(S3H, new_pt[1]))
                x_min = new_pt[1] / S3
                x_max = 1 - new_pt[1] / S3
                new_pt[0] = max(x_min, min(x_max, new_pt[0]))
                
                old_pt = sa2_pts[i].copy()
                sa2_pts[i] = new_pt
                new_area = _min_area_fast(sa2_pts)
                
                delta = new_area - sa2_area
                if delta > 0 or rng.random() < np.exp(delta / T):
                    sa2_area = new_area
                    if new_area > best_area:
                        best_area = new_area
                        best_pts = sa2_pts.copy()
                else:
                    sa2_pts[i] = old_pt
        
        # Final polish
        if _elapsed_ok(5):
            best_uv = _cart_to_bary(best_pts)
            polished_uv = _optimize_bary(best_uv.flatten(), [800.0, 1500.0, 3000.0], maxiter=200)
            polished_cart = _bary_to_cart(polished_uv)
            _update_best(polished_cart)

    if best_pts is None:
        return _bary_to_cart(init_uv)

    best_pts = _project_cartesian(best_pts)
    return best_pts
# EVOLVE-BLOCK-END
