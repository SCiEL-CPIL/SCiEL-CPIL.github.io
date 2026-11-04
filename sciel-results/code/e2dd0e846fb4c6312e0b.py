"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, minimize_scalar, differential_evolution

# Pre-compute indices for N=14 for fast triangle area calculation
_IDX_CACHE = {}
def _get_indices(n):
    if n not in _IDX_CACHE:
        i_idx, j_idx, k_idx = [], [], []
        for i in range(n-2):
            for j in range(i+1, n-1):
                for k in range(j+1, n):
                    i_idx.append(i)
                    j_idx.append(j)
                    k_idx.append(k)
        _IDX_CACHE[n] = (np.array(i_idx), np.array(j_idx), np.array(k_idx))
    return _IDX_CACHE[n]

def _min_triangle_area(pts):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    return np.min(areas)

def _convex_hull_area(pts):
    try:
        hull = ConvexHull(pts)
        return hull.volume
    except Exception:
        return 1e-10

def _score(pts):
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-9:
        return 0.0
    min_area = _min_triangle_area(pts)
    return min_area / hull_area

def _soft_min_area(pts, beta):
    n = len(pts)
    if n < 3:
        return 0.0
    i_idx, j_idx, k_idx = _get_indices(n)
    
    pi = pts[i_idx]
    pj = pts[j_idx]
    pk = pts[k_idx]
    
    v1 = pj - pi
    v2 = pk - pi
    
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_a = np.min(areas)
    shifted = areas - min_a
    log_sum = np.log(np.sum(np.exp(-beta * shifted)))
    return min_a - (1.0 / beta) * log_sum

def _two_ring_score(theta_inner, theta_outer, r_inner):
    """
    Compute score for a two-ring configuration.
    theta_inner: angles for inner ring (len 7)
    theta_outer: angles for outer ring (len 7)
    r_inner: radius of inner ring (outer is 1.0)
    """
    inner_pts = np.column_stack([r_inner * np.cos(theta_inner), r_inner * np.sin(theta_inner)])
    outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
    pts = np.vstack([inner_pts, outer_pts])
    return _score(pts)

def construct():
    start_time = time.time()
    deadline = start_time + 115.0
    
    best_score = -np.inf
    best_pts = None
    
    # Helper to update best
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # 1. Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)
    
    # 2. Two-ring optimization
    # Inner ring: 7 points, Outer ring: 7 points
    # By symmetry, we can fix one point and optimize the rest
    # But let's try a parametric approach first
    
    # Symmetric two-ring: inner at angles 0, 2pi/7, ..., outer at pi/7, 3pi/7, ...
    # Optimize r_inner
    def neg_two_ring_score(r):
        theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        theta_outer = theta_inner + np.pi / 7
        inner_pts = np.column_stack([r * np.cos(theta_inner), r * np.sin(theta_inner)])
        outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
        pts = np.vstack([inner_pts, outer_pts])
        return -_score(pts)
    
    if time.time() < deadline - 30.0:
        try:
            res = minimize_scalar(neg_two_ring_score, bounds=(0.1, 0.9), method='bounded',
                                   options={'xatol': 1e-6})
            r_opt = res.x
            theta_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False)
            theta_outer = theta_inner + np.pi / 7
            inner_pts = np.column_stack([r_opt * np.cos(theta_inner), r_opt * np.sin(theta_inner)])
            outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
            pts = np.vstack([inner_pts, outer_pts])
            update_best(pts)
        except Exception:
            pass
    
    # 3. More general two-ring optimization with angle optimization
    # We fix the outer ring to be regular (by symmetry it's likely optimal)
    # and optimize the inner ring angles and radius
    # Actually, let's use differential evolution for the full 14-point problem
    # but with a good initial guess and tight bounds
    
    if time.time() < deadline - 40.0:
        # Use differential evolution on a subset of points or full?
        # Full 28-dim DE is slow. Let's try optimizing the inner ring only.
        
        def neg_inner_ring_opt(x):
            r = x[0]
            theta_inner = x[1:8]
            theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
            inner_pts = np.column_stack([r * np.cos(theta_inner), r * np.sin(theta_inner)])
            outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
            pts = np.vstack([inner_pts, outer_pts])
            return -_score(pts)
        
        # Bounds: r in [0.2, 0.8], angles in [0, 2pi]
        bounds = [(0.2, 0.8)] + [(0, 2 * np.pi)] * 7
        
        # Initial guess: symmetric
        x0 = np.array([0.5] + list(np.linspace(0, 2 * np.pi, 7, endpoint=False)))
        
        try:
            res = differential_evolution(
                neg_inner_ring_opt, 
                bounds, 
                seed=42, 
                maxiter=200, 
                tol=1e-6,
                polish=True
            )
            if res.fun < 0:
                r_opt = res.x[0]
                theta_inner = res.x[1:8]
                theta_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
                inner_pts = np.column_stack([r_opt * np.cos(theta_inner), r_opt * np.sin(theta_inner)])
                outer_pts = np.column_stack([np.cos(theta_outer), np.sin(theta_outer)])
                pts = np.vstack([inner_pts, outer_pts])
                update_best(pts)
        except Exception:
            pass
    
    # 4. Local refinement using L-BFGS-B on the full 28-dim space
    # Start from the best point found so far
    
    if best_pts is not None and time.time() < deadline - 20.0:
        def neg_score_full(x):
            pts = x.reshape(-1, 2)
            return -_score(pts)
        
        def neg_soft_score_full(x, beta):
            pts = x.reshape(-1, 2)
            hull_area = _convex_hull_area(pts)
            if hull_area < 1e-9:
                return 1e10
            soft_min = _soft_min_area(pts, beta)
            return -(soft_min / hull_area)
        
        current_pts = best_pts.copy()
        current_score = best_score
        
        # Continuation
        betas = [10.0, 50.0, 200.0]
        for beta in betas:
            if time.time() > deadline - 10.0:
                break
            try:
                res = minimize(
                    neg_soft_score_full,
                    current_pts.flatten(),
                    args=(beta,),
                    method='L-BFGS-B',
                    options={'maxiter': 100, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass
        
        # Final Powell
        if time.time() < deadline - 5.0:
            try:
                res = minimize(
                    neg_score_full,
                    current_pts.flatten(),
                    method='Powell',
                    options={'maxiter': 200, 'xtol': 1e-9, 'ftol': 1e-12}
                )
                if res.fun < -current_score:
                    current_pts = res.x.reshape(-1, 2)
                    current_score = -res.fun
                    update_best(current_pts)
            except Exception:
                pass
    
    # 5. If time remains, try a few random restarts with local optimization
    if time.time() < deadline - 25.0:
        rng = np.random.RandomState(999)
        for restart in range(3):
            if time.time() > deadline - 15.0:
                break
            
            # Random start near the regular polygon
            noise = rng.normal(0, 0.1, size=(N, 2))
            init_pts = reg_pts + noise
            update_best(init_pts)
            
            # Quick local refinement
            try:
                res = minimize(
                    neg_score_full,
                    init_pts.flatten(),
                    method='Nelder-Mead',
                    options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < -best_score:
                    update_best(res.x.reshape(-1, 2))
            except Exception:
                pass
    
    if best_pts is None:
        return reg_pts
        
    return best_pts
# EVOLVE-BLOCK-END
