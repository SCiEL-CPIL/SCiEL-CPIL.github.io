"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
from scipy.optimize import minimize
N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize_scalar, differential_evolution

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
        if np.allclose(pts[:, 0], pts[0, 0]) or np.allclose(pts[:, 1], pts[0, 1]):
            return 1e-10
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

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 120s limit
    
    best_score = -np.inf
    best_pts = None
    
    def update_best(pts):
        nonlocal best_score, best_pts
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
            return True
        return False

    # 1. Baseline: Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    reg_pts = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(reg_pts)

    # 2. Two-Ring Optimization with Analytic Inner Radius
    # Strategy: For a given rotation angle alpha, find the optimal inner radius r
    # that maximizes the minimum triangle area. Then optimize over alpha in 1D.
    
    def _two_ring_points_analytic(alpha):
        """
        Generate two-ring points with optimized inner radius for given alpha.
        Outer ring: 7 points at radius 1.0
        Inner ring: 7 points at radius r (optimized)
        
        We optimize r to balance "pure" and "mixed" triangle areas.
        """
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_inner = angles_outer + alpha
        
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        
        # Optimize r for this alpha
        # The optimal r balances:
        # 1. Pure outer triangles (fixed area)
        # 2. Pure inner triangles (scales with r^2)
        # 3. Mixed triangles (scales with r)
        
        # We can find r by maximizing the minimum area over all triangle types
        # For efficiency, we'll use a 1D search over r
        
        def min_area_for_r(r):
            inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
            all_pts = np.vstack([outer_pts, inner_pts])
            return _min_triangle_area(all_pts)
        
        # Search for optimal r in [0.1, 0.95]
        # Use minimize_scalar to find max of min_area (i.e., min of -min_area)
        
        # First, do a coarse grid to find a good range
        r_coarse = np.linspace(0.2, 0.9, 20)
        areas_coarse = [min_area_for_r(r) for r in r_coarse]
        best_idx = np.argmax(areas_coarse)
        
        # Refine around the best coarse value
        r_min = max(0.1, r_coarse[max(0, best_idx-1)])
        r_max = min(0.95, r_coarse[min(len(r_coarse)-1, best_idx+1)])
        
        if r_max - r_min < 0.01:
            r_opt = r_coarse[best_idx]
        else:
            # Use minimize_scalar for fine search
            def neg_min_area(r):
                return -min_area_for_r(r)
            
            result = minimize_scalar(neg_min_area, bounds=(r_min, r_max), method='bounded',
                                    options={'xatol': 1e-6})
            r_opt = result.x
        
        inner_pts = np.column_stack([r_opt * np.cos(angles_inner), r_opt * np.sin(angles_inner)])
        return np.vstack([outer_pts, inner_pts])
    
    def neg_two_ring_analytic(alpha):
        """Objective function for 1D optimization over alpha."""
        pts = _two_ring_points_analytic(alpha)
        return -_score(pts)
    
    # Optimize over alpha in [0, 2*pi/7]
    # The symmetry period is 2*pi/7 since we have 7-fold symmetry
    
    if time.time() < deadline - 30.0:
        try:
            # Coarse search first
            alpha_coarse = np.linspace(0, 2 * np.pi / 7, 50)
            scores_coarse = [_score(_two_ring_points_analytic(a)) for a in alpha_coarse]
            best_alpha_idx = np.argmax(scores_coarse)
            
            # Refine around the best coarse value
            a_min = alpha_coarse[max(0, best_alpha_idx-1)]
            a_max = alpha_coarse[min(len(alpha_coarse)-1, best_alpha_idx+1)]
            
            if a_max - a_min < 0.01:
                alpha_opt = alpha_coarse[best_alpha_idx]
            else:
                result = minimize_scalar(neg_two_ring_analytic, bounds=(a_min, a_max), 
                                        method='bounded', options={'xatol': 1e-8})
                alpha_opt = result.x
            
            pts = _two_ring_points_analytic(alpha_opt)
            update_best(pts)
            
        except Exception:
            pass

    # 3. Alternative: Direct 2D optimization (r, alpha) as fallback/comparison
    def _two_ring_points_direct(r, alpha):
        angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
        angles_inner = angles_outer + alpha
        
        outer_pts = np.column_stack([np.cos(angles_outer), np.sin(angles_outer)])
        inner_pts = np.column_stack([r * np.cos(angles_inner), r * np.sin(angles_inner)])
        
        return np.vstack([outer_pts, inner_pts])
    
    def neg_two_ring_direct(params):
        r, alpha = params
        pts = _two_ring_points_direct(r, alpha)
        return -_score(pts)
    
    if time.time() < deadline - 20.0:
        try:
            # Use differential evolution for the 2D parameter space
            bounds = [(0.2, 0.95), (0.0, 2 * np.pi / 7)]
            res = differential_evolution(
                neg_two_ring_direct, 
                bounds, 
                seed=42, 
                maxiter=50, 
                tol=1e-6,
                polish=True
            )
            if res.fun < 0:
                r_opt, alpha_opt = res.x
                pts = _two_ring_points_direct(r_opt, alpha_opt)
                update_best(pts)
        except Exception:
            pass

    # 4. Local refinement if time permits
    if best_pts is not None and time.time() < deadline - 10.0:
        try:
            def neg_score_full(x):
                pts = x.reshape(-1, 2)
                return -_score(pts)
            
            res = minimize(
                neg_score_full,
                best_pts.flatten(),
                method='Powell',
                options={'maxiter': 50, 'xtol': 1e-8, 'ftol': 1e-10}
            )
            
            if res.fun < -best_score:
                new_pts = res.x.reshape(-1, 2)
                update_best(new_pts)
        except Exception:
            pass

    if best_pts is None:
        return reg_pts
        
    return best_pts
# EVOLVE-BLOCK-END
