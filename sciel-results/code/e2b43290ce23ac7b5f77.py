import numpy as np
import scipy.spatial
import scipy.optimize
import time
from itertools import combinations

N = 14

# Pre-compute triangle indices for speed
def _get_triangle_indices(n):
    indices = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                indices.append((i, j, k))
    return np.array(indices, dtype=np.int32)

_TRI_INDICES = _get_triangle_indices(N)
_I = _TRI_INDICES[:, 0]
_J = _TRI_INDICES[:, 1]
_K = _TRI_INDICES[:, 2]

def _min_triangle_area(points):
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return 0.5 * np.abs(cross).min()

def _hull_area(points):
    try:
        hull = scipy.spatial.ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def _hull_area_and_grad(points):
    """
    Computes the area of the convex hull and its gradient with respect to points.
    Returns (area, grad) where grad is (N, 2).
    """
    try:
        hull = scipy.spatial.ConvexHull(points)
        area = hull.volume
        hull_idx = hull.vertices
        n_hull = len(hull_idx)
        
        grad = np.zeros_like(points)
        
        if n_hull < 3:
            return area, grad
            
        # For each hull vertex, find its neighbors in the hull cycle
        # hull_idx is in CCW order
        for i in range(n_hull):
            idx_curr = hull_idx[i]
            idx_prev = hull_idx[(i - 1) % n_hull]
            idx_next = hull_idx[(i + 1) % n_hull]
            
            p_curr = points[idx_curr]
            p_prev = points[idx_prev]
            p_next = points[idx_next]
            
            # Contribution to area from edges (prev, curr) and (curr, next)
            # Area = 0.5 * sum (x_i y_{i+1} - x_{i+1} y_i)
            # Terms involving p_curr:
            # 0.5 * (x_prev * y_curr - y_prev * x_curr) + 0.5 * (x_curr * y_next - y_curr * x_next)
            # d/dx_curr: 0.5 * (-y_prev + y_next)
            # d/dy_curr: 0.5 * (x_prev - x_next)
            
            grad[idx_curr, 0] = 0.5 * (p_next[1] - p_prev[1])
            grad[idx_curr, 1] = 0.5 * (p_prev[0] - p_next[0])
            
        return area, grad
    except Exception:
        return 0.0, np.zeros_like(points)

def _score(points):
    try:
        if len(points) < 3:
            return 0.0
        ha = _hull_area(points)
        if ha < 1e-12:
            return 0.0
        mta = _min_triangle_area(points)
        return mta / ha
    except Exception:
        return 0.0

def _normalize(points):
    try:
        ha = _hull_area(points)
        if ha < 1e-12:
            return points
        return points / np.sqrt(ha)
    except Exception:
        return points

def _regular_polygon(n, radius=1.0, rotation=0.0):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rotation
    return np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

def _generate_three_ring_config(n1, n2, n3, r1, r2, r3, rot1, rot2, rot3):
    """
    Generate points on three concentric circles.
    """
    pts1 = _regular_polygon(n1, r1, rot1)
    pts2 = _regular_polygon(n2, r2, rot2)
    pts3 = _regular_polygon(n3, r3, rot3)
    return np.vstack([pts1, pts2, pts3])

def _smooth_min_triangle_area_and_grad(points, beta=100.0):
    """
    Smooth approximation of min triangle area using log-sum-exp.
    Returns (value, gradient).
    """
    pi = points[_I]
    pj = points[_J]
    pk = points[_K]
    v1 = pj - pi
    v2 = pk - pi
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # For numerical stability, shift by max
    max_area = np.max(areas)
    if max_area < 1e-15:
        return 0.0, np.zeros_like(points)
    
    # Stable log-sum-exp: log(sum(exp(-beta * areas))) = -beta*max_area + log(sum(exp(-beta*(areas - max_area))))
    shifted = areas - max_area
    exp_vals = np.exp(-beta * shifted)
    log_sum = -beta * max_area + np.log(np.sum(exp_vals))
    smooth_min = -log_sum / beta
    
    # Gradient of smooth_min
    sum_exp = np.sum(exp_vals)
    weights = exp_vals / sum_exp
    
    # Gradients of cross product
    # cross = (xj - xi)(yk - yi) - (yj - yi)(xk - xi)
    # d/dxi = yj - yk
    # d/dyi = xj - xk
    # d/dxj = yk - yi
    # d/dyj = -(xk - xi)
    # d/dxk = -(yj - yi)
    # d/dyk = xj - xi
    
    g_pi = np.column_stack([points[_J, 1] - points[_K, 1], points[_J, 0] - points[_K, 0]])
    g_pj = np.column_stack([points[_K, 1] - points[_I, 1], -(points[_K, 0] - points[_I, 0])])
    g_pk = np.column_stack([-(points[_J, 1] - points[_I, 1]), points[_J, 0] - points[_I, 0]])
    
    signs = np.sign(cross)
    signs[signs == 0] = 1.0
    
    coeff = 0.5 * weights * signs
    
    grad_points = np.zeros_like(points)
    
    np.add.at(grad_points, _I, coeff[:, np.newaxis] * g_pi)
    np.add.at(grad_points, _J, coeff[:, np.newaxis] * g_pj)
    np.add.at(grad_points, _K, coeff[:, np.newaxis] * g_pk)
    
    return smooth_min, grad_points

def _smooth_score_and_grad(points, beta=100.0):
    """
    Smooth version of the score: smooth_min_triangle_area / hull_area
    Returns (score, gradient).
    """
    try:
        if len(points) < 3:
            return 0.0, np.zeros_like(points)
        
        ha, grad_ha = _hull_area_and_grad(points)
        if ha < 1e-12:
            return 0.0, np.zeros_like(points)
            
        smta, grad_smta = _smooth_min_triangle_area_and_grad(points, beta)
        
        score = smta / ha
        
        # d(S/A) = (dS*A - S*dA) / A^2
        grad_score = (grad_smta * ha - score * grad_ha) / (ha * ha)
        
        return score, grad_score
    except Exception:
        return 0.0, np.zeros_like(points)

def _neg_smooth_score_and_grad(points_flat, beta=100.0):
    """
    Returns (-smooth_score, gradient) for L-BFGS-B.
    """
    points = points_flat.reshape(-1, 2)
    score, grad = _smooth_score_and_grad(points, beta)
    return -score, -grad

def _refine_points(points, time_limit=10.0):
    """
    Refine a point configuration using gradient-based optimization with smooth surrogate.
    """
    start = time.time()
    deadline = start + time_limit
    
    init_norm = _normalize(points)
    x = init_norm.flatten()
    
    # Anneal beta
    n_stages = 3
    betas = [50.0, 200.0, 800.0]
    
    for beta in betas:
        if time.time() > deadline:
            break
        
        def _obj_and_grad(x_flat, beta_val=beta):
            return _neg_smooth_score_and_grad(x_flat, beta_val)
        
        try:
            res = scipy.optimize.minimize(
                _obj_and_grad,
                x,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': 300, 'gtol': 1e-8, 'ftol': 1e-12}
            )
            x = res.x
        except Exception:
            pass
    
    # Final polish with Nelder-Mead on exact score
    if time.time() < deadline - 1:
        try:
            def _neg_exact_score(x_flat):
                return -_score(x_flat.reshape(-1, 2))
            
            res = scipy.optimize.minimize(
                _neg_exact_score,
                x,
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-12, 'fatol': 1e-14, 'adaptive': True}
            )
            x = res.x
        except Exception:
            pass
    
    return x.reshape(-1, 2)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin

    best_score = 0.0
    best_points = None

    def _update_best(points):
        nonlocal best_score, best_points
        pts = _normalize(points)
        sc = _score(pts)
        if sc > best_score:
            best_score = sc
            best_points = pts.copy()
        return sc

    # --- Structural Change: Three-Ring Symmetric Configurations ---
    # Represent 14 points as union of 3 concentric regular polygons.
    # Sizes: (7, 5, 2), (6, 6, 2), (8, 4, 2), (7, 4, 3), etc.
    
    # We fix the outer ring radius to 1.0 (scale invariance handled by normalization).
    # Parameters to optimize:
    # - r2, r3: radii of inner rings (relative to outer)
    # - rot2, rot3: rotations of inner rings relative to outer (outer rot fixed at 0)
    # Total 4 parameters for each split.
    
    splits = [
        (7, 5, 2),
        (6, 6, 2),
        (8, 4, 2),
        (7, 4, 3),
        (5, 6, 3),
        (4, 5, 5),
        (3, 5, 6),
        (2, 6, 6),
    ]
    
    rng = np.random.RandomState(42)
    
    for n1, n2, n3 in splits:
        if time.time() > deadline - 5:
            break
            
        def _three_ring_score(params, n1=n1, n2=n2, n3=n3):
            r2 = params[0]
            r3 = params[1]
            rot2 = params[2]
            rot3 = params[3]
            
            # Ensure radii are positive and ordered (outer > inner)
            # r1 = 1.0
            if r2 <= 0.05 or r3 <= 0.05:
                return 1e6 # Penalty
            
            # We want r1 > r2 > r3 typically, but the optimizer might find other orderings.
            # Let's just let it explore.
            
            pts = _generate_three_ring_config(n1, n2, n3, 1.0, r2, r3, 0.0, rot2, rot3)
            sc = _score(_normalize(pts))
            return -sc

        # Differential Evolution for global search in 4D parameter space
        # Bounds:
        # r2 in [0.1, 0.95]
        # r3 in [0.05, 0.95]
        # rot2 in [0, 2*pi]
        # rot3 in [0, 2*pi]
        
        bounds = [
            (0.1, 0.95),
            (0.05, 0.95),
            (0.0, 2 * np.pi),
            (0.0, 2 * np.pi)
        ]
        
        # Try a few different initial populations or seeds? DE is stochastic.
        # We can run DE for a limited number of generations.
        
        try:
            # Use differential evolution
            # maxiter is number of generations. 
            # popsize is multiple of dimension (4). Let's use 10*4=40.
            # Time budget per split: ~10s max.
            # DE can be slow. Let's limit generations.
            
            # We will run DE with a time callback to stop early if needed.
            # scipy's differential evolution doesn't have a direct time limit, 
            # but we can check time in a callback? No, callback is for iteration.
            # We'll just run a fixed number of iterations and check time after.
            
            # Let's try 30 iterations with popsize 8 (32 individuals).
            # 30 * 32 * (cost of 1 eval) ~ 30 * 32 * 0.01s = 9.6s. Might be tight.
            # Let's use popsize 6 (24 individuals) and 20 iterations.
            # 20 * 24 * 0.01 = 4.8s.
            
            res = scipy.optimize.differential_evolution(
                _three_ring_score,
                bounds,
                maxiter=20,
                popsize=6,
                seed=42,
                tol=0.01,
                polish=True # Polish with L-BFGS-B at the end
            )
            
            if res.success or res.fun < 1e5:
                best_params = res.x
                pts = _generate_three_ring_config(n1, n2, n3, 1.0, best_params[0], best_params[1], 0.0, best_params[2], best_params[3])
                sc = _update_best(pts)
                
                # If this is a good candidate, refine the full 28D coordinates
                if sc > best_score * 0.9 and time.time() < deadline - 15:
                    refined = _refine_points(pts, time_limit=10.0)
                    _update_best(refined)
                    
        except Exception:
            pass

    # --- Fallback: Standard 2-Ring and Regular Polygon searches if time remains ---
    # The 3-ring approach is the primary new structural change.
    # If it fails or time is left, we can add the previous 2-ring logic.
    
    if time.time() < deadline - 20:
        # Add back some 2-ring exploration as it was effective
        splits_2 = [(7, 7), (6, 8), (8, 6)]
        
        for n1, n2 in splits_2:
            if time.time() > deadline - 10:
                break
                
            def _two_ring_score(params, n1=n1, n2=n2):
                r2 = params[0]
                rot2 = params[1]
                if r2 <= 0.05:
                    return 1e6
                pts = _generate_three_ring_config(n1, n2, 0, 1.0, r2, 0.0, 0.0, rot2, 0.0)
                # Wait, _generate_three_ring_config with n3=0 might fail or behave weirdly.
                # Let's make a specific 2-ring generator or handle n3=0.
                # Actually, _regular_polygon(0, ...) returns empty array?
                # np.linspace(0, 2*pi, 0) is empty. vstack works.
                # But _hull_area needs at least 3 points.
                # If n1+n2 >= 3, it's fine.
                sc = _score(_normalize(pts))
                return -sc

            bounds = [
                (0.1, 0.95),
                (0.0, 2 * np.pi)
            ]
            
            try:
                res = scipy.optimize.differential_evolution(
                    _two_ring_score,
                    bounds,
                    maxiter=20,
                    popsize=6,
                    seed=42,
                    tol=0.01,
                    polish=True
                )
                if res.success or res.fun < 1e5:
                    r2, rot2 = res.x
                    pts = _generate_three_ring_config(n1, n2, 0, 1.0, r2, 0.0, 0.0, rot2, 0.0)
                    sc = _update_best(pts)
                    
                    if sc > best_score * 0.9 and time.time() < deadline - 10:
                        refined = _refine_points(pts, time_limit=5.0)
                        _update_best(refined)
            except Exception:
                pass

    # Regular polygon fallback
    reg = _regular_polygon(N, 1.0)
    _update_best(reg)
    
    # Final check
    if best_points is None:
        best_points = _normalize(_regular_polygon(N, 1.0))
    else:
        best_points = _normalize(best_points)
        
    return best_points
