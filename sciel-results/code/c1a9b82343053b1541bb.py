# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time
import itertools

def _precompute_indices(n):
    """Precompute triangle indices for efficiency."""
    return np.array(list(itertools.combinations(range(n), 3)))

def _compute_dets(points, idx):
    """Compute 2 * signed area (determinant) for all triangles."""
    p1 = points[idx[:, 0]]
    p2 = points[idx[:, 1]]
    p3 = points[idx[:, 2]]
    
    v1 = p2 - p1
    v2 = p3 - p1
    det = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    return det

def _compute_hull_area(points):
    """Compute convex hull area safely."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 0.0

def compute_score(points):
    """
    Compute the Heilbronn score: A_min / A_hull.
    Returns 0 if invalid.
    """
    n = len(points)
    if n < 3:
        return 0.0
    
    if not np.all(np.isfinite(points)):
        return 0.0
        
    A_hull = _compute_hull_area(points)
    if A_hull <= 1e-12:
        return 0.0
        
    idx = _precompute_indices(n)
    dets = _compute_dets(points, idx)
    A_min = 0.5 * np.min(np.abs(dets))
    
    return A_min / A_hull

def heilbronn_convex13() -> np.ndarray:
    n = 13
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    idx = _precompute_indices(n)
    
    # Define the smooth objective for the epigraph formulation
    # We minimize t subject to:
    # 1. 0.5 * |det_i| / A_hull <= t  (for all i)
    # 2. Points are in [-1, 1] x [-1, 1]
    #
    # To make it differentiable, we use a soft-min approximation or direct constraints.
    # SLSQP handles inequality constraints well.
    # Let x = [p1x, p1y, ..., p13x, p13y, t]
    # Objective: minimize t
    # Constraints:
    #   g_i(x) = t - (0.5 * |det_i(x)| / A_hull(x)) >= 0
    #   h_j(x) = 1 - |p_j|_inf >= 0 (bounds handled by SLSQP bounds, but let's use explicit if needed)
    
    # Since |det| is not differentiable at 0, and A_hull is not differentiable,
    # direct SLSQP might struggle.
    # However, the "soft-min" inspiration suggests using a smooth approximation.
    # Let's try a hybrid: use a smooth max approximation for the constraint violation.
    # Actually, a better approach for SLSQP is to minimize the "soft" maximum of the normalized areas.
    # f(x) = logsumexp( [log(0.5*|det_i|/A_hull) for all i] ) / log(N)  -- this is a smooth max.
    # But log(0) is -inf. We need to handle small areas.
    # Alternatively, minimize the sum of squares of violations? No, that's not min-max.
    
    # Let's stick to the epigraph with a smooth surrogate for the non-smooth parts.
    # A common trick: minimize t + lambda * sum(soft_plus(t - area_i))^2 ?
    # Or simply: minimize t, with constraints area_i >= t - epsilon?
    
    # Let's try a direct smooth optimization of the "soft max" of the normalized triangle areas.
    # Score = 1 / (SoftMax_Norm_Areas)
    # We want to maximize Score -> Minimize SoftMax_Norm_Areas.
    
    # Soft Max of a vector v: logsumexp(v) / log(len(v)) is not quite right for min.
    # We want the MIN of areas. So we want to MINIMIZE the MIN.
    # This is equivalent to MAXIMIZING the MIN.
    # Let's define the objective as the "soft minimum" of the normalized areas.
    # SoftMin(v, beta) = -1/beta * log(sum(exp(-beta * v_i)))
    # As beta -> inf, SoftMin -> min(v).
    # We want to MAXIMIZE SoftMin(v).
    # So we MINIMIZE -SoftMin(v).
    
    # But v_i = 0.5 * |det_i| / A_hull.
    # A_hull is non-smooth.
    
    # Let's use a different strategy:
    # 1. Start with a good initial guess (regular polygon).
    # 2. Use a local optimizer (SLSQP or Nelder-Mead) on the objective:
    #    -score(points) = - (min_i 0.5*|det_i|) / A_hull
    #    To make it smooth, replace min with soft-min and |det| with sqrt(det^2) approx or just handle sign carefully.
    #    Actually, 0.5*|det| is smooth everywhere except det=0.
    #    A_hull is smooth almost everywhere.
    
    # Let's try a multi-start local optimization with a smooth objective function.
    
    def smooth_objective(x):
        points = x.reshape((n, 2))
        if not np.all(np.isfinite(points)):
            return 1e6
        
        A_hull = _compute_hull_area(points)
        if A_hull <= 1e-12:
            return 1e6
            
        dets = _compute_dets(points, idx)
        # Normalize areas
        areas = 0.5 * np.abs(dets) / A_hull
        
        # Soft minimum of areas
        # beta controls sharpness. Higher beta = closer to true min.
        beta = 50.0
        # To avoid overflow/underflow in exp, subtract max
        max_area = np.max(areas)
        # We want min, so we use -beta * areas
        # soft_min = -1/beta * log(sum(exp(-beta * areas)))
        # Let's use the log-sum-exp trick for stability
        # log(sum(exp(-beta * areas))) = -beta * max(areas) + log(sum(exp(-beta * (areas - max(areas)))))
        shifted = -beta * (areas - max_area)
        # Clip shifted to avoid overflow in exp
        shifted = np.clip(shifted, -50, 50)
        sum_exp = np.sum(np.exp(shifted))
        if sum_exp == 0:
            return 1e6
        log_sum = np.log(sum_exp)
        soft_min = -1/beta * (log_sum - beta * max_area)
        
        # We want to MAXIMIZE soft_min, so MINIMIZE -soft_min
        return -soft_min

    # Initial points: Regular polygon
    angles = np.linspace(0, 2*np.pi, n, endpoint=False)
    init_points = np.column_stack([np.cos(angles), np.sin(angles)])
    
    # Run local optimization from multiple starts
    # 1. Regular polygon
    # 2. Slightly perturbed regular polygon
    # 3. Points on a circle with random radii? No, convex hull area matters.
    
    starts = [
        init_points.flatten(),
        (init_points + 0.1 * np.random.randn(n, 2)).flatten(),
        (init_points + 0.05 * np.random.randn(n, 2)).flatten()
    ]
    
    for i, x0 in enumerate(starts):
        if time.time() > deadline - 30:
            break
            
        try:
            # Use Nelder-Mead for robustness on non-smooth-ish functions
            # Or SLSQP if we trust the smoothness
            res = minimize(
                smooth_objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            pts = res.x.reshape((n, 2))
            score = compute_score(pts)
            
            if score > best_score:
                best_score = score
                best_points = pts.copy()
                
        except Exception as e:
            pass

    # If we have time, try to polish the best point with a higher beta (sharper min)
    if best_points is not None and time.time() < deadline - 60:
        def sharp_objective(x):
            points = x.reshape((n, 2))
            if not np.all(np.isfinite(points)):
                return 1e6
            A_hull = _compute_hull_area(points)
            if A_hull <= 1e-12:
                return 1e6
            dets = _compute_dets(points, idx)
            areas = 0.5 * np.abs(dets) / A_hull
            beta = 100.0
            max_area = np.max(areas)
            shifted = -beta * (areas - max_area)
            shifted = np.clip(shifted, -50, 50)
            sum_exp = np.sum(np.exp(shifted))
            if sum_exp == 0:
                return 1e6
            log_sum = np.log(sum_exp)
            soft_min = -1/beta * (log_sum - beta * max_area)
            return -soft_min

        try:
            res_sharp = minimize(
                sharp_objective, 
                best_points.flatten(), 
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'adaptive': True}
            )
            pts_sharp = res_sharp.x.reshape((n, 2))
            score_sharp = compute_score(pts_sharp)
            if score_sharp > best_score:
                best_score = score_sharp
                best_points = pts_sharp.copy()
        except Exception:
            pass

    # Fallback
    if best_points is None or best_score < 0.01:
        best_points = init_points
        best_score = compute_score(best_points)

    return best_points

# EVOLVE-BLOCK-END
