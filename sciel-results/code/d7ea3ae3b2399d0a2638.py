import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial import ConvexHull
import time
import itertools

N = 14

# Precompute triangle indices for vectorized area calculation
_IDX = np.array([(i, j, k) for i in range(N) for j in range(i + 1, N) for k in range(j + 1, N)])
_I, _J, _K = _IDX[:, 0], _IDX[:, 1], _IDX[:, 2]

def compute_score(points):
    """
    Compute min triangle area / convex hull area, vectorized.
    """
    n = len(points)
    if n < 3:
        return 0.0
    try:
        # Check for degenerate points
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    min_area = np.min(areas)
    if min_area < 1e-12:
        return 0.0
    return min_area / hull_area

def neg_score_full(x):
    """
    Objective function for full 28-dimensional optimization.
    """
    points = x.reshape((N, 2))
    return -compute_score(points)

def smooth_min_score(points, temperature=0.03):
    """
    Smooth surrogate for min triangle area / hull area using log-sum-exp.
    Returns a smooth approximation of the score that is differentiable.
    """
    n = len(points)
    if n < 3:
        return 0.0
    try:
        if np.any(np.isnan(points)) or np.any(np.isinf(points)):
            return 0.0
        hull = ConvexHull(points)
        hull_area = hull.volume
        if hull_area <= 1e-12:
            return 0.0
    except Exception:
        return 0.0
    
    p_i = points[_I]
    p_j = points[_J]
    p_k = points[_K]
    
    v1 = p_j - p_i
    v2 = p_k - p_i
    cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    areas = 0.5 * np.abs(cross)
    
    # Soft-min via log-sum-exp: approximates min(areas) smoothly
    # soft_min = -T * log(sum(exp(-a_i / T)))
    scaled = -areas / temperature
    max_scaled = np.max(scaled)
    log_sum_exp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    soft_min_area = -temperature * log_sum_exp
    
    if soft_min_area < 1e-12:
        return 0.0
    return soft_min_area / hull_area

def neg_smooth_score(x, temperature=0.03):
    """
    Smooth objective for gradient-based optimization.
    """
    points = x.reshape((N, 2))
    return -smooth_min_score(points, temperature)

def heptagon_plus_seven_configuration(params):
    """
    Configuration: 7 points on outer heptagon + 7 points on inner heptagon (possibly rotated).
    params: [r_outer, r_inner, theta_offset, scale_inner]
    """
    r_outer, r_inner, theta_offset, scale_inner = params
    
    angles_outer = np.linspace(0, 2*np.pi, 7, endpoint=False)
    outer_pts = np.column_stack((r_outer*np.cos(angles_outer), r_outer*np.sin(angles_outer)))
    
    angles_inner = theta_offset + np.linspace(0, 2*np.pi, 7, endpoint=False)
    inner_pts = np.column_stack((r_inner*scale_inner*np.cos(angles_inner), r_inner*scale_inner*np.sin(angles_inner)))
    
    return np.vstack([outer_pts, inner_pts])

def neg_heptagon_plus_seven_score(params):
    """
    Objective for heptagon+7 ring parameter optimization.
    """
    try:
        points = heptagon_plus_seven_configuration(params)
        return -compute_score(points)
    except Exception:
        return 1e10

def ring_configuration(params):
    """
    Generate points from ring parameters.
    params: [r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4]
    n1+n2+n3+n4 must be 14.
    """
    r1, r2, r3, r4, th1, th2, th3, th4, n1, n2, n3, n4 = params
    n1 = int(round(n1))
    n2 = int(round(n2))
    n3 = int(round(n3))
    n4 = int(round(n4))
    
    # Ensure we have exactly 14 points
    # Simple adjustment: if sum != 14, adjust the last ring
    total = n1 + n2 + n3 + n4
    if total != 14:
        # Distribute difference to n4, keeping it >= 0
        diff = 14 - total
        n4 = max(0, n4 + diff)
        # If n4 became 0 and we still need points, steal from others
        # This is a fallback, DE bounds should prevent extreme cases
        while n1 + n2 + n3 + n4 < 14:
            n4 += 1
        while n1 + n2 + n3 + n4 > 14:
            if n4 > 1: n4 -= 1
            elif n3 > 1: n3 -= 1
            elif n2 > 1: n2 -= 1
            else: n1 -= 1
    
    points = []
    
    # Ring 1 (outermost)
    if n1 > 0:
        angles = th1 + np.linspace(0, 2*np.pi, n1, endpoint=False)
        points.append(np.column_stack((r1*np.cos(angles), r1*np.sin(angles))))
    
    # Ring 2
    if n2 > 0:
        angles = th2 + np.linspace(0, 2*np.pi, n2, endpoint=False)
        points.append(np.column_stack((r2*np.cos(angles), r2*np.sin(angles))))
    
    # Ring 3
    if n3 > 0:
        angles = th3 + np.linspace(0, 2*np.pi, n3, endpoint=False)
        points.append(np.column_stack((r3*np.cos(angles), r3*np.sin(angles))))
    
    # Ring 4 (innermost)
    if n4 > 0:
        angles = th4 + np.linspace(0, 2*np.pi, n4, endpoint=False)
        points.append(np.column_stack((r4*np.cos(angles), r4*np.sin(angles))))
    
    if not points:
        return np.zeros((N, 2))
    
    result = np.vstack(points)
    
    # Pad or trim to exactly N points if needed
    if len(result) < N:
        extra = np.zeros((N - len(result), 2))
        result = np.vstack([result, extra])
    elif len(result) > N:
        result = result[:N]
    
    return result

def neg_ring_score(params):
    """
    Objective function for ring parameter optimization.
    """
    try:
        points = ring_configuration(params)
        return -compute_score(points)
    except Exception:
        return 0.0

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin before 150s kill limit
    
    best_points = None
    best_score = -float('inf')
    rng = np.random.default_rng(42)
    
    def try_update(points):
        nonlocal best_points, best_score
        if points is None:
            return
        s = compute_score(points)
        if s > best_score:
            best_score = s
            best_points = points.copy()
        return s
    
    # Phase 1a: Try structured configurations quickly
    # Regular 14-gon
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_pts = np.column_stack((np.cos(angles), np.sin(angles)))
    try_update(initial_pts)
    
    # Heptagon + 7 inner heptagon (various offsets)
    hept_bounds = [
        (0.5, 1.5),   # r_outer
        (0.1, 1.0),   # r_inner
        (0, 2*np.pi), # theta_offset
        (0.3, 1.0),   # scale_inner
    ]
    
    if time.time() < deadline - 50:
        try:
            de_hept = differential_evolution(
                neg_heptagon_plus_seven_score,
                hept_bounds,
                seed=42,
                maxiter=30,
                tol=1e-4,
                popsize=12,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            if de_hept.fun < 1e9:
                points = heptagon_plus_seven_configuration(de_hept.x)
                try_update(points)
        except Exception:
            pass
    
    # Phase 1b: Global Search using Differential Evolution on Ring Parameters
    bounds = [
        (0.5, 1.5),  # r1
        (0.1, 1.2),  # r2
        (0.0, 0.8),  # r3
        (0.0, 0.5),  # r4
        (0, 2*np.pi),  # th1
        (0, 2*np.pi),  # th2
        (0, 2*np.pi),  # th3
        (0, 2*np.pi),  # th4
        (1, 14),  # n1
        (1, 14),  # n2
        (1, 14),  # n3
        (1, 14),  # n4
    ]
    
    if time.time() < deadline - 45:
        try:
            de_res = differential_evolution(
                neg_ring_score,
                bounds,
                seed=42,
                maxiter=50,
                tol=1e-5,
                popsize=15,
                mutation=(0.3, 1.0),
                recombination=0.8,
                polish=False,
                init='latinhypercube'
            )
            
            if de_res.fun < 0:
                best_params = de_res.x
                points = ring_configuration(best_params)
                try_update(points)
        except Exception:
            pass
    
    # If nothing good found, ensure we have a reasonable starting point
    if best_points is None or best_score < 0.1:
        try_update(initial_pts)
    
    # Phase 2: Local Refinement in Full Space using Nelder-Mead
    if best_points is not None and time.time() < deadline - 25:
        x0 = best_points.flatten()
        
        nm_options = {
            'maxiter': 5000,
            'xatol': 1e-8,
            'fatol': 1e-8,
            'disp': False
        }
        
        try:
            res = minimize(neg_score_full, x0, method='Nelder-Mead', options=nm_options)
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape((N, 2))
        except Exception:
            pass
        
        # Random perturbations to escape local minima
        for _ in range(3):
            if time.time() > deadline - 15:
                break
            
            noise = rng.normal(0, 0.05, size=(N, 2))
            x0_perturbed = (best_points + noise).flatten()
            
            try:
                res = minimize(neg_score_full, x0_perturbed, method='Nelder-Mead', options=nm_options)
                if res.fun < -best_score:
                    best_score = -res.fun
                    best_points = res.x.reshape((N, 2))
            except Exception:
                pass
    
    # Phase 3: Smooth-guided refinement using L-BFGS-B
    # Use the smooth surrogate to guide optimization with gradient information
    if best_points is not None and time.time() < deadline - 10:
        x0 = best_points.flatten()
        
        # Try multiple temperatures for the soft-min
        for temp in [0.05, 0.03, 0.01]:
            if time.time() > deadline - 8:
                break
            
            try:
                res = minimize(
                    lambda x: neg_smooth_score(x, temperature=temp),
                    x0,
                    method='L-BFGS-B',
                    options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                if res.fun < -best_score:
                    candidate = res.x.reshape((N, 2))
                    s = compute_score(candidate)
                    if s > best_score:
                        best_score = s
                        best_points = candidate.copy()
            except Exception:
                pass
    
    # Phase 4: Final high-precision Nelder-Mead polish
    if best_points is not None and time.time() < deadline - 3:
        x0 = best_points.flatten()
        polish_options = {
            'maxiter': 3000,
            'xatol': 1e-10,
            'fatol': 1e-10,
            'disp': False
        }
        try:
            res = minimize(neg_score_full, x0, method='Nelder-Mead', options=polish_options)
            if res.fun < -best_score:
                best_score = -res.fun
                best_points = res.x.reshape((N, 2))
        except Exception:
            pass
    
    # Phase 5: Final smooth polish with very low temperature
    if best_points is not None and time.time() < deadline - 1:
        x0 = best_points.flatten()
        try:
            res = minimize(
                lambda x: neg_smooth_score(x, temperature=0.005),
                x0,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            candidate = res.x.reshape((N, 2))
            s = compute_score(candidate)
            if s > best_score:
                best_score = s
                best_points = candidate.copy()
        except Exception:
            pass

    if best_points is None:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        best_points = np.column_stack((np.cos(angles), np.sin(angles)))

    return best_points
