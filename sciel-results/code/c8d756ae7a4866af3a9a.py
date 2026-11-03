"""
Strategy: Support-function parameterization for strict convexity, with multi-start
Nelder-Mead/Powell refinement and a soft-min surrogate for smooth optimization.
The 14 points are vertices of a convex polygon defined by a trigonometric polynomial
support function h(θ) = h0 + Σ_{k=1}^{K} (a_k cos(kθ) + b_k sin(kθ)), with small
coefficients to guarantee strict convexity.
"""
import time
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull

N = 14
_DEADLINE = 145.0  # Safety margin before 150s kill

def _min_triangle_area(points):
    """
    Compute the minimum area of any triangle formed by 3 points.
    Vectorized implementation for speed.
    """
    n = len(points)
    min_area = float('inf')
    x = points[:, 0]
    y = points[:, 1]
    
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            areas = 0.5 * np.abs(cross)
            
            if len(areas) > 0:
                local_min = np.min(areas)
                if local_min < min_area:
                    min_area = local_min
                    
    return min_area

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0
    
    if hull_area < 1e-10:
        return 0.0
        
    min_area = _min_triangle_area(points)
    return min_area / hull_area

def _all_triangle_areas(points):
    """
    Compute all triangle areas (vectorized).
    Returns array of shape (C(n,3),) with all triangle areas.
    """
    n = len(points)
    x = points[:, 0]
    y = points[:, 1]
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            if j + 1 >= n:
                continue
            dx_ji = x[j] - x[i]
            dy_ji = y[j] - y[i]
            dx_ki = x[j+1:] - x[i]
            dy_ki = y[j+1:] - y[i]
            cross = dx_ji * dy_ki - dx_ki * dy_ji
            a = 0.5 * np.abs(cross)
            areas.extend(a)
    return np.array(areas)

def _points_from_support(params, n_samples=14):
    """
    Reconstruct N points from support function parameters.
    params: array of shape (1 + 2*K,) where K is the number of harmonics.
    h(θ) = h0 + Σ_{k=1}^{K} (a_k cos(kθ) + b_k sin(kθ))
    
    The points are placed at equally spaced angles θ_i = 2πi/N, and their
    positions are determined by the support function: for a convex polygon,
    the vertex at angle θ is at position h(θ) * (cos(θ), sin(θ)) only if
    the polygon is star-shaped w.r.t. origin. But for general convex polygons,
    we need to use the fact that the point on the boundary at normal angle θ
    is h(θ) * (cos θ, sin θ) only for the support function interpretation.
    
    Actually, for a convex body, the point on the boundary with outward normal
    direction u = (cos θ, sin θ) is x = h(θ) u. But the vertices of the polygon
    are not generally at these positions unless the polygon is a circle.
    
    A better approach: use the support function to define the polygon implicitly,
    then sample N points along the boundary. For optimization purposes, we can
    approximate the boundary by N points at equally spaced angles, where each
    point is h(θ_i) * (cos θ_i, sin θ_i). This gives a star-shaped polygon
    (convex if h is such that the resulting curve is convex).
    
    To ensure convexity, we require h(θ) + h''(θ) > 0 for all θ. With small
    harmonics, this is satisfied if |a_k|, |b_k| < h0 / k^2 for each k.
    """
    K = (len(params) - 1) // 2
    h0 = params[0]
    
    # Ensure h0 is positive
    if h0 < 0.1:
        h0 = 0.1
    
    # Compute points at N equally spaced angles
    theta = np.linspace(0, 2 * np.pi, n_samples, endpoint=False)
    
    h = np.full(n_samples, h0)
    for k in range(1, K + 1):
        a_k = params[2 * k - 1]
        b_k = params[2 * k]
        h += a_k * np.cos(k * theta) + b_k * np.sin(k * theta)
    
    # Ensure h is positive
    h = np.maximum(h, 0.05)
    
    points = np.column_stack((h * np.cos(theta), h * np.sin(theta)))
    
    return points

def _objective(params):
    """
    Objective function to minimize (negative score).
    """
    points = _points_from_support(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    score = _score(points)
    
    if score < 1e-6:
        return 1e10
        
    return -score

def _objective_soft(params, beta=80.0):
    """
    Objective using a soft-min for the triangle areas to smooth the landscape.
    """
    points = _points_from_support(params)
    
    if np.any(np.isnan(points)) or np.any(np.isinf(points)):
        return 1e10
        
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 1e10
        
    if hull_area < 1e-10:
        return 1e10
        
    areas = _all_triangle_areas(points)
    if len(areas) == 0:
        return 1e10
    
    # Soft min: -1/beta * log(sum(exp(-beta * areas)))
    min_a = np.min(areas)
    shifted = areas - min_a
    soft_min_val = -1.0/beta * np.log(np.sum(np.exp(-beta * shifted))) + min_a
    
    score = soft_min_val / hull_area
    return -score

def _generate_initial_configs(K=4):
    """
    Generate diverse initial configurations for multi-start optimization.
    Returns list of parameter arrays.
    """
    configs = []
    
    # 1. Regular polygon: h0 = 1, all other coefficients 0
    params = np.zeros(1 + 2 * K)
    params[0] = 1.0
    configs.append(params.copy())
    
    # 2. Small random perturbations of the regular polygon
    for trial in range(8):
        params = np.zeros(1 + 2 * K)
        params[0] = 1.0 + np.random.uniform(-0.05, 0.05)
        for k in range(1, K + 1):
            params[2 * k - 1] = np.random.uniform(-0.05, 0.05)
            params[2 * k] = np.random.uniform(-0.05, 0.05)
        configs.append(params.copy())
    
    # 3. Configurations with dominant low-frequency harmonics
    for k_dom in [1, 2, 3]:
        for phase in [0.0, np.pi / 4, np.pi / 2]:
            params = np.zeros(1 + 2 * K)
            params[0] = 1.0
            if k_dom <= K:
                params[2 * k_dom - 1] = 0.1 * np.cos(phase)
                params[2 * k_dom] = 0.1 * np.sin(phase)
            configs.append(params.copy())
    
    # 4. Alternating pattern: h0 = 1, a_1 = 0.15
    for amp in [0.1, 0.15, 0.2]:
        params = np.zeros(1 + 2 * K)
        params[0] = 1.0
        params[1] = amp  # a_1
        configs.append(params.copy())
    
    # 5. More varied random configs
    for _ in range(5):
        params = np.zeros(1 + 2 * K)
        params[0] = 1.0 + np.random.uniform(-0.1, 0.1)
        for k in range(1, K + 1):
            params[2 * k - 1] = np.random.uniform(-0.08, 0.08)
            params[2 * k] = np.random.uniform(-0.08, 0.08)
        configs.append(params.copy())
    
    return configs

def construct():
    start_time = time.time()
    deadline = start_time + _DEADLINE
    
    K = 4  # Number of harmonics
    init_configs = _generate_initial_configs(K)
    
    best_params = init_configs[0].copy()
    best_score_val = -_objective(best_params)
    
    # Phase 1: Multi-start Nelder-Mead with hard objective
    phase1_budget = 55.0
    time_per_start = phase1_budget / len(init_configs)
    
    for i, init in enumerate(init_configs):
        if time.time() > deadline - 50:
            break
        
        try:
            nm_res = minimize(
                _objective,
                init,
                method='Nelder-Mead',
                options={
                    'maxiter': 800,
                    'xatol': 1e-9,
                    'fatol': 1e-11,
                    'adaptive': True
                }
            )
            if nm_res.fun < _objective(best_params):
                best_params = nm_res.x
                best_score_val = -nm_res.fun
        except Exception:
            pass
    
    # Phase 2: Powell's method for more aggressive local search
    if time.time() < deadline - 30:
        try:
            pow_res = minimize(
                _objective,
                best_params,
                method='Powell',
                options={
                    'maxiter': 500,
                    'xtol': 1e-10,
                    'ftol': 1e-12,
                    'adaptive': True
                }
            )
            if pow_res.fun < _objective(best_params):
                best_params = pow_res.x
                best_score_val = -pow_res.fun
        except Exception:
            pass
    
    # Phase 3: Soft objective polish with multiple beta values
    if time.time() < deadline - 20:
        for beta in [50.0, 80.0, 120.0]:
            if time.time() > deadline - 10:
                break
            try:
                nm_soft_res = minimize(
                    lambda p, b=beta: _objective_soft(p, beta=b),
                    best_params,
                    method='Nelder-Mead',
                    options={
                        'maxiter': 400,
                        'xatol': 1e-10,
                        'fatol': 1e-12,
                        'adaptive': True
                    }
                )
                hard_score_soft_params = -_objective(nm_soft_res.x)
                if hard_score_soft_params > best_score_val:
                    best_params = nm_soft_res.x
                    best_score_val = hard_score_soft_params
            except Exception:
                pass
    
    # Phase 4: Final high-precision polish
    if time.time() < deadline - 5:
        try:
            final_res = minimize(
                _objective,
                best_params,
                method='Nelder-Mead',
                options={
                    'maxiter': 1000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True
                }
            )
            if final_res.fun < _objective(best_params):
                best_params = final_res.x
                best_score_val = -final_res.fun
        except Exception:
            pass

    # Reconstruct final points
    points = _points_from_support(best_params)
    
    # Normalize to unit area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale
        return final_points
        
    if hull_area > 1e-10:
        scale = np.sqrt(1.0 / hull_area)
        final_points = points * scale
    else:
        angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
        final_points = np.column_stack([np.cos(angles), np.sin(angles)])
        hull = ConvexHull(final_points)
        hull_area = hull.volume
        scale = np.sqrt(1.0 / hull_area)
        final_points = final_points * scale

    return final_points
