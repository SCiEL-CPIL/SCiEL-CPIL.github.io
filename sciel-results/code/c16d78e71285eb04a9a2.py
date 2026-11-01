N = 14

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_DEADLINE = 135.0  # safety margin before 150s kill

def _convex_hull_area(pts):
    """Compute convex hull area using Andrew's monotone chain."""
    n = len(pts)
    if n < 3:
        return 0.0
    pts = np.array(pts, dtype=float)
    pts = pts[np.lexsort((pts[:, 0], pts[:, 1]))]
    
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 1e-12:
            lower.pop()
        lower.append(p)
    
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 1e-12:
            upper.pop()
        upper.append(p)
    
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return 0.0
    
    # Shoelace formula
    x = hull[:, 0]
    y = hull[:, 1]
    area = 0.5 * abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1]))
    return area

def _min_triangle_area(pts):
    """Compute minimum triangle area over all triples."""
    n = len(pts)
    min_area = np.inf
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - 
                             (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]))
                if a < min_area:
                    min_area = a
    return min_area

def _score(pts):
    """Score = min triangle area / convex hull area."""
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 0.0
    min_tri = _min_triangle_area(pts)
    return min_tri / hull_area

def _soft_min_objective(flat_pts, tau):
    """
    Objective to minimize: -soft_min(triangle_areas) / hull_area
    We negate because minimize() minimizes.
    """
    pts = flat_pts.reshape(-1, 2)
    n = len(pts)
    
    # Compute all triangle areas
    areas = []
    for i in range(n):
        for j in range(i + 1, n):
            for k in range(j + 1, n):
                a = 0.5 * abs((pts[j][0] - pts[i][0]) * (pts[k][1] - pts[i][1]) - 
                             (pts[k][0] - pts[i][0]) * (pts[j][1] - pts[i][1]))
                areas.append(a)
    areas = np.array(areas)
    
    # Soft min: -log-sum-exp(-areas/tau)
    max_a = np.max(areas)
    shifted = areas - max_a
    log_sum_exp = np.log(np.sum(np.exp(shifted / tau))) + max_a / tau
    soft_min_val = max_a - tau * log_sum_exp
    
    hull_area = _convex_hull_area(pts)
    if hull_area < 1e-12:
        return 1e6
    
    return -soft_min_val / hull_area

def _regular_polygon(n, radius=1.0, center=(0.0, 0.0)):
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    xs = center[0] + radius * np.cos(angles)
    ys = center[1] + radius * np.sin(angles)
    return np.column_stack([xs, ys])

def construct():
    start_time = time.time()
    
    best_pts = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    init_configs = []
    
    # 1. Regular 14-gon
    init_configs.append(_regular_polygon(14, radius=1.0))
    
    # 2. Regular 14-gon with slight radial perturbation
    rng = np.random.RandomState(42)
    angles = np.linspace(0, 2 * np.pi, 14, endpoint=False)
    radii = 1.0 + 0.05 * rng.randn(14)
    radii = np.clip(radii, 0.5, 1.5)
    xs = radii * np.cos(angles)
    ys = radii * np.sin(angles)
    init_configs.append(np.column_stack([xs, ys]))
    
    # 3. Two concentric rings: 7 inner, 7 outer
    angles_outer = np.linspace(0, 2 * np.pi, 7, endpoint=False)
    angles_inner = np.linspace(0, 2 * np.pi, 7, endpoint=False) + np.pi / 7
    r_outer = 1.0
    r_inner = 0.5
    xs = np.concatenate([r_outer * np.cos(angles_outer), r_inner * np.cos(angles_inner)])
    ys = np.concatenate([r_outer * np.sin(angles_outer), r_inner * np.sin(angles_inner)])
    init_configs.append(np.column_stack([xs, ys]))
    
    # 4. Random well-spread points
    for seed in range(5):
        rng = np.random.RandomState(seed)
        pts = rng.rand(14, 2) * 2 - 1  # in [-1,1]^2
        init_configs.append(pts)
    
    # 5. Regular polygon with different rotation
    for rot in [0.1, 0.3, 0.5]:
        angles = np.linspace(0, 2 * np.pi, 14, endpoint=False) + rot
        xs = np.cos(angles)
        ys = np.sin(angles)
        init_configs.append(np.column_stack([xs, ys]))
    
    # Evaluate initial configs
    for pts in init_configs:
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_pts = pts.copy()
    
    # Optimize each initial config
    tau_values = [0.01, 0.005, 0.001]
    
    for cfg_idx, init_pts in enumerate(init_configs):
        if time.time() - start_time > _DEADLINE - 20:
            break
        
        for tau in tau_values:
            if time.time() - start_time > _DEADLINE - 15:
                break
            
            x0 = init_pts.flatten()
            
            try:
                result = minimize(
                    _soft_min_objective,
                    x0,
                    args=(tau,),
                    method='Nelder-Mead',
                    options={
                        'maxiter': 2000,
                        'xatol': 1e-8,
                        'fatol': 1e-10,
                        'adaptive': True
                    }
                )
                
                if result.success or result.fun < 0:
                    opt_pts = result.x.reshape(-1, 2)
                    s = _score(opt_pts)
                    if s > best_score:
                        best_score = s
                        best_pts = opt_pts.copy()
                        
            except Exception:
                pass
        
        # Also try a final polish with small tau using the best point from this config
        if time.time() - start_time > _DEADLINE - 10:
            break
        
        # Use best found so far for this config as starting point
        x0 = init_pts.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                args=(0.0005,),
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            if result.success:
                opt_pts = result.x.reshape(-1, 2)
                s = _score(opt_pts)
                if s > best_score:
                    best_score = s
                    best_pts = opt_pts.copy()
        except Exception:
            pass
    
    # Final check
    if best_pts is None:
        best_pts = _regular_polygon(14, radius=1.0)
    
    return best_pts
# EVOLVE-BLOCK-END
