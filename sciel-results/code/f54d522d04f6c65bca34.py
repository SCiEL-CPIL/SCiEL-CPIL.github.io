import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import time

N = 13

# EVOLVE-BLOCK-START

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by 3 points."""
    n = len(points)
    min_area = float('inf')
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                if area < min_area:
                    min_area = area
    return min_area

def _hull_area(points):
    """Compute the area of the convex hull."""
    try:
        hull = ConvexHull(points)
        return hull.volume
    except Exception:
        return 1e-12

def _score(points):
    """Compute the score: min triangle area / hull area."""
    points = np.asarray(points)
    if points.shape != (N, 2):
        return 0.0
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    min_a = _min_triangle_area(points)
    return min_a / hull_a

def _soft_min_area(points, temperature=0.01):
    """Soft approximation of min triangle area using log-sum-exp."""
    n = len(points)
    areas = []
    for i in range(n):
        for j in range(i+1, n):
            for k in range(j+1, n):
                a = points[i]
                b = points[j]
                c = points[k]
                area = 0.5 * abs((b[0]-a[0])*(c[1]-a[1]) - (c[0]-a[0])*(b[1]-a[1]))
                areas.append(area)
    areas = np.array(areas)
    # Soft min: -logsumexp(-areas / T) * T
    neg_areas = -areas / temperature
    max_neg = np.max(neg_areas)
    logsumexp_val = max_neg + np.log(np.sum(np.exp(neg_areas - max_neg)))
    return -logsumexp_val * temperature

def _soft_score(points, temperature=0.01):
    """Soft version of score for gradient-based optimization."""
    points = np.asarray(points)
    hull_a = _hull_area(points)
    if hull_a < 1e-12:
        return 0.0
    soft_min_a = _soft_min_area(points, temperature)
    return soft_min_a / hull_a

def _negative_soft_score(points, temperature=0.01):
    """Negative soft score for minimization."""
    return -_soft_score(points, temperature)

def _regular_polygon(n, radius=1.0, center=(0,0), rotation=0.0):
    """Generate n points on a regular polygon."""
    angles = np.linspace(0, 2*np.pi, n, endpoint=False) + rotation
    x = center[0] + radius * np.cos(angles)
    y = center[1] + radius * np.sin(angles)
    return np.column_stack([x, y])

def construct():
    start_time = time.time()
    deadline = start_time + 140  # Safety margin
    
    best_points = None
    best_score = 0.0
    
    # Generate multiple initial configurations
    initial_configs = []
    
    # 1. Regular 13-gon
    initial_configs.append(_regular_polygon(N, radius=1.0))
    
    # 2. Regular 13-gon with different rotation
    initial_configs.append(_regular_polygon(N, radius=1.0, rotation=np.pi/13))
    
    # 3. Regular 12-gon + center
    pts12 = _regular_polygon(12, radius=1.0)
    center = np.array([[0.0, 0.0]])
    initial_configs.append(np.vstack([pts12, center]))
    
    # 4. Jittered regular polygon
    for seed in [1, 2, 3]:
        rng = np.random.RandomState(seed)
        base = _regular_polygon(N, radius=1.0)
        jitter = rng.normal(0, 0.1, size=(N, 2))
        initial_configs.append(base + jitter)
    
    # 5. Random points in unit circle
    for seed in [4, 5, 6]:
        rng = np.random.RandomState(seed)
        angles = rng.uniform(0, 2*np.pi, N)
        radii = rng.uniform(0.5, 1.0, N)
        x = radii * np.cos(angles)
        y = radii * np.sin(angles)
        initial_configs.append(np.column_stack([x, y]))
    
    # 6. Grid-like pattern
    # 4x4 grid minus one corner, scaled
    grid_pts = []
    for i in range(4):
        for j in range(4):
            if len(grid_pts) < N:
                grid_pts.append([i/3.0, j/3.0])
    initial_configs.append(np.array(grid_pts[:N]))
    
    # Evaluate all initial configs
    for idx, config in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
        config = np.asarray(config, dtype=float)
        if config.shape != (N, 2):
            continue
        s = _score(config)
        if s > best_score:
            best_score = s
            best_points = config.copy()
    
    # Optimize from the best initial config and a few others
    configs_to_optimize = [best_points.copy()] if best_points is not None else []
    # Add a couple more promising configs
    for idx, config in enumerate(initial_configs[:5]):
        config = np.asarray(config, dtype=float)
        if config.shape == (N, 2):
            configs_to_optimize.append(config.copy())
    
    for opt_idx, init in enumerate(configs_to_optimize):
        if time.time() > deadline - 20:
            break
        
        # Try with different temperatures
        for temp in [0.05, 0.02, 0.01]:
            if time.time() > deadline - 10:
                break
            
            x0 = init.flatten()
            
            try:
                result = minimize(
                    _negative_soft_score,
                    x0,
                    args=(temp,),
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'ftol': 1e-10, 'gtol': 1e-8}
                )
                
                if result.success or result.fun < 0:
                    x_opt = result.x.reshape(N, 2)
                    s = _score(x_opt)
                    if s > best_score:
                        best_score = s
                        best_points = x_opt.copy()
                        
            except Exception:
                pass
        
        # Also try Powell method (derivative-free)
        if time.time() > deadline - 10:
            break
        
        x0 = init.flatten()
        try:
            result = minimize(
                _negative_soft_score,
                x0,
                args=(0.01,),
                method='Powell',
                options={'maxiter': 100, 'ftol': 1e-10}
            )
            
            if result.success or result.fun < 0:
                x_opt = result.x.reshape(N, 2)
                s = _score(x_opt)
                if s > best_score:
                    best_score = s
                    best_points = x_opt.copy()
                    
        except Exception:
            pass
    
    # Final polish: if we have a good point, try a high-precision local optimization
    if best_points is not None and time.time() < deadline - 5:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _negative_soft_score,
                x0,
                args=(0.005,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            
            x_opt = result.x.reshape(N, 2)
            s = _score(x_opt)
            if s > best_score:
                best_score = s
                best_points = x_opt.copy()
                
        except Exception:
            pass
    
    # Return the best points found
    if best_points is None:
        # Fallback: regular polygon
        best_points = _regular_polygon(N, radius=1.0)
    
    return best_points

# EVOLVE-BLOCK-END
