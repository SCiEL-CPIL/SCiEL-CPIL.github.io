import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

def _compute_covering_radius(points):
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
        
    facets = hull.simplices
    radii = np.zeros(len(facets))
    
    for k, tri in enumerate(facets):
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[k] = 0.0
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[k] = np.arccos(dot_val)
        
    return np.max(radii)

def _soft_max_covering_radius(points, beta=50.0):
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1e6
        
    facets = hull.simplices
    radii = np.zeros(len(facets))
    
    for k, tri in enumerate(facets):
        a, b, c = points[tri[0]], points[tri[1]], points[tri[2]]
        n = np.cross(b - a, c - a)
        norm_n = np.linalg.norm(n)
        if norm_n < 1e-10:
            radii[k] = 0.0
            continue
        n = n / norm_n
        if np.dot(n, a) < 0:
            n = -n
        dot_val = np.clip(np.dot(n, a), -1.0, 1.0)
        radii[k] = np.arccos(dot_val)
        
    max_r = np.max(radii)
    if max_r == 0:
        return 0.0
    sum_exp = np.sum(np.exp(beta * (radii - max_r)))
    return max_r + np.log(sum_exp) / beta

def construct():
    N = 50
    best_points = None
    best_radius = np.inf
    
    # Fibonacci spiral initialization
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    fib_points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

    # List of initial points
    inits = [fib_points]
    for _ in range(9):
        p = fib_points + np.random.normal(0, 0.1, (N, 3))
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        inits.append(p)

    for init_p in inits:
        x0 = init_p.flatten()
        
        def objective(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            if np.any(norms < 1e-10):
                return 1e6
            p = p / norms
            return _soft_max_covering_radius(p, beta=50.0)
        
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 300, 'disp': False})
            if res.x is not None:
                opt_points = res.x.reshape(N, 3)
                opt_points /= np.linalg.norm(opt_points, axis=1, keepdims=True)
                exact_rad = _compute_covering_radius(opt_points)
                if exact_rad < best_radius:
                    best_radius = exact_rad
                    best_points = opt_points.copy()
        except Exception:
            continue

    # Final refinement on the best point found
    if best_points is not None:
        x0 = best_points.flatten()
        def objective_sharp(x):
            p = x.reshape(N, 3)
            norms = np.linalg.norm(p, axis=1, keepdims=True)
            if np.any(norms < 1e-10):
                return 1e6
            p = p / norms
            return _soft_max_covering_radius(p, beta=200.0)
            
        try:
            res = minimize(objective_sharp, x0, method='Powell', options={'maxiter': 100, 'disp': False})
            if res.x is not None:
                opt_points = res.x.reshape(N, 3)
                opt_points /= np.linalg.norm(opt_points, axis=1, keepdims=True)
                exact_rad = _compute_covering_radius(opt_points)
                if exact_rad < best_radius:
                    best_radius = exact_rad
                    best_points = opt_points.copy()
        except Exception:
            pass

    return best_points
