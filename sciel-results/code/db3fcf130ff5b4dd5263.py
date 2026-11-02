import numpy as np
import time
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import itertools

N = 14

def _score(points):
    """
    Compute the minimum triangle area divided by the convex hull area.
    """
    points = np.asarray(points, dtype=float)
    if points.shape != (N, 2):
        return 0.0

    # Compute convex hull area
    try:
        hull = ConvexHull(points)
        hull_area = hull.volume
    except Exception:
        return 0.0

    if hull_area < 1e-12:
        return 0.0

    # Precompute triangle indices
    if not hasattr(_score, '_indices'):
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _score._indices = (i[mask], j[mask], k[mask])
    
    i, j, k = _score._indices
    
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    
    min_area = np.min(areas)
    
    return min_area / hull_area

def _all_triangle_areas(points):
    """
    Compute all triangle areas.
    """
    if not hasattr(_score, '_indices'):
        i, j, k = np.indices((N, N, N))
        mask = (i < j) & (j < k)
        _score._indices = (i[mask], j[mask], k[mask])
    
    i, j, k = _score._indices
    p1 = points[i]
    p2 = points[j]
    p3 = points[k]
    
    v1 = p2 - p1
    v2 = p3 - p1
    areas = 0.5 * np.abs(v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0])
    return areas

def _convexity_penalty(points, mu=0.01):
    """
    Log-barrier penalty for convexity.
    We assume points are in CCW order.
    For a convex polygon, all cross products of consecutive edges should be positive.
    cross = (p_{i+1} - p_i) x (p_{i+2} - p_{i+1})
    If cross <= 0, it's not convex (or collinear).
    Penalty = -mu * sum(log(cross_i))
    To avoid log(0) or log(negative), we use log(max(cross, eps)) or a smooth approximation.
    Actually, if cross < 0, the log is undefined. We should penalize heavily if cross <= 0.
    A common barrier is -mu * sum(log(cross_i)) for cross_i > 0.
    If cross_i <= 0, we return a large penalty.
    
    However, L-BFGS might step into non-convex regions.
    We can use a soft barrier: -mu * sum(log(cross_i + delta)) where delta is small?
    No, if cross is negative, log is invalid.
    
    Better: Use a penalty that is smooth and goes to infinity as cross -> 0+.
    If cross <= 0, return a large constant.
    """
    n = len(points)
    cross_prods = np.zeros(n)
    for i in range(n):
        p0 = points[i]
        p1 = points[(i + 1) % n]
        p2 = points[(i + 2) % n]
        v1 = p1 - p0
        v2 = p2 - p1
        cross_prods[i] = v1[0] * v2[1] - v1[1] * v2[0]
    
    # If any cross product is <= 0, the polygon is not strictly convex (or is CW).
    # We assume CCW. If CW, all are negative.
    # Check if all are same sign.
    if np.all(cross_prods < 0):
        # CW order. Flip points? No, we can just take abs or negate.
        # But the barrier should be symmetric or we enforce CCW.
        # Let's just use abs for the barrier calculation if consistent, but 
        # standard convexity requires consistent orientation.
        # Let's enforce CCW by checking the signed area.
        signed_area = 0.5 * np.sum(points[:, 0] * np.roll(points[:, 1], -1) - np.roll(points[:, 0], -1) * points[:, 1])
        if signed_area < 0:
            # It's CW. We can reverse the points to make it CCW for the barrier?
            # Or just use abs(cross) in the barrier? 
            # If we use abs, we allow CW convex polygons too.
            # But the objective function (triangle areas) is invariant to orientation.
            # So we can just use abs(cross_prods) for the barrier.
            pass
            
    abs_cross = np.abs(cross_prods)
    
    # Barrier term
    # To avoid log(0), we clip to a small epsilon
    eps = 1e-6
    abs_cross_clipped = np.maximum(abs_cross, eps)
    
    barrier = -mu * np.sum(np.log(abs_cross_clipped))
    
    # If any cross product is very small (close to 0), the barrier is large.
    # If any is negative (which is handled by abs), we don't penalize direction, only magnitude.
    # This allows the optimizer to flip orientation if it wants, but keeps it convex.
    
    return barrier

def _objective(flat_points, mu=0.01):
    points = flat_points.reshape((N, 2))
    score = _score(points)
    if score < 1e-10:
        return 1e10 + _convexity_penalty(points, mu)
    
    barrier = _convexity_penalty(points, mu)
    
    # Maximize score, so minimize -score + barrier
    return -score + barrier

def _objective_grad(flat_points, mu=0.01):
    """
    Approximate gradient using finite differences or analytical?
    Analytical is hard due to min and hull.
    L-BFGS can use numerical gradient, but it's expensive.
    Let's try L-BFGS with numerical gradient first.
    """
    points = flat_points.reshape((N, 2))
    score = _score(points)
    if score < 1e-10:
        # Return large value and zero gradient? Or approximate gradient of penalty?
        # If invalid, just return large value.
        return 1e10, np.zeros_like(flat_points)
    
    barrier = _convexity_penalty(points, mu)
    val = -score + barrier
    
    # Numerical gradient
    eps = 1e-7
    grad = np.zeros_like(flat_points)
    for i in range(len(flat_points)):
        xp = flat_points.copy()
        xp[i] += eps
        fp = _objective(xp, mu)
        
        xm = flat_points.copy()
        xm[i] -= eps
        fm = _objective(xm, mu)
        
        grad[i] = (fp - fm) / (2 * eps)
        
    return val, grad

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    best_points = None
    best_score = -np.inf

    def update_best(pts):
        nonlocal best_points, best_score
        s = _score(pts)
        if s > best_score:
            best_score = s
            best_points = pts.copy()

    # 1. Regular 14-gon (CCW)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    initial_points = np.column_stack([np.cos(angles), np.sin(angles)])
    update_best(initial_points)

    x0 = initial_points.flatten()
    
    # 2. L-BFGS with log-barrier
    # Start with a small mu and increase it? Or start with a moderate mu?
    # Small mu allows exploring the boundary, but might violate convexity.
    # Large mu keeps it strictly convex but might be too restrictive.
    # Let's try a few values of mu.
    
    mus = [0.001, 0.01, 0.1]
    
    for mu in mus:
        if time.time() > deadline - 10:
            break
            
        try:
            # Use L-BFGS-B or L-BFGS
            # L-BFGS is better for unconstrained (with barrier)
            res = minimize(
                _objective,
                x0,
                args=(mu,),
                method='L-BFGS-B',
                options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-8, 'maxls': 50}
            )
            if res.fun < _objective(x0, mu):
                pts = res.x.reshape((N, 2))
                update_best(pts)
                x0 = res.x
        except Exception:
            pass

    # 3. Nelder-Mead refinement on the best point found
    if time.time() < deadline - 5:
        try:
            # Use a small mu for the final polish to allow more flexibility
            res_nm = minimize(
                _objective,
                x0,
                args=(0.001,),
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8, 'adaptive': True}
            )
            if res_nm.fun < _objective(x0, 0.001):
                pts = res_nm.x.reshape((N, 2))
                update_best(pts)
        except Exception:
            pass

    # 4. Random restarts with L-BFGS if time allows
    rng = np.random.default_rng(42)
    
    # Generate some random convex polygons
    for _ in range(5):
        if time.time() > deadline - 15:
            break
            
        # Random angles, sorted
        angles_rand = np.sort(rng.uniform(0, 2 * np.pi, N))
        # Random radii
        radii = rng.uniform(0.5, 1.0, N)
        
        pts = np.column_stack([radii * np.cos(angles_rand), radii * np.sin(angles_rand)])
        
        # Ensure CCW
        # Check signed area
        signed_area = 0.5 * np.sum(pts[:, 0] * np.roll(pts[:, 1], -1) - np.roll(pts[:, 0], -1) * pts[:, 1])
        if signed_area < 0:
            pts = pts[::-1]
            
        update_best(pts)
        
        x0_rand = pts.flatten()
        
        try:
            res = minimize(
                _objective,
                x0_rand,
                args=(0.01,),
                method='L-BFGS-B',
                options={'maxiter': 100, 'ftol': 1e-6, 'gtol': 1e-6}
            )
            if res.fun < _objective(x0_rand, 0.01):
                pts_opt = res.x.reshape((N, 2))
                update_best(pts_opt)
        except Exception:
            pass

    if best_points is None:
        return initial_points

    return best_points

# Pre-compute indices
i, j, k = np.indices((N, N, N))
mask = (i < j) & (j < k)
_score._indices = (i[mask], j[mask], k[mask])
