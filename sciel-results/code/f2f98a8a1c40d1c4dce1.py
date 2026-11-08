"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 13

# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

# Deadline guard: kill with safety margin
DEADLINE = 140.0  # seconds (10s safety margin from 150s limit)
_START_TIME = time.time()

def _time_remaining():
    return DEADLINE - (time.time() - _START_TIME)

def _min_triangle_area(points):
    """Compute the minimum area of any triangle formed by triples of points."""
    n = len(points)
    if n < 3:
        return 0.0
    # Vectorized computation for speed
    min_area = float('inf')
    # Precompute for efficiency
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            # Vectorize over k
            pk = points[j + 1:]
            # Area = 0.5 * |cross(pj - pi, pk - pi)|
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas = 0.5 * np.abs(cross)
            local_min = np.min(areas)
            if local_min < min_area:
                min_area = local_min
    return min_area

def _convex_hull_area(points):
    """Compute the area of the convex hull of the points."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(points)
        return hull.volume  # In 2D, volume is area
    except Exception:
        # Fallback: if all points collinear or degenerate
        return 1e-10

def _score(points):
    """Score: minimum triangle area divided by convex hull area."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    min_tri = _min_triangle_area(points)
    return min_tri / hull_area

def _soft_min_triangle_area(points, beta=100.0):
    """Soft surrogate for the minimum triangle area using log-sum-exp."""
    n = len(points)
    if n < 3:
        return 0.0
    # Collect all triangle areas
    areas = []
    for i in range(n - 2):
        pi = points[i]
        for j in range(i + 1, n - 1):
            pj = points[j]
            pk = points[j + 1:]
            v1 = pj - pi
            v2 = pk - pi
            cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
            areas.append(0.5 * np.abs(cross))
    areas = np.concatenate(areas)
    # Soft min via log-sum-exp: -logsumexp(-beta * areas) / beta
    # This approximates min(areas) as beta -> inf
    scaled = -beta * areas
    max_scaled = np.max(scaled)
    logsumexp = max_scaled + np.log(np.sum(np.exp(scaled - max_scaled)))
    return -logsumexp / beta

def _negative_soft_score(points, beta=50.0):
    """Negative of soft score for minimization."""
    hull_area = _convex_hull_area(points)
    if hull_area < 1e-10:
        return 0.0
    soft_min_tri = _soft_min_triangle_area(points, beta)
    return -(soft_min_tri / hull_area)

def _init_regular_polygon():
    """Initialize with a regular 13-gon."""
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    points = np.column_stack([np.cos(angles), np.sin(angles)])
    return points

def _random_points(rng):
    """Generate random points in a disk."""
    angles = rng.uniform(0, 2 * np.pi, N)
    radii = rng.uniform(0.5, 1.0, N)
    points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    return points

def _local_polish(points, max_iter=200, tol=1e-10):
    """Local polish using L-BFGS-B with soft surrogate."""
    x0 = points.flatten()
    
    def obj(x):
        pts = x.reshape(N, 2)
        return _negative_soft_score(pts, beta=100.0)
    
    def grad(x):
        # Numerical gradient
        eps = 1e-7
        g = np.zeros_like(x)
        for i in range(len(x)):
            xp = x.copy()
            xm = x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (obj(xp) - obj(xm)) / (2 * eps)
        return g
    
    result = minimize(obj, x0, method='L-BFGS-B', jac=grad, 
                      options={'maxiter': max_iter, 'ftol': tol, 'gtol': 1e-8})
    return result.x.reshape(N, 2)

def _refine_with_exact_min(points, max_iter=100):
    """Refine using the exact (nonsmooth) minimum triangle area with subgradient method."""
    pts = points.copy()
    for _ in range(max_iter):
        if _time_remaining() < 5.0:
            break
        min_tri = _min_triangle_area(pts)
        hull_area = _convex_hull_area(pts)
        if hull_area < 1e-10:
            break
        
        # Find the triple achieving the minimum
        n = len(pts)
        min_val = float('inf')
        best_triple = None
        for i in range(n - 2):
            pi = pts[i]
            for j in range(i + 1, n - 1):
                pj = pts[j]
                pk = pts[j + 1:]
                v1 = pj - pi
                v2 = pk - pi
                cross = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
                areas = 0.5 * np.abs(cross)
                idx = np.argmin(areas)
                if areas[idx] < min_val:
                    min_val = areas[idx]
                    k = j + 1 + idx
                    best_triple = (i, j, k)
        
        if best_triple is None:
            break
        
        i, j, k = best_triple
        pi, pj, pk = pts[i], pts[j], pts[k]
        v1 = pj - pi
        v2 = pk - pi
        cross_val = v1[0] * v2[1] - v1[1] * v2[0]
        
        # Subgradient of min triangle area w.r.t. points
        # The area is 0.5 * |cross|, so gradient depends on sign
        sign = np.sign(cross_val) if cross_val != 0 else 1.0
        
        # Gradient of 0.5 * sign * (v1[0]*v2[1] - v1[1]*v2[0]) w.r.t. pi, pj, pk
        # Let A = 0.5 * sign * (v1[0]*v2[1] - v1[1]*v2[0])
        # dA/dpi = -0.5 * sign * (v2[1], -v2[0])  ... need to be careful
        # Actually, let's use the soft surrogate for gradient
        
        # Use numerical subgradient via soft surrogate
        pts_flat = pts.flatten()
        eps = 1e-6
        grad = np.zeros(2 * N)
        for idx in range(2 * N):
            xp = pts_flat.copy()
            xp[idx] += eps
            pts_p = xp.reshape(N, 2)
            val_p = _soft_min_triangle_area(pts_p, beta=200.0) / _convex_hull_area(pts_p)
            grad[idx] = val_p / eps
        
        # Simple gradient ascent on the score
        step_size = 0.01
        new_pts = (pts_flat + step_size * grad).reshape(N, 2)
        
        # Check if score improved
        old_score = _score(pts)
        new_score = _score(new_pts)
        
        if new_score > old_score:
            pts = new_pts
        else:
            # Try smaller step
            step_size *= 0.5
            new_pts = (pts_flat + step_size * grad).reshape(N, 2)
            new_score = _score(new_pts)
            if new_score > old_score:
                pts = new_pts
            else:
                break
    
    return pts

def construct():
    global _START_TIME
    _START_TIME = time.time()
    
    rng = np.random.RandomState(N)
    
    # Start with regular polygon
    best_points = _init_regular_polygon()
    best_score = _score(best_points)
    
    # Try several initializations
    init_candidates = [best_points]
    for _ in range(10):
        if _time_remaining() < 30.0:
            break
        cand = _random_points(rng)
        init_candidates.append(cand)
    
    # For each candidate, do local polish
    for cand in init_candidates:
        if _time_remaining() < 15.0:
            break
        
        # Local polish with soft surrogate
        polished = _local_polish(cand, max_iter=100, tol=1e-8)
        
        # Refine with exact min
        refined = _refine_with_exact(polished, max_iter=50)
        
        score = _score(refined)
        if score > best_score:
            best_score = score
            best_points = refined
    
    # Final polish on the best point
    if _time_remaining() > 10.0:
        final = _local_polish(best_points, max_iter=200, tol=1e-10)
        final = _refine_with_exact(final, max_iter=50)
        if _score(final) > best_score:
            best_points = final
    
    return best_points

# EVOLVE-BLOCK-END
