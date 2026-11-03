"""Optimized covering of sphere with 50 points using multi-phase approach."""
import numpy as np
from scipy.spatial import ConvexHull
import time
import sys

N = 50
_TIME_LIMIT = 135.0
_DEADLINE = time.time() + _TIME_LIMIT


def _time_remaining():
    return _DEADLINE - time.time()


def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius(pts):
    """Compute covering radius using convex hull facet circumradii."""
    try:
        hull = ConvexHull(pts)
        max_cr = 0.0
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            t = np.dot(a, n)
            circumcenter_3d = t * n
            cc_norm = np.linalg.norm(circumcenter_3d)
            if cc_norm < 1e-15:
                continue
            circumcenter_sphere = circumcenter_3d / cc_norm
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(circumcenter_sphere, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                if ang > max_cr:
                    max_cr = ang
        return np.degrees(max_cr)
    except Exception:
        return 180.0


def _covering_radius_grid(pts, n_grid=80):
    """Estimate covering radius on a grid for fast evaluation."""
    # Create a grid of points on the sphere
    theta = np.linspace(0, np.pi, n_grid)
    phi = np.linspace(0, 2 * np.pi, n_grid, endpoint=False)
    theta, phi = np.meshgrid(theta, phi)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    grid_pts = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    
    # For each grid point, find nearest configuration point
    # Use cdist for efficiency
    from scipy.spatial.distance import cdist
    dists = cdist(grid_pts, pts)
    min_dists = dists.min(axis=1)
    max_dist = min_dists.max()
    # Convert Euclidean distance to angular distance
    # For points on unit sphere, angular distance = 2 * arcsin(d/2)
    angular = 2 * np.arcsin(np.clip(max_dist / 2.0, 0, 1))
    return np.degrees(angular)


def _fibonacci_points(jitter=0.0):
    """Golden-angle Fibonacci spiral points on the unit sphere."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    if jitter > 0:
        noise = np.random.randn(N, 3) * jitter
        pts = pts + noise
        pts = _normalize(pts)
    return pts


def _jittered_fibonacci(seed=None, jitter=0.1):
    """Fibonacci points with random jitter for diversity."""
    if seed is not None:
        np.random.seed(seed)
    return _fibonacci_points(jitter=jitter)


def _random_points(seed=None):
    """Random points on sphere."""
    if seed is not None:
        np.random.seed(seed)
    pts = np.random.randn(N, 3)
    return _normalize(pts)


def _spherical_lloyd(pts, n_iters=50, tol=1e-8):
    """Spherical Lloyd's algorithm: move each point to centroid of its Voronoi cell."""
    pts = _normalize(pts.copy())
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        new_pts = pts.copy()
        for i in range(N):
            cell_points = set()
            for simplex in hull.simplices:
                if i in simplex:
                    for j in simplex:
                        cell_points.add(j)
            if len(cell_points) < 3:
                continue
            cell_arr = np.array([pts[j] for j in cell_points])
            centroid = np.mean(cell_arr, axis=0)
            c_norm = np.linalg.norm(centroid)
            if c_norm > 1e-12:
                new_pts[i] = centroid / c_norm
        
        diff = np.max(np.linalg.norm(new_pts - pts, axis=1))
        pts = _normalize(new_pts)
        if diff < tol:
            break
    
    return pts


def _soft_covering_radius(pts, beta=100.0):
    """Soft covering radius using softmax over angular distances."""
    # Sample points on sphere for estimation
    n_samples = 500
    sample_pts = np.random.randn(n_samples, 3)
    sample_pts = _normalize(sample_pts)
    
    # Compute angular distances
    dots = sample_pts @ pts.T  # (n_samples, N)
    cos_dists = np.clip(dots, -1.0, 1.0)
    min_cos = cos_dists.max(axis=1)  # max dot product = min angular distance
    min_ang = np.arccos(min_cos)
    
    # Soft max to get a smooth estimate
    scaled = beta * min_ang
    max_scaled = scaled.max()
    exp_scaled = np.exp(scaled - max_scaled)
    soft_max = max_scaled + np.log(exp_scaled.sum()) / beta
    return np.degrees(soft_max)


def _gradient_descent(pts, n_iters=1000, beta=500.0, lr=0.01):
    """Gradient descent on soft covering radius."""
    pts = _normalize(pts.copy())
    current_cr = _soft_covering_radius(pts, beta)
    best_pts = pts.copy()
    best_cr = current_cr
    
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        
        # Compute gradient of soft covering radius
        n_samples = 300
        sample_pts = np.random.randn(n_samples, 3)
        sample_pts = _normalize(sample_pts)
        
        dots = sample_pts @ pts.T  # (n_samples, N)
        cos_dists = np.clip(dots, -1.0, 1.0)
        min_cos_idx = cos_dists.argmax(axis=1)
        min_cos = cos_dists[np.arange(n_samples), min_cos_idx]
        min_ang = np.arccos(min_cos)
        
        # Soft max weights
        scaled = beta * min_ang
        max_scaled = scaled.max()
        exp_scaled = np.exp(scaled - max_scaled)
        weights = exp_scaled / exp_scaled.sum()
        
        # Gradient with respect to each point
        grad = np.zeros_like(pts)
        for k in range(N):
            mask = (min_cos_idx == k)
            if mask.sum() == 0:
                continue
            # d(min_ang)/d(pts[k]) for samples where k is nearest
            # min_ang = arccos(min_cos), d(arccos(x))/dx = -1/sqrt(1-x^2)
            # min_cos = sample_pts @ pts[k], d(min_cos)/d(pts[k]) = sample_pts
            # So d(min_ang)/d(pts[k]) = -sample_pts / sqrt(1 - min_cos^2)
            mc = min_cos[mask]
            denom = np.sqrt(np.clip(1.0 - mc * mc, 1e-10, 1.0))
            d_ang_d_pts = -sample_pts[mask] / denom[:, None]
            # Weighted sum
            w = weights[mask]
            grad[k] = (w[:, None] * d_ang_d_pts).sum(axis=0)
        
        # Normalize gradient to be on sphere
        # Project gradient to tangent space
        for i in range(N):
            grad[i] -= np.dot(grad[i], pts[i]) * pts[i]
        
        # Update
        new_pts = pts - lr * grad
        new_pts = _normalize(new_pts)
        
        new_cr = _soft_covering_radius(new_pts, beta)
        if new_cr < current_cr:
            current_cr = new_cr
            pts = new_pts
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
        else:
            lr *= 0.95
        
        if it % 100 == 0:
            # Occasionally use true covering radius
            true_cr = _covering_radius(pts)
            if true_cr < best_cr:
                best_cr = true_cr
                best_pts = pts.copy()
    
    return best_pts


def _facet_shrink_polish(pts, n_iters=200):
    """Polish by moving points toward circumcenters of worst facets."""
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # Find the facet with the largest circumradius
        max_cr = 0.0
        worst_facet = None
        worst_cc = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            t = np.dot(a, n)
            cc_3d = t * n
            cc_norm = np.linalg.norm(cc_3d)
            if cc_norm < 1e-15:
                continue
            cc = cc_3d / cc_norm
            # Compute angular circumradius
            cr = 0.0
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(cc, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                cr = max(cr, ang)
            if cr > max_cr:
                max_cr = cr
                worst_facet = simplex
                worst_cc = cc
        
        if worst_facet is None:
            break
        
        # Move the vertices of the worst facet toward the circumcenter
        step = 0.05 * (1.0 - it / n_iters)
        for idx in worst_facet:
            pts[idx] += step * (worst_cc - pts[idx])
        pts = _normalize(pts)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts


def _worst_facet_projection(pts):
    """Find the worst facet and return its circumcenter."""
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None
    
    max_cr = 0.0
    worst_cc = None
    for simplex in hull.simplices:
        tri = pts[simplex]
        a, b, c = tri
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            continue
        n = n / n_norm
        t = np.dot(a, n)
        cc_3d = t * n
        cc_norm = np.linalg.norm(cc_3d)
        if cc_norm < 1e-15:
            continue
        cc = cc_3d / cc_norm
        cr = 0.0
        for v in [a, b, c]:
            cos_ang = np.clip(np.dot(cc, v), -1.0, 1.0)
            ang = np.arccos(cos_ang)
            cr = max(cr, ang)
        if cr > max_cr:
            max_cr = cr
            worst_cc = cc
    
    return worst_cc


def _optimize_facet_shrink(pts, n_iters=100):
    """Iteratively shrink the worst facet by moving its vertices."""
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_cr = _covering_radius(pts)
    
    for it in range(n_iters):
        if _time_remaining() < 5.0:
            break
        
        worst_cc = _worst_facet_projection(pts)
        if worst_cc is None:
            break
        
        # Find the facet with largest circumradius and move its vertices
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        max_cr = 0.0
        worst_facet = None
        for simplex in hull.simplices:
            tri = pts[simplex]
            a, b, c = tri
            n = np.cross(b - a, c - a)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-15:
                continue
            n = n / n_norm
            t = np.dot(a, n)
            cc_3d = t * n
            cc_norm = np.linalg.norm(cc_3d)
            if cc_norm < 1e-15:
                continue
            cc = cc_3d / cc_norm
            cr = 0.0
            for v in [a, b, c]:
                cos_ang = np.clip(np.dot(cc, v), -1.0, 1.0)
                ang = np.arccos(cos_ang)
                cr = max(cr, ang)
            if cr > max_cr:
                max_cr = cr
                worst_facet = simplex
        
        if worst_facet is None:
            break
        
        step = 0.03
        for idx in worst_facet:
            pts[idx] += step * (worst_cc - pts[idx])
        pts = _normalize(pts)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    return best_pts


def construct():
    """Construct 50 points on unit sphere minimizing covering radius."""
    global _DEADLINE
    _DEADLINE = time.time() + _TIME_LIMIT
    
    np.random.seed(42)
    
    best_pts = None
    best_cr = 180.0
    
    # Phase 1: Multiple diverse starts with multi-stage pipeline
    starts = []
    for seed in range(8):
        starts.append(_jittered_fibonacci(seed=seed, jitter=0.1))
        starts.append(_random_points(seed=100 + seed))
    
    for s_idx, start in enumerate(starts):
        if _time_remaining() < 20.0:
            break
        
        # Stage 1: Lloyd's algorithm
        pts = _spherical_lloyd(start, n_iters=30)
        
        # Stage 2: Gradient descent on soft covering radius
        pts = _gradient_descent(pts, n_iters=500, beta=500.0, lr=0.01)
        
        # Stage 3: Facet shrink polish
        pts = _optimize_facet_shrink(pts, n_iters=100)
        
        cr = _covering_radius(pts)
        if cr < best_cr:
            best_cr = cr
            best_pts = pts.copy()
    
    # Phase 2: Final polish with local search
    if best_pts is not None and _time_remaining() > 15.0:
        # Additional Lloyd's iterations
        best_pts = _spherical_lloyd(best_pts, n_iters=20)
        cr = _covering_radius(best_pts)
        if cr < best_cr:
            best_cr = cr
        
        # Final facet shrink
        best_pts = _facet_shrink_polish(best_pts, n_iters=50)
        cr = _covering_radius(best_pts)
        if cr < best_cr:
            best_cr = cr
    
    # Ensure valid output
    if best_pts is None:
        best_pts = _fibonacci_points()
    
    best_pts = _normalize(best_pts)
    
    # Final validation
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10):
        best_pts = _fibonacci_points()
    
    # Check not in closed hemisphere
    try:
        hull = ConvexHull(best_pts)
        inside = True
        for simplex in hull.simplices:
            tri = best_pts[simplex]
            n = np.cross(tri[1] - tri[0], tri[2] - tri[0])
            n_norm = np.linalg.norm(n)
            if n_norm > 1e-15:
                n = n / n_norm
                if np.dot(n, tri[0]) < -1e-10:
                    inside = False
                    break
        if not inside:
            best_pts = _fibonacci_points()
    except Exception:
        best_pts = _fibonacci_points()
    
    return best_pts
