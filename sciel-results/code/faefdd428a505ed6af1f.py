"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree

def _covering_radius(points):
    """Compute the covering radius in radians using the convex hull."""
    points = np.asarray(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # Fallback: use random sampling
        from scipy.spatial import cKDTree
        tree = cKDTree(points)
        rng = np.random.RandomState(42)
        n_samples = 10000
        samples = rng.randn(n_samples, 3)
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)
        dists, _ = tree.query(samples)
        # Convert Euclidean distance to angular distance for unit vectors
        # d = 2*sin(theta/2), so theta = 2*arcsin(d/2)
        return np.max(2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1)))
    
    max_r = 0.0
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            continue
        a, b, c = pts[0], pts[1], pts[2]
        # Compute circumcenter of the spherical triangle
        # The circumcenter x satisfies x·a = x·b = x·c
        # x is parallel to (b-a) × (c-a)
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            continue
        v = v / v_norm
        
        # Determine correct sign: circumcenter should be on the same side as origin
        # For a convex hull facet, the outward normal points away from origin
        # The circumcenter of the empty cap is in the direction of the outward normal
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        max_r = max(max_r, r)
    
    return max_r


def _fibonacci_points(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def _spherical_centroid(points):
    """Compute the spherical centroid (mean of unit vectors, then normalize)."""
    mean = np.mean(points, axis=0)
    norm = np.linalg.norm(mean)
    if norm < 1e-12:
        return points[0].copy()
    return mean / norm


def _lloyd_iteration(points, max_iters=2000, tol=1e-10, time_limit=None, start_time=None):
    """Spherical Lloyd's algorithm: move each point to the centroid of its Voronoi cell.
    
    The Voronoi cell on the sphere is determined by the convex hull.
    Each point moves to the centroid of its adjacent hull facets' circumcenters.
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # For each point, find adjacent facets and compute centroid of their circumcenters
        new_points = np.zeros_like(points)
        
        # Build adjacency: point index -> list of facet indices
        point_to_facets = [[] for _ in range(len(points))]
        for fi, simplex in enumerate(hull.simplices):
            for idx in simplex:
                point_to_facets[idx].append(fi)
        
        for i in range(len(points)):
            if not point_to_facets[i]:
                # Point is not on hull (shouldn't happen for convex position)
                new_points[i] = points[i]
                continue
            
            # Compute circumcenters of adjacent facets
            circumcenters = []
            for fi in point_to_facets[i]:
                simplex = hull.simplices[fi]
                pts = points[simplex]
                a, b, c = pts[0], pts[1], pts[2]
                v = np.cross(b - a, c - a)
                v_norm = np.linalg.norm(v)
                if v_norm < 1e-12:
                    continue
                v = v / v_norm
                dot_a = np.dot(v, a)
                if dot_a < 0:
                    v = -v
                circumcenters.append(v)
            
            if not circumcenters:
                new_points[i] = points[i]
                continue
            
            # Spherical centroid: average the circumcenter vectors and normalize
            centroid = np.mean(circumcenters, axis=0)
            centroid_norm = np.linalg.norm(centroid)
            if centroid_norm < 1e-12:
                new_points[i] = points[i]
            else:
                new_points[i] = centroid / centroid_norm
        
        # Check convergence
        diff = np.max(np.linalg.norm(new_points - points, axis=1))
        if diff < tol:
            points = new_points
            break
        
        points = new_points
        cr = _covering_radius(points)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    return best_points, best_cr


def _gradient_descent(points, max_iters=5000, lr_init=0.005, lr_min=1e-7, time_limit=None, start_time=None):
    """Gradient descent to minimize covering radius.
    
    The subgradient of the covering radius w.r.t. each point on the worst facet
    points away from the circumcenter (projected onto tangent plane).
    """
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms < 1e-15] = 1.0
    points = points / norms
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    lr = lr_init
    
    for it in range(max_iters):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        try:
            hull = ConvexHull(points)
        except Exception:
            break
        
        # Find worst facet
        worst_simplex = None
        worst_r = 0.0
        worst_center = None
        
        for simplex in hull.simplices:
            pts = points[simplex]
            if len(pts) != 3:
                continue
            a, b, c = pts[0], pts[1], pts[2]
            v = np.cross(b - a, c - a)
            v_norm = np.linalg.norm(v)
            if v_norm < 1e-12:
                continue
            v = v / v_norm
            dot_a = np.dot(v, a)
            if dot_a < 0:
                v = -v
                dot_a = -dot_a
            cos_r = np.clip(dot_a, -1.0, 1.0)
            r = np.arccos(cos_r)
            if r > worst_r:
                worst_r = r
                worst_simplex = simplex
                worst_center = v
        
        if worst_simplex is None:
            break
        
        # Compute step: move worst facet vertices away from circumcenter
        step = np.zeros_like(points)
        for idx in worst_simplex:
            p = points[idx]
            d = worst_center - p
            d_tangent = d - np.dot(d, p) * p
            step[idx] += d_tangent
        
        step_norm = np.linalg.norm(step)
        if step_norm < 1e-15:
            break
        step /= step_norm
        
        new_points = points + lr * step
        new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
        
        new_cr = _covering_radius(new_points)
        if new_cr < best_cr - 1e-12:
            best_cr = new_cr
            best_points = new_points.copy()
            points = new_points
            lr = max(lr * 0.995, lr_min)
        else:
            lr = max(lr * 0.95, lr_min)
            if lr < lr_min * 1.01:
                break
    
    return best_points, best_cr


def _perturb_and_refine(points, n_perturbs=10, time_limit=None, start_time=None):
    """Perturb random points and refine with local optimization."""
    points = np.array(points, dtype=float)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    rng = np.random.RandomState(42)
    
    for _ in range(n_perturbs):
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        
        # Perturb a few random points
        n_pert = rng.randint(1, 4)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise = rng.randn(3) * 0.05
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        # Refine with Lloyd + gradient descent
        refined, cr1 = _lloyd_iteration(perturbed, max_iters=500, time_limit=5, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        refined, cr2 = _gradient_descent(refined, max_iters=2000, time_limit=5, start_time=start_time)
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr


def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s limit
    
    # Start with Fibonacci spiral
    points = _fibonacci_points(N)
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Phase 1: Lloyd's algorithm from Fibonacci start
    remaining = deadline - time.time()
    if remaining > 20:
        points, cr = _lloyd_iteration(points, max_iters=3000, time_limit=remaining * 0.3, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 2: Gradient descent polish
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _gradient_descent(best_points, max_iters=5000, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 3: Perturb and refine to escape local minima
    remaining = deadline - time.time()
    if remaining > 10:
        points, cr = _perturb_and_refine(best_points, n_perturbs=8, time_limit=remaining * 0.5, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Phase 4: Final Lloyd + gradient polish
    remaining = deadline - time.time()
    if remaining > 5:
        points, cr = _lloyd_iteration(best_points, max_iters=2000, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
        remaining = deadline - time.time()
        if remaining > 3:
            points, cr = _gradient_descent(best_points, max_iters=3000, lr_init=0.002, time_limit=remaining * 0.8, start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
