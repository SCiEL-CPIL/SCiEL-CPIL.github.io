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


def _facet_circumdata(points, hull):
    """Compute circumcenter and circumradius for each triangular facet."""
    facet_centers = []
    facet_radii = []
    for simplex in hull.simplices:
        pts = points[simplex]
        if len(pts) != 3:
            facet_centers.append(None)
            facet_radii.append(0.0)
            continue
        a, b, c = pts[0], pts[1], pts[2]
        v = np.cross(b - a, c - a)
        v_norm = np.linalg.norm(v)
        if v_norm < 1e-12:
            facet_centers.append(None)
            facet_radii.append(0.0)
            continue
        v = v / v_norm
        dot_a = np.dot(v, a)
        if dot_a < 0:
            v = -v
            dot_a = -dot_a
        cos_r = np.clip(dot_a, -1.0, 1.0)
        r = np.arccos(cos_r)
        facet_centers.append(v)
        facet_radii.append(r)
    return np.array(facet_radii, dtype=float), facet_centers


def _soft_max_gradient(points, hull, beta=50.0):
    """Compute the gradient of the soft-max of facet circumradii.
    
    soft_max(r_i) = (1/beta) * log(sum(exp(beta * r_i)))
    d(soft_max)/d(r_i) = exp(beta * r_i) / sum(exp(beta * r_j))
    
    Then chain rule: d(soft_max)/d(point_k) = sum_i w_i * d(r_i)/d(point_k)
    where d(r_i)/d(point_k) for facet i containing point k is the tangent component
    of (circumcenter_i - point_k) / sin(r_i) (derivative of arccos).
    """
    radii, centers = _facet_circumdata(points, hull)
    
    # Soft-max weights
    max_r = np.max(radii)
    exp_vals = np.exp(beta * (radii - max_r))
    weights = exp_vals / np.sum(exp_vals)
    
    # Gradient: for each point, accumulate contributions from facets it belongs to
    grad = np.zeros_like(points)
    
    for fi, simplex in enumerate(hull.simplices):
        w = weights[fi]
        if w < 1e-10:
            continue
        center = centers[fi]
        r = radii[fi]
        if center is None or r < 1e-10:
            continue
        
        # d(arccos(x))/dx = -1/sqrt(1-x^2) = -1/sin(arccos(x)) = -1/sin(r)
        # But we want to MINIMIZE r, so gradient points in direction that increases r
        # Actually, we want the gradient of the objective, so we move AGAINST it
        # d(r)/d(point) for point p on facet: the circumcenter moves when p moves
        # Approximation: moving p in direction d_p changes r by approximately
        # (center - p) projected tangent to sphere, dotted with d_p / sin(r)
        # More precisely, if we move p by epsilon*d_p (tangent), the new circumcenter
        # shifts, and dr ≈ (center - p)·d_p / (|center - p| * sin(r)) ... 
        # Simpler: the gradient of r w.r.t. p is the tangent projection of 
        # (center - p) normalized, scaled by 1/sin(r)
        sin_r = np.sin(r)
        if sin_r < 1e-10:
            continue
        
        for idx in simplex:
            p = points[idx]
            d = center - p
            d_tangent = d - np.dot(d, p) * p
            # Gradient contribution: this is d(r)/d(p) direction
            grad[idx] += w * d_tangent / sin_r
    
    return grad, np.max(radii)


def _gradient_descent(points, max_iters=5000, lr_init=0.005, lr_min=1e-7, time_limit=None, start_time=None, beta=50.0):
    """Gradient descent using soft-max surrogate for covering radius.
    
    Uses the soft-max of facet circumradii as a smooth surrogate, enabling
    stable gradient-based optimization.
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
        
        # Compute soft-max gradient
        grad, current_max_r = _soft_max_gradient(points, hull, beta=beta)
        
        # Normalize gradient
        grad_norm = np.linalg.norm(grad)
        if grad_norm < 1e-15:
            break
        grad /= grad_norm
        
        # Adaptive learning rate with line search
        best_lr = lr
        best_new_cr = best_cr
        best_new_points = points.copy()
        
        for lr_try in [lr, lr * 0.5, lr * 0.25, lr * 0.1]:
            new_points = points - lr_try * grad  # Move against gradient (minimize)
            new_points = new_points / np.linalg.norm(new_points, axis=1, keepdims=True)
            
            new_cr = _covering_radius(new_points)
            if new_cr < best_new_cr:
                best_new_cr = new_cr
                best_new_points = new_points.copy()
                best_lr = lr_try
        
        if best_new_cr < best_cr - 1e-12:
            best_cr = best_new_cr
            best_points = best_new_points.copy()
            points = best_new_points
            lr = best_lr * 1.05  # Slight increase if we made progress
            lr = min(lr, 0.02)
        else:
            lr = max(lr * 0.8, lr_min)
            if lr < lr_min * 1.01:
                break
        
        # Increase beta over time to sharpen the soft-max toward true max
        beta = min(beta * 1.02, 200.0)
    
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
        
        # Perturb a few random points with varying magnitudes
        n_pert = rng.randint(1, 5)
        indices = rng.choice(N, size=n_pert, replace=False)
        perturbed = points.copy()
        for idx in indices:
            noise_mag = rng.uniform(0.02, 0.1)
            noise = rng.randn(3) * noise_mag
            perturbed[idx] = points[idx] + noise
            perturbed[idx] /= np.linalg.norm(perturbed[idx])
        
        # Refine with Lloyd + gradient descent
        refined, cr1 = _lloyd_iteration(perturbed, max_iters=300, time_limit=3, start_time=start_time)
        if time_limit is not None and start_time is not None:
            if time.time() - start_time > time_limit:
                break
        refined, cr2 = _gradient_descent(refined, max_iters=1500, time_limit=4, start_time=start_time)
        
        if cr2 < best_cr:
            best_cr = cr2
            best_points = refined.copy()
    
    return best_points, best_cr


def _random_fibonacci(start_offset=0.0):
    """Generate Fibonacci spiral points with a phase offset for diversity."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + start_offset
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s limit
    
    # Try multiple diverse starts and keep the best
    starts = [_fibonacci_points(N), _random_fibonacci(0.5), _random_fibonacci(1.0)]
    rng = np.random.RandomState(123)
    for _ in range(2):
        random_pts = rng.randn(N, 3)
        random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        starts.append(random_pts)
    
    best_points = None
    best_cr = np.inf
    
    for start_pts in starts:
        if time.time() > deadline - 15:
            break
        
        # Lloyd's algorithm
        remaining = deadline - time.time()
        if remaining > 15:
            points, cr = _lloyd_iteration(start_pts, max_iters=1500, time_limit=min(remaining * 0.35, 20), start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
        
        # Gradient descent with soft-max
        remaining = deadline - time.time()
        if remaining > 10:
            points, cr = _gradient_descent(best_points, max_iters=3000, lr_init=0.004, time_limit=min(remaining * 0.4, 25), start_time=start_time)
            if cr < best_cr:
                best_cr = cr
                best_points = points.copy()
    
    # Phase: Perturb and refine to escape local minima
    remaining = deadline - time.time()
    if remaining > 15:
        points, cr = _perturb_and_refine(best_points, n_perturbs=12, time_limit=remaining * 0.5, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Final high-precision polish: Lloyd with tight tolerance
    remaining = deadline - time.time()
    if remaining > 8:
        points, cr = _lloyd_iteration(best_points, max_iters=3000, tol=1e-12, time_limit=remaining * 0.4, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Final gradient polish with small steps
    remaining = deadline - time.time()
    if remaining > 4:
        points, cr = _gradient_descent(best_points, max_iters=5000, lr_init=0.001, lr_min=1e-8, time_limit=remaining * 0.85, start_time=start_time)
        if cr < best_cr:
            best_cr = cr
            best_points = points.copy()
    
    # Normalize final result
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
