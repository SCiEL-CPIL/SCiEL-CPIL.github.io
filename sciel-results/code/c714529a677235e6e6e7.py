"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize, differential_evolution

N = 130
TIME_LIMIT = 115.0  # safety margin before 120s kill


def _covering_radius(points):
    """Compute the covering radius in degrees for points on the unit sphere."""
    points = np.asarray(points)
    if points.shape != (N, 3):
        return np.inf
    if not np.all(np.isfinite(points)):
        return np.inf
    # Check no coincident points
    norms = np.linalg.norm(points, axis=1)
    if np.any(norms < 1e-10):
        return np.inf
    points = points / norms[:, None]
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return np.inf
    
    # Get facets
    simplices = hull.simplices
    max_rad = 0.0
    for simplex in simplices:
        # For a spherical triangle (3 vertices on unit sphere),
        # the covering radius of the cap is determined by the circumradius
        # of the spherical triangle (the point farthest from all three vertices
        # within that facet)
        verts = points[simplex]  # shape (3, 3)
        
        # The center of the spherical triangle's circumscribed cap is the
        # normalized sum of vertices for small triangles, but more accurately:
        # We need the point on the sphere within this spherical triangle that
        # maximizes the minimum angular distance to the three vertices.
        
        # For a spherical triangle, the circumcenter (equidistant point) can be found
        # as the normalized cross product of edge vectors, or more robustly:
        # The pole of the plane through the three vertices gives a point equidistant
        # from all three (in the sense of the great-circle circumcircle).
        
        a, b, c = verts[0], verts[1], verts[2]
        
        # Compute the circumcenter of the spherical triangle
        # The circumcenter is in the direction of a×b + b×c + c×a (for small triangles)
        # More precisely, it's the normalized vector perpendicular to the plane
        # through a, b, c (if the triangle doesn't contain the antipode)
        
        # Plane normal
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            # Degenerate
            max_rad = np.max(np.arccos(np.clip(np.sum(points * verts, axis=1), -1, 1)))
            continue
        n = n / n_norm
        
        # Choose the side: the circumcenter should be on the same side as the origin
        # relative to the plane, i.e., dot(n, a) should give us the right sign
        # Actually, for a convex hull facet, the outward normal points away from origin.
        # The circumcenter of the spherical triangle (the point maximizing min dist to vertices)
        # is in the direction of -n if n points outward, or we need to check.
        
        # The covering radius for this facet is the maximum over the spherical triangle
        # of the minimum angular distance to the three vertices.
        # For a spherical triangle, this maximum occurs at the circumcenter
        # (if it lies inside the triangle) or at a vertex.
        
        # The angular distance from circumcenter to each vertex:
        # cos(angle) = dot(circumcenter, vertex)
        
        # The circumcenter direction is ±n. We pick the one closer to the vertices.
        if np.dot(n, a) > 0:
            circ_dir = n
        else:
            circ_dir = -n
        
        cos_dist = np.dot(circ_dir, verts)
        cos_dist = np.clip(cos_dist, -1, 1)
        ang_dist = np.arccos(cos_dist)
        
        # Check if circumcenter is inside the spherical triangle
        # For a convex hull, the circumcenter of the facet's spherical triangle
        # is the point we care about. The covering radius contribution is the max ang_dist.
        
        facet_rad = np.max(ang_dist)
        max_rad = max(max_rad, facet_rad)
    
    return np.degrees(max_rad)


def _objective_and_grad(x):
    """Objective: smooth surrogate for covering radius using log-sum-exp on facet circumradii."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1e6, np.zeros_like(x)
    
    simplices = hull.simplices
    facet_rads = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            facet_rads[idx] = np.pi / 2
            continue
        n = n / n_norm
        if np.dot(n, a) > 0:
            circ_dir = n
        else:
            circ_dir = -n
        cos_dist = np.clip(np.dot(circ_dir, verts), -1, 1)
        ang_dist = np.arccos(cos_dist)
        facet_rads[idx] = np.max(ang_dist)
    
    # Smooth max via log-sum-exp
    beta = 50.0
    shifted = beta * facet_rads
    shifted -= np.max(shifted)
    logsumexp = np.max(shifted) / beta + np.log(np.sum(np.exp(shifted - np.max(shifted)))) / beta
    
    # Gradient: approximate via finite differences is too slow.
    # Instead, we'll use a different approach: minimize sum of squared facet radii
    # which is differentiable.
    
    return np.sum(facet_rads ** 2), None


def _smooth_objective(x):
    """Smooth surrogate: sum of squared facet circumradii."""
    points = x.reshape(N, 3)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    try:
        hull = ConvexHull(points)
    except Exception:
        return 1e6
    
    simplices = hull.simplices
    total = 0.0
    for simplex in simplices:
        verts = points[simplex]
        a, b, c = verts[0], verts[1], verts[2]
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-15:
            total += (np.pi / 2) ** 2
            continue
        n = n / n_norm
        if np.dot(n, a) > 0:
            circ_dir = n
        else:
            circ_dir = -n
        cos_dist = np.clip(np.dot(circ_dir, verts), -1, 1)
        ang_dist = np.arccos(cos_dist)
        total += np.max(ang_dist) ** 2
    
    return total


def _gradient_approx(x, eps=1e-6):
    """Approximate gradient of smooth objective via central differences on a few random directions."""
    points = x.reshape(N, 3)
    f0 = _smooth_objective(x)
    grad = np.zeros_like(x)
    
    # Use a few random directions for gradient estimation (stochastic gradient)
    rng = np.random.default_rng(42)
    for _ in range(3):
        d = rng.standard_normal(x.shape)
        d /= np.linalg.norm(d)
        f_plus = _smooth_objective(x + eps * d)
        f_minus = _smooth_objective(x - eps * d)
        grad += ((f_plus - f_minus) / (2 * eps)) * d
    
    return grad / 3


def _project_to_sphere(points):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms = np.where(norms < 1e-10, 1.0, norms)
    return points / norms


def _local_optimize(points, time_budget):
    """Local optimization using projected gradient descent on smooth objective."""
    points = _project_to_sphere(points.copy())
    x = points.flatten()
    
    start_time = time.time()
    lr = 0.01
    best_x = x.copy()
    best_f = _smooth_objective(x)
    best_cr = _covering_radius(points)
    
    for iteration in range(10000):
        if time.time() - start_time > time_budget:
            break
        
        # Compute gradient
        grad = _gradient_approx(x, eps=1e-5)
        
        # Projected gradient step: move in direction that reduces objective
        x_new = x - lr * grad
        
        # Project back to sphere
        points_new = _project_to_sphere(x_new.reshape(N, 3))
        x_new = points_new.flatten()
        
        f_new = _smooth_objective(x_new)
        
        if f_new < best_f:
            best_x = x_new.copy()
            best_f = f_new
            best_cr = _covering_radius(points_new)
        
        # Adaptive learning rate
        if f_new < best_f:
            lr = min(lr * 1.1, 0.1)
        else:
            lr = max(lr * 0.7, 1e-4)
    
    return _project_to_sphere(best_x.reshape(N, 3)), best_cr


def construct():
    start_time = time.time()
    
    # Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_cr = _covering_radius(points)
    
    # Try multiple random restarts with local optimization
    rng = np.random.default_rng(12345)
    
    # First, do local optimization from the Fibonacci start
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 10:
        time_budget = min(20, TIME_LIMIT - elapsed - 5)
        opt_points, opt_cr = _local_optimize(points, time_budget)
        if opt_cr < best_cr:
            best_points = opt_points.copy()
            best_cr = opt_cr
    
    # Try a few random restarts
    for restart in range(5):
        elapsed = time.time() - start_time
        if elapsed > TIME_LIMIT - 15:
            break
        
        # Generate random initial points
        rand_points = rng.standard_normal((N, 3))
        rand_points = rand_points / np.linalg.norm(rand_points, axis=1, keepdims=True)
        
        time_budget = min(15, TIME_LIMIT - elapsed - 5)
        if time_budget <= 2:
            break
        
        opt_points, opt_cr = _local_optimize(rand_points, time_budget)
        if opt_cr < best_cr:
            best_points = opt_points.copy()
            best_cr = opt_cr
    
    # Final polish from best point
    elapsed = time.time() - start_time
    if elapsed < TIME_LIMIT - 5:
        time_budget = min(10, TIME_LIMIT - elapsed - 2)
        if time_budget > 2:
            opt_points, opt_cr = _local_optimize(best_points, time_budget)
            if opt_cr < best_cr:
                best_points = opt_points.copy()
                best_cr = opt_cr
    
    return _project_to_sphere(best_points)
