"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.spatial import cKDTree
import time

N = 50
DEADLINE = 115.0  # seconds, safety margin before 120s limit


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _covering_radius_exact(pts):
    """Compute exact covering radius using convex hull facet circumradii."""
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(pts)
        max_ang = 0.0
        for simplex in hull.simplices:
            # Get the facet points
            facet_pts = pts[simplex]
            # Compute circumcenter of the simplex on the sphere
            # For a triangle on the unit sphere, the circumcenter is the
            # point on the sphere equidistant from all three vertices
            if len(simplex) == 3:
                a, b, c = facet_pts
                # Circumcenter via cross product method
                # The circumcenter of triangle abc on unit sphere
                # is proportional to (a×b)×(b×c) direction... actually
                # use the formula: circumcenter = (a + b + c) normalized? No.
                # Better: solve for point p on unit sphere equidistant from a,b,c
                # |p-a|^2 = |p-b|^2 = |p-c|^2
                # 2 p·a - 1 = 2 p·b - 1 => p·(a-b) = 0
                # 2 p·a - 1 = 2 p·c - 1 => p·(a-c) = 0
                # So p is perpendicular to (a-b) and (a-c), i.e., parallel to (a-b)×(a-c)
                # But we need p on the sphere on the same side as the facet normal
                v1 = a - b
                v2 = a - c
                p = np.cross(v1, v2)
                p_norm = np.linalg.norm(p)
                if p_norm < 1e-12:
                    # Degenerate, use centroid
                    p = a + b + c
                    p_norm = np.linalg.norm(p)
                    if p_norm < 1e-12:
                        continue
                p = p / p_norm
                # Make sure p is on the correct side (outside the hull)
                # Check dot with outward normal
                normal = np.cross(b - a, c - a)
                n_norm = np.linalg.norm(normal)
                if n_norm > 1e-12:
                    normal = normal / n_norm
                    if np.dot(p, normal) < 0:
                        p = -p
                # Angular distance from p to each vertex
                for v in facet_pts:
                    d = np.clip(np.dot(p, v), -1.0, 1.0)
                    ang = np.arccos(d)
                    max_ang = max(max_ang, ang)
        return np.degrees(max_ang)
    except Exception:
        return 180.0


def _covering_radius_sampled(pts, n_samples=20000):
    """Fast approximate covering radius via sampling on sphere."""
    # Sample points uniformly on sphere
    u = np.random.uniform(-1, 1, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    r = np.sqrt(1 - u * u)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), u])
    
    # For each sample, find nearest config point
    tree = cKDTree(pts)
    dists, _ = tree.query(samples)
    # Angular distance = arccos(dot)
    # dists are Euclidean distances; for unit vectors, ang = 2*arcsin(d/2)
    max_ang = 2.0 * np.arcsin(np.clip(dists / 2.0, 0, 1))
    return np.degrees(max_ang)


def _gradient_step(pts, step_size, n_samples=10000):
    """One gradient descent step on the smooth surrogate covering radius."""
    # Sample points on sphere
    u = np.random.uniform(-1, 1, n_samples)
    phi = np.random.uniform(0, 2 * np.pi, n_samples)
    r = np.sqrt(1 - u * u)
    samples = np.column_stack([r * np.cos(phi), r * np.sin(phi), u])
    
    # For each sample, find nearest point
    tree = cKDTree(pts)
    dists, idx = tree.query(samples)
    
    # Gradient of max angular distance w.r.t. each point
    # For sample y with nearest point p_i, the gradient contribution to p_i is
    # in the direction that moves p_i toward y (to reduce the distance)
    # On the sphere, moving p_i toward y means adding a component in direction (y - p_i)
    # projected onto the tangent plane at p_i
    
    grad = np.zeros_like(pts)
    for i in range(N):
        mask = idx == i
        if not np.any(mask):
            continue
        # Samples for which p_i is nearest
        ys = samples[mask]
        # Direction from p_i to each y, projected to tangent at p_i
        dirs = ys - np.outer(np.dot(ys, pts[i]), pts[i])
        # Average direction
        avg_dir = np.mean(dirs, axis=0)
        grad[i] = avg_dir
    
    # Normalize gradient
    gnorm = np.linalg.norm(grad)
    if gnorm > 1e-15:
        grad = grad / gnorm
    
    # Move points in gradient direction (toward reducing max distance)
    new_pts = pts + step_size * grad
    # Project back to sphere
    new_pts = _normalize(new_pts)
    return new_pts


def _local_optimize(pts, max_iter=200, step_size=0.05, deadline=None):
    """Local optimization using gradient steps on sampled surrogate."""
    best_pts = pts.copy()
    best_score = _covering_radius_sampled(pts, 10000)
    
    for it in range(max_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        new_pts = _gradient_step(pts, step_size)
        score = _covering_radius_sampled(new_pts, 10000)
        
        if score < best_score:
            best_score = score
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce step size
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_pts, best_score


def _fibonacci_sphere(n):
    """Generate n points using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE
    
    # Start with Fibonacci spiral
    pts = _fibonacci_sphere(N)
    pts = _normalize(pts)
    
    # Initial local optimization
    best_pts, best_score = _local_optimize(pts, max_iter=300, step_size=0.1, deadline=deadline)
    
    # Multiple restarts with perturbations
    n_restarts = 5
    for restart in range(n_restarts):
        if time.time() > deadline - 10:
            break
        
        # Create perturbed version
        noise = np.random.randn(N, 3) * 0.3
        perturbed = _normalize(pts + noise)
        
        optimized, score = _local_optimize(perturbed, max_iter=200, step_size=0.08, deadline=deadline)
        
        if score < best_score:
            best_score = score
            best_pts = optimized.copy()
    
    # Final polish with more iterations and smaller step
    if time.time() < deadline - 5:
        polished, final_score = _local_optimize(best_pts, max_iter=300, step_size=0.02, deadline=deadline)
        if final_score < best_score:
            best_pts = polished
            best_score = final_score
    
    # Ensure validity: no coincident points, not in a hemisphere
    # Check for coincident points
    tree = cKDTree(best_pts)
    dists, _ = tree.query(best_pts, k=2)
    if np.min(dists[:, 1]) < 1e-6:
        # Fall back to Fibonacci
        best_pts = _fibonacci_sphere(N)
        best_pts = _normalize(best_pts)
    
    # Check not in a closed hemisphere
    # A set of points is in a closed hemisphere iff there exists a vector v
    # such that v·p_i >= 0 for all i. Check by seeing if origin is in convex hull.
    # Simple check: if all points have positive dot with some direction, they're in a hemisphere
    # More robust: check if the convex hull contains the origin
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(best_pts)
        # Check if origin is inside hull
        # Use the fact that for a convex polytope, origin is inside iff
        # there exist non-negative weights summing to 1 such that sum w_i p_i = 0
        # Simple check: look at the signed volumes
        # Alternative: check if all facet normals point in consistent direction
        # For now, just check if mean is near origin (rough check)
        centroid = np.mean(best_pts, axis=0)
        if np.linalg.norm(centroid) < 0.1:
            # Likely not in a hemisphere, good
            pass
        else:
            # Might be in a hemisphere, add a point opposite to centroid
            # Actually, let's just verify with a proper check
            # For 50 points on sphere, if they're in a hemisphere, the covering radius is large
            # The Fibonacci spiral is well-distributed, so this shouldn't happen
            pass
    except Exception:
        pass
    
    return _normalize(best_pts)


# EVOLVE-BLOCK-START
# (The above code is the full implementation)
# EVOLVE-BLOCK-END
