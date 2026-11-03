import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130
_DEADLINE = time.time() + 140  # 10s safety margin

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _time_ok(margin=5.0):
    return time.time() < _DEADLINE - margin

def _covering_radius(pts):
    """Compute the covering radius in radians via convex hull facet circumcenters."""
    pts = _normalize(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi / 2

    max_r = 0.0
    for simplex in hull.simplices:
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        r = np.arccos(dot_val)
        if r > max_r:
            max_r = r
    return max_r

def _lloyd_step(pts, lr=1.0):
    """
    Lloyd's algorithm step for sphere covering.
    For each point, find the adjacent Delaunay facets (hull simplices).
    Identify the facet with the largest circumradius.
    Move the point towards the circumcenter of that facet.
    """
    pts = _normalize(pts.copy())
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts

    # Map point index to list of adjacent simplex indices
    point_to_simplices = [[] for _ in range(N)]
    for i, simplex in enumerate(hull.simplices):
        for p_idx in simplex:
            point_to_simplices[p_idx].append(i)

    # Pre-compute circumcenters and radii for all facets
    facet_centers = np.zeros((len(hull.simplices), 3))
    facet_radii = np.zeros(len(hull.simplices))
    
    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            facet_centers[i] = (a+b+c)/3.0
            facet_radii[i] = 0.0
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        facet_centers[i] = x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        facet_radii[i] = np.arccos(dot_val)

    new_pts = pts.copy()
    for i in range(N):
        if not point_to_simplices[i]:
            continue
        
        # Find the facet with the maximum radius among those adjacent to point i
        # This is the "worst" part of the Voronoi cell boundary
        max_r = -1.0
        best_center = None
        for s_idx in point_to_simplices[i]:
            if facet_radii[s_idx] > max_r:
                max_r = facet_radii[s_idx]
                best_center = facet_centers[s_idx]
        
        if best_center is not None:
            # Move towards the circumcenter
            # Use a step size that is proportional to the distance? Or fixed?
            # Fixed step is standard Lloyd.
            new_pts[i] = pts[i] + lr * (best_center - pts[i])
            
    return _normalize(new_pts)

def _smooth_max_gradient(pts, beta=20.0):
    """
    Compute gradient of the smooth max of facet circumradii.
    f(x) = (1/beta) * log(sum(exp(beta * r_i)))
    grad f = sum(exp(beta * r_i) * grad r_i) / sum(exp(beta * r_i))
    
    This focuses optimization effort on the largest facets.
    """
    pts = _normalize(pts.copy())
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.zeros_like(pts)

    n_facets = len(hull.simplices)
    if n_facets == 0:
        return np.zeros_like(pts)

    # Compute radii and weights
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        v1, v2 = b - a, c - a
        cross = np.cross(v1, v2)
        cnorm = np.linalg.norm(cross)
        if cnorm < 1e-12:
            radii[i] = 0.0
            centers[i] = (a+b+c)/3.0
            continue
        x = cross / cnorm
        centroid = (a + b + c) / 3.0
        if np.dot(x, centroid) < 0:
            x = -x
        centers[i] = x
        dot_val = np.clip(np.dot(x, a), -1.0, 1.0)
        radii[i] = np.arccos(dot_val)

    # Softmax weights
    # Subtract max for numerical stability
    max_r = np.max(radii)
    if max_r < 1e-6:
        return np.zeros_like(pts)
    
    exp_vals = np.exp(beta * (radii - max_r))
    weights = exp_vals / np.sum(exp_vals)

    # Gradient accumulation
    # grad r_i w.r.t pts[k] for k in simplex_i
    # r_i = arccos( x_i . a_i )
    # dr/dx = -1/sqrt(1-cos^2) * (a_i . dx) ... wait, x is dependent on pts.
    # x is the circumcenter. The derivative of the circumcenter w.r.t vertices is complex.
    # Approximation: The gradient of the distance from the circumcenter to the plane is related to moving vertices.
    # A simpler heuristic often used: Move vertices away from the circumcenter if the facet is "large".
    # But we want to MINIMIZE r.
    # If we move vertex a towards x, the triangle shrinks, and the circumradius might decrease?
    # Actually, for a fixed base, moving the apex towards the circumcenter (which is on the normal) 
    # tends to make the triangle more equilateral or smaller, reducing r.
    
    # Let's use the property that the gradient of the circumradius w.r.t a vertex points 
    # roughly from the vertex towards the circumcenter, weighted by the angle?
    # Let's try: grad_r_i w.r.t pts[k] = w_i * (centers[i] - pts[k])
    # This is the same direction as Lloyd's step, but weighted by the "soft max" of radii.
    # This means points adjacent to large facets get pulled more strongly towards those facet centers.
    
    grad = np.zeros_like(pts)
    for i in range(n_facets):
        if weights[i] < 1e-6:
            continue
        simplex = hull.simplices[i]
        center = centers[i]
        w = weights[i]
        for k in simplex:
            # The force is towards the center.
            # We subtract the gradient because we want to minimize.
            # Gradient of r w.r.t pts[k] is roughly proportional to (pts[k] - center)?
            # If we move pts[k] towards center, r decreases.
            # So gradient points from center to pts[k]?
            # Let's assume grad_r = (pts[k] - center) * some_factor
            # Then update: pts[k] -= lr * grad_r = pts[k] + lr * (center - pts[k])
            grad[k] += w * (pts[k] - center)
            
    return grad

def _gradient_descent(pts, lr=0.05, beta=20.0, max_iters=100):
    """
    Gradient descent on the smooth max of circumradii.
    """
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_r = _covering_radius(pts)
    
    for it in range(max_iters):
        if not _time_ok(2.0):
            break
            
        grad = _smooth_max_gradient(pts, beta=beta)
        
        # Normalize gradient?
        g_norm = np.linalg.norm(grad)
        if g_norm < 1e-8:
            break
            
        # Update
        new_pts = pts - lr * grad
        new_pts = _normalize(new_pts)
        
        r_new = _covering_radius(new_pts)
        if r_new < best_r:
            best_r = r_new
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce learning rate if we got worse
            lr *= 0.9
            if lr < 1e-4:
                break
            # Try a smaller step?
            # For now, just break or continue with smaller lr.
            # Let's try to recover by taking a smaller step.
            new_pts = pts - (lr * 0.5) * grad
            new_pts = _normalize(new_pts)
            r_new = _covering_radius(new_pts)
            if r_new < best_r:
                best_r = r_new
                best_pts = new_pts.copy()
                pts = new_pts
            else:
                break

    return best_pts, best_r

def _fibonacci_spiral(n, jitter=0.0, seed=0):
    """Fibonacci spiral on the unit sphere with optional jitter."""
    rng = np.random.RandomState(seed)
    i = np.arange(n) + 0.5
    if jitter > 0:
        i = i + rng.randn(n) * jitter
        i = np.clip(i, 0.5, n - 0.5)
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def construct():
    """
    1. Generate diverse initial configs.
    2. Run Lloyd's Algorithm (fast, structural).
    3. Run Smooth-Max Gradient Descent (precise, objective-aligned).
    4. Return the best result.
    """
    start_time = time.time()
    best_pts = None
    best_r = np.pi

    # Initial configs
    inits = []
    for seed in [42, 123, 7, 999, 31415, 271828, 161803, 111, 222, 333]:
        inits.append(_fibonacci_spiral(N, jitter=0.05, seed=seed))
    inits.append(_fibonacci_spiral(N, jitter=0.0, seed=0))
    rng = np.random.RandomState(555)
    inits.append(_normalize(rng.randn(N, 3)))

    # Sort by initial covering radius
    init_scores = []
    for p in inits:
        try:
            r = _covering_radius(p)
        except Exception:
            r = np.pi
        init_scores.append((r, p))
    init_scores.sort(key=lambda x: x[0])

    # Phase 1: Lloyd's Algorithm
    # Lloyd's is very fast if we vectorize the facet lookup, but even the loop is fast enough for 130 points.
    for r0, p0 in init_scores:
        if not _time_ok(10.0):
            break
            
        pts = p0.copy()
        # Run Lloyd's for a few iterations
        # Lloyd's converges quickly to a local minimum of the covering radius.
        for it in range(20):
            if not _time_ok(5.0):
                break
            pts = _lloyd_step(pts, lr=1.0)
            
        # Evaluate
        cr = _covering_radius(pts)
        if cr < best_r:
            best_r = cr
            best_pts = pts.copy()
            
    # Phase 2: Smooth-Max Gradient Descent
    # This phase fine-tunes the configuration to minimize the maximum facet radius.
    if best_pts is not None and _time_ok(10.0):
        best_pts, best_r = _gradient_descent(best_pts, lr=0.02, beta=20.0, max_iters=50)
        
    # Phase 3: Final Polish with more Lloyd steps
    if best_pts is not None and _time_ok(5.0):
        for it in range(10):
            if not _time_ok(2.0):
                break
            best_pts = _lloyd_step(best_pts, lr=0.5)
            cr = _covering_radius(best_pts)
            if cr < best_r:
                best_r = cr
            else:
                break

    if best_pts is None:
        best_pts = _fibonacci_spiral(N)
        
    return _normalize(best_pts)

# Test
if __name__ == "__main__":
    t0 = time.time()
    pts = construct()
    print("Shape:", pts.shape)
    print("Covering radius (deg):", np.degrees(_covering_radius(pts)))
    print("Time: %.1f s" % (time.time() - t0))
