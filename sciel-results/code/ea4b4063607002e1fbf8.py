"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull


def _normalize(pts):
    """Normalize points to unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms


def _slerp(a, b, t):
    """Spherical linear interpolation between unit vectors a and b."""
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)
    cos_theta = np.clip(np.dot(a, b), -1.0, 1.0)
    theta = np.arccos(cos_theta)
    if theta < 1e-10:
        return a
    sin_theta = np.sin(theta)
    return (np.sin((1 - t) * theta) / sin_theta) * a + (np.sin(t * theta) / sin_theta) * b


def _compute_facet_data(pts):
    """
    Compute convex hull and for each facet: circumcenter direction, circumradius,
    and the three vertex indices.
    Returns: (hull, facet_normals, facet_radii, vertex_indices)
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None, None
    
    n_facets = len(hull.simplices)
    facet_normals = np.zeros((n_facets, 3))
    facet_radii = np.zeros(n_facets)
    vertex_indices = hull.simplices  # shape (n_facets, 3)
    
    for i, simplex in enumerate(hull.simplices):
        a, b, c = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
        
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            continue
        n = n / n_norm
        
        # Choose sign: circumcenter should be on the "inside" of the hull
        # The outward normal from ConvexHull points away from the interior
        # The circumcenter for covering is on the opposite side (toward interior)
        centroid = (a + b + c) / 3.0
        centroid_norm = np.linalg.norm(centroid)
        if centroid_norm < 1e-12:
            continue
        centroid = centroid / centroid_norm
        
        # For a convex hull of points on unit sphere, the circumcenter of a facet
        # (as seen from outside) should be on the same side as the centroid
        if np.dot(n, centroid) < 0:
            n = -n
        
        facet_normals[i] = n
        cos_R = np.clip(np.dot(n, a), -1.0, 1.0)
        facet_radii[i] = np.arccos(cos_R)
    
    return hull, facet_normals, facet_radii, vertex_indices


def _covering_radius(pts):
    """Compute the covering radius (max circumradius of hull facets)."""
    _, _, facet_radii, _ = _compute_facet_data(pts)
    if facet_radii is None or len(facet_radii) == 0:
        return np.pi
    return np.max(facet_radii)


def _covering_radius_degrees(pts):
    return np.degrees(_covering_radius(pts))


def _lloyd_iteration(pts, n_iters=30):
    """
    Lloyd's algorithm for sphere: replace each point with the centroid
    of its Voronoi cell (approximated by averaging neighbors within a radius).
    """
    pts = _normalize(pts.copy())
    
    for it in range(n_iters):
        new_pts = np.zeros_like(pts)
        for i in range(len(pts)):
            # Find neighbors within some angular distance
            # Use a simple approach: points whose dot product with pts[i] > threshold
            dots = np.dot(pts, pts[i])
            # Adaptive threshold: use the median distance to nearest neighbor
            dists = np.arccos(np.clip(dots, -1.0, 1.0))
            # Use a radius that captures roughly 1/N of the sphere area
            # For 50 points, each cell should have solid angle ~4*pi/50
            # Angular radius ~ sqrt(4*pi/50) ~ 0.5 radians
            radius = 0.6
            mask = dists < radius
            # Always include self
            mask[i] = True
            
            if np.sum(mask) == 0:
                new_pts[i] = pts[i]
            else:
                # Weighted average (Voronoi centroid approximation)
                # Points closer to pts[i] get more weight
                weights = np.exp(-dists[mask] ** 2 / (2 * 0.15 ** 2))
                weights /= np.sum(weights)
                new_pts[i] = np.sum(pts[mask] * weights[:, None], axis=0)
        
        new_pts = _normalize(new_pts)
        pts = new_pts
    
    return pts


def _gradient_optimize(pts, max_time=30.0):
    """
    Gradient-based optimization: for each facet with large circumradius,
    move the three vertices toward the circumcenter to reduce that facet's radius.
    Uses face-normal-based updates.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    learning_rate = 0.05
    n_outer = 0
    
    while time.time() - start_time < max_time:
        n_outer += 1
        hull, facet_normals, facet_radii, vertex_indices = _compute_facet_data(pts)
        if facet_radii is None:
            break
        
        # Adaptive learning rate: smaller as we converge
        lr = learning_rate * (0.95 ** n_outer)
        
        # Compute gradient for each point: sum of contributions from facets it belongs to
        grad = np.zeros_like(pts)
        
        for i in range(len(facet_radii)):
            R = facet_radii[i]
            # Only optimize facets that are "active" (larger than median)
            median_R = np.median(facet_radii)
            if R < median_R * 0.9:
                continue
            
            n = facet_normals[i]
            verts = vertex_indices[i]
            
            # The gradient of the circumradius with respect to each vertex
            # Moving vertex a toward n reduces the circumradius
            # Gradient direction for vertex a: (n - a) projected tangent to sphere
            for j in range(3):
                v = pts[verts[j]]
                # Tangent component of (n - v)
                tangent = n - np.dot(n, v) * v
                grad[verts[j]] += tangent * (R - median_R)
        
        # Normalize gradient and apply step
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms < 1e-12] = 1.0
        grad = grad / grad_norms
        
        # Apply update with slerp-like step
        new_pts = pts + lr * grad
        new_pts = _normalize(new_pts)
        
        # Check improvement
        new_rad = _covering_radius(new_pts)
        if new_rad < best_rad - 1e-12:
            best_rad = new_rad
            best_pts = new_pts.copy()
            pts = new_pts
        else:
            # Reduce learning rate further
            learning_rate *= 0.9
            if learning_rate < 1e-6:
                break
    
    return best_pts, best_rad


def _high_precision_polish(pts, max_time=20.0):
    """
    Final polish: coordinate descent with small slerp steps on each point.
    """
    start_time = time.time()
    pts = _normalize(pts.copy())
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    step_size = 0.02
    
    while time.time() - start_time < max_time:
        improved = False
        for idx in range(len(pts)):
            if time.time() - start_time > max_time:
                break
            
            orig = pts[idx].copy()
            best_local = orig
            best_local_rad = _covering_radius(pts)
            
            # Try small perturbations in random tangent directions
            for trial in range(20):
                if time.time() - start_time > max_time:
                    break
                
                # Random tangent direction
                rand_vec = np.random.randn(3)
                tangent = rand_vec - np.dot(rand_vec, orig) * orig
                tangent_norm = np.linalg.norm(tangent)
                if tangent_norm < 1e-12:
                    continue
                tangent /= tangent_norm
                
                # Move in both directions
                for sign in [1, -1]:
                    new_pt = orig + sign * step_size * tangent
                    new_pt = new_pt / np.linalg.norm(new_pt)
                    pts[idx] = new_pt
                    rad = _covering_radius(pts)
                    if rad < best_local_rad - 1e-12:
                        best_local_rad = rad
                        best_local = new_pt.copy()
                
                pts[idx] = best_local
            
            if best_local_rad < best_rad - 1e-12:
                best_rad = best_local_rad
                best_pts = pts.copy()
                improved = True
        
        if not improved:
            step_size *= 0.7
            if step_size < 1e-5:
                break
    
    return best_pts, best_rad


def _fibonacci_spiral(n):
    """Generate Fibonacci spiral points on the unit sphere."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)


def construct():
    # Deadline guard
    start_time = time.time()
    deadline = 140.0  # 10s safety margin
    
    np.random.seed(42)
    
    # Phase 1: Initialize with Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_rad = _covering_radius(pts)
    
    # Phase 2: Lloyd's algorithm for better initial distribution
    if time.time() < deadline - 60:
        pts = _lloyd_iteration(pts, n_iters=20)
        rad = _covering_radius(pts)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    # Phase 3: Gradient-based facet optimization
    if time.time() < deadline - 40:
        pts, rad = _gradient_optimize(pts, max_time=30.0)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    # Phase 4: High-precision polish
    if time.time() < deadline - 15:
        pts, rad = _high_precision_polish(best_pts, max_time=15.0)
        if rad < best_rad:
            best_rad = rad
            best_pts = pts.copy()
    
    return best_pts
# EVOLVE-BLOCK-END
