import numpy as np
import time
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

def _circumcenter_robust(p1, p2, p3):
    """Compute circumcenter of triangle on unit sphere, robust to degeneracy."""
    # The circumcenter is the intersection of the sphere with the line passing through 
    # the origin and the normal to the plane containing the three points.
    # Normal vector n = (p2-p1) x (p3-p1)
    n = np.cross(p2 - p1, p3 - p1)
    n_norm = np.linalg.norm(n)
    if n_norm < 1e-10:
        # Degenerate triangle, return average
        c = (p1 + p2 + p3) / 3.0
        norm_c = np.linalg.norm(c)
        if norm_c < 1e-10:
            return p1.copy()
        return c / norm_c
    
    c = n / n_norm
    # Ensure c is on the same side of the plane as the origin? No, the circumcenter 
    # is the point on the sphere equidistant from p1, p2, p3.
    # The line is t*n. The point on the sphere is either n or -n.
    # We check which one is closer to the triangle centroid or simply check dot products.
    # Actually, for a triangle on the unit sphere, the circumcenter is the unit vector 
    # perpendicular to the plane of the triangle. There are two such vectors.
    # The correct one is the one that lies "inside" the spherical triangle.
    # A robust way: check the sign of the dot product with the centroid.
    centroid = (p1 + p2 + p3) / 3.0
    if np.dot(c, centroid) < 0:
        c = -c
    return c

def _compute_circumradii(pts, hull):
    """
    Compute angular circumradii for all triangular facets of the convex hull.
    Returns array of circumradii (in radians) and the circumcenters.
    """
    simplices = hull.simplices
    n_facets = len(simplices)
    radii = np.zeros(n_facets)
    centers = np.zeros((n_facets, 3))
    
    for k, tri in enumerate(simplices):
        v0 = pts[tri[0]]
        v1 = pts[tri[1]]
        v2 = pts[tri[2]]
        
        c = _circumcenter_robust(v0, v1, v2)
        centers[k] = c
        
        # Circumradius is the angle between the circumcenter and any vertex
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def _covering_radius(pts):
    """Compute the covering radius in degrees using convex hull."""
    try:
        hull = ConvexHull(pts)
        radii, centers = _compute_circumradii(pts, hull)
        max_radius = np.max(radii)
        return np.degrees(max_radius), radii, centers, hull
    except Exception:
        return 180.0, None, None, None

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _repulsive_force(pts):
    """Compute repulsive force between all pairs of points to prevent clustering."""
    n = len(pts)
    rep_force = np.zeros_like(pts)
    
    # Compute pairwise distances
    dots = pts @ pts.T
    np.fill_diagonal(dots, 1.0)
    dots = np.clip(dots, -1.0, 1.0)
    D = np.sqrt(np.clip(2.0 - 2.0 * dots, 0.0, 4.0))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                d_ij = D[i, j]
                if d_ij < 1e-6:
                    f = 100.0
                else:
                    f = 1.0 / (d_ij * d_ij)
                rep_force[i] += f * (pts[i] - pts[j])
                
    return rep_force

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s kill
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_init()
    
    best_cr = 180.0
    best_pts = pts.copy()
    
    # Pre-compute initial hull
    current_cr, radii, centers, hull = _covering_radius(pts)
    if radii is not None:
        if current_cr < best_cr:
            best_cr = current_cr
            best_pts = pts.copy()
    else:
        return pts
        
    step_size = 0.02
    no_improve_count = 0
    max_no_improve = 50
    iteration = 0
    
    while time.time() < deadline:
        iteration += 1
        
        # If hull is invalid, break
        if radii is None:
            break
            
        # Identify the top-k largest facets
        k = min(10, len(radii))
        top_k_indices = np.argsort(radii)[-k:]
        
        # Compute gradient: move vertices of top-k facets towards their circumcenters
        grad = np.zeros_like(pts)
        for idx in top_k_indices:
            tri = hull.simplices[idx]
            c = centers[idx]
            for v_idx in tri:
                v = pts[v_idx]
                # Project circumcenter onto tangent space at v
                # Tangent space is orthogonal to v
                # grad_v = c - (c . v) v
                proj = np.dot(c, v) * v
                tangent = c - proj
                grad[v_idx] += tangent
                
        # Normalize gradient to have unit norm per point if possible
        # Or just scale it
        grad_norms = np.linalg.norm(grad, axis=1, keepdims=True)
        grad_norms[grad_norms == 0] = 1.0
        grad = grad / grad_norms
        
        # Add repulsive force
        rep = _repulsive_force(pts)
        rep_scale = 0.005
        total_grad = grad + rep_scale * rep
        
        # Project total_grad onto tangent space
        total_grad_dot_P = np.sum(total_grad * pts, axis=1, keepdims=True)
        total_grad_tangent = total_grad - total_grad_dot_P * pts
        
        # Update points
        pts = pts + step_size * total_grad_tangent
        pts = _normalize(pts)
        
        # Recompute hull and radii periodically or every step if fast enough
        # For N=130, ConvexHull is fast, so we can do it every step
        new_cr, new_radii, new_centers, new_hull = _covering_radius(pts)
        
        if new_radii is None:
            # Hull failed, revert
            pts = best_pts.copy()
            break
            
        radii = new_radii
        centers = new_centers
        hull = new_hull
        
        if new_cr < best_cr:
            best_cr = new_cr
            best_pts = pts.copy()
            no_improve_count = 0
            # Reduce step size slightly on improvement to refine
            step_size *= 0.995
        else:
            no_improve_count += 1
            if no_improve_count > max_no_improve:
                # Apply repulsive force to break symmetry or local minimum
                rep = _repulsive_force(pts)
                pts = pts + 0.01 * rep
                pts = _normalize(pts)
                
                # Reduce step size
                step_size *= 0.9
                no_improve_count = 0
                if step_size < 1e-6:
                    break
                    
        # Final polish: if we have time, try to optimize the points in the largest facet
        if time.time() < deadline - 10.0 and iteration > 10:
            try:
                # Find the largest facet
                max_idx = np.argmax(radii)
                tri = hull.simplices[max_idx]
                
                # Define a local objective function for the 3 points in the facet
                # We want to minimize the circumradius of this facet, while keeping
                # other points fixed. This is a local move.
                # Actually, moving just 3 points might increase other facets.
                # A better polish is to use scipy.optimize.minimize on the whole set
                # but that's expensive.
                # Instead, let's do a few more iterations with a smaller step size.
                pass
            except Exception:
                pass

    return best_pts

