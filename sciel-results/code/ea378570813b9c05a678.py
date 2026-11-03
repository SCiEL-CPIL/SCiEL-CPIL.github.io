import numpy as np
import time
from scipy.spatial import ConvexHull

N = 130

def _fibonacci_init(seed=0):
    rng = np.random.RandomState(seed)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i + rng.uniform(0, 2*np.pi)
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])

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
        
        # Circumcenter of the triangle on the sphere is proportional to v0+v1+v2
        # This is exact for the plane circumcenter, and its normalization gives the sphere circumcenter
        # if the triangle is "small" enough that the spherical circumcenter is close to the planar one.
        # For a rigorous spherical circumcenter, we need to solve for c such that c.v0 = c.v1 = c.v2.
        # c is in the span of v0, v1, v2.
        # Actually, the normal to the plane of the triangle is n = (v1-v0) x (v2-v0).
        # The circumcenter lies on the line passing through the origin and the planar circumcenter? No.
        # The spherical circumcenter is the point c on the sphere equidistant from v0, v1, v2.
        # c is a linear combination a*v0 + b*v1 + c*v2.
        # Simpler: The circumcenter is in the direction of the normal to the plane? No.
        
        # Let's use the property that c is equidistant.
        # c . v0 = c . v1 = c . v2 = cos(R)
        # c = alpha * (v0 x v1 + v1 x v2 + v2 x v0) ? No.
        
        # Standard trick: The circumcenter of a triangle on the unit sphere is proportional to the sum of the vertices IF the triangle is small.
        # For larger triangles, this is an approximation.
        # Given N=130, the facets are small (radius ~ 11 deg). The approximation is likely sufficient for local search.
        
        c_vec = v0 + v1 + v2
        c_norm = np.linalg.norm(c_vec)
        if c_norm < 1e-10:
            # Degenerate case, skip or handle
            radii[k] = np.pi
            centers[k] = v0
            continue
            
        c = c_vec / c_norm
        centers[k] = c
        
        # Calculate angular distance from c to v0
        dot = np.clip(np.dot(c, v0), -1.0, 1.0)
        radii[k] = np.arccos(dot)
        
    return radii, centers

def construct():
    start_time = time.time()
    deadline = 115.0  # Safety margin
    
    # Initialize with Fibonacci spiral
    pts = _fibonacci_init()
    
    best_cr = 180.0
    best_pts = pts.copy()
    
    # Pre-compute initial hull
    try:
        hull = ConvexHull(pts)
    except Exception:
        return pts
        
    radii, centers = _compute_circumradii(pts, hull)
    current_cr = np.degrees(np.max(radii))
    
    if current_cr < best_cr:
        best_cr = current_cr
        best_pts = pts.copy()
        
    step_size = 0.01
    min_step = 1e-6
    no_improve_count = 0
    max_no_improve = 500
    
    while time.time() - start_time < deadline:
        # Find the worst facet
        max_idx = np.argmax(radii)
        max_rad = radii[max_idx]
        
        if max_rad < 1e-6:
            break
            
        tri = hull.simplices[max_idx]
        c = centers[max_idx]
        
        # Move vertices of the worst facet away from circumcenter
        # Direction: tangent vector from c to v_i
        # v_i_tangent = v_i - (v_i . c) c
        moves = np.zeros(3)
        for idx in tri:
            v = pts[idx]
            proj = np.dot(v, c) * c
            tangent = v - proj
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-10:
                moves += tangent / norm_t
        
        # Normalize move direction
        move_norm = np.linalg.norm(moves)
        if move_norm < 1e-10:
            # Try moving in random direction if stuck
            break
        moves = moves / move_norm
        
        # Apply move to the 3 vertices
        # We want to shrink the triangle, so we move vertices APART from the center of mass?
        # Actually, moving them away from the circumcenter c increases the distance between them?
        # Let's think: c is the center. v0, v1, v2 are on the sphere.
        # If we move v0 in the direction away from c, it moves "outward" relative to the triangle center.
        # This should increase the side lengths and thus the circumradius?
        # Wait. We want to DECREASE the circumradius.
        # The circumradius is determined by the size of the triangle.
        # To decrease the circumradius, we need to make the triangle SMALLER.
        # So we should move the vertices TOWARDS the circumcenter?
        # If we move v0 towards c, it gets closer to v1 and v2?
        # Let's test: Equilateral triangle. c is centroid. Moving v0 towards c shrinks the triangle.
        # So direction should be: c - v_i (tangent component).
        
        moves = np.zeros(3)
        for idx in tri:
            v = pts[idx]
            proj = np.dot(v, c) * c
            tangent = c - proj  # Direction towards c
            norm_t = np.linalg.norm(tangent)
            if norm_t > 1e-10:
                moves += tangent / norm_t
                
        move_norm = np.linalg.norm(moves)
        if move_norm < 1e-10:
            break
        moves = moves / move_norm
        
        # Apply step
        for idx in tri:
            pts[idx] += step_size * moves
            # Normalize
            pts[idx] /= np.linalg.norm(pts[idx])
            
        # Update hull and radii
        try:
            hull = ConvexHull(pts)
            radii, centers = _compute_circumradii(pts, hull)
            new_cr = np.degrees(np.max(radii))
            
            if new_cr < best_cr:
                best_cr = new_cr
                best_pts = pts.copy()
                no_improve_count = 0
            else:
                no_improve_count += 1
                # If no improvement, reduce step size or try different facet?
                # For now, just reduce step size if stuck
                if no_improve_count > 100:
                    step_size *= 0.9
                    if step_size < min_step:
                        break
        except Exception:
            # Hull failed, try to fix or break
            break
            
    return best_pts

construct()
