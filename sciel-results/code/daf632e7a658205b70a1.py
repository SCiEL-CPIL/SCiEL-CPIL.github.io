"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0  # safety margin before 150s kill

def _to_sphere(pts):
    """Project points onto unit sphere."""
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return pts / norms

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradius for each facet.
    Returns:
        hull: ConvexHull object or None
        radii: array of circumradii for each facet
        max_radius: maximum circumradius
        critical_indices: indices of facets with circumradius close to max
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, np.pi, []
    
    radii = np.zeros(len(hull.simplices))
    max_radius = 0.0
    critical_indices = []
    
    for idx, simplex in enumerate(hull.simplices):
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        v1 = a - b
        v2 = a - c
        cross = np.cross(v1, v2)
        norm_cross = np.linalg.norm(cross)
        
        if norm_cross < 1e-12:
            radii[idx] = 0.0
            continue
            
        # Normal vector to the facet
        n = cross / norm_cross
        
        # Ensure normal points outward (away from origin)
        if np.dot(n, a) < 0:
            n = -n
            
        # The circumcenter of a spherical triangle on the unit sphere is the normalized normal vector of the plane containing the triangle.
        # The angular distance (circumradius) is arccos(dot(n, a)).
        d = np.dot(n, a)
        d = np.clip(d, -1.0, 1.0)
        r = np.arccos(d)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            critical_indices = [idx]
        elif abs(r - max_radius) < 1e-6:
            critical_indices.append(idx)
            
    return hull, radii, max_radius, critical_indices

def _spherical_voronoi_centroid_iteration(pts, max_iter=20, damping=1.0):
    """
    Perform Spherical Voronoi Centroid (SVC) iterations.
    Each point moves to the centroid of its Voronoi cell on the sphere.
    The Voronoi cell vertices are the circumcenters of the incident facets.
    """
    pts = _to_sphere(pts.copy())
    
    for _ in range(max_iter):
        if time.time() > _DEADLINE - 10:
            break
            
        hull, radii, max_r, _ = _compute_hull_and_radii(pts)
        if hull is None:
            break
            
        # Map point index to list of incident facets
        point_to_facets = {i: [] for i in range(N)}
        for idx, simplex in enumerate(hull.simplices):
            for p_idx in simplex:
                point_to_facets[p_idx].append(idx)
                
        new_pts = np.zeros_like(pts)
        for i in range(N):
            facets = point_to_facets[i]
            if not facets:
                new_pts[i] = pts[i]
                continue
                
            # Compute the spherical centroid of the Voronoi cell.
            # The Voronoi cell is a spherical polygon. Its vertices are the circumcenters of the incident facets.
            # We approximate the centroid by the normalized sum of the circumcenters (vertices).
            # A more accurate method would weight by edge lengths or areas, but unweighted vertex sum is a standard robust approximation.
            
            centroid_vec = np.zeros(3)
            for f_idx in facets:
                simplex = hull.simplices[f_idx]
                a = pts[simplex[0]]
                b = pts[simplex[1]]
                c = pts[simplex[2]]
                
                v1 = a - b
                v2 = a - c
                n = np.cross(v1, v2)
                nn = np.linalg.norm(n)
                if nn > 1e-12:
                    n = n / nn
                    if np.dot(n, a) < 0:
                        n = -n
                    centroid_vec += n
                    
            norm_c = np.linalg.norm(centroid_vec)
            if norm_c > 1e-12:
                # Interpolate between current point and centroid to prevent oscillation
                new_pts[i] = (1 - damping) * pts[i] + damping * (centroid_vec / norm_c)
            else:
                new_pts[i] = pts[i]
                
        new_pts = _to_sphere(new_pts)
        
        # Check for improvement in covering radius
        _, _, new_obj, _ = _compute_hull_and_radii(new_pts)
        
        if new_obj < max_r:
            pts = new_pts
        else:
            # If no improvement, try with smaller damping
            if damping > 0.1:
                # Retry with smaller damping in next iteration? Or just break?
                # For robustness, let's just break if we don't improve with current damping.
                # Or we could reduce damping and continue. Let's reduce damping.
                damping *= 0.8
            else:
                break
                
    return pts

def _refine_with_svc(pts):
    """
    Multi-stage SVC refinement with annealing.
    """
    best_pts = _to_sphere(pts.copy())
    best_obj = _compute_hull_and_radii(best_pts)[2]
    
    # Stage 1: Coarse optimization
    current_pts = best_pts.copy()
    for i in range(10):
        if time.time() > _DEADLINE - 20:
            break
        current_pts = _spherical_voronoi_centroid_iteration(current_pts, max_iter=20, damping=0.5)
        _, _, obj, _ = _compute_hull_and_radii(current_pts)
        if obj < best_obj:
            best_obj = obj
            best_pts = current_pts.copy()
        else:
            # If stuck, perturb slightly
            noise = np.random.randn(N, 3) * 0.01
            current_pts = _to_sphere(best_pts + noise)
            
    # Stage 2: Fine optimization
    current_pts = best_pts.copy()
    for i in range(20):
        if time.time() > _DEADLINE - 10:
            break
        current_pts = _spherical_voronoi_centroid_iteration(current_pts, max_iter=50, damping=0.2)
        _, _, obj, _ = _compute_hull_and_radii(current_pts)
        if obj < best_obj:
            best_obj = obj
            best_pts = current_pts.copy()
        else:
            # If stuck, perturb slightly
            noise = np.random.randn(N, 3) * 0.005
            current_pts = _to_sphere(best_pts + noise)
            
    return best_pts, best_obj

def _fibonacci_spiral(n, offset=0.0):
    """Generate points using Fibonacci spiral with an offset."""
    i = np.arange(n) + 0.5 + offset
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _to_sphere(pts)

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    np.random.seed(42)
    
    best_pts = None
    best_obj = np.pi
    
    # Define a set of diverse initializations
    # 1. Standard Fibonacci
    # 2. Rotated Fibonacci
    # 3. Perturbed Fibonacci
    # 4. Random points (for diversity, though SVC usually works best with good init)
    
    inits = []
    inits.append(_fibonacci_spiral(N, 0.0))
    inits.append(_fibonacci_spiral(N, 0.5))
    
    # Add a rotated version
    rot_angle = np.pi / 4
    rot_mat = np.array([
        [np.cos(rot_angle), -np.sin(rot_angle), 0],
        [np.sin(rot_angle), np.cos(rot_angle), 0],
        [0, 0, 1]
    ])
    inits.append(_to_sphere(_fibonacci_spiral(N, 0.0) @ rot_mat.T))
    
    # Add a perturbed Fibonacci
    inits.append(_to_sphere(_fibonacci_spiral(N, 0.0) + np.random.randn(N, 3) * 0.1))
    
    # Add another rotation
    rot_angle2 = np.pi / 6
    rot_mat2 = np.array([
        [1, 0, 0],
        [0, np.cos(rot_angle2), -np.sin(rot_angle2)],
        [0, np.sin(rot_angle2), np.cos(rot_angle2)]
    ])
    inits.append(_to_sphere(_fibonacci_spiral(N, 0.0) @ rot_mat2.T))
    
    # Process each initialization
    for init in inits:
        if time.time() > _DEADLINE - 30:
            break
            
        refined_pts, obj = _refine_with_svc(init)
        
        if obj < best_obj:
            best_obj = obj
            best_pts = refined_pts.copy()
            
    # If we didn't find any valid points (shouldn't happen), fallback to Fibonacci
    if best_pts is None:
        best_pts = _fibonacci_spiral(N, 0.0)
        
    # Final Validation
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        # Separate coincident points
        for i in range(N):
            for j in range(i+1, N):
                if np.dot(best_pts[i], best_pts[j]) > 0.999999:
                    # Move j slightly
                    best_pts[j] = _to_sphere(best_pts[j] + np.random.randn(3) * 0.01)
                    
    return best_pts
# EVOLVE-BLOCK-END
