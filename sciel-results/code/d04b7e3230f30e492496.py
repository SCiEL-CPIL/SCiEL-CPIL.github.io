import numpy as np
from scipy.spatial import ConvexHull
import time
from scipy.optimize import minimize

N = 130

def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _fibonacci_spiral(n, seed=42):
    """Generate n points on unit sphere using Fibonacci spiral."""
    i = np.arange(n) + 0.5
    z = 1.0 - 2.0 * i / n
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    return _normalize(pts)

def _spherical_circumcenter(tri):
    """
    Compute the spherical circumcenter of a triangle on the unit sphere.
    tri: (3, 3) array of unit vectors.
    Returns unit vector circumcenter.
    """
    a, b, c = tri
    # Vector v = (a-b) x (b-c) is parallel to the spherical circumcenter
    v = np.cross(a - b, b - c)
    norm = np.linalg.norm(v)
    if norm < 1e-10:
        # Degenerate: points are collinear or coincident
        # Use centroid
        center = (a + b + c) / 3.0
        norm = np.linalg.norm(center)
        if norm < 1e-10:
            return a # Fallback
        return center / norm
    return v / norm

def _compute_hull_and_radii(pts):
    """
    Compute convex hull and circumradii for all facets.
    Returns hull, simplices, radii (array of length num_simplices).
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, None, None
        
    simplices = hull.simplices
    radii = np.zeros(len(simplices))
    
    for idx, simplex in enumerate(simplices):
        tri = pts[simplex]
        center = _spherical_circumcenter(tri)
        dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
        radii[idx] = np.arccos(dot)
        
    return hull, simplices, radii

def _get_max_radius(radii):
    return np.max(radii) if len(radii) > 0 else 0.0

def _lloyd_iteration_fast(pts, hull, simplices, radii):
    """
    Perform one Lloyd's algorithm iteration using precomputed hull data.
    Moves each point towards the circumcenter of its worst incident facet.
    Returns new_pts.
    """
    new_pts = pts.copy()
    
    # Map point index to list of simplex indices
    point_to_simplices = {i: [] for i in range(len(pts))}
    for idx, simplex in enumerate(simplices):
        for v in simplex:
            point_to_simplices[v].append(idx)
            
    for i in range(len(pts)):
        simplices_for_i = point_to_simplices[i]
        if not simplices_for_i:
            continue
            
        # Find the worst facet for this point (max radius)
        worst_s_idx = -1
        worst_radius = -1.0
        for s_idx in simplices_for_i:
            if radii[s_idx] > worst_radius:
                worst_radius = radii[s_idx]
                worst_s_idx = s_idx
                
        if worst_s_idx != -1:
            simplex = simplices[worst_s_idx]
            tri = pts[simplex]
            center = _spherical_circumcenter(tri)
            
            # Move towards the center
            # Step size 0.5
            diff = center - pts[i]
            tangent = diff - np.dot(diff, pts[i]) * pts[i]
            new_pts[i] = pts[i] + 0.5 * tangent
            new_pts[i] = new_pts[i] / np.linalg.norm(new_pts[i])
            
    return new_pts

def _polish_worst_facets(pts, hull, simplices, radii, deadline):
    """
    Polish the configuration by applying Nelder-Mead to the vertices
    of the worst facet, using fast radius updates.
    """
    current_pts = pts.copy()
    current_radii = radii.copy()
    best_radius = _get_max_radius(current_radii)
    
    while time.time() < deadline:
        # Find worst facet
        worst_idx = np.argmax(current_radii)
        worst_radius = current_radii[worst_idx]
        
        if worst_radius < 0.197: # ~11.3 degrees in radians, stop if we hit the known best approx
             break
             
        worst_facet = simplices[worst_idx]
        i, j, k = worst_facet
        
        # Define objective function
        def objective(params):
            # params: 6 floats (dx, dy, dz for i, j, k)
            p_i = current_pts[i] + params[0:3]
            p_j = current_pts[j] + params[3:6]
            p_k = current_pts[k] + params[6:9]
            
            # Normalize
            ni = np.linalg.norm(p_i)
            nj = np.linalg.norm(p_j)
            nk = np.linalg.norm(p_k)
            
            if ni < 1e-10 or nj < 1e-10 or nk < 1e-10:
                return np.pi
                
            p_i = p_i / ni
            p_j = p_j / nj
            p_k = p_k / nk
            
            # Update points in a temporary array
            temp_pts = current_pts.copy()
            temp_pts[i] = p_i
            temp_pts[j] = p_j
            temp_pts[k] = p_k
            
            # Update radii for affected facets
            # Affected facets are those containing i, j, or k
            changed = {i, j, k}
            temp_radii = current_radii.copy()
            
            # Recompute radii for facets involving i, j, k
            for idx, simplex in enumerate(simplices):
                if i in simplex or j in simplex or k in simplex:
                    tri = temp_pts[simplex]
                    center = _spherical_circumcenter(tri)
                    dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
                    temp_radii[idx] = np.arccos(dot)
            
            return _get_max_radius(temp_radii)
            
        # Initial params: 0
        x0 = np.zeros(9)
        
        # Run Nelder-Mead
        try:
            res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 50, 'xatol': 1e-5, 'fatol': 1e-5})
        except Exception:
            break
            
        if res.fun < best_radius:
            # Apply update
            current_pts[i] = (current_pts[i] + res.x[0:3])
            current_pts[j] = (current_pts[j] + res.x[3:6])
            current_pts[k] = (current_pts[k] + res.x[6:9])
            current_pts[i] = current_pts[i] / np.linalg.norm(current_pts[i])
            current_pts[j] = current_pts[j] / np.linalg.norm(current_pts[j])
            current_pts[k] = current_pts[k] / np.linalg.norm(current_pts[k])
            
            # Update radii
            changed = {i, j, k}
            current_radii = current_radii.copy()
            for idx, simplex in enumerate(simplices):
                if i in simplex or j in simplex or k in simplex:
                    tri = current_pts[simplex]
                    center = _spherical_circumcenter(tri)
                    dot = np.clip(np.dot(center, tri[0]), -1.0, 1.0)
                    current_radii[idx] = np.arccos(dot)
                    
            best_radius = _get_max_radius(current_radii)
        else:
            # If no improvement, break
            break
            
    return current_pts, current_radii

def construct():
    start_time = time.time()
    deadline = start_time + 140.0 # Safety margin
    
    # 1. Initialize
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    
    hull, simplices, radii = _compute_hull_and_radii(best_pts)
    if hull is None:
        return best_pts
        
    best_radius = _get_max_radius(radii)
    
    # 2. Lloyd's Algorithm Phase
    lloyd_deadline = start_time + 80.0
    no_improve_count = 0
    
    while time.time() < lloyd_deadline:
        new_pts = _lloyd_iteration_fast(best_pts, hull, simplices, radii)
        
        # Recompute hull and radii to ensure validity and handle topology changes
        new_hull, new_simplices, new_radii = _compute_hull_and_radii(new_pts)
        if new_hull is None:
             break
             
        new_radius = _get_max_radius(new_radii)
        
        if new_radius < best_radius:
            best_radius = new_radius
            best_pts = new_pts
            hull = new_hull
            simplices = new_simplices
            radii = new_radii
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 5:
                break
            # If no improvement, we might still want to try again with a smaller step?
            # For now, break to save time for polishing
            break
            
    # 3. Nelder-Mead Polishing Phase
    nm_deadline = deadline - 5.0
    polished_pts, polished_radii = _polish_worst_facets(best_pts, hull, simplices, radii, nm_deadline)
    polished_radius = _get_max_radius(polished_radii)
    
    if polished_radius < best_radius:
        best_pts = polished_pts
        best_radius = polished_radius
        
    return _normalize(best_pts)
