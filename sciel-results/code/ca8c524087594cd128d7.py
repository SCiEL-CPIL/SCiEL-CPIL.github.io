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

def _covering_radius_and_facets(pts):
    """
    Compute the covering radius and identify the facet(s) contributing to it.
    Returns (radius, list_of_critical_simplex_indices)
    """
    pts = _to_sphere(pts)
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi, [] # Fallback: infinite radius, no facets
    
    max_radius = 0.0
    critical_simplexes = []
    
    # Pre-compute circumradii for all facets
    radii = np.zeros(len(hull.simplices))
    
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
            
        x = cross / norm_cross
        
        # Determine correct circumcenter direction
        n = np.cross(b - a, c - a)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-12:
            radii[idx] = 0.0
            continue
        n = n / n_norm
        if np.dot(n, a) < 0:
            n = -n
            
        if np.dot(x, n) > 0:
            cc = x
        else:
            cc = -x
            
        dot_a = np.clip(np.dot(cc, a), -1.0, 1.0)
        r = np.arccos(dot_a)
        radii[idx] = r
        
        if r > max_radius:
            max_radius = r
            critical_simplexes = [idx]
        elif abs(r - max_radius) < 1e-6:
            critical_simplexes.append(idx)
            
    # Also check antipodes of points (these are also "facets" in a sense, or rather vertices of the dual)
    # Actually, the covering radius is max(facet circumradii, distance to antipodes of points if points are isolated?)
    # No, the covering radius is the max distance from any y to nearest point.
    # The points on the sphere farthest from the set are the circumcenters of the Delaunay triangles.
    # So the facet circumradii cover it.
    
    return max_radius, critical_simplexes

def _objective(pts):
    """Compute the objective (covering radius in radians)."""
    r, _ = _covering_radius_and_facets(pts)
    return r

def _gradient_step(pts, critical_simplexes, step_size=0.01):
    """
    Perform a gradient step on the points involved in the critical facets.
    For a critical facet (a,b,c) with circumcenter cc, moving a, b, c towards cc
    reduces the circumradius.
    """
    pts = pts.copy()
    # Initialize gradients
    grad = np.zeros_like(pts)
    
    for idx in critical_simplexes:
        simplex = pts.shape[0] and [] # Placeholder, need actual simplex
        # We need the actual simplex indices from the hull, which we don't have here.
        # This function needs to be refactored to take hull or recompute.
        # For efficiency, we should pass the hull or recompute inside.
        pass
        
    return pts

def _local_optimize_critical(pts, max_time=10.0):
    """
    Local optimization by targeting critical facets.
    """
    pts = _to_sphere(pts.copy())
    current_obj = _objective(pts)
    best_pts = pts.copy()
    best_obj = current_obj
    
    start = time.time()
    step_size = 0.05
    
    while time.time() - start < max_time:
        r, critical_idxs = _covering_radius_and_facets(pts)
        if not critical_idxs:
            break
            
        # Get the actual simplex indices
        # We need the hull to get the simplexes. 
        # Let's restructure: compute hull once, get radii, find criticals.
        # Then move points.
        
        try:
            hull = ConvexHull(pts)
        except:
            break
            
        # Identify critical simplexes
        max_r = 0
        crit_simplexes = []
        for idx, simplex in enumerate(hull.simplices):
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            nc = np.linalg.norm(cross)
            if nc < 1e-12: continue
            x = cross / nc
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n = n / nn
            if np.dot(n, a) < 0: n = -n
            
            cc = x if np.dot(x, n) > 0 else -x
            r = np.arccos(np.clip(np.dot(cc, a), -1.0, 1.0))
            
            if r > max_r:
                max_r = r
                crit_simplexes = [simplex]
            elif abs(r - max_r) < 1e-5:
                crit_simplexes.append(simplex)
                
        if not crit_simplexes:
            break
            
        # Compute movement direction for each point
        # Move each vertex of the critical triangles towards the circumcenter
        move = np.zeros_like(pts)
        for simplex in crit_simplexes:
            i, j, k = simplex
            a = pts[i]
            b = pts[j]
            c = pts[k]
            
            v1 = a - b
            v2 = a - c
            cross = np.cross(v1, v2)
            nc = np.linalg.norm(cross)
            if nc < 1e-12: continue
            x = cross / nc
            
            n = np.cross(b - a, c - a)
            nn = np.linalg.norm(n)
            if nn < 1e-12: continue
            n = n / nn
            if np.dot(n, a) < 0: n = -n
            
            cc = x if np.dot(x, n) > 0 else -x
            
            # Move a, b, c towards cc
            move[i] += (cc - a)
            move[j] += (cc - b)
            move[k] += (cc - c)
            
        # Normalize movement
        move_norms = np.linalg.norm(move, axis=1, keepdims=True)
        move_norms[move_norms == 0] = 1.0
        move = move / move_norms
        
        new_pts = _to_sphere(pts + step_size * move)
        new_obj = _objective(new_pts)
        
        if new_obj < current_obj:
            pts = new_pts
            current_obj = new_obj
            if current_obj < best_obj:
                best_obj = current_obj
                best_pts = pts.copy()
            # Decay step size slightly
            step_size *= 0.95
        else:
            step_size *= 0.5
            if step_size < 1e-5:
                break
                
    return best_pts, best_obj

def construct():
    """
    Construct 50 points on the unit sphere minimizing the covering radius.
    """
    # Start with a good initial configuration (Fibonacci spiral)
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _to_sphere(pts)
    
    best_pts = pts.copy()
    best_obj = _objective(pts)
    
    deadline = _START_TIME + 145.0
    
    # Phase 1: Critical Facet Local Optimization
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize_critical(pts, max_time=min(remaining - 5, 30))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
    # Phase 2: Simulated Annealing to escape local minima
    remaining = deadline - time.time()
    if remaining > 10:
        np.random.seed(42)
        for restart in range(3):
            if time.time() > deadline - 10:
                break
                
            # Perturb
            perturbed = _to_sphere(best_pts + np.random.randn(N, 3) * 0.2)
            
            # SA
            current_obj = _objective(perturbed)
            sa_pts = perturbed.copy()
            temp = 0.05
            start = time.time()
            while time.time() - start < 15:
                noise = np.random.randn(N, 3) * 0.05
                new_pts = _to_sphere(sa_pts + noise)
                new_obj = _objective(new_pts)
                delta = new_obj - current_obj
                if delta < 0 or np.random.random() < np.exp(-delta / temp):
                    sa_pts = new_pts
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_obj = current_obj
                        best_pts = sa_pts.copy()
                temp *= 0.999
                if temp < 0.001:
                    break
                    
            # Refine SA result with Critical Facet Optimization
            remaining = deadline - time.time()
            if remaining > 5:
                pts, obj = _local_optimize_critical(sa_pts, max_time=min(remaining - 5, 15))
                if obj < best_obj:
                    best_obj = obj
                    best_pts = pts.copy()
                    
    # Final Polish
    remaining = deadline - time.time()
    if remaining > 5:
        pts, obj = _local_optimize_critical(best_pts, max_time=min(remaining - 5, 10))
        if obj < best_obj:
            best_obj = obj
            best_pts = pts.copy()
            
    # Final Validation and Normalization
    best_pts = _to_sphere(best_pts)
    
    # Check for coincident points
    dots = best_pts @ best_pts.T
    np.fill_diagonal(dots, 0)
    if np.max(np.abs(dots)) > 0.999999:
        best_pts = _to_sphere(best_pts + np.random.randn(N, 3) * 0.01)
        
    return best_pts
# EVOLVE-BLOCK-END
