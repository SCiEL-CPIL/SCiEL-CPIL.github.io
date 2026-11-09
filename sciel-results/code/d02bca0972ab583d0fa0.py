import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def _covering_radius(points):
    """
    Compute the covering radius of the points on the sphere.
    Returns the radius in radians.
    """
    if len(points) < 3:
        return np.pi # Invalid or trivial
    
    try:
        hull = ConvexHull(points)
    except Exception:
        # If hull fails, fall back to max distance to nearest neighbor?
        # Or just return a large value
        return np.pi
    
    max_rad = 0.0
    # Facets are indices of vertices
    for face in hull.simplices:
        u = points[face[0]]
        v = points[face[1]]
        w = points[face[2]]
        
        # Normal of the face (outward)
        n = np.cross(v - u, w - u)
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-10:
            continue
        n = n / n_norm
        
        # Circumcenter candidate
        c = np.cross(u - v, v - w)
        c_norm = np.linalg.norm(c)
        if c_norm < 1e-10:
            continue
        c = c / c_norm
        
        # Ensure c is on the same side as the outward normal?
        # Actually, for a convex polytope containing the origin, the facet normal points out.
        # The circumcenter of the spherical triangle should be the one "closer" to the triangle interior.
        # The vector c is perpendicular to the chords.
        # Let's check the dot product with the normal.
        # If the triangle is small, c is roughly in the direction of the centroid.
        # The centroid g = (u+v+w)/3.
        # c dot g should be positive.
        # Alternatively, c dot n should be positive?
        # Let's test with a simple triangle in the x-y plane near z=1.
        # u=(1,0,0), v=(0,1,0), w=(0,0,1).
        # n = (v-u)x(w-u) = (-1,1,0)x(-1,0,1) = (1,1,1).
        # c = (u-v)x(v-w) = (1,-1,0)x(0,1,-1) = (1,1,1).
        # c dot n = 3 > 0.
        # So yes, c should have positive dot product with n.
        
        if np.dot(c, n) < 0:
            c = -c
            
        # Distance from c to u (or v or w)
        # d = arccos(c dot u)
        dot_val = np.dot(c, u)
        dot_val = np.clip(dot_val, -1.0, 1.0)
        rad = np.arccos(dot_val)
        
        if rad > max_rad:
            max_rad = rad
            
    return max_rad

def construct():
    # Initial Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_rad = _covering_radius(best_points)
    
    start_time = time.time()
    timeout = 110.0 # Safety margin
    
    step_size = 0.05
    stagnation_counter = 0
    max_stagnation = 500
    
    while time.time() - start_time < timeout:
        # Find worst face
        try:
            hull = ConvexHull(points)
        except Exception:
            break
            
        worst_rad = 0.0
        worst_face_idx = -1
        worst_c = None
        worst_u, worst_v, worst_w = None, None, None
        
        for idx, face in enumerate(hull.simplices):
            u = points[face[0]]
            v = points[face[1]]
            w = points[face[2]]
            
            n = np.cross(v - u, w - u)
            n_norm = np.linalg.norm(n)
            if n_norm < 1e-10:
                continue
            n = n / n_norm
            
            c = np.cross(u - v, v - w)
            c_norm = np.linalg.norm(c)
            if c_norm < 1e-10:
                continue
            c = c / c_norm
            
            if np.dot(c, n) < 0:
                c = -c
                
            dot_val = np.dot(c, u)
            dot_val = np.clip(dot_val, -1.0, 1.0)
            rad = np.arccos(dot_val)
            
            if rad > worst_rad:
                worst_rad = rad
                worst_face_idx = idx
                worst_c = c
                worst_u, worst_v, worst_w = u, v, w
                worst_face = face

        if worst_face_idx == -1:
            break
            
        # Select a point to move
        # Move the point farthest from the circumcenter
        d_u = np.arccos(np.clip(np.dot(worst_u, worst_c), -1.0, 1.0))
        d_v = np.arccos(np.clip(np.dot(worst_v, worst_c), -1.0, 1.0))
        d_w = np.arccos(np.clip(np.dot(worst_w, worst_c), -1.0, 1.0))
        
        dists = [d_u, d_v, d_w]
        move_idx = np.argmax(dists)
        
        orig_idx = worst_face[move_idx]
        orig_point = points[orig_idx].copy()
        
        # Move towards circumcenter
        # Spherical interpolation or simple vector addition
        # new_point = orig_point + step_size * (worst_c - orig_point)
        # This is a chordal move.
        new_point = orig_point + step_size * (worst_c - orig_point)
        
        # Normalize
        norm = np.linalg.norm(new_point)
        if norm < 1e-10:
            continue
        new_point = new_point / norm
        
        # Check if it's too close to other points (coincident)
        # Simple check: distance to nearest other point
        # If we move a point, we should ensure it doesn't jump over another.
        # For now, just try it.
        
        points[orig_idx] = new_point
        
        new_rad = _covering_radius(points)
        
        if new_rad < best_rad:
            best_rad = new_rad
            best_points = points.copy()
            stagnation_counter = 0
            # Maybe decrease step size slightly?
            # step_size *= 0.99
        else:
            # Revert
            points[orig_idx] = orig_point
            stagnation_counter += 1
            
            if stagnation_counter > max_stagnation:
                # Perturb randomly
                # Pick a random point and move it randomly
                rand_idx = np.random.randint(N)
                rand_dir = np.random.randn(3)
                rand_dir /= np.linalg.norm(rand_dir)
                points[rand_idx] = points[rand_idx] + 0.1 * rand_dir
                points[rand_idx] /= np.linalg.norm(points[rand_idx])
                
                # Check if this perturbation improved things?
                # Or just accept it as a jump.
                # Let's evaluate
                perturbed_rad = _covering_radius(points)
                if perturbed_rad < best_rad:
                    best_rad = perturbed_rad
                    best_points = points.copy()
                else:
                    # Revert to best
                    points = best_points.copy()
                    
                stagnation_counter = 0
                max_stagnation = np.random.randint(200, 500)
                
    return best_points
