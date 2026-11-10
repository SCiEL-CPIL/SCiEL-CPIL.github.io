"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 130
import time
import numpy as np
from scipy.spatial import ConvexHull, Delaunay
from scipy.optimize import minimize

# EVOLVE-BLOCK-START
def _normalize(pts):
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    return pts / norms

def _covering_radius(pts):
    """Compute the covering radius in radians using convex hull facets."""
    if pts.shape[0] < 4:
        return np.pi
    try:
        # Find convex hull in 3D
        hull = ConvexHull(pts)
        facets = hull.simplices  # each row is a triangle (3 indices)
        
        # For each facet, find the circumcenter (pole) and its distance
        # The circumradius of a facet (triangle on unit sphere) is the angular
        # distance from the facet's circumcenter (pole) to any vertex
        # The covering radius is the max over all facets of this circumradius
        
        radii = []
        for tri in facets:
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            
            # Circumcenter of triangle abc on the sphere
            # For a triangle with vertices a,b,c on unit sphere, the circumcenter
            # is in the direction of a cross (b cross c) + b cross (c cross a) + c cross (a cross b)
            # More precisely, the pole direction is:
            pole = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            pn = np.linalg.norm(pole)
            if pn < 1e-10:
                # Degenerate - use centroid
                pole = (a + b + c) / 3.0
                pn = np.linalg.norm(pole)
                if pn < 1e-10:
                    continue
                pole = pole / pn
            else:
                pole = pole / pn
            
            # Angular distance from pole to vertex a
            cos_angle = np.clip(np.dot(pole, a), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            radii.append(angle)
        
        if not radii:
            return np.pi
        return max(radii)
    except Exception:
        return np.pi

def construct():
    start_time = time.time()
    DEADLINE = 140.0  # safety margin before 150s limit
    
    # Step 1: Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    pts = _normalize(pts)
    
    current_radius = _covering_radius(pts)
    
    # Step 2: Iterative local optimization - move points to reduce covering radius
    rng = np.random.default_rng(42)
    
    max_iterations = 5000
    step_size = 0.02
    
    for iteration in range(max_iterations):
        if time.time() - start_time > DEADLINE:
            break
        
        # Find the worst facet(s)
        try:
            hull = ConvexHull(pts)
            facets = hull.simplices
        except Exception:
            break
        
        # Compute radius for each facet and find the worst
        facet_radii = []
        facet_indices = []
        for idx, tri in enumerate(facets):
            a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
            pole = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            pn = np.linalg.norm(pole)
            if pn < 1e-10:
                continue
            pole = pole / pn
            cos_angle = np.clip(np.dot(pole, a), -1.0, 1.0)
            angle = np.arccos(cos_angle)
            facet_radii.append(angle)
            facet_indices.append(idx)
        
        if not facet_radii:
            break
        
        max_radius = max(facet_radii)
        
        # Move the points in the worst facet toward the pole to reduce the cap
        worst_idx = np.argmax(facet_radii)
        tri = facets[facet_indices[worst_idx]]
        a, b, c = pts[tri[0]].copy(), pts[tri[1]].copy(), pts[tri[2]].copy()
        
        pole = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
        pn = np.linalg.norm(pole)
        if pn < 1e-10:
            break
        pole = pole / pn
        
        # Move each vertex slightly toward the pole
        alpha = step_size * 0.1
        new_a = a + alpha * (pole - a)
        new_b = b + alpha * (pole - b)
        new_c = c + alpha * (pole - c)
        
        # Re-normalize
        new_a = new_a / np.linalg.norm(new_a)
        new_b = new_b / np.linalg.norm(new_b)
        new_c = new_c / np.linalg.norm(new_c)
        
        trial_pts = pts.copy()
        trial_pts[tri[0]] = new_a
        trial_pts[tri[1]] = new_b
        trial_pts[tri[2]] = new_c
        
        trial_radius = _covering_radius(trial_pts)
        
        if trial_radius < current_radius - 1e-12:
            pts = trial_pts
            current_radius = trial_radius
        else:
            # Try random perturbations of a few random points
            num_tries = 3
            improved = False
            for _ in range(num_tries):
                trial_pts = pts.copy()
                k = rng.integers(0, N)
                # Random perturbation
                pert = rng.normal(0, step_size * 0.05, 3)
                trial_pts[k] += pert
                trial_pts[k] /= np.linalg.norm(trial_pts[k])
                
                trial_radius = _covering_radius(trial_pts)
                if trial_radius < current_radius - 1e-12:
                    pts = trial_pts
                    current_radius = trial_radius
                    improved = True
                    break
            
            if not improved:
                # Reduce step size
                step_size *= 0.95
                if step_size < 1e-5:
                    break
    
    # Step 3: Final polish - more careful local optimization
    if time.time() - start_time < DEADLINE - 10:
        step_size = 0.001
        for iteration in range(2000):
            if time.time() - start_time > DEADLINE:
                break
            
            try:
                hull = ConvexHull(pts)
                facets = hull.simplices
            except Exception:
                break
            
            facet_radii = []
            facet_indices = []
            for idx, tri in enumerate(facets):
                a, b, c = pts[tri[0]], pts[tri[1]], pts[tri[2]]
                pole = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
                pn = np.linalg.norm(pole)
                if pn < 1e-10:
                    continue
                pole = pole / pn
                cos_angle = np.clip(np.dot(pole, a), -1.0, 1.0)
                angle = np.arccos(cos_angle)
                facet_radii.append(angle)
                facet_indices.append(idx)
            
            if not facet_radii:
                break
            
            max_radius = max(facet_radii)
            worst_idx = np.argmax(facet_radii)
            tri = facets[facet_indices[worst_idx]]
            a, b, c = pts[tri[0]].copy(), pts[tri[1]].copy(), pts[tri[2]].copy()
            
            pole = np.cross(a, b) + np.cross(b, c) + np.cross(c, a)
            pn = np.linalg.norm(pole)
            if pn < 1e-10:
                break
            pole = pole / pn
            
            alpha = step_size * 0.1
            new_a = a + alpha * (pole - a)
            new_b = b + alpha * (pole - b)
            new_c = c + alpha * (pole - c)
            
            new_a = new_a / np.linalg.norm(new_a)
            new_b = new_b / np.linalg.norm(new_b)
            new_c = new_c / np.linalg.norm(new_c)
            
            trial_pts = pts.copy()
            trial_pts[tri[0]] = new_a
            trial_pts[tri[1]] = new_b
            trial_pts[tri[2]] = new_c
            
            trial_radius = _covering_radius(trial_pts)
            
            if trial_radius < current_radius - 1e-12:
                pts = trial_pts
                current_radius = trial_radius
            else:
                step_size *= 0.95
                if step_size < 1e-6:
                    break
    
    return _normalize(pts)
# EVOLVE-BLOCK-END
