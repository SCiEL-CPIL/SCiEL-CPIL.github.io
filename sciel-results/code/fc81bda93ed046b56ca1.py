import numpy as np
from scipy.spatial import ConvexHull
import time

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

def _random_sphere(n, seed=0):
    """Generate n random points on the unit sphere."""
    np.random.seed(seed)
    pts = np.random.randn(n, 3)
    norms = np.linalg.norm(pts, axis=1, keepdims=True)
    norms[norms < 1e-10] = 1.0
    return pts / norms

def _circumcenter_robust(tri):
    """
    Compute the circumcenter of a spherical triangle (3 points on unit sphere).
    Returns the point on the unit sphere that is equidistant from all three vertices.
    This is the center of the circumscribed spherical cap.
    """
    a, b, c = tri
    # Use the formula for circumcenter of spherical triangle
    # The circumcenter is proportional to a*sin(A) + b*sin(B) + c*sin(C)
    # where A, B, C are the angles at vertices a, b, c
    
    # Compute side lengths (angular distances)
    def ang_dist(p, q):
        dot = np.clip(np.dot(p, q), -1.0, 1.0)
        return np.arccos(dot)
    
    A = ang_dist(b, c)  # angle at a
    B = ang_dist(a, c)  # angle at b
    C = ang_dist(a, b)  # angle at c
    
    # Circumcenter direction
    center = a * np.sin(A) + b * np.sin(B) + c * np.sin(C)
    norm = np.linalg.norm(center)
    if norm < 1e-10:
        # Degenerate case: use centroid
        center = (a + b + c) / 3.0
        norm = np.linalg.norm(center)
    return center / norm

def _covering_radius(pts):
    """
    Compute the covering radius of a set of points on the unit sphere.
    This is the maximum circumradius of all facets of the convex hull.
    """
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi  # Fallback: worst case
    
    max_radius = 0.0
    for simplex in hull.simplices:
        # Each simplex is a triangle (3 points)
        tri = pts[simplex]
        center = _circumcenter_robust(tri)
        
        # The circumradius is the angular distance from center to any vertex
        # Check if the cap is empty (center should be on the same side as the hull)
        # For a convex hull, the circumcenter should be outside the hull
        # The covering radius contribution is the angular distance from center to nearest vertex
        
        # Compute distances from center to each vertex
        dists = []
        for v in tri:
            dot = np.clip(np.dot(center, v), -1.0, 1.0)
            dists.append(np.arccos(dot))
        
        circumradius = max(dists)
        
        # Check if the cap is empty: the center should be such that no other point
        # is closer to the center than the circumradius
        # This is automatically satisfied for convex hull facets if the center is
        # on the correct side. We just take the circumradius.
        
        max_radius = max(max_radius, circumradius)
    
    return max_radius

def _repulsive_force(pts, k=1.0):
    """
    Compute repulsive forces between points on the sphere.
    Points repel each other with force proportional to 1/r^2 (Coulomb-like).
    The force is projected onto the tangent space of the sphere.
    """
    n = len(pts)
    forces = np.zeros_like(pts)
    
    for i in range(n):
        for j in range(i + 1, n):
            diff = pts[i] - pts[j]
            dist = np.linalg.norm(diff)
            if dist < 1e-10:
                # Points are coincident: add random perturbation
                rand_vec = np.random.randn(3)
                forces[i] += rand_vec
                forces[j] -= rand_vec
                continue
            
            # Coulomb force: F = k * diff / dist^3 (magnitude k/dist^2, direction diff/dist)
            force_mag = k / (dist * dist)
            force_dir = diff / dist
            force = force_mag * force_dir
            
            forces[i] += force
            forces[j] -= force
    
    # Project forces onto tangent space
    for i in range(n):
        p = pts[i]
        f = forces[i]
        # Tangent component: f - (f . p) * p
        forces[i] = f - np.dot(f, p) * p
    
    return forces

def _opti_cov(pts, deadline, verbose=False):
    """
    Optimize points to minimize the covering radius using a combination of:
    1. Repulsive force simulation (Lennard-Jones-like potential)
    2. Direct minimization of covering radius via active set method
    """
    current_pts = pts.copy()
    best_pts = pts.copy()
    best_radius = _covering_radius(current_pts)
    
    # Phase 1: Repulsive force simulation to get a good initial distribution
    step_size = 0.01
    no_improve_count = 0
    max_no_improve = 100
    
    while time.time() < deadline - 30.0:
        # Compute repulsive forces
        forces = _repulsive_force(current_pts, k=1.0)
        
        # Update positions
        new_pts = current_pts.copy()
        for idx in range(N):
            p = current_pts[idx]
            f = forces[idx]
            f_norm = np.linalg.norm(f)
            if f_norm > 1e-10:
                f = f / f_norm
            new_pts[idx] = p + step_size * f
            norm = np.linalg.norm(new_pts[idx])
            if norm < 1e-10:
                new_pts[idx] = p
            else:
                new_pts[idx] = new_pts[idx] / norm
        
        # Evaluate new configuration
        new_radius = _covering_radius(new_pts)
        
        if new_radius < best_radius - 1e-10:
            best_radius = new_radius
            best_pts = new_pts.copy()
            current_pts = new_pts.copy()
            no_improve_count = 0
            step_size = min(step_size * 0.99, 0.05)
        else:
            no_improve_count += 1
            step_size = max(step_size * 0.95, 0.001)
            
            if no_improve_count >= max_no_improve:
                # Try random perturbation
                if time.time() < deadline - 40.0:
                    current_pts = best_pts.copy()
                    n_perturb = max(5, N // 10)
                    idxs = np.random.choice(N, n_perturb, replace=False)
                    for idx in idxs:
                        p = current_pts[idx]
                        rand_vec = np.random.randn(3)
                        tangent = rand_vec - np.dot(rand_vec, p) * p
                        tangent_norm = np.linalg.norm(tangent)
                        if tangent_norm > 1e-10:
                            current_pts[idx] = p + 0.05 * tangent / tangent_norm
                            current_pts[idx] = current_pts[idx] / np.linalg.norm(current_pts[idx])
                    no_improve_count = 0
                    step_size = 0.01
                else:
                    break
    
    # Phase 2: Direct minimization of covering radius using active set
    # Find the facet with maximum circumradius and move points to reduce it
    while time.time() < deadline:
        try:
            hull = ConvexHull(current_pts)
        except Exception:
            break
        
        # Find the facet with maximum circumradius
        max_circumradius = 0.0
        max_facet_idx = -1
        max_facet_center = None
        max_facet_vertices = None
        
        for facet_idx, simplex in enumerate(hull.simplices):
            tri = current_pts[simplex]
            center = _circumcenter_robust(tri)
            
            # Compute circumradius
            dists = []
            for v in tri:
                dot = np.clip(np.dot(center, v), -1.0, 1.0)
                dists.append(np.arccos(dot))
            circumradius = max(dists)
            
            if circumradius > max_circumradius:
                max_circumradius = circumradius
                max_facet_idx = facet_idx
                max_facet_center = center
                max_facet_vertices = simplex
        
        if max_facet_idx == -1 or max_circumradius < 1e-10:
            break
        
        # Move the vertices of the worst facet towards the circumcenter
        # to reduce the circumradius
        for vertex_idx in max_facet_vertices:
            p = current_pts[vertex_idx]
            center = max_facet_center
            
            # Direction from vertex to center (projected to tangent space)
            diff = center - p
            diff_tangent = diff - np.dot(diff, p) * p
            diff_norm = np.linalg.norm(diff_tangent)
            
            if diff_norm > 1e-10:
                step = 0.01 * diff_tangent / diff_norm
                new_p = p + step
                new_norm = np.linalg.norm(new_p)
                if new_norm > 1e-10:
                    current_pts[vertex_idx] = new_p / new_norm
        
        # Evaluate
        new_radius = _covering_radius(current_pts)
        if new_radius < best_radius - 1e-10:
            best_radius = new_radius
            best_pts = current_pts.copy()
        else:
            # Revert if it didn't help
            current_pts = best_pts.copy()
            break
    
    return best_pts

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit
    
    # Try multiple initializations and pick the best
    best_overall = None
    best_overall_radius = np.inf
    
    # Initialization 1: Fibonacci spiral
    pts1 = _fibonacci_spiral(N, seed=42)
    r1 = _covering_radius(pts1)
    if r1 < best_overall_radius:
        best_overall_radius = r1
        best_overall = pts1.copy()
    
    # Initialization 2: Random
    pts2 = _random_sphere(N, seed=7)
    r2 = _covering_radius(pts2)
    if r2 < best_overall_radius:
        best_overall_radius = r2
        best_overall = pts2.copy()
    
    # Initialization 3: Fibonacci with different seed
    pts3 = _fibonacci_spiral(N, seed=123)
    r3 = _covering_radius(pts3)
    if r3 < best_overall_radius:
        best_overall_radius = r3
        best_overall = pts3.copy()
    
    # Optimize the best initialization
    try:
        optimized = _opti_cov(best_overall, deadline)
        if np.all(np.isfinite(optimized)):
            final_radius = _covering_radius(optimized)
            if final_radius < best_overall_radius:
                best_overall = optimized
                best_overall_radius = final_radius
    except Exception:
        pass
    
    return _normalize(best_overall)
