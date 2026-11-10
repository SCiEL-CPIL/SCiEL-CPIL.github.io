"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50
import time

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull


def _circumcenter_of_triangle(a, b, c):
    """
    Compute the circumcenter of a spherical triangle with vertices a, b, c on the unit sphere.
    The circumcenter is the point on the sphere equidistant from a, b, c.
    It lies in the direction of the normal to the plane containing a, b, c, adjusted.
    
    For three points on the unit sphere, the circumcenter direction is:
    n = a x b x c (triple product direction), but we need the one on the same side.
    Actually, the circumcenter is in the direction of:
    (a * sin(2*angle_a) + b * sin(2*angle_b) + c * sin(2*angle_c))
    where angle_a is the angle at vertex a, etc.
    
    A simpler robust method: solve for the point p on the sphere such that
    |p-a| = |p-b| = |p-c|. This means p·a = p·b = p·c.
    So p is orthogonal to (a-b) and (a-c), i.e., p is parallel to (a-b) x (a-c).
    But that's the normal to the plane, not necessarily on the sphere in the right direction.
    
    Actually, p·a = p·b = p·c means p is in the null space of [a-b; a-c] (as row vectors),
    so p is parallel to (a-b) x (a-c). Let n = (a-b) x (a-c). Then p = ±n/|n|.
    We choose the sign so that p is on the same side of the triangle as the origin... 
    actually for the covering radius, we want the circumcenter that is farthest from the vertices,
    which is the one on the "outside" of the triangle.
    
    Let me use a more direct approach: the circumcenter of three points on the unit sphere
    can be found as follows. Let M = [a, b, c] as columns. The circumcenter p satisfies
    p·a = p·b = p·c = cos(r) for some r. So p is in the row space of [a-b, a-c]... 
    no, p is orthogonal to a-b and a-c, so p is parallel to (a-b) x (a-c).
    
    Let n = (a-b) x (a-c). Then p = n / |n| or p = -n / |n|.
    We pick the sign that gives the larger angular distance to the vertices (the "outer" circumcenter).
    """
    n = np.cross(a - b, a - c)
    norm_n = np.linalg.norm(n)
    if norm_n < 1e-15:
        # Degenerate triangle, return centroid
        centroid = (a + b + c) / 3.0
        norm_c = np.linalg.norm(centroid)
        if norm_c < 1e-15:
            return a
        return centroid / norm_c
    
    p1 = n / norm_n
    p2 = -p1
    
    # Choose the one that gives larger angular distance (smaller dot product)
    d1 = min(np.dot(p1, a), np.dot(p1, b), np.dot(p1, c))
    d2 = min(np.dot(p2, a), np.dot(p2, b), np.dot(p2, c))
    
    # We want the circumcenter that is farthest from the vertices (largest angle)
    # That corresponds to the smaller dot product
    if d1 < d2:
        return p1
    else:
        return p2


def _cover_radius_fast(pts):
    """
    Compute the covering radius using the convex hull facets.
    For each triangular facet, compute the circumcenter and the angular distance to the nearest vertex.
    The covering radius is the maximum over all facets.
    """
    if len(pts) < 3:
        return np.pi
    
    try:
        hull = ConvexHull(pts)
    except Exception:
        return np.pi
    
    max_angle = 0.0
    
    for simplex in hull.simplices:
        a = pts[simplex[0]]
        b = pts[simplex[1]]
        c = pts[simplex[2]]
        
        cc = _circumcenter_of_triangle(a, b, c)
        
        # Angular distance from circumcenter to nearest vertex
        dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
        min_dot = np.min(dots)
        min_dot = np.clip(min_dot, -1.0, 1.0)
        angle = np.arccos(min_dot)
        
        if angle > max_angle:
            max_angle = angle
    
    return max_angle


def _force_based_optimize(pts, max_iters=3000, initial_step=0.02):
    """
    Force-based optimization: for each point, compute the net force from the
    circumcenters of the facets it belongs to. Points are pushed away from
    circumcenters that are far from them.
    """
    pts = pts.copy()
    step = initial_step
    best_radius = _cover_radius_fast(pts)
    no_improve_count = 0
    
    for it in range(max_iters):
        try:
            hull = ConvexHull(pts)
        except Exception:
            break
        
        # For each facet, compute circumcenter and the angular distance
        # Build a force field: each point gets pushed away from the circumcenter
        # of facets it belongs to, proportional to how far the circumcenter is
        
        forces = np.zeros_like(pts)
        facet_radii = []
        
        for simplex in hull.simplices:
            a = pts[simplex[0]]
            b = pts[simplex[1]]
            c = pts[simplex[2]]
            
            cc = _circumcenter_of_triangle(a, b, c)
            
            dots = np.array([np.dot(cc, a), np.dot(cc, b), np.dot(cc, c)])
            min_dot = np.min(dots)
            min_dot = np.clip(min_dot, -1.0, 1.0)
            angle = np.arccos(min_dot)
            facet_radii.append(angle)
            
            # The force on each vertex of this facet is in the direction from cc to the vertex
            # Weighted by how much this facet contributes to the covering radius
            # Simpler: push each vertex away from cc if cc is far
            for idx in simplex:
                v = pts[idx]
                # Direction from cc to v
                dir_vec = v - cc
                dir_norm = np.linalg.norm(dir_vec)
                if dir_norm > 1e-15:
                    dir_vec = dir_vec / dir_norm
                    # Force magnitude proportional to the angle (worse facets push harder)
                    forces[idx] += angle * dir_vec
        
        # Normalize forces and apply
        force_norms = np.linalg.norm(forces, axis=1, keepdims=True)
        force_norms = np.where(force_norms < 1e-15, 1.0, force_norms)
        forces = forces / force_norms
        
        # Apply forces
        new_pts = pts + step * forces
        # Normalize back to unit sphere
        new_pts_norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
        new_pts_norms = np.where(new_pts_norms < 1e-15, 1.0, new_pts_norms)
        new_pts = new_pts / new_pts_norms
        
        new_radius = _cover_radius_fast(new_pts)
        
        if new_radius < best_radius - 1e-12:
            pts = new_pts
            best_radius = new_radius
            no_improve_count = 0
            # Slightly increase step if we're making good progress
            if it % 100 == 0 and step < initial_step:
                step = min(step * 1.05, initial_step)
        else:
            no_improve_count += 1
            step *= 0.95
            if step < 1e-7:
                break
        
        if no_improve_count > 50:
            break
    
    return pts, best_radius


def _fibonacci_spiral(N):
    """Generate points on the unit sphere using the Fibonacci spiral."""
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(np.maximum(0, 1.0 - z * z))
    return np.column_stack([r * np.cos(phi), r * np.sin(phi), z])


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Initial configuration: Fibonacci spiral
    pts = _fibonacci_spiral(N)
    best_pts = pts.copy()
    best_radius = _cover_radius_fast(best_pts)
    
    # Force-based optimization from Fibonacci start
    if time.time() < deadline - 20:
        optimized, opt_radius = _force_based_optimize(pts, max_iters=5000, initial_step=0.02)
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Try multiple random restarts
    rng = np.random.RandomState(42)
    for restart in range(20):
        if time.time() > deadline - 15:
            break
        
        # Generate a random initial configuration with some structure
        # Mix of random and Fibonacci-like
        if restart % 2 == 0:
            # Perturbed Fibonacci
            base = _fibonacci_spiral(N)
            noise = rng.randn(N, 3) * 0.1
            random_pts = base + noise
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        else:
            # Pure random
            random_pts = rng.randn(N, 3)
            random_pts /= np.linalg.norm(random_pts, axis=1, keepdims=True)
        
        remaining = deadline - 15 - time.time()
        if remaining < 5:
            break
        
        # Estimate iterations based on remaining time
        # Roughly 0.01s per iteration for force-based (with hull computation)
        est_iters = int(remaining * 80)
        est_iters = max(500, min(est_iters, 5000))
        
        optimized, opt_radius = _force_based_optimize(random_pts, max_iters=est_iters, initial_step=0.02)
        
        if opt_radius < best_radius:
            best_pts = optimized
            best_radius = opt_radius
    
    # Final check: ensure points are valid
    norms = np.linalg.norm(best_pts, axis=1)
    if np.any(norms < 1e-10) or not np.all(np.isfinite(best_pts)):
        # Fallback to Fibonacci
        best_pts = _fibonacci_spiral(N)
    
    # Normalize to ensure unit sphere
    best_pts = best_pts / np.linalg.norm(best_pts, axis=1, keepdims=True)
    
    return best_pts
# EVOLVE-BLOCK-END
