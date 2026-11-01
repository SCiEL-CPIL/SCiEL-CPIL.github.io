"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull, cKDTree

def construct():
    start_time = time.time()
    deadline = start_time + 118.0  # Safety margin for 120s limit
    
    # 1. Initialize with Fibonacci spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    def get_covering_info(pts):
        """
        Computes the convex hull, finds the facet with the largest circumradius,
        and returns the circumcenter of that facet (the worst-case point on the sphere).
        Also returns the covering radius (angular distance).
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            # Fallback if hull fails (should not happen for valid points)
            # Use a grid-based estimate as a rough fallback
            tree = cKDTree(pts)
            # Sample some points
            grid_lat = np.linspace(-np.pi/2, np.pi/2, 50)
            grid_lon = np.linspace(0, 2*np.pi, 100)
            lat, lon = np.meshgrid(grid_lat, grid_lon)
            gp = np.column_stack([
                (np.cos(lat)*np.cos(lon)).ravel(),
                (np.cos(lat)*np.sin(lon)).ravel(),
                (np.sin(lat)).ravel()
            ])
            dists, _ = tree.query(gp)
            angles = 2.0 * np.arcsin(np.clip(dists/2.0, 0, 1))
            return np.max(angles), gp[np.argmax(angles)]

        # For each facet, find the circumcenter
        # Facet indices are in hull.simplices
        # We want the facet with the largest circumradius.
        # Circumcenter of a triangle (a,b,c) on the unit sphere:
        # It is the point c such that |c-a| = |c-b| = |c-c| and c is on the sphere.
        # Actually, the circumcenter of the spherical triangle is the intersection of the sphere
        # and the plane passing through a, b, c? No, that's the origin if they are orthonormal.
        # The circumcenter of a facet defined by vertices v1, v2, v3 is the point on the sphere
        # equidistant from v1, v2, v3.
        # Let c be the circumcenter. c is in the span of v1, v2, v3.
        # c = alpha*v1 + beta*v2 + gamma*v3
        # |c-v1|^2 = |c-v2|^2 = |c-v3|^2
        # This is a linear system.
        
        # Alternative: The normal vector to the facet is n = cross(v1-v2, v3-v2).
        # The circumcenter lies on the line passing through the centroid? No.
        # The circumcenter is the point c on the sphere such that c dot (v1-v2) = 0 and c dot (v1-v3) = 0?
        # No, |c-v1|^2 = 2 - 2 c.dot(v1). So c.dot(v1) = c.dot(v2) = c.dot(v3) = k.
        # So c is the solution to c.dot(vi) = k for i=1,2,3.
        # This implies c is in the span of {v1, v2, v3}.
        # Let c = A x, where A = [v1, v2, v3] (3x3 matrix).
        # c.dot(v1) = x^T A^T v1 = x^T (A^T v1).
        # We want A^T c = k * ones.
        # A^T (A x) = k * ones => (A^T A) x = k * ones.
        # This doesn't fix k.
        
        # Simpler: The circumcenter c is the normalized vector:
        # c = normalize( v1/|v1-v2|^2 * (v1-v3) + ... ) ? No.
        
        # Let's use the property: c is the intersection of the sphere and the plane
        # perpendicular to the chord v1-v2 at its midpoint? No, that's for 2D.
        # In 3D, c is equidistant from v1, v2, v3.
        # c . v1 = c . v2 = c . v3.
        # This means c is orthogonal to (v1-v2) and (v1-v3).
        # So c is parallel to cross(v1-v2, v1-v3).
        # Wait, if c is parallel to the normal of the facet, does it lie on the sphere?
        # Yes, we just normalize it.
        # But is it equidistant?
        # If c is parallel to n = cross(v1-v2, v1-v3), then c . (v1-v2) = 0 and c . (v1-v3) = 0.
        # This implies c . v1 = c . v2 and c . v1 = c . v3.
        # So yes, the circumcenter is simply the normalized cross product of two edge vectors of the facet.
        
        max_radius = 0
        worst_center = None
        
        for simplex in hull.simplices:
            v1 = pts[simplex[0]]
            v2 = pts[simplex[1]]
            v3 = pts[simplex[2]]
            
            # Check if the triangle is degenerate
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
                
            c = n / norm_n
            # Ensure c is on the correct side (pointing outwards)
            # The centroid of the facet is (v1+v2+v3)/3
            # The normal should point away from the origin
            if np.dot(c, v1) < 0:
                c = -c
                
            # Calculate angular distance from c to v1 (or v2, v3)
            # d = 2 * arcsin(||c - v1|| / 2)
            d = np.linalg.norm(c - v1)
            radius = 2.0 * np.arcsin(np.clip(d / 2.0, 0.0, 1.0))
            
            if radius > max_radius:
                max_radius = radius
                worst_center = c
                
        if worst_center is None:
            # Fallback
            tree = cKDTree(pts)
            # Just pick a random point far from others?
            return 0.0, np.random.randn(3) / np.linalg.norm(np.random.randn(3))
            
        return max_radius, worst_center

    def calc_covering_radius(pts):
        radius, _ = get_covering_info(pts)
        return radius

    # Initialize
    current_points = points.copy()
    best_points = current_points.copy()
    best_radius = calc_covering_radius(current_points)
    
    # Optimization Loop: Repulsive Force / Lloyd's Algorithm
    # We move each point towards the circumcenter of its Voronoi cell.
    # However, computing Voronoi cells on the sphere is complex.
    # Approximation: Move each point towards the "worst" point? No, that collapses them.
    # Better: Use the gradient of the covering radius.
    # The covering radius is determined by the facet F* with circumcenter C*.
    # The vertices of F* are v1, v2, v3.
    # To reduce the radius, we should move v1, v2, v3 towards C*.
    # Moving them towards C* reduces the distance to C*.
    # What about other points? They might become the new worst case.
    # A common heuristic is to move ALL points slightly towards the global centroid? No.
    # Let's use a simple gradient step on the vertices of the worst facet.
    
    step_size = 0.1
    decay = 0.99
    
    for iter in range(1000):
        if time.time() > deadline - 10:
            break
            
        radius, worst_center = get_covering_info(current_points)
        
        if radius < best_radius:
            best_radius = radius
            best_points = current_points.copy()
            
        # Find the facet corresponding to worst_center
        # Re-compute hull to find which simplex has this circumcenter
        try:
            hull = ConvexHull(current_points)
        except:
            break
            
        target_indices = None
        for simplex in hull.simplices:
            v1 = current_points[simplex[0]]
            v2 = current_points[simplex[1]]
            v3 = current_points[simplex[2]]
            edge1 = v2 - v1
            edge2 = v3 - v1
            n = np.cross(edge1, edge2)
            norm_n = np.linalg.norm(n)
            if norm_n < 1e-10:
                continue
            c = n / norm_n
            if np.dot(c, v1) < 0:
                c = -c
            # Check if this is the worst center
            if np.dot(c, worst_center) > 0.9999:
                target_indices = simplex
                break
                
        if target_indices is None:
            # If we can't find the facet, break
            break
            
        # Move the vertices of the target facet towards worst_center
        # Use a small step
        for idx in target_indices:
            v = current_points[idx]
            # Direction to worst_center
            dir_vec = worst_center - v
            # Normalize direction
            norm_dir = np.linalg.norm(dir_vec)
            if norm_dir > 1e-10:
                dir_vec /= norm_dir
            # Move
            current_points[idx] = v + step_size * dir_vec
            # Normalize to sphere
            current_points[idx] /= np.linalg.norm(current_points[idx])
            
        # Also, to prevent points from clustering, add a small repulsive force
        # between all pairs? That's O(N^2) but N=50 is small.
        # Let's add a small repulsion to maintain spread.
        # Actually, the "worst facet" method tends to spread points naturally
        # because as you fix one hole, another appears.
        
        # Decay step size
        step_size *= decay
        
        # If step size is too small, restart with random perturbation?
        if step_size < 0.001:
            # Add small noise
            current_points += np.random.randn(N, 3) * 0.01
            current_points /= np.linalg.norm(current_points, axis=1, keepdims=True)
            step_size = 0.05

    # Final Polish: Coordinate Descent / Random Search around best
    if time.time() < deadline - 5:
        current_best = best_points.copy()
        current_score = best_radius
        step = 0.05
        
        for i in range(200):
            if time.time() > deadline - 5:
                break
            improved = False
            for j in range(N):
                if time.time() > deadline - 5:
                    break
                for _ in range(5):
                    rand_vec = np.random.randn(3)
                    rand_vec /= np.linalg.norm(rand_vec)
                    
                    old_p = current_best[j].copy()
                    new_p = current_best[j] + step * rand_vec
                    new_p /= np.linalg.norm(new_p)
                    
                    current_best[j] = new_p
                    new_score = calc_covering_radius(current_best)
                    
                    if new_score < current_score:
                        current_score = new_score
                        improved = True
                    else:
                        current_best[j] = old_p
                        
            if not improved:
                break
                
        if current_score < best_radius:
            best_points = current_best
            best_radius = current_score

    # Final normalization
    best_points = best_points / np.linalg.norm(best_points, axis=1, keepdims=True)
    
    return best_points
# EVOLVE-BLOCK-END
