"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial import ConvexHull

def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin for 150s limit
    
    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms < 1e-10] = 1.0
        return pts / norms

    def fibonacci_spiral(phase_offset=0.0, jitter=0.0):
        i = np.arange(N) + 0.5
        z = 1.0 - 2.0 * i / N
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i + phase_offset
        r = np.sqrt(1.0 - z * z)
        pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        if jitter > 0:
            # Add small random perturbations to break symmetry
            rng = np.random.default_rng(int(phase_offset * 1000))
            noise = rng.normal(0, jitter, pts.shape)
            pts += noise
            pts = normalize(pts)
        return pts

    def compute_covering_radius(pts):
        """
        Computes the exact covering radius using the convex hull.
        The covering radius is the maximum over all facets (triangles) of the 
        circumradius of the spherical triangle formed by the facet vertices.
        If the circumcenter is outside the triangle, the max distance is to the 
        midpoint of the longest edge.
        """
        try:
            hull = ConvexHull(pts)
        except Exception:
            return np.pi  # Invalid or degenerate hull, return large value
            
        simplices = hull.simplices
        n_facets = len(simplices)
        
        if n_facets == 0:
            return np.pi

        max_r = 0.0
        
        # Vectorized-ish loop for robustness, but N is small (50), facets ~ 2N
        for simplex in simplices:
            p1 = pts[simplex[0]]
            p2 = pts[simplex[1]]
            p3 = pts[simplex[2]]
            
            # Compute circumcenter of the spherical triangle
            # We can use the vector formula for the circumcenter of a spherical triangle
            # Or simply find the point y on the sphere equidistant from p1, p2, p3.
            # This is equivalent to finding the intersection of the sphere with the 
            # perpendicular bisector planes of the chords.
            
            # Let y be the circumcenter.
            # y . p1 = y . p2 = y . p3 = cos(r)
            # This implies y is in the span of p1, p2, p3.
            # y = a*p1 + b*p2 + c*p3
            # And |y|=1.
            
            # A simpler numerical approach for a single triangle:
            # The circumcenter is the normalized vector:
            # y = (p1 x p2) x (p2 x p3) + ... no, that's the normal.
            
            # Let's solve the linear system for the coefficients of the circumcenter
            # in the basis of the face normal? No, just use the property that
            # the circumcenter is the normalized vector:
            # y = (p1 x p2) x (p2 x p3) ... wait.
            
            # Actually, the circumcenter of a spherical triangle is the normalized 
            # vector:
            # y = (p1 x p2) x (p2 x p3) + (p2 x p3) x (p3 x p1) + (p3 x p1) x (p1 x p2)
            # This is the vector normal to the plane of the triangle, but scaled?
            # No, the vector (p1 x p2) x (p2 x p3) is parallel to the normal of the plane 
            # containing p1, p2, p3. The circumcenter is the intersection of this line 
            # with the sphere? NO. The circumcenter is NOT on the normal to the plane 
            # unless the triangle is isosceles.
            
            # Correct approach:
            # The circumcenter y satisfies:
            # y . p1 = y . p2 = y . p3
            # Let y = a*p1 + b*p2 + c*p3.
            # a + b*(p1.p2) + c*(p1.p3) = a*(p2.p1) + b + c*(p2.p3) = ...
            
            # Even simpler: Use the formula for the circumcenter of a triangle on the sphere.
            # It is the normalized vector:
            # y = (p1 x p2) x (p2 x p3) + (p2 x p3) x (p3 x p1) + (p3 x p1) x (p1 x p2)
            # Let's verify this.
            # Let n12 = p1 x p2, n23 = p2 x p3, n31 = p3 x p1.
            # The vector y = n12 x n23 + n23 x n31 + n31 x n12
            # This vector is perpendicular to p1, p2, p3? No.
            
            # Let's just use a small numerical solver for each facet, or a robust vectorized formula.
            # Formula:
            # y = ( (p2 x p3) x (p3 x p1) ) / ||...|| ?
            
            # Let's use the fact that the circumcenter is the normalized vector:
            # y = (p1 x p2) x (p2 x p3) + (p2 x p3) x (p3 x p1) + (p3 x p1) x (p1 x p2)
            # Actually, a known formula is:
            # y = (p1 x p2) x (p2 x p3) + ... is the normal.
            
            # Let's use the barycentric coordinates method.
            # The circumcenter in barycentric coordinates (alpha, beta, gamma) satisfies:
            # alpha * p1 + beta * p2 + gamma * p3 = y
            # And alpha, beta, gamma are proportional to sin(2*A), sin(2*B), sin(2*C)
            # where A, B, C are the angles of the spherical triangle.
            
            # This is getting complicated. Let's use a robust numerical method for the single triangle.
            # We want to minimize max(||y-p1||, ||y-p2||, ||y-p3||) on the sphere.
            # This is the Chebyshev center of the 3 points on the sphere.
            
            # For 3 points, the circumcenter is either the intersection of the perpendicular bisectors 
            # (which is the normalized vector n = (p1 x p2) x (p2 x p3) ... wait, the line through the origin 
            # perpendicular to the plane of the triangle intersects the sphere at two points. 
            # One of them is the circumcenter IF the triangle is acute? No.
            
            # The circumcenter of a spherical triangle is the point on the sphere equidistant from the vertices.
            # It lies on the line passing through the origin and the circumcenter of the planar triangle 
            # formed by the points? No.
            
            # Let's use the vector formula:
            # y = (p1 x p2) x (p2 x p3) + (p2 x p3) x (p3 x p1) + (p3 x p1) x (p1 x p2)
            # Let's test this mentally. If p1, p2, p3 are orthogonal (octant), p1=(1,0,0), p2=(0,1,0), p3=(0,0,1).
            # p1 x p2 = (0,0,1)
            # p2 x p3 = (1,0,0)
            # p3 x p1 = (0,1,0)
            # (0,0,1) x (1,0,0) = (0,1,0)
            # (1,0,0) x (0,1,0) = (0,0,1)
            # (0,1,0) x (0,0,1) = (1,0,0)
            # Sum = (1,1,1). Normalized = (1,1,1)/sqrt(3).
            # Distance to (1,0,0): cos(theta) = 1/sqrt(3). theta = 54.7 deg.
            # This is correct.
            
            # So the formula is:
            n12 = np.cross(p1, p2)
            n23 = np.cross(p2, p3)
            n31 = np.cross(p3, p1)
            
            y = np.cross(n12, n23) + np.cross(n23, n31) + np.cross(n31, n12)
            
            y_norm = np.linalg.norm(y)
            if y_norm < 1e-10:
                # Degenerate triangle
                continue
            
            y = y / y_norm
            
            # Check if y is inside the spherical triangle.
            # The circumcenter is inside if the triangle is "acute" in some sense.
            # A robust check: The distance from y to the vertices should be the same, and 
            # y should be in the same "hemisphere" relative to the edges as the triangle interior.
            # Specifically, y should have positive dot product with the inward normals of the edges?
            # Or simpler: Check if the barycentric coordinates of y with respect to p1, p2, p3 are positive.
            
            # Solve [p1 p2 p3] * [alpha beta gamma]^T = y
            M = np.column_stack([p1, p2, p3])
            try:
                coeffs = np.linalg.solve(M, y)
            except np.linalg.LinAlgError:
                coeffs = np.array([1/3, 1/3, 1/3])
                
            # If all coeffs are positive, the circumcenter is inside the triangle.
            if np.all(coeffs > -1e-8):
                # The radius is the distance from y to any vertex
                r = np.arccos(np.clip(np.dot(y, p1), -1.0, 1.0))
            else:
                # If outside, the farthest point is the midpoint of the longest edge.
                # Find the longest edge
                d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
                d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
                d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
                
                max_d = max(d12, d23, d31)
                if max_d == d12:
                    mid = (p1 + p2) / np.linalg.norm(p1 + p2)
                elif max_d == d23:
                    mid = (p2 + p3) / np.linalg.norm(p2 + p3)
                else:
                    mid = (p3 + p1) / np.linalg.norm(p3 + p1)
                    
                r = np.arccos(np.clip(np.dot(mid, p1), -1.0, 1.0))
                
            if r > max_r:
                max_r = r
                
        return max_r

    def local_search(pts, max_iters=200, step_size=0.1):
        """
        Stochastic local search.
        1. Find the point on the sphere with the largest distance to the nearest config point (the "hole").
        2. Move the nearest config point towards the hole.
        3. Accept if the covering radius decreases.
        """
        current_pts = pts.copy()
        current_r = compute_covering_radius(current_pts)
        
        rng = np.random.default_rng(42)
        
        for i in range(max_iters):
            if time.time() > deadline - 10:
                break
                
            # Find the "hole": the point on the sphere farthest from any current point.
            # We approximate this by checking the circumcenters of the facets (as in the exact method).
            # But we can also just sample the sphere? No, that's slow.
            # Let's reuse the facet logic to find the worst facet.
            
            try:
                hull = ConvexHull(current_pts)
            except Exception:
                break
                
            worst_facet_idx = -1
            worst_r = 0.0
            worst_y = None
            
            for idx, simplex in enumerate(hull.simplices):
                p1 = current_pts[simplex[0]]
                p2 = current_pts[simplex[1]]
                p3 = current_pts[simplex[2]]
                
                n12 = np.cross(p1, p2)
                n23 = np.cross(p2, p3)
                n31 = np.cross(p3, p1)
                
                y = np.cross(n12, n23) + np.cross(n23, n31) + np.cross(n31, n12)
                y_norm = np.linalg.norm(y)
                if y_norm < 1e-10:
                    continue
                y = y / y_norm
                
                M = np.column_stack([p1, p2, p3])
                try:
                    coeffs = np.linalg.solve(M, y)
                except:
                    coeffs = np.array([1/3, 1/3, 1/3])
                    
                if np.all(coeffs > -1e-8):
                    r = np.arccos(np.clip(np.dot(y, p1), -1.0, 1.0))
                else:
                    d12 = np.arccos(np.clip(np.dot(p1, p2), -1.0, 1.0))
                    d23 = np.arccos(np.clip(np.dot(p2, p3), -1.0, 1.0))
                    d31 = np.arccos(np.clip(np.dot(p3, p1), -1.0, 1.0))
                    max_d = max(d12, d23, d31)
                    if max_d == d12:
                        mid = (p1 + p2) / np.linalg.norm(p1 + p2)
                    elif max_d == d23:
                        mid = (p2 + p3) / np.linalg.norm(p2 + p3)
                    else:
                        mid = (p3 + p1) / np.linalg.norm(p3 + p1)
                    r = np.arccos(np.clip(np.dot(mid, p1), -1.0, 1.0))
                    y = mid # Update y to the actual worst point for this facet
                    
                if r > worst_r:
                    worst_r = r
                    worst_facet_idx = idx
                    worst_y = y
                    
            if worst_y is None:
                break
            
            # Find the nearest point to worst_y
            dists = np.sum(current_pts * worst_y, axis=1)
            nearest_idx = np.argmax(dists)
            
            # Move the nearest point towards worst_y
            # Step size decay
            step = step_size * (0.95 ** i)
            
            # Spherical interpolation (SLERP) or simple vector addition
            # Simple vector addition and normalization is robust enough for small steps
            new_pt = current_pts[nearest_idx] + step * (worst_y - current_pts[nearest_idx])
            new_pt = new_pt / np.linalg.norm(new_pt)
            
            # Try the move
            trial_pts = current_pts.copy()
            trial_pts[nearest_idx] = new_pt
            
            trial_r = compute_covering_radius(trial_pts)
            
            if trial_r < current_r:
                current_pts = trial_pts
                current_r = trial_r
            else:
                # If it didn't improve, try a random kick to escape local minima
                if rng.random() < 0.1:
                    kick_idx = rng.integers(0, N)
                    kick_dir = rng.normal(size=3)
                    kick_dir = kick_dir / np.linalg.norm(kick_dir)
                    trial_pts = current_pts.copy()
                    trial_pts[kick_idx] = current_pts[kick_idx] + 0.1 * (kick_dir - current_pts[kick_idx])
                    trial_pts[kick_idx] = trial_pts[kick_idx] / np.linalg.norm(trial_pts[kick_idx])
                    trial_r = compute_covering_radius(trial_pts)
                    if trial_r < current_r + 1e-4: # Allow slight worsening for exploration
                        current_pts = trial_pts
                        current_r = trial_r
                        
        return current_pts, current_r

    # Main loop
    best_pts = None
    best_r = np.pi
    
    # Multiple restarts with different initializations
    n_restarts = 5
    restart_budget = (deadline - start_time - 15) / n_restarts
    
    for i in range(n_restarts):
        if time.time() > deadline - 15:
            break
            
        # Vary the phase offset and add jitter
        phase = i * np.pi / 3
        jitter = 0.05 if i > 0 else 0.0
        
        init_pts = fibonacci_spiral(phase, jitter)
        
        # Run local search
        # Allocate time for local search based on remaining time
        # We'll just let the local search run until its internal deadline check triggers
        refined_pts, r = local_search(init_pts)
        
        if r < best_r:
            best_r = r
            best_pts = refined_pts
            
        # If we have time, try a second pass of local search on the best found so far
        if best_pts is not None and time.time() < deadline - 10:
            # Re-run local search on the best result to polish
            polished_pts, polished_r = local_search(best_pts, max_iters=100, step_size=0.05)
            if polished_r < best_r:
                best_r = polished_r
                best_pts = polished_pts

    if best_pts is None:
        best_pts = fibonacci_spiral(0.0)
        
    return normalize(best_pts)

# EVOLVE-BLOCK-END
