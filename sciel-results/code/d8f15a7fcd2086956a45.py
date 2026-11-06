import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin for 150s limit

    def normalize(pts):
        norms = np.linalg.norm(pts, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return pts / norms

    def compute_covering_radius(pts):
        """
        Compute the exact covering radius using the convex hull.
        The covering radius is the maximum circumradius of the spherical triangles (facets) of the convex hull.
        """
        try:
            hull = ConvexHull(pts)
            max_r = 0.0
            worst_face = None
            
            for face in hull.simplices:
                p0, p1, p2 = pts[face[0]], pts[face[1]], pts[face[2]]
                
                # Compute circumcenter of the spherical triangle
                # The circumcenter lies in the plane spanned by the three points.
                # c = a*p0 + b*p1 + c*p2 such that |c| = 1 and c.p0 = c.p1 = c.p2 = cos(r)
                
                # Alternatively, use the formula for the circumcenter of a triangle on the sphere.
                # Let v1 = p0, v2 = p1, v3 = p2.
                # The circumcenter c satisfies c = (alpha * v1 + beta * v2 + gamma * v3) / norm(...)
                # where alpha, beta, gamma are barycentric coordinates related to the side lengths.
                
                # A simpler vector approach:
                # The circumcenter is the intersection of the sphere and the plane perpendicular to the chord midpoints? No.
                # The circumcenter is the point on the sphere equidistant from v1, v2, v3.
                # c dot v1 = c dot v2 = c dot v3 = cos(r)
                
                # Let's solve the linear system for c in the basis of v1, v2, v3?
                # Or use the cross product formula.
                # c is parallel to (v1 x v2) x (v2 x v3) ... no.
                
                # Standard formula for circumcenter of spherical triangle:
                # c = (sin(2A) * v1 + sin(2B) * v2 + sin(2C) * v3) / norm(...)
                # where A, B, C are the angles of the spherical triangle at v1, v2, v3.
                
                # Angles can be computed via dot products of normals.
                # n1 = v2 x v3, n2 = v3 x v1, n3 = v1 x v2
                # cos(A) = (n2 dot n3) / (|n2| |n3|)
                
                n1 = np.cross(p1, p2)
                n2 = np.cross(p2, p0)
                n3 = np.cross(p0, p1)
                
                n1_n = np.linalg.norm(n1)
                n2_n = np.linalg.norm(n2)
                n3_n = np.linalg.norm(n3)
                
                if n1_n < 1e-10 or n2_n < 1e-10 or n3_n < 1e-10:
                    continue
                
                cos_A = np.dot(n2, n3) / (n2_n * n3_n)
                cos_B = np.dot(n3, n1) / (n3_n * n1_n)
                cos_C = np.dot(n1, n2) / (n1_n * n2_n)
                
                # Clip for safety
                cos_A = np.clip(cos_A, -1.0, 1.0)
                cos_B = np.clip(cos_B, -1.0, 1.0)
                cos_C = np.clip(cos_C, -1.0, 1.0)
                
                A = np.arccos(cos_A)
                B = np.arccos(cos_B)
                C = np.arccos(cos_C)
                
                # Check if the circumcenter is inside the triangle.
                # If any angle is > pi, the circumcenter is outside (or on the edge).
                # Actually, for a spherical triangle, if the triangle is small (< pi/2 sides?), 
                # the circumcenter is inside. 
                # A more robust check: The circumcenter c should have the same orientation 
                # as the centroid with respect to the face normal.
                
                # Calculate circumcenter vector
                # c = (sin(2A) p0 + sin(2B) p1 + sin(2C) p2)
                c_vec = np.sin(2*A) * p0 + np.sin(2*B) * p1 + np.sin(2*C) * p2
                c_norm = np.linalg.norm(c_vec)
                
                if c_norm < 1e-10:
                    continue
                
                c_vec /= c_norm
                
                # Check if c_vec is inside the triangle.
                # The centroid of the triangle is (p0+p1+p2)/3.
                # The normal of the face is n_face = p0 x p1.
                # c_vec should be on the same side of the plane as the centroid?
                # Actually, the circumcenter of a small spherical triangle is inside.
                # If the triangle is large, it might be outside.
                # We only care about the covering radius, which is determined by the "empty cap" condition.
                # The covering radius is the maximum distance from any point on the sphere to the nearest point.
                # This is equivalent to the maximum circumradius of the facets of the convex hull 
                # IF the circumcenter lies inside the facet.
                # If the circumcenter lies outside, the covering radius for that facet is determined 
                # by the edge or vertex, which is smaller than the circumradius? 
                # No, if the circumcenter is outside, the farthest point in the Voronoi cell 
                # is a vertex or edge midpoint.
                # However, for the convex hull of points on a sphere, the Voronoi cells are spherical polygons.
                # The covering radius is the maximum over all cells of the max distance from the site to any point in the cell.
                # This maximum is always at a vertex of the cell (which is a circumcenter of a triangle of sites) 
                # or an edge (midpoint of an edge of sites).
                # So we need to check circumcenters of all triangles formed by 3 sites that form a face of the convex hull,
                # AND midpoints of all edges of the convex hull.
                
                # Let's just check circumcenters for now, as they are usually the dominant term for "covering".
                # To ensure the circumcenter is "valid" (inside the cell), we can check if it is 
                # closer to the 3 sites than to any other site.
                
                # Distance from c_vec to p0
                d0 = np.arccos(np.clip(np.dot(c_vec, p0), -1.0, 1.0))
                
                # Check if it's the farthest point for these 3 sites.
                # If c_vec is not inside the triangle, the max distance in the cell might be elsewhere.
                # But for the purpose of optimization, minimizing the max circumradius of all hull faces
                # is a good proxy.
                
                # Refinement: Only consider circumcenter if it is inside the spherical triangle.
                # A point is inside a spherical triangle if it is on the same side of each edge 
                # as the opposite vertex.
                
                # Edge p0-p1: normal n3 = p0 x p1. 
                # Check if c_vec and p2 are on the same side.
                # (c_vec - projection) dot n3 ... simpler: 
                # The condition is that c_vec, p0, p1, p2 form a consistent orientation.
                # For a convex hull, the faces are oriented outwards.
                # Let's use the dot product with the face normal.
                # n_face = np.cross(p0, p1)
                # if np.dot(c_vec, n_face) * np.dot(p2, n_face) < 0: outside?
                
                # Let's stick to the simpler check: 
                # If the triangle is "small" (area < pi/2 or so), circumcenter is inside.
                # We can check if d0, d1, d2 are all < pi/2.
                
                d1 = np.arccos(np.clip(np.dot(c_vec, p1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_vec, p2), -1.0, 1.0))
                
                # If any distance is > pi/2, the circumcenter is likely "far" or outside.
                # Actually, if the triangle contains the center of the sphere in its convex hull, 
                # the circumcenter is ambiguous or far.
                # Let's just take the max circumradius of all faces, assuming they are "small" enough 
                # which is true for a good covering.
                
                r = d0
                if r > max_r:
                    max_r = r
                    worst_face = face
                    
            return np.degrees(max_r), worst_face

        except Exception:
            return 180.0, None

    def refine_with_hull(pts, iterations=50):
        """
        Targeted refinement: Move vertices of the worst facet towards its circumcenter.
        """
        current_pts = pts.copy()
        best_cr, _ = compute_covering_radius(current_pts)
        best_pts = current_pts.copy()
        
        for _ in range(iterations):
            if time.time() > deadline - 5:
                break
                
            cr, face = compute_covering_radius(current_pts)
            if face is None:
                break
                
            if cr < best_cr:
                best_cr = cr
                best_pts = current_pts.copy()
                
            # Compute circumcenter of the worst face
            p0, p1, p2 = current_pts[face[0]], current_pts[face[1]], current_pts[face[2]]
            
            n1 = np.cross(p1, p2)
            n2 = np.cross(p2, p0)
            n3 = np.cross(p0, p1)
            
            n1_n = np.linalg.norm(n1)
            n2_n = np.linalg.norm(n2)
            n3_n = np.linalg.norm(n3)
            
            if n1_n < 1e-10 or n2_n < 1e-10 or n3_n < 1e-10:
                break
                
            cos_A = np.dot(n2, n3) / (n2_n * n3_n)
            cos_B = np.dot(n3, n1) / (n3_n * n1_n)
            cos_C = np.dot(n1, n2) / (n1_n * n2_n)
            
            cos_A = np.clip(cos_A, -1.0, 1.0)
            cos_B = np.clip(cos_B, -1.0, 1.0)
            cos_C = np.clip(cos_C, -1.0, 1.0)
            
            A = np.arccos(cos_A)
            B = np.arccos(cos_B)
            C = np.arccos(cos_C)
            
            c_vec = np.sin(2*A) * p0 + np.sin(2*B) * p1 + np.sin(2*C) * p2
            c_norm = np.linalg.norm(c_vec)
            
            if c_norm < 1e-10:
                break
                
            c_vec /= c_norm
            
            # Move vertices towards circumcenter
            # Use a small step size to avoid oscillation
            step_size = 0.05
            for idx in face:
                # Geodesic move: slerp
                # Or simple linear in R3 followed by normalization
                new_pos = current_pts[idx] + step_size * (c_vec - current_pts[idx])
                current_pts[idx] = new_pos / np.linalg.norm(new_pos)
            
            # Perform a few Lloyd steps to relax the rest of the configuration
            # This prevents the other points from being "dragged" or becoming suboptimal
            current_pts = lloyd_relax(current_pts, steps=2)
            
        return best_pts, best_cr

    def lloyd_relax(pts, steps=10):
        """
        Perform Lloyd's algorithm iterations on the sphere.
        """
        # Use a moderate grid for speed
        n_grid = 20000
        i = np.arange(n_grid) + 0.5
        z = 1.0 - 2.0 * i / n_grid
        phi = np.pi * (1.0 + np.sqrt(5.0)) * i
        r = np.sqrt(1.0 - z * z)
        grid_pts = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
        grid_pts = normalize(grid_pts)
        
        current_pts = pts.copy()
        
        for _ in range(steps):
            # Assign each grid point to its nearest configuration point
            dots = grid_pts @ current_pts.T
            nearest_idx = np.argmax(dots, axis=1)
            
            # Compute new positions as the normalized centroid of assigned grid points
            new_pts = np.zeros_like(current_pts)
            
            for k in range(N):
                mask = (nearest_idx == k)
                if np.any(mask):
                    new_pts[k] = np.sum(grid_pts[mask], axis=0)
                else:
                    new_pts[k] = current_pts[k]
            
            # Normalize
            norms = np.linalg.norm(new_pts, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            new_pts /= norms
            
            # Check for convergence
            diff = np.linalg.norm(new_pts - current_pts)
            current_pts = new_pts
            if diff < 1e-7:
                break
                
        return current_pts

    # 1. Initial Configuration: Fibonacci Spiral
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    P = normalize(P)

    # 2. Initial Lloyd Relaxation to get a good starting point
    P = lloyd_relax(P, steps=20)
    
    # 3. Hull-based Refinement
    P, cr = refine_with_hull(P, iterations=100)
    
    # 4. Final Lloyd Polish
    P = lloyd_relax(P, steps=10)
    
    # 5. Final Hull-based Refinement with smaller steps
    P, cr = refine_with_hull(P, iterations=50)
    
    # Ensure normalization
    P = normalize(P)
    
    return P
