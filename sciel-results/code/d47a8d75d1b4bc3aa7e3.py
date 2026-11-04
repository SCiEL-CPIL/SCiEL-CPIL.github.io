import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import minimize

N = 130

# EVOLVE-BLOCK-START
import numpy as np
from scipy.spatial import ConvexHull

def construct():
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    P = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    # 2. Local Search Loop
    # We will iterate, finding the Delaunay triangulation via ConvexHull,
    # identifying the worst triangle for each point, and moving the point towards that triangle's circumcenter.
    
    for it in range(2000):
        # Compute Convex Hull to get Delaunay triangulation
        # The faces of the convex hull of points on the sphere correspond to Delaunay triangles
        try:
            hull = ConvexHull(P)
            simplices = hull.simplices
        except Exception:
            # If hull fails (e.g., points coplanar or duplicate), break or handle
            break
            
        # Build adjacency: for each point, list of triangle indices containing it
        point_to_tris = [[] for _ in range(N)]
        for tri_idx, tri in enumerate(simplices):
            for p_idx in tri:
                point_to_tris[p_idx].append(tri_idx)
                
        # For each point, find the "worst" triangle (largest circumradius)
        # and the circumcenter of that triangle.
        
        # Precompute circumcenters and radii for all triangles
        # Circumcenter of a spherical triangle is the normalized cross product of the two edge vectors?
        # Or solve linear system.
        # For a triangle with vertices A, B, C on unit sphere:
        # Circumcenter U is proportional to (A x B) x (B x C) ... wait, that's not quite right for general spherical.
        # Simpler: The circumcenter of the plane triangle ABC projected onto the sphere?
        # Actually, the circumcenter of the spherical triangle is the point U on the sphere such that dist(U,A)=dist(U,B)=dist(U,C).
        # This U is the normalized vector (A x B) x (B x C) is incorrect.
        # The vector perpendicular to the plane ABC is N = (A-B) x (A-C).
        # The circumcenter of the planar triangle lies on the line passing through the centroid and perpendicular to the plane.
        # Let's use a robust method:
        # The circumcenter U satisfies U . A = U . B = U . C = k.
        # This is a linear system:
        # [A; B; C] U = [k; k; k]
        # We can solve for U and k.
        
        centers = np.zeros((len(simplices), 3))
        radii = np.zeros(len(simplices))
        
        for idx, tri in enumerate(simplices):
            A = P[tri[0]]
            B = P[tri[1]]
            C = P[tri[2]]
            
            # Solve [A B C]^T U = [1 1 1]^T ? No, U . A = cos(theta).
            # We want U to be the unit vector equidistant from A, B, C.
            # The vector U is parallel to the normal of the plane defined by A, B, C? No.
            # The circumcenter of the spherical triangle is the intersection of the sphere with the line passing through the origin and the circumcenter of the planar triangle formed by A, B, C? No.
            # Actually, the circumcenter of the spherical triangle is the normalized vector:
            # U = (A x B) x (B x C) is wrong.
            # Correct formula: U is the normalized solution to:
            # U . A = U . B = U . C
            # This implies U is orthogonal to (A-B) and (B-C).
            # So U is parallel to (A-B) x (B-C).
            # Let's check:
            # (A-B) x (B-C) is the normal to the plane containing A, B, C.
            # Wait, the circumcenter of the *spherical* triangle is indeed the projection of the circumcenter of the *planar* triangle?
            # No, the circumcenter of the spherical triangle is the point on the sphere equidistant from A, B, C.
            # The locus of points equidistant from A and B is the great circle bisecting AB.
            # The intersection of these bisectors is the circumcenter.
            # Vectorially, U is proportional to (A x B) x (B x C) is NOT correct.
            # U is proportional to (A-B) x (B-C) IS the normal to the plane.
            # Does the normal to the plane ABC give the circumcenter?
            # Let's test with equilateral triangle on equator. A=(1,0,0), B=(-1/2, sqrt(3)/2, 0), C=(-1/2, -sqrt(3)/2, 0).
            # Normal is (0,0,1). Dist from (0,0,1) to A is 90 deg. Dist to B is 90 deg.
            # Yes, for an equilateral triangle on the equator, the pole is the circumcenter.
            # Is it always true?
            # The normal vector N = (A-B) x (B-C) is perpendicular to the plane.
            # The point on the sphere in the direction of N is equidistant from A, B, C?
            # Let U = N / ||N||.
            # U . A = U . B ?
            # N . (A-B) = ((A-B) x (B-C)) . (A-B) = 0. Yes.
            # So the normal to the plane of the triangle, normalized, is the circumcenter of the spherical triangle!
            
            N_vec = np.cross(A - B, B - C)
            norm_N = np.linalg.norm(N_vec)
            if norm_N < 1e-9:
                # Degenerate triangle
                centers[idx] = (A + B + C) / 3.0
                centers[idx] /= np.linalg.norm(centers[idx])
                radii[idx] = np.max(np.arccos(np.clip(np.dot([A, B, C], centers[idx]), -1, 1)))
            else:
                U = N_vec / norm_N
                # Check orientation: if U points away from the triangle centroid, flip?
                # The circumcenter should be "inside" the triangle or close to it.
                # If the triangle is obtuse, the circumcenter is outside.
                # We just need the point.
                centers[idx] = U
                # Calculate circumradius (angular distance)
                # dist = arccos(U . A)
                dot_val = np.dot(U, A)
                radii[idx] = np.arccos(np.clip(dot_val, -1, 1))
                
        # For each point, find the incident triangle with the largest radius
        # and move the point towards that triangle's circumcenter.
        
        moves = np.zeros_like(P)
        
        for i in range(N):
            tris = point_to_tris[i]
            if not tris:
                continue
            
            # Find max radius triangle
            max_rad_idx = max(tris, key=lambda t: radii[t])
            target_center = centers[max_rad_idx]
            
            # Move towards target_center
            # We want to reduce the distance from P[i] to target_center?
            # No, we want P[i] to become the circumcenter?
            # If we move P[i] to the circumcenter of the worst triangle, the radius of that triangle becomes 0?
            # No, if P[i] moves, the triangle changes.
            # Heuristic: Move P[i] towards the circumcenter of the worst triangle.
            # This pulls the center of the Voronoi cell towards the "far" boundary.
            
            # Step size: adaptive
            # dist = angular distance
            # We can just add a small fraction of the vector difference.
            # Project target onto tangent plane at P[i]?
            # Or just simple vector addition and normalization.
            
            # Simple update:
            # P_new = P[i] + alpha * (target_center - P[i])
            # Then normalize.
            
            # Let's use a fixed small step for stability, or decay.
            # The "error" is the radius.
            # Let's just move a fixed fraction of the way to the circumcenter.
            # If we move too far, we might overshoot.
            
            # A better heuristic: Move towards the centroid of the Voronoi cell.
            # But we only have Delaunay.
            # Let's try moving towards the circumcenter of the worst triangle.
            
            # Step size: 0.1 seems reasonable?
            # Let's try to be aggressive early, conservative late.
            step = 0.05
            
            delta = target_center - P[i]
            # Project delta to tangent plane at P[i] to avoid moving radially (which does nothing after normalization)
            # delta_tan = delta - (delta . P[i]) * P[i]
            dot_p = np.dot(delta, P[i])
            delta_tan = delta - dot_p * P[i]
            
            moves[i] = step * delta_tan
            
        P = P + moves
        P = P / np.linalg.norm(P, axis=1, keepdims=True)
        
    return P
# EVOLVE-BLOCK-END
