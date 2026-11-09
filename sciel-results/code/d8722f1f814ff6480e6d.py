import numpy as np
import time
from scipy.spatial import ConvexHull

N = 50

def construct():
    start_time = time.time()
    time_limit = 110.0  # Safety margin for 120s limit
    
    # 1. Initialization: Fibonacci Sphere
    i = np.arange(N) + 0.5
    z = 1.0 - 2.0 * i / N
    phi = np.pi * (1.0 + np.sqrt(5.0)) * i
    r = np.sqrt(1.0 - z * z)
    points = np.column_stack([r * np.cos(phi), r * np.sin(phi), z])
    
    best_points = points.copy()
    best_radius = np.inf
    
    # Helper to compute covering radius from points
    def compute_covering_radius(pts):
        try:
            hull = ConvexHull(pts)
            max_rad = 0.0
            for simplex in hull.simplices:
                # simplex contains indices of 3 points
                v1, v2, v3 = pts[simplex[0]], pts[simplex[1]], pts[simplex[2]]
                
                # Compute circumcenter of the triangle v1, v2, v3 on the sphere
                # The circumcenter c satisfies |c - v1| = |c - v2| = |c - v3| and c is on the sphere?
                # Actually, the circumcenter of the spherical triangle is the intersection of the sphere 
                # and the plane perpendicular to the chord?
                # Easier: The circumcenter of the 3 points in 3D space (not necessarily on sphere) 
                # lies on the line passing through the origin and the circumcenter of the planar triangle?
                # No. The points are on the unit sphere.
                # The circumcenter of the spherical triangle is the normalized vector of the 
                # circumcenter of the planar triangle formed by v1, v2, v3?
                # Let's find the circumcenter of the planar triangle v1, v2, v3.
                
                # Solve for c such that c dot (v1-v2) = 0.5 * (|v1|^2 - |v2|^2) = 0
                # c dot (v1-v3) = 0
                # This gives the circumcenter in the plane of the triangle.
                # Let c = a * v1 + b * v2 + c * v3? No, c is in the plane spanned by v1, v2, v3.
                # Since |v1|=|v2|=|v3|=1, the circumcenter of the planar triangle is the origin?
                # No, the origin is the center of the sphere. The distance from origin to v1 is 1.
                # So the circumcenter of the planar triangle is NOT the origin unless the triangle is equilateral and "flat"?
                # Wait, v1, v2, v3 are on the unit sphere. The circumcircle of the planar triangle v1v2v3 
                # has a center O' and radius R'. The distance from O' to v1 is R'.
                # The spherical circumcenter is the projection of O' onto the sphere?
                # Actually, the spherical circumcenter is the point on the sphere equidistant from v1, v2, v3.
                # This point is along the vector O' (if O' is not 0).
                
                # Let's calculate O' (planar circumcenter)
                # Let v1, v2, v3 be the vectors.
                # O' = v1 + s * (v2 - v1) + t * (v3 - v1)
                # |O' - v1|^2 = |O' - v2|^2 => 2 O' dot (v2 - v1) = |v2|^2 - |v1|^2 = 0
                # 2 O' dot (v3 - v1) = 0
                # So O' is orthogonal to (v2-v1) and (v3-v1).
                # This means O' is parallel to the normal of the plane?
                # No, O' is in the plane of v1, v2, v3.
                # The vector orthogonal to (v2-v1) and (v3-v1) is the normal n.
                # So O' must be 0?
                # Let's check. If O' is in the plane, O' = a v1 + b v2 + c v3.
                # O' dot (v2 - v1) = 0.
                # This implies the circumcenter of the planar triangle is the origin ONLY if the triangle is equilateral?
                # No, for any triangle on a sphere centered at origin, the circumcenter of the planar triangle is NOT the origin.
                # Example: v1=(1,0,0), v2=(0,1,0), v3=(0,0,1).
                # Plane x+y+z=1.
                # Circumcenter of (1,0,0),(0,1,0),(0,0,1) is (1/3, 1/3, 1/3).
                # Distance to v1 is sqrt(4/9+1/9+1/9) = sqrt(6)/3.
                # The spherical circumcenter is normalize(1/3, 1/3, 1/3) = (1,1,1)/sqrt(3).
                # Angular distance from (1,1,1)/sqrt(3) to (1,0,0) is arccos(1/sqrt(3)).
                
                # So, compute planar circumcenter O'.
                # O' is the intersection of the perpendicular bisectors.
                # In 2D coordinates of the triangle, it's easy. In 3D, we can use the formula:
                # O' = v1 + ( |v1|^2 (v2-v3) + |v2|^2 (v3-v1) + |v3|^2 (v1-v2) ) / (2 * |v1 x v2 + v2 x v3 + v3 x v1| ? No.
                
                # Simpler: Use the fact that O' = (v1 + v2 + v3) / 3 is the centroid, not circumcenter.
                # Let's use the standard 3D circumcenter formula for points on a sphere.
                # The spherical circumcenter C is the normalized vector of the planar circumcenter O'.
                # O' can be found by solving:
                # O' dot (v1 - v2) = 0
                # O' dot (v1 - v3) = 0
                # O' = a * v1 + b * v2 + c * v3 (in the plane)
                
                # Alternative: The spherical circumcenter is the intersection of the sphere and the line 
                # passing through the origin and the planar circumcenter.
                
                # Let's compute the planar circumcenter using cross products.
                # Let A=v1, B=v2, C=v3.
                # d = 2 * (A x B) dot (B - C) ... this is for 2D.
                
                # Robust method:
                # The circumcenter O' satisfies O' = A + s*(B-A) + t*(C-A)
                # |O'-A|^2 = |O'-B|^2 => 2 O' dot (B-A) = |B|^2 - |A|^2 = 0
                # |O'-A|^2 = |O'-C|^2 => 2 O' dot (C-A) = 0
                # So O' is orthogonal to B-A and C-A.
                # This implies O' is parallel to (B-A) x (C-A).
                # Wait, if O' is in the plane of ABC, and orthogonal to two vectors in the plane, O' must be 0?
                # No, the vectors B-A and C-A are in the plane. The vector orthogonal to them is the normal.
                # So O' cannot be in the plane AND orthogonal to vectors in the plane, unless O'=0.
                # My previous deduction was flawed.
                # O' is in the plane. O' dot (B-A) = 0 is WRONG.
                # The condition is |O'-A|^2 = |O'-B|^2.
                # O'^2 - 2 O' dot A + A^2 = O'^2 - 2 O' dot B + B^2
                # 2 O' dot (B - A) = B^2 - A^2 = 1 - 1 = 0.
                # So O' IS orthogonal to B-A.
                # And O' IS orthogonal to C-A.
                # So O' is parallel to the normal of the plane.
                # But O' is IN the plane.
                # The only vector in the plane that is orthogonal to the plane is 0.
                # So O' = 0?
                # If O' = 0, then |0-A| = 1, |0-B| = 1.
                # So the origin is equidistant from A, B, C.
                # So the planar circumcenter IS the origin?
                # Let's check the example: A=(1,0,0), B=(0,1,0), C=(0,0,1).
                # Origin (0,0,0). Distance to A is 1. Distance to B is 1. Distance to C is 1.
                # So the circumcenter of the planar triangle A-B-C is NOT the origin.
                # The circumcenter of the planar triangle is (1/3, 1/3, 1/3).
                # Distance to A: sqrt((2/3)^2 + (1/3)^2 + (1/3)^2) = sqrt(6)/3 != 1.
                # So the origin is NOT the circumcenter of the planar triangle.
                # Where did I go wrong?
                # |O'-A|^2 = |O'-B|^2 implies O' lies on the perpendicular bisector plane of AB.
                # The perpendicular bisector of the chord AB passes through the origin and is perpendicular to AB.
                # So O' lies on the plane defined by the origin and the normal to AB?
                # No, the set of points equidistant from A and B is the plane passing through the midpoint of AB and perpendicular to AB.
                # This plane contains the origin because |O-A|=|O-B|=1.
                # So O' lies in the plane P_AB: {x | x dot (B-A) = 0}.
                # Similarly O' lies in P_AC: {x | x dot (C-A) = 0}.
                # The intersection of P_AB and P_AC is the line passing through the origin and the normal to the plane ABC.
                # So O' lies on the line generated by the normal N = (B-A) x (C-A).
                # But O' is also in the plane of ABC.
                # The intersection of the line (origin + t*N) and the plane (containing A,B,C) is a single point.
                # So O' = t * N.
                # We need to find t such that O' is in the plane.
                # Actually, we just need the direction. The spherical circumcenter is normalize(O').
                # So Spherical Circumcenter = normalize( (B-A) x (C-A) )?
                # Let's check the example.
                # A=(1,0,0), B=(0,1,0), C=(0,0,1).
                # B-A = (-1, 1, 0). C-A = (-1, 0, 1).
                # Cross product = (1, 1, 1).
                # Normalize(1,1,1) = (1,1,1)/sqrt(3).
                # This matches the spherical circumcenter we found earlier.
                
                # So, Spherical Circumcenter C_s = normalize( (v2-v1) x (v3-v1) )
                # Wait, is this always correct?
                # The vector (v2-v1) x (v3-v1) is perpendicular to the plane of the triangle.
                # The spherical circumcenter lies on this normal line.
                # Yes.
                
                normal = np.cross(v2 - v1, v3 - v1)
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue # Degenerate
                c_s = normal / norm
                
                # The angular distance from c_s to v1 is the circumradius of this facet.
                # cos(theta) = c_s dot v1
                # theta = arccos(c_s dot v1)
                
                # Check if the circumcenter is "inside" the triangle?
                # For the covering radius, we care about the largest empty cap.
                # If the triangle is obtuse, the circumcenter is outside the spherical triangle.
                # In that case, the largest empty cap is centered at the midpoint of the longest edge.
                # We should check the edges as well.
                
                # Calculate angular distances to vertices
                d1 = np.arccos(np.clip(np.dot(c_s, v1), -1.0, 1.0))
                d2 = np.arccos(np.clip(np.dot(c_s, v2), -1.0, 1.0))
                d3 = np.arccos(np.clip(np.dot(c_s, v3), -1.0, 1.0))
                
                # If any distance is > pi/2, the triangle is obtuse (or right).
                # In that case, the circumcenter is not the center of the largest empty cap for this facet.
                # The center is the midpoint of the longest edge.
                
                # Edge 1-2
                e12 = v1 + v2
                e12_norm = np.linalg.norm(e12)
                if e12_norm > 1e-10:
                    m12 = e12 / e12_norm
                    d_e12 = np.arccos(np.clip(np.dot(m12, v1), -1.0, 1.0))
                else:
                    d_e12 = 0.0
                
                # Edge 2-3
                e23 = v2 + v3
                e23_norm = np.linalg.norm(e23)
                if e23_norm > 1e-10:
                    m23 = e23 / e23_norm
                    d_e23 = np.arccos(np.clip(np.dot(m23, v2), -1.0, 1.0))
                else:
                    d_e23 = 0.0

                # Edge 3-1
                e31 = v3 + v1
                e31_norm = np.linalg.norm(e31)
                if e31_norm > 1e-10:
                    m31 = e31 / e31_norm
                    d_e31 = np.arccos(np.clip(np.dot(m31, v3), -1.0, 1.0))
                else:
                    d_e31 = 0.0

                # The local covering radius for this facet is the max of these candidates?
                # Actually, the covering radius of the sphere is the max over ALL facets of the "worst" point.
                # For a specific facet, the "worst" point is the one furthest from the vertices.
                # If the circumcenter is inside the triangle (acute), the worst point is c_s, distance d_circ.
                # If the circumcenter is outside (obtuse), the worst point is the midpoint of the longest edge, distance d_edge.
                
                # How to determine if c_s is inside?
                # If d1, d2, d3 are all < pi/2, it's acute.
                # If any is > pi/2, it's obtuse.
                
                max_d_circ = max(d1, d2, d3)
                max_d_edge = max(d_e12, d_e23, d_e31)
                
                # If acute, radius is max_d_circ (which is equal for all).
                # If obtuse, radius is max_d_edge.
                # Actually, if obtuse, the circumcenter is outside, so the distance from c_s to the "other" vertices is > pi/2?
                # No, c_s is the circumcenter. The distance to all vertices is the same in the spherical sense?
                # No, c_s is the center of the circle passing through v1, v2, v3.
                # So d1=d2=d3.
                # If d1 > pi/2, the center is on the "back" side.
                # The empty cap is bounded by the edges. The center of the largest empty cap is the midpoint of the longest edge.
                
                if max_d_circ <= np.pi / 2.0:
                    local_rad = max_d_circ
                    center = c_s
                else:
                    local_rad = max_d_edge
                    # Find which edge is longest to determine center?
                    # The max_d_edge is the radius. The center is the midpoint of that edge.
                    if d_e12 == max_d_edge:
                        center = m12
                    elif d_e23 == max_d_edge:
                        center = m23
                    else:
                        center = m31

                if local_rad > max_rad:
                    max_rad = local_rad
                    worst_center = center
                    worst_simplex = simplex
            return max_rad, worst_center, worst_simplex
        except Exception:
            return np.inf, None, None

    # Optimization Loop
    step_size = 0.01
    while time.time() - start_time < time_limit:
        rad, center, simplex = compute_covering_radius(points)
        
        if rad < best_radius:
            best_radius = rad
            best_points = points.copy()
            
        if simplex is None:
            break
            
        # Move the 3 vertices of the worst simplex towards the worst center
        # This reduces the size of the cap
        v1, v2, v3 = points[simplex[0]], points[simplex[1]], points[simplex[2]]
        
        # Move towards center
        # We want to move the points such that the cap shrinks.
        # Moving vertices towards the center of the cap reduces the circumradius.
        
        new_v1 = v1 + step_size * (center - v1)
        new_v2 = v2 + step_size * (center - v2)
        new_v3 = v3 + step_size * (center - v3)
        
        # Normalize
        new_v1 /= np.linalg.norm(new_v1)
        new_v2 /= np.linalg.norm(new_v2)
        new_v3 /= np.linalg.norm(new_v3)
        
        points[simplex[0]] = new_v1
        points[simplex[1]] = new_v2
        points[simplex[2]] = new_v3
        
        # Add small random perturbation to escape local minima
        points += np.random.normal(0, 0.001, points.shape)
        points /= np.linalg.norm(points, axis=1, keepdims=True)
        
        # Check time
        if time.time() - start_time > time_limit - 5.0:
            break
            
    return best_points

# EVOLVE-BLOCK-END
