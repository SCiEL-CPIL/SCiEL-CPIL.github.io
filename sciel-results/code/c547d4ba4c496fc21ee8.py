import numpy as np
import time
from scipy.optimize import minimize_scalar

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def polygons_disjoint_sat(p1, p2, eps=1e-5):
        """
        Returns True if two convex polygons are disjoint (allowing touching with eps tolerance).
        Uses Separating Axis Theorem (SAT).
        """
        for i in range(len(p1)):
            j = (i + 1) % len(p1)
            edge = p1[j] - p1[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        for i in range(len(p2)):
            j = (i + 1) % len(p2)
            edge = p2[j] - p2[i]
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) < 1e-10:
                continue
            normal /= np.linalg.norm(normal)
            
            proj1 = p1.dot(normal)
            proj2 = p2.dot(normal)
            
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2 - eps or max2 < min1 - eps:
                return True
        
        return False

    def check_validity(positions, angles):
        """
        positions: (12, 2) array of (x, y).
        angles: (12,) array of rotation angles in degrees.
        Returns True if all are pairwise disjoint.
        """
        vertices_list = []
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            verts = hexagon_vertices(x, y, 1.0, a)
            vertices_list.append(verts)
        
        for i in range(N):
            for j in range(i + 1, N):
                if not polygons_disjoint_sat(vertices_list[i], vertices_list[j]):
                    return False
        return True

    def get_min_enclosing_hexagon_side(positions, angles):
        """
        Calculates the side length of the smallest regular hexagon (centered at origin, aligned with axes initially, but we can rotate the frame)
        Wait, the outer hexagon can also rotate. 
        The problem statement says: "outer hexagon center (x, y), side length and rotation".
        However, the score is just the side length.
        We can assume the outer hexagon is centered at the centroid of the inner hexagons for optimal packing?
        Actually, for a set of convex shapes, the minimum enclosing regular hexagon side length is determined by the support functions.
        
        Let's simplify: 
        The outer hexagon is a regular hexagon. We can choose its center and rotation.
        To minimize the side length S, we should center it at the centroid of the inner configuration? 
        Not necessarily, but for symmetric clusters like the honeycomb, yes.
        
        Let's assume the outer hexagon is centered at the origin (we can translate the whole system to make the centroid 0) and we optimize its rotation angle theta_outer.
        
        For a fixed configuration of inner hexagons (positions and rotations), and a fixed outer hexagon rotation theta_outer, 
        the required side length S is:
        S = max over all inner vertices v of: 
            max( 
                v . n1, 
                v . n2, 
                ... 
            )
        where nk are the normals to the sides of the outer hexagon.
        
        Wait, the condition for a point p to be inside a regular hexagon of side S, centered at c, rotated by theta is:
        For each of the 6 sides, the projection of (p-c) onto the outward normal must be <= S * cos(30) = S * sqrt(3)/2?
        No.
        Let's derive the support function of a regular hexagon of side S.
        The apothem (distance from center to side) is h = S * sqrt(3) / 2.
        The condition is: max_k ( (p - c) . n_k ) <= h, where n_k are the unit normals to the sides.
        So S = (2 / sqrt(3)) * max_k ( (p - c) . n_k ).
        
        We want to minimize S over c and theta.
        For a fixed theta, the optimal c is such that the max and min projections are balanced?
        Actually, for a convex body, the minimum enclosing regular polygon (with fixed orientation) is centered at the midpoint of the "diameter" parallel to the normal? 
        It's easier to just optimize c and theta. But that adds 3 variables.
        
        Alternative: 
        The "kinematic closure" strategy suggests using a rigid cluster.
        If the cluster is rigid, we can compute its minimal enclosing regular hexagon side length by optimizing the orientation of the outer hexagon relative to the cluster.
        Since the cluster is fixed (up to global rotation), we can just try all 60 degrees of outer hexagon rotation relative to the cluster? 
        No, the outer hexagon can rotate continuously.
        
        However, for a symmetric cluster (like the 7-hexagon honeycomb), the minimal enclosing hexagon is likely aligned with the cluster or offset by 30 degrees.
        
        Let's implement a function that, given a configuration of vertices (from inner hexagons), finds the minimal S over outer rotation theta and center c.
        Actually, if we assume the center c is the centroid of the vertices, we can just optimize theta.
        
        Let's assume the outer hexagon is centered at the centroid of the inner hexagon centers.
        Then S(theta) = (2/sqrt(3)) * max_{v, k} ( (v - centroid) . n_k(theta) ).
        This is a function of theta. We can minimize it using scipy.optimize.minimize_scalar on theta in [0, 60).
        """
        # Collect all vertices
        verts = []
        for i in range(N):
            x, y = positions[i]
            a = angles[i]
            v = hexagon_vertices(x, y, 1.0, a)
            verts.append(v)
        all_verts = np.vstack(verts)
        
        centroid = np.mean(all_verts, axis=0)
        verts_centered = all_verts - centroid
        
        # Normals for outer hexagon at rotation theta
        # Side normals are at theta + 30, theta + 90, ...
        # Actually, if vertices are at theta, normals are at theta+30.
        # Let's define normals directly.
        
        def calc_S(theta):
            theta_rad = np.radians(theta)
            # Normals angles: theta_rad + np.pi/6 + np.arange(6)*np.pi/3
            normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi/3
            normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
            
            # Project all vertices onto all normals
            # (72, 2) dot (2, 6) -> (72, 6)
            projections = verts_centered @ normals.T
            max_proj = np.max(projections)
            return max_proj * 2.0 / np.sqrt(3.0)
        
        # Minimize S(theta) over theta in [0, 60)
        # The function is periodic with period 60 degrees.
        res = minimize_scalar(calc_S, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
        return res.fun, centroid

    # Pre-defined cluster shapes (local coordinates)
    # Shape 1: Honeycomb (1 center + 6 around + 5 more?) 
    # The standard honeycomb for 7 hexagons is very compact.
    # For 12, we can take a 7-hexagon cluster and add 5 more.
    # Or a 3x3 grid minus corners?
    
    # Let's generate a few candidate rigid configurations.
    # Configuration A: 7-hexagon honeycomb + 5 attached to the "outside"
    # Centers of 7-hexagon honeycomb (side 1, distance between centers sqrt(3)):
    # (0,0)
    # (sqrt(3), 0), (0, sqrt(3)), (-sqrt(3), sqrt(3)), (-sqrt(3), 0), (0, -sqrt(3)), (sqrt(3), -sqrt(3))
    # This is not a regular hexagon of centers. 
    # The centers of a honeycomb of hexagons are on a triangular lattice.
    # Let's use axial coordinates (q, r) for the centers.
    # Distance between centers = sqrt(3).
    # x = sqrt(3) * (q + r/2)
    # y = 1.5 * r
    
    def gen_honeycomb_12():
        # A compact shape. 
        # Row 0: 3 hexagons
        # Row 1: 4 hexagons
        # Row 2: 3 hexagons
        # Row 3: 2 hexagons
        # Total 12.
        # Let's try to make it more symmetric.
        # Maybe a hexagon of hexagons? 
        # A hexagon of side 2 (in terms of hexagons) has 1 + 6 + 12 = 19 hexagons. Too many.
        # A hexagon of side 1 has 7.
        # So we need 5 more.
        
        # Let's place 7 in a cluster, and 5 in a line? No, that's long.
        # Let's place them in a 4-3-3-2 pattern?
        
        # Let's try a specific set of axial coordinates that form a compact shape.
        # (0,0)
        # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)  <- 7 total
        # Add:
        # (2,0), (1,1), (0,2), (-1,2), (-2,1) <- 5 more. Total 12.
        # This forms a larger hexagon-like shape but missing the bottom corner?
        # Let's check the shape.
        coords = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1)
        ]
        positions = []
        for q, r in coords:
            x = np.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            positions.append([x, y])
        return np.array(positions)

    def gen_circle_12():
        # 1 center, 11 around? No, 12 total.
        # 1 center + 11 on a circle? 
        # Or 12 on a circle?
        # Let's try 12 on a circle of radius R.
        # Distance between neighbors on circle: 2*R*sin(pi/12) >= sqrt(3).
        # R >= sqrt(3) / (2*sin(pi/12)).
        # sin(15 deg) = (sqrt(6)-sqrt(2))/4 ~ 0.2588.
        # R >= 1.732 / 0.5176 ~ 3.35.
        # This is likely larger than the honeycomb.
        return None

    # We will test the honeycomb configuration with different global rotations.
    # The inner hexagons can also be rotated. 
    # In the honeycomb lattice, all hexagons are aligned. 
    # Is it beneficial to rotate some inner hexagons?
    # Probably not for a rigid cluster, as it might increase the bounding box.
    # Let's assume all inner hexagons are aligned (angle 0) relative to the cluster frame.
    # The cluster frame is rotated by alpha.
    # So inner hexagon i has angle alpha.
    
    base_positions = gen_honeycomb_12()
    base_angles = np.zeros(N) # All aligned
    
    # We can also try a configuration where the outer hexagon is rotated relative to the cluster.
    # But the `get_min_enclosing_hexagon_side` function already optimizes the outer hexagon rotation.
    # So we just need to optimize the global rotation of the cluster (alpha).
    
    def objective_cluster_rotation(alpha):
        """
        alpha: global rotation of the cluster in degrees.
        Returns the minimal outer hexagon side length.
        """
        # Rotate base positions by alpha
        alpha_rad = np.radians(alpha)
        c = np.cos(alpha_rad)
        s = np.sin(alpha_rad)
        rot_mat = np.array([[c, -s], [s, c]])
        
        rotated_positions = base_positions @ rot_mat.T
        
        # Inner hexagons are also rotated by alpha
        rotated_angles = base_angles + alpha
        
        # Check validity (should be valid since it's a rigid rotation of a valid cluster)
        # We assume the base cluster is valid.
        
        S, centroid = get_min_enclosing_hexagon_side(rotated_positions, rotated_angles)
        return S

    # Optimize alpha in [0, 60)
    # The system has 6-fold symmetry? 
    # The honeycomb cluster of 12 might not have 6-fold symmetry.
    # Let's check the shape:
    # (0,0)
    # (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)
    # (2,0), (1,1), (0,2), (-1,2), (-2,1)
    # This shape is NOT 6-fold symmetric. It is somewhat elongated.
    # So we should search the full 0-360 range, or 0-60 if we assume symmetry?
    # No, search 0-360 to be safe. But the function is periodic with 360.
    # Actually, the outer hexagon optimization handles the outer symmetry.
    # The cluster rotation alpha changes the shape relative to the outer hexagon.
    
    res_cluster = minimize_scalar(objective_cluster_rotation, bounds=(0, 360), method='bounded', options={'xatol': 1e-4})
    
    best_alpha = res_cluster.x
    best_S = res_cluster.fun
    
    # Get the final configuration
    alpha_rad = np.radians(best_alpha)
    c = np.cos(alpha_rad)
    s = np.sin(alpha_rad)
    rot_mat = np.array([[c, -s], [s, c]])
    
    final_positions = base_positions @ rot_mat.T
    final_angles = base_angles + best_alpha
    
    # Final validation
    if not check_validity(final_positions, final_angles):
        # Fallback: use the base configuration (alpha=0)
        final_positions = base_positions
        final_angles = base_angles
        S, centroid = get_min_enclosing_hexagon_side(final_positions, final_angles)
        best_S = S
    else:
        S, centroid = get_min_enclosing_hexagon_side(final_positions, final_angles)
        best_S = S
        
    # The outer hexagon center is the centroid
    outer_center = (float(centroid[0]), float(centroid[1]))
    
    # The outer hexagon angle is not returned by get_min_enclosing_hexagon_side currently.
    # We need to find the optimal outer angle theta_outer for the final configuration.
    # Let's re-calculate it.
    
    verts = []
    for i in range(N):
        x, y = final_positions[i]
        a = final_angles[i]
        v = hexagon_vertices(x, y, 1.0, a)
        verts.append(v)
    all_verts = np.vstack(verts)
    verts_centered = all_verts - centroid
    
    def calc_S_theta(theta):
        theta_rad = np.radians(theta)
        normal_angles = theta_rad + np.pi/6 + np.arange(6) * np.pi/3
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])
        projections = verts_centered @ normals.T
        max_proj = np.max(projections)
        return max_proj * 2.0 / np.sqrt(3.0)

    res_outer = minimize_scalar(calc_S_theta, bounds=(0, 60), method='bounded', options={'xatol': 1e-6})
    outer_angle = res_outer.x
    final_S = res_outer.fun
    
    # Construct inner list
    inner = []
    for i in range(N):
        inner.append([float(final_positions[i, 0]), float(final_positions[i, 1]), float(final_angles[i])])
        
    return inner, outer_center, float(final_S), float(outer_angle)

# EVOLVE-BLOCK-END
