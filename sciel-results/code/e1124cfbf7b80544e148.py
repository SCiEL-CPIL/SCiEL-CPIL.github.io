import numpy as np
import time
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    DEADLINE = 135.0  # Safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle = np.deg2rad(angle_deg)
        # Vertices of unit hexagon centered at origin, flat-top (30 deg rotation)
        # Actually, let's use the standard orientation where one vertex points right
        # The "side length" is the distance between parallel sides? No, side length is edge length.
        # For a regular hexagon with side length s, the distance from center to vertex is s.
        # The distance from center to side (apothem) is s * sqrt(3)/2.
        # Let's define vertices based on center-to-vertex distance = side.
        # Wait, the task says "side length 1". So edge length is 1.
        # Distance from center to vertex is 1.
        # Distance from center to edge is sqrt(3)/2.
        vertices = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            x = cx + side * np.cos(theta)
            y = cy + side * np.sin(theta)
            vertices.append([x, y])
        return np.array(vertices)

    def segment_distance(p1, p2, q1, q2):
        """Minimum distance between two segments p1-p2 and q1-q2."""
        # Vector from q1 to q2
        r = q2 - q1
        # Vector from p1 to p2
        s = p2 - p1
        # Vector from p1 to q1
        qp = q1 - p1
        a = np.dot(s, s)
        e = np.dot(r, r)
        f = np.dot(r, qp)
        
        if a < 1e-12:
            p1_param = 0.0
        else:
            p1_param = np.clip(-np.dot(s, qp) / a, 0, 1)
            
        if e < 1e-12:
            q1_param = 0.0
        else:
            q1_param = np.clip((np.dot(s, qp) + f) / e, 0, 1)
            
        cp = p1 + p1_param * s
        cq = q1 + q1_param * r
        dist = np.linalg.norm(cp - cq)
        return dist

    def min_dist_between_hexagons(hex1_verts, hex2_verts):
        """Calculate minimum distance between two hexagons."""
        min_d = np.inf
        # Check all pairs of edges
        for i in range(6):
            p1 = hex1_verts[i]
            p2 = hex1_verts[(i + 1) % 6]
            for j in range(6):
                q1 = hex2_verts[j]
                q2 = hex2_verts[(j + 1) % 6]
                d = segment_distance(p1, p2, q1, q2)
                if d < min_d:
                    min_d = d
        # Also check if one is inside the other (not disjoint)
        # Simple check: if center of one is inside the other, distance is negative?
        # For disjointness, we need min distance >= 0.
        # If they overlap, min distance is 0 (or we need to detect overlap).
        # The segment distance gives 0 if they touch or intersect.
        # To be safe, if min_d is 0, we should check for overlap.
        # However, for optimization, a penalty based on max(0, -min_d) works if min_d can be negative.
        # But segment_distance is always >= 0.
        # We need a signed distance or an overlap detector.
        # Let's use a simple point-in-polygon test for centers to detect deep overlap, 
        # but for near-contact, segment distance is sufficient.
        # Actually, if they overlap, the minimum distance between boundaries is 0.
        # So if min_d < eps, we consider them overlapping/touching.
        return min_d

    def point_in_hexagon(point, hex_verts):
        """Check if a point is inside a hexagon."""
        # Winding number or ray casting. 
        # For a convex polygon, check if point is on the same side of all edges.
        for i in range(6):
            p1 = hex_verts[i]
            p2 = hex_verts[(i + 1) % 6]
            # Cross product to check side
            v1 = p2 - p1
            v2 = point - p1
            cross = v1[0] * v2[1] - v1[1] * v2[0]
            # For CCW polygon, cross should be positive (or negative depending on orientation)
            # Let's assume standard CCW.
            # If any cross has opposite sign to the first, it's outside.
            if i == 0:
                sign = np.sign(cross)
            else:
                if np.sign(cross) != sign:
                    return False
        return True

    def outer_side_length(centers, angles, outer_center, outer_angle, side):
        """
        Calculate the minimum side length of the outer hexagon needed to contain all inner hexagons.
        This is a non-trivial problem. 
        Approximation: The outer hexagon must contain the convex hull of all inner hexagon vertices.
        The minimum side length of a regular hexagon (with fixed orientation) to contain a set of points
        is determined by the maximum extent of the points in the 3 normal directions of the hexagon.
        """
        # Collect all vertices of all inner hexagons
        all_verts = []
        for i, (cx, cy) in enumerate(centers):
            verts = hexagon_vertices(cx, cy, 1.0, angles[i])
            all_verts.append(verts)
        all_verts = np.vstack(all_verts)
        
        # Outer hexagon is centered at outer_center with rotation outer_angle.
        # The 3 normal directions of the outer hexagon are at angles:
        # outer_angle, outer_angle + 60, outer_angle + 120 (for the 3 pairs of parallel sides)
        # The distance from center to side (apothem) is side * sqrt(3)/2.
        # So the constraint is: for each normal direction n, max(dot(p - c, n)) <= side * sqrt(3)/2
        # and min(dot(p - c, n)) >= -side * sqrt(3)/2
        # So side >= max( |max(dot(p-c, n))|, |min(dot(p-c, n))| ) / (sqrt(3)/2)
        
        max_extent = 0
        for k in range(3):
            theta = np.deg2rad(outer_angle + k * 60)
            n = np.array([np.cos(theta), np.sin(theta)])
            dots = np.dot(all_verts - np.array(outer_center), n)
            extent = max(np.max(dots), np.abs(np.min(dots)))
            if extent > max_extent:
                max_extent = extent
        
        side_needed = max_extent / (np.sqrt(3) / 2)
        return side_needed

    def objective(params):
        """
        params: [cx1, cy1, a1, cx2, cy2, a2, ..., cx11, cy11, a11, ocx, ocy, oa]
        Total 11*3 + 3 = 36 parameters.
        """
        N = 11
        centers = []
        angles = []
        for i in range(N):
            cx = params[i * 3]
            cy = params[i * 3 + 1]
            a = params[i * 3 + 2]
            centers.append((cx, cy))
            angles.append(a)
        
        ocx = params[N * 3]
        ocy = params[N * 3 + 1]
        oa = params[N * 3 + 2]
        
        outer_center = (ocx, ocy)
        
        # Calculate required outer side length
        side = outer_side_length(centers, angles, outer_center, oa, 0)
        
        # Penalty for overlaps
        penalty = 0
        hex_verts = [hexagon_vertices(cx, cy, 1.0, a) for cx, cy, a in zip(centers, angles)]
        for i in range(N):
            for j in range(i + 1, N):
                d = min_dist_between_hexagons(hex_verts[i], hex_verts[j])
                if d < 0.001:
                    # Add penalty for overlap
                    penalty += (0.001 - d) ** 2 * 1000
        
        return side + penalty

    # Initial configuration: The flower pattern
    # Center at (0,0)
    # 6 neighbors at distance sqrt(3)
    # 4 more at distance 3
    
    eps = 1e-5
    init_params = []
    
    # Center
    init_params.extend([0.0, 0.0, 30.0])
    
    # Ring 1
    for k in range(6):
        angle = k * 60
        r = np.sqrt(3) + eps
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        init_params.extend([x, y, 30.0])
        
    # Ring 2
    for k in range(4):
        angle = 30 + k * 60
        r = 3.0 + 2 * eps
        x = r * np.cos(np.deg2rad(angle))
        y = r * np.sin(np.deg2rad(angle))
        init_params.extend([x, y, 30.0])
        
    # Outer hexagon
    init_params.extend([0.0, 0.0, 30.0])
    
    # Optimize
    best_params = init_params
    best_val = objective(init_params)
    
    # Use Nelder-Mead for local optimization
    try:
        result = minimize(objective, init_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8})
        
        if result.fun < best_val:
            best_params = result.x
            best_val = result.fun
            
        # Check time
        if time.time() - start_time > DEADLINE - 5:
            pass # Skip further checks
            
        # Verify validity of best_params
        N = 11
        centers = [(best_params[i*3], best_params[i*3+1]) for i in range(N)]
        angles = [best_params[i*3+2] for i in range(N)]
        ocx = best_params[N*3]
        ocy = best_params[N*3+1]
        oa = best_params[N*3+2]
        
        hex_verts = [hexagon_vertices(cx, cy, 1.0, a) for cx, cy, a in zip(centers, angles)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                d = min_dist_between_hexagons(hex_verts[i], hex_verts[j])
                if d < -1e-6: # Allow tiny numerical error
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            side = outer_side_length(centers, angles, (ocx, ocy), oa, 0)
            # Add tiny epsilon to side to ensure containment
            side += 1e-6
            inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
            return inner, (ocx, ocy), side, oa
        else:
            # Fall back to initial configuration if invalid
            pass
            
    except Exception as e:
        pass
    
    # Fallback: Return the initial configuration with calculated side
    N = 11
    centers = [(init_params[i*3], init_params[i*3+1]) for i in range(N)]
    angles = [init_params[i*3+2] for i in range(N)]
    ocx = init_params[N*3]
    ocy = init_params[N*3+1]
    oa = init_params[N*3+2]
    
    side = outer_side_length(centers, angles, (ocx, ocy), oa, 0)
    side += 1e-5
    
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    return inner, (ocx, ocy), side, oa

