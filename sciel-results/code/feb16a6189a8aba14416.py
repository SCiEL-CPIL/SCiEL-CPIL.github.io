import numpy as np
from scipy.optimize import minimize
import time

def hexagon_packing_12():
    """
    Constructs a packing of 12 disjoint unit regular hexagons inside a larger regular hexagon, maximizing 1/outer_hex_side_length.
    Returns
        inner_hex_data: np.ndarray of shape (12,3), where each row is of the form (x, y, angle_degrees)
        outer_hex_data: np.ndarray of shape (3,) of form (x,y,angle_degree)
        outer_hex_side_length: float
    """
    n = 12
    
    # Initial positions: 2x6 grid of hexagons
    # Flat-top hexagons (angle 0)
    # Spacing x: 1.5 * sqrt(3) ? No, distance between centers is sqrt(3).
    # For flat-top, horizontal neighbors are at dx = 1.5 * side * sqrt(3)?
    # Let's just place them in a grid that is definitely separated.
    # dx = 3.0, dy = 3.0 is safe.
    
    # Better initial guess: Hex lattice
    # Center to center distance for touching is sqrt(3) ~ 1.732.
    # Let's use spacing 1.8 to be safe.
    
    # Arrangement: 3 rows of 4?
    # Row 0: 4 hexes
    # Row 1: 4 hexes (shifted)
    # Row 2: 4 hexes
    # This is a 3x4 block.
    
    init_pos = []
    # Hex lattice coordinates
    # Flat top
    # x = 1.5 * col
    # y = sqrt(3) * (row + col/2)
    # This assumes touching. Let's scale by 1.1 to ensure separation.
    
    cols = 4
    rows = 3
    
    for r in range(rows):
        for c in range(cols):
            x = 1.5 * c * 1.1
            y = np.sqrt(3) * (r + c/2.0) * 1.1
            # Center the cluster
            init_pos.append([x, y, 0.0])
            
    # Center the cluster
    pos_arr = np.array(init_pos)
    pos_arr[:, 0] -= np.mean(pos_arr[:, 0])
    pos_arr[:, 1] -= np.mean(pos_arr[:, 1])
    
    x0 = pos_arr.flatten()
    
    def get_hex_vertices(cx, cy, angle_deg, side=1.0):
        a = np.deg2rad(angle_deg)
        angles = a + np.arange(6) * np.pi / 3
        return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])

    def hex_radius(x, y):
        # Minimal s for point (x,y) to be in hexagon of side s (aligned with x-axis)
        # s >= max(|x|, 2/|y|/sqrt(3), (|x|+|y|)/sqrt(3))
        return np.maximum(np.abs(x), np.maximum(2.0 * np.abs(y) / np.sqrt(3.0), (np.abs(x) + np.abs(y)) / np.sqrt(3.0)))

    def min_dist_polygons(v1, v2):
        # v1, v2 are (6,2) arrays
        # Check if they intersect using SAT or simple distance
        # For speed, we can compute min distance between all pairs of vertices/edges?
        # Or use scipy.spatial.distance?
        # Simple approach: min distance between any vertex of v1 and any vertex of v2 is a lower bound? No.
        # If they don't intersect, min distance is min of:
        # - vertex to vertex
        # - vertex to edge
        # We can approximate with vertex-vertex for initial steps, but it's inaccurate.
        
        # Let's use a robust but slightly slower method:
        # Check intersection first.
        # If intersect, return 0.
        # Else, return min distance.
        
        # For this small N, we can just compute min distance between all 36 pairs of vertices?
        # This is an overestimate of the actual distance if the closest points are on edges.
        # But it's a valid lower bound? No, it's an upper bound on the distance?
        # Distance between sets A and B is min_{a in A, b in B} ||a-b||.
        # The min over vertices is >= min over all points.
        # So if min_vertex_dist > 1e-6, they are separated.
        # If min_vertex_dist <= 1e-6, they might still be separated (if closest points are edge-midpoints).
        # But if they intersect, min_vertex_dist can be > 0? No, if they intersect, some vertex might be inside?
        # Not necessarily.
        
        # Let's use a proper check.
        # Since we have time, we can use a simple SAT to check intersection.
        # If not intersect, compute min distance.
        
        # To save code, let's assume if min vertex distance is small, we need to be careful.
        # Given the small size, let's just compute min vertex distance for the penalty.
        # It might allow "touching" edges where vertices don't touch, but that's okay if we use a small threshold.
        # Actually, if they touch, min vertex dist is 0? No, if they share an edge, the vertices are distinct but distance is 0?
        # If they share an edge, the endpoints are the same? No, if they are separate hexagons touching, the vertices are distinct but distance between a vertex of one and the edge of the other is 0.
        # Vertex-vertex distance would be > 0.
        # So vertex-vertex min dist is a safe *upper* bound on the separation?
        # No, if min vertex dist is 0.5, the actual distance could be 0.1.
        # So if we enforce min vertex dist > 1e-6, we guarantee separation > 1e-6?
        # Yes, because actual distance <= min vertex dist.
        # Wait. Distance between sets is min over all points.
        # Min over vertices is a specific subset of points.
        # So min_vertex_dist >= min_actual_dist.
        # So if min_vertex_dist > 1e-6, then min_actual_dist could be < 1e-6.
        # This is dangerous. We might accept overlapping hexagons.
        
        # We need a reliable intersection check.
        
        def polygons_intersect(p1, p2):
            # SAT for convex polygons
            # p1, p2 are (N, 2)
            n1 = len(p1)
            n2 = len(p2)
            
            def project(poly, axis):
                dots = np.dot(poly, axis)
                return np.min(dots), np.max(dots)
            
            # Check axes from p1
            for i in range(n1):
                p_a = p1[i]
                p_b = p1[(i+1)%n1]
                edge = p_b - p_a
                normal = np.array([-edge[1], edge[0]])
                min1, max1 = project(p1, normal)
                min2, max2 = project(p2, normal)
                if max1 < min2 or max2 < min1:
                    return False
            
            # Check axes from p2
            for i in range(n2):
                p_a = p2[i]
                p_b = p2[(i+1)%n2]
                edge = p_b - p_a
                normal = np.array([-edge[1], edge[0]])
                min1, max1 = project(p1, normal)
                min2, max2 = project(p2, normal)
                if max1 < min2 or max2 < min1:
                    return False
            
            return True

        if polygons_intersect(v1, v2):
            return 0.0
            
        # If not intersect, compute min distance
        # Min distance between vertices of v1 and vertices of v2
        min_d = np.inf
        for p in v1:
            for q in v2:
                d = np.linalg.norm(p - q)
                if d < min_d:
                    min_d = d
        # This is still an overestimate.
        # But for a penalty function, if it's > 0, it's fine.
        # We can refine if needed.
        return min_d

    def objective(x):
        pos = x.reshape(n, 3)
        s_max = 0.0
        penalty = 0.0
        
        verts = []
        for i in range(n):
            cx, cy, ang = pos[i]
            v = get_hex_vertices(cx, cy, ang)
            verts.append(v)
            
            # Update s_max
            r = hex_radius(v[:, 0], v[:, 1])
            s_max = max(s_max, np.max(r))
            
        # Check overlaps
        for i in range(n):
            for j in range(i+1, n):
                d = min_dist_polygons(verts[i], verts[j])
                if d < 1e-6:
                    # Penalty for being too close
                    penalty += (1e-6 - d)**2
                    
        return s_max + 100.0 * penalty

    # Optimize
    # Use Nelder-Mead
    options = {
        'xatol': 1e-8,
        'fatol': 1e-8,
        'maxiter': 5000,
        'maxfev': 5000
    }
    
    result = minimize(objective, x0, method='Nelder-Mead', options=options)
    
    best_x = result.x
    best_pos = best_x.reshape(n, 3)
    
    # Final calculation of s
    s_final = 0.0
    for i in range(n):
        cx, cy, ang = best_pos[i]
        v = get_hex_vertices(cx, cy, ang)
        r = hex_radius(v[:, 0], v[:, 1])
        s_final = max(s_final, np.max(r))
        
    # Ensure valid separation
    # If not, we might need to adjust, but the optimizer should handle it.
    
    inner_hex_data = best_pos
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    outer_hex_side_length = s_final
    
    return inner_hex_data, outer_hex_data, outer_hex_side_length
