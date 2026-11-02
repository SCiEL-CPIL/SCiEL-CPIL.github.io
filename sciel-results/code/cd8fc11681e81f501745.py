"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    """
    Place N=11 unit regular hexagons inside a regular hexagon, minimizing outer side length.
    Uses a jittered lattice approach with continuous optimization.
    """
    start_time = time.time()
    deadline = 150.0  # seconds
    
    # Generate hexagon vertices for a unit regular hexagon centered at origin
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    hex_angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_verts = np.column_stack([np.cos(hex_angles), np.sin(hex_angles)])
    
    def hexagon_vertices(center, angle_deg, side=1.0):
        """Return vertices of a regular hexagon with given center, rotation, and side length."""
        theta = np.radians(angle_deg)
        R = np.array([[np.cos(theta), -np.sin(theta)],
                      [np.sin(theta), np.cos(theta)]])
        return center + side * (R @ hex_verts.T).T
    
    def polygon_area_signed(verts):
        """Compute signed area of a polygon given its vertices."""
        n = len(verts)
        area = 0.0
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return 0.5 * area
    
    def polygon_intersection_area(verts1, verts2):
        """
        Compute area of intersection of two convex polygons using Sutherland-Hodgman clipping.
        Returns 0 if no intersection.
        """
        # Clip polygon1 by each edge of polygon2
        def clip_polygon(subject, clip_edge_start, clip_edge_end):
            if len(subject) == 0:
                return []
            
            def is_inside(p):
                # Check if point p is inside (to the left of) the directed line from clip_edge_start to clip_edge_end
                dx = clip_edge_end[0] - clip_edge_start[0]
                dy = clip_edge_end[1] - clip_edge_start[1]
                px = p[0] - clip_edge_start[0]
                py = p[1] - clip_edge_start[1]
                return dx * py - dy * px >= -1e-10
            
            def line_intersection(p1, p2, q1, q2):
                # Find intersection of line p1-p2 with line q1-q2
                x1, y1 = p1
                x2, y2 = p2
                x3, y3 = q1
                x4, y4 = q2
                denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
                if abs(denom) < 1e-12:
                    return None
                t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
                if t < -1e-10 or t > 1 + 1e-10:
                    return None
                ix = x1 + t * (x2 - x1)
                iy = y1 + t * (y2 - y1)
                return np.array([ix, iy])
            
            output = []
            input_list = list(subject)
            n = len(input_list)
            for i in range(n):
                current = input_list[i]
                next_pt = input_list[(i + 1) % n]
                current_inside = is_inside(current)
                next_inside = is_inside(next_pt)
                
                if current_inside:
                    if next_inside:
                        output.append(current)
                    else:
                        inter = line_intersection(current, next_pt, clip_edge_start, clip_edge_end)
                        if inter is not None:
                            output.append(inter)
                else:
                    if next_inside:
                        inter = line_intersection(current, next_pt, clip_edge_start, clip_edge_end)
                        if inter is not None:
                            output.append(inter)
                        output.append(next_pt)
            return output
        
        # Initialize with polygon1
        clipped = [v.copy() for v in verts1]
        
        # Clip by each edge of polygon2
        for i in range(len(verts2)):
            if len(clipped) == 0:
                break
            edge_start = verts2[i]
            edge_end = verts2[(i + 1) % len(verts2)]
            clipped = clip_polygon(clipped, edge_start, edge_end)
        
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(np.array(clipped)))
    
    def overlap_penalty(centers, angles, side=1.0):
        """
        Compute total overlap area between all pairs of inner hexagons.
        """
        n = len(centers)
        total_overlap = 0.0
        for i in range(n):
            vi = hexagon_vertices(centers[i], angles[i], side)
            for j in range(i + 1, n):
                vj = hexagon_vertices(centers[j], angles[j], side)
                # Quick bounding box check
                if (max(vi[:, 0].max(), vj[:, 0].max()) - min(vi[:, 0].min(), vj[:, 0].min()) < 0.1 or
                    max(vi[:, 1].max(), vj[:, 1].max()) - min(vi[:, 1].min(), vj[:, 1].min()) < 0.1):
                    continue
                total_overlap += polygon_intersection_area(vi, vj)
        return total_overlap
    
    def enclosing_hexagon_side_length(centers, angles, outer_center, outer_angle, side=1.0):
        """
        Compute the minimum side length of a regular hexagon (with given center and orientation)
        that contains all inner hexagons.
        """
        # The enclosing hexagon has 6 sides. For a regular hexagon with side length L,
        # the apothem (distance from center to side) is L * sqrt(3)/2.
        # A point is inside if its projection onto each of the 6 normal directions is within the apothem.
        
        # The 6 normal directions for a hexagon rotated by outer_angle
        normals = []
        for k in range(6):
            angle = np.radians(outer_angle + k * 60)
            normals.append(np.array([np.cos(angle), np.sin(angle)]))
        
        max_proj = 0.0
        for i in range(len(centers)):
            vi = hexagon_vertices(centers[i], angles[i], side)
            for v in vi:
                for n_vec in normals:
                    proj = np.dot(v - outer_center, n_vec)
                    if proj > max_proj:
                        max_proj = proj
        
        # apothem = L * sqrt(3)/2, so L = max_proj / (sqrt(3)/2) = 2*max_proj/sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def objective(params):
        """
        Objective function: outer side length + penalty for overlaps.
        params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_x, outer_y, outer_angle]
        """
        n_inner = N
        # Extract inner hexagon parameters
        centers = []
        angles = []
        for i in range(n_inner):
            x = params[3 * i]
            y = params[3 * i + 1]
            a = params[3 * i + 2]
            centers.append(np.array([x, y]))
            angles.append(a)
        
        outer_center = np.array([params[3 * n_inner], params[3 * n_inner + 1]])
        outer_angle = params[3 * n_inner + 2]
        
        # Compute outer side length
        side_len = enclosing_hexagon_side_length(centers, angles, outer_center, outer_angle)
        
        # Compute overlap penalty
        overlap = overlap_penalty(centers, angles)
        
        # Penalize overlaps heavily
        penalty = 100.0 * overlap
        
        return side_len + penalty
    
    # Initial configuration: standard hexagonal lattice of 11 hexagons
    # Center hexagon at origin, 6 around it, 4 more in the second ring
    # Axial coordinates for a 11-hex cluster:
    # (0,0)
    # (1,0), (-1,0), (0,1), (0,-1), (1,-1), (-1,1)
    # (2,0), (-2,0), (1,1), (-1,-1)
    # Wait, that's 1+6+4=11. Let me use proper axial to Cartesian.
    
    def axial_to_cartesian(q, r, size=1.0):
        """Convert axial coordinates to Cartesian for a hexagon grid."""
        x = size * 3.0 / 2.0 * q
        y = size * np.sqrt(3) * (r + q / 2.0)
        return np.array([x, y])
    
    # Define the 11 hexagon positions in axial coordinates
    axial_coords = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, 0), (-2, 0), (1, 1), (-1, -1)
    ]
    
    # Initial parameters
    init_params = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        init_params.extend([x, y, 0.0])
    init_params.extend([0.0, 0.0, 0.0])  # outer center and angle
    
    # Try multiple random perturbations and optimize
    best_result = None
    best_score = float('inf')
    
    np.random.seed(42)
    
    num_attempts = 50
    for attempt in range(num_attempts):
        if time.time() > deadline - 10:
            break
        
        # Perturb initial configuration
        params = init_params.copy()
        
        # Small random perturbations to break symmetry
        for i in range(3 * N):
            if i % 3 < 2:  # x and y
                params[i] += np.random.uniform(-0.1, 0.1)
            else:  # angle
                params[i] += np.random.uniform(-10, 10)
        
        # Random outer angle
        params[3 * N + 2] = np.random.uniform(0, 60)
        
        # Optimize
        try:
            result = minimize(objective, params, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6})
            
            if result.success or result.fun < best_score:
                # Check for overlaps
                centers = []
                angles = []
                for i in range(N):
                    centers.append(np.array([result.x[3 * i], result.x[3 * i + 1]]))
                    angles.append(result.x[3 * i + 2])
                
                outer_center = np.array([result.x[3 * N], result.x[3 * N + 1]])
                outer_angle = result.x[3 * N + 2]
                
                overlap = overlap_penalty(centers, angles)
                if overlap < 1e-6:
                    side_len = enclosing_hexagon_side_length(centers, angles, outer_center, outer_angle)
                    if side_len < best_score:
                        best_score = side_len
                        best_result = (result.x, side_len)
        except Exception as e:
            continue
    
    # If no valid result found, use initial configuration
    if best_result is None:
        # Use the initial lattice without optimization
        centers = []
        angles = []
        for q, r in axial_coords:
            x, y = axial_to_cartesian(q, r)
            centers.append(np.array([x, y]))
            angles.append(0.0)
        
        outer_center = np.array([0.0, 0.0])
        outer_angle = 0.0
        side_len = enclosing_hexagon_side_length(centers, angles, outer_center, outer_angle)
        
        inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
        return inner, (0.0, 0.0), side_len, 0.0
    
    # Extract best result
    params, side_len = best_result
    centers = []
    angles = []
    for i in range(N):
        centers.append(np.array([params[3 * i], params[3 * i + 1]]))
        angles.append(params[3 * i + 2])
    
    outer_center = np.array([params[3 * N], params[3 * N + 1]])
    outer_angle = params[3 * N + 2]
    
    inner = [[centers[i][0], centers[i][1], angles[i]] for i in range(N)]
    return inner, (outer_center[0], outer_center[1]), side_len, outer_angle
# EVOLVE-BLOCK-END
