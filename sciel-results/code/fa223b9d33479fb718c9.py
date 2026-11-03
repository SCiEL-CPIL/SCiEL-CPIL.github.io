"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
# Deadline guard
START_TIME = time.time()
TIME_LIMIT = 140.0  # Safety margin before 150s

def hex_vertices(side, center, angle_deg):
    """
    Compute vertices of a regular hexagon.
    side: side length
    center: (x, y)
    angle_deg: rotation in degrees
    Returns array of 6 vertices (x, y)
    """
    angle_rad = np.deg2rad(angle_deg)
    # Unit hexagon vertices (flat-top orientation at 0 deg? or pointy?)
    # Standard regular hexagon with side 1, center at origin.
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees if pointy-top?
    # Let's use flat-top: vertices at 30, 90, 150, 210, 270, 330?
    # Actually, the orientation doesn't matter for the shape, only the relative angles.
    # Let's define a base hexagon with vertices at (1, 0), (0.5, sqrt(3)/2), etc.
    # This is a pointy-top hexagon (vertex at (1,0)).
    angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    xs = np.cos(angles)
    ys = np.sin(angles)
    
    # Rotate
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    rx = cos_a * xs - sin_a * ys
    ry = sin_a * xs + cos_a * ys
    
    # Scale and Translate
    rx = rx * side + center[0]
    ry = ry * side + center[1]
    return np.column_stack((rx, ry))

def max_distance_to_lines(vertices, center, side, angle_deg):
    """
    Compute the minimum side length 'S' of an outer hexagon (center, side=S, angle)
    such that all vertices of 'vertices' are inside or on the boundary.
    
    The outer hexagon has 6 sides. The distance from the center to each side is the apothem.
    For a regular hexagon of side S, the apothem is S * sqrt(3) / 2.
    So S = 2 * apothem / sqrt(3).
    
    The condition for a point P to be inside the hexagon is that its distance to each of the 6 supporting lines (inwards) is >= 0.
    Equivalently, the projection of (P - center) onto the outward normal of each side must be <= apothem.
    Or: max over all 6 sides of (dot(P - center, n_i)) <= apothem.
    So apothem_required = max_i dot(P - center, n_i).
    S = 2 * apothem_required / sqrt(3).
    """
    # Outer hexagon normals
    # If vertices are at angles 0, 60, ..., then sides are perpendicular to angles 30, 90, ...
    # Wait, for a hexagon with vertices at (1,0), (1/2, sqrt(3)/2)...
    # Side 1 connects (1,0) and (1/2, sqrt(3)/2).
    # Midpoint is (3/4, sqrt(3)/4). Vector is (-1/2, sqrt(3)/2).
    # Normal is (sqrt(3)/2, 1/2) which is at 30 degrees.
    # So normals are at 30, 90, 150, 210, 270, 330.
    
    angle_rad = np.deg2rad(angle_deg)
    base_normals = np.deg2rad([30, 90, 150, 210, 270, 330])
    normals = base_normals + angle_rad
    
    nx = np.cos(normals)
    ny = np.sin(normals)
    
    # vertices: (N_inner * 6, 2)
    # center: (2,)
    # relative vertices
    rel_v = vertices - center
    
    # dot products: (N_inner * 6, 6)
    # rel_v[:, 0] * nx[None, :] + rel_v[:, 1] * ny[None, :]
    dots = rel_v[:, 0] * nx[None, :] + rel_v[:, 1] * ny[None, :]
    
    max_dot = np.max(dots)
    
    if max_dot <= 0:
        return 0.0 # Should not happen unless all inside center? No, side length must be positive.
    
    side_length = 2.0 * max_dot / np.sqrt(3.0)
    return side_length

def compute_min_outer_side(inner, center, angle_deg):
    """
    inner: list of (x, y, angle_deg)
    center: (x, y)
    angle_deg: outer rotation
    Returns: min side length of outer hexagon.
    """
    all_vertices = []
    for x, y, a in inner:
        verts = hex_vertices(1.0, (x, y), a)
        all_vertices.append(verts)
    
    if not all_vertices:
        return 0.0
        
    all_vertices = np.vstack(all_vertices)
    
    return max_distance_to_lines(all_vertices, center, 1.0, angle_deg)

def check_collision(inner, tol=1e-5):
    """
    Check if any two inner hexagons overlap.
    Returns True if no collision, False otherwise.
    """
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            v1 = hex_vertices(1.0, (inner[i][0], inner[i][1]), inner[i][2])
            v2 = hex_vertices(1.0, (inner[j][0], inner[j][1]), inner[j][2])
            
            # Check edge intersections
            # For each edge of v1, check against each edge of v2
            # Edge is defined by two points p1, p2
            # Segment intersection test
            
            for k in range(6):
                p1 = v1[k]
                p2 = v1[(k+1)%6]
                for m in range(6):
                    p3 = v2[m]
                    p4 = v2[(m+1)%6]
                    
                    if segments_intersect(p1, p2, p3, p4):
                        # Check if intersection is strictly interior or just touching?
                        # For packing, touching is allowed. Overlapping is not.
                        # If they intersect, we need to check if they overlap.
                        # A simple check: if the intersection point is strictly inside both polygons?
                        # Or just if the segments cross?
                        # If two convex polygons intersect, either an edge of one crosses an edge of the other,
                        # or one vertex of one is inside the other.
                        # If segments cross, they definitely overlap (unless just touching at endpoints).
                        # We treat crossing as collision.
                        return False
            
            # Check if any vertex of v1 is strictly inside v2
            for k in range(6):
                if point_in_polygon(v1[k], v2, tol):
                    return False
            # Check if any vertex of v2 is strictly inside v1
            for k in range(6):
                if point_in_polygon(v2[k], v1, tol):
                    return False
                    
    return True

def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

def on_segment(p, q, r):
    # Check if point q lies on segment pr
    if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and 
        q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
        return True
    return False

def segments_intersect(p1, p2, p3, p4):
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
        
    if d1 == 0 and on_segment(p3, p1, p4): return True
    if d2 == 0 and on_segment(p3, p2, p4): return True
    if d3 == 0 and on_segment(p1, p3, p2): return True
    if d4 == 0 and on_segment(p1, p4, p2): return True
        
    return False

def point_in_polygon(point, polygon, tol=1e-5):
    # Ray casting
    x, y = point
    n = len(polygon)
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = polygon[i]
        xj, yj = polygon[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi - tol):
            inside = not inside
        j = i
    return inside

# EVOLVE-BLOCK-START

def construct():
    """
    Construct 12 disjoint unit hexagons inside a regular hexagon, minimizing outer side length.
    """
    global START_TIME
    start_time = time.time()
    
    # Initial configuration: 12 hexagons in a 3x4 grid? Or 4x3?
    # Hexagonal lattice spacing:
    # For flat-top hexagons, horizontal distance between centers is 1.5 * side? No.
    # Let's use the standard hex packing.
    # Distance between centers of adjacent hexagons is 2 * apothem = sqrt(3).
    
    # Let's try a 3x4 grid of hexagons.
    # Rows: 0, 1, 2. Cols: 0, 1, 2, 3.
    # But hexagons in adjacent rows are offset.
    
    # Let's generate positions for 12 hexagons in a compact cluster.
    # A 3x4 rectangular block of hexagons in hex lattice.
    
    # Coordinates for hex lattice (pointy-top):
    # x = 1.5 * col
    # y = sqrt(3) * (row + 0.5 * (col % 2))
    
    # Let's center this block at (0,0).
    # Cols 0..3, Rows 0..2.
    # x values: 0, 1.5, 3.0, 4.5. Center x = 2.25.
    # y values depend on col.
    
    positions = []
    for r in range(3):
        for c in range(4):
            x = 1.5 * c
            y = np.sqrt(3) * (r + 0.5 * (c % 2))
            positions.append([x, y, 0.0])
            
    # Center the positions
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
        
    inner = positions
    
    # Verify no collision
    if not check_collision(inner):
        # Fallback: spread them out more
        # This shouldn't happen with correct lattice spacing
        pass
        
    # Optimize positions and angles
    # Variables: x1, y1, a1, ..., x12, y12, a12, cx, cy, outer_angle
    # Total 12*3 + 2 + 1 = 39 variables.
    
    def objective(vars):
        # vars: [x1..x12, y1..y12, a1..a12, cx, cy, outer_angle]
        inner_vars = vars[:36]
        cx, cy = vars[36], vars[37]
        outer_angle = vars[38]
        
        # Check collision penalty
        # If collision, return large penalty
        # For speed, we might skip collision check in objective and rely on constraints?
        # But scipy minimize doesn't handle constraints well with many variables.
        # Let's use a penalty method.
        
        # Reconstruct inner list
        inner_list = []
        for i in range(12):
            x = inner_vars[i*3]
            y = inner_vars[i*3+1]
            a = inner_vars[i*3+2]
            inner_list.append([x, y, a])
            
        # Collision check (expensive)
        # To speed up, we can check only close pairs?
        # Or just compute distance between centers. If < 2*sqrt(3) - eps, they might collide.
        # Actually, min distance between centers is sqrt(3) for touching.
        # So if dist < sqrt(3) - eps, collision.
        
        collision_penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                d = np.sqrt((xi-xj)**2 + (yi-yj)**2)
                if d < np.sqrt(3) - 1e-4:
                    collision_penalty += (np.sqrt(3) - d) ** 2
        
        # Compute outer side
        side = compute_min_outer_side(inner_list, (cx, cy), outer_angle)
        
        return side + 100.0 * collision_penalty
    
    # Initial guess
    x0 = []
    for p in inner:
        x0.append(p[0])
        x0.append(p[1])
        x0.append(p[2])
    x0.append(0.0) # cx
    x0.append(0.0) # cy
    x0.append(0.0) # outer angle
    
    best_vars = x0
    best_score = objective(x0)
    
    # Use Nelder-Mead or Powell? Nelder-Mead is derivative-free and robust.
    # But it can be slow. Let's try Powell.
    
    # We need to be careful with time.
    # Let's do a few iterations of local search.
    
    # To improve speed, we can vectorize the collision check?
    # Or just reduce the number of function evaluations.
    
    # Let's try a simple grid search for outer angle first?
    # No, let's just run minimize.
    
    # Constraints: None (handled by penalty)
    
    # To avoid getting stuck, we can try multiple random restarts if time permits.
    
    # For now, one optimization run.
    
    # Check time
    if time.time() - start_time > TIME_LIMIT * 0.5:
        # Not enough time for optimization, return initial
        pass
    else:
        # Run minimize
        # To speed up, we can reduce the collision check cost.
        # The current check_collision is O(N^2 * 36) which is fast enough.
        # But objective calls compute_min_outer_side which is O(N*6 * 6).
        
        # Let's use Powell method
        try:
            res = minimize(objective, x0, method='Powell', options={'maxiter': 200, 'xtol': 1e-6, 'ftol': 1e-6})
            if res.success or res.fun < best_score:
                if res.fun < best_score:
                    best_vars = res.x
                    best_score = res.fun
        except Exception as e:
            pass
            
    # Extract best solution
    best_inner = []
    for i in range(12):
        x = best_vars[i*3]
        y = best_vars[i*3+1]
        a = best_vars[i*3+2]
        best_inner.append([x, y, a])
        
    best_cx = best_vars[36]
    best_cy = best_vars[37]
    best_outer_angle = best_vars[38]
    
    # Final verification and polish
    # Check if solution is valid
    if check_collision(best_inner):
        final_side = compute_min_outer_side(best_inner, (best_cx, best_cy), best_outer_angle)
    else:
        # If invalid, fall back to initial
        final_side = compute_min_outer_side(inner, (0.0, 0.0), 0.0)
        best_inner = inner
        best_cx, best_cy = 0.0, 0.0
        best_outer_angle = 0.0
        
    return best_inner, (best_cx, best_cy), final_side, best_outer_angle

# EVOLVE-BLOCK-END
