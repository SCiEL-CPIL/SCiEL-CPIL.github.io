"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial import ConvexHull
import time

N = 11

def hexagon_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with side length 'side' rotated by angle_deg degrees."""
    angle_rad = np.radians(angle_deg)
    # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base_angles = np.array([0, 60, 120, 180, 240, 300]) + angle_deg
    base_angles_rad = np.radians(base_angles)
    x = side * np.cos(base_angles_rad)
    y = side * np.sin(base_angles_rad)
    return np.column_stack([x + cx, y + cy])

def distance_point_to_polygon(px, py, vertices):
    """Compute the minimum distance from point (px, py) to the boundary of a polygon given by vertices.
    If the point is inside, return negative distance (penalty)."""
    n = len(vertices)
    min_dist = np.inf
    inside = 0
    
    # Check if point is inside using ray casting
    x, y = px, py
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # Ray casting
        if ((y1 > y) != (y2 > y)) and (x < (x2 - x1) * (y - y1) / (y2 - y1) + x1):
            inside = 1 - inside
    
    # Distance to each edge
    for i in range(n):
        x1, y1 = vertices[i]
        x2, y2 = vertices[(i + 1) % n]
        # Project point onto line segment
        dx = x2 - x1
        dy = y2 - y1
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            d = np.sqrt((px - x1)**2 + (py - y1)**2)
        else:
            t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            d = np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
        min_dist = min(min_dist, d)
    
    if inside:
        return -min_dist
    else:
        return min_dist

def min_distance_between_hexagons(verts1, verts2):
    """Compute minimum distance between two convex polygons. Negative if they overlap."""
    # Check if any vertex of poly1 is inside poly2 or vice versa
    for v in verts1:
        d = distance_point_to_polygon(v[0], v[1], verts2)
        if d < 0:
            return d  # Overlapping
    for v in verts2:
        d = distance_point_to_polygon(v[0], v[1], verts1)
        if d < 0:
            return d  # Overlapping
    
    # Compute minimum distance between edges
    min_dist = np.inf
    n1 = len(verts1)
    n2 = len(verts2)
    for i in range(n1):
        x1, y1 = verts1[i]
        x2, y2 = verts1[(i + 1) % n1]
        for j in range(n2):
            x3, y3 = verts2[j]
            x4, y4 = verts2[(j + 1) % n2]
            # Distance between two line segments
            d = segment_distance(x1, y1, x2, y2, x3, y3, x4, y4)
            min_dist = min(min_dist, d)
    return min_dist

def segment_distance(x1, y1, x2, y2, x3, y3, x4, y4):
    """Compute minimum distance between two line segments."""
    def point_segment_distance(px, py, ax, ay, bx, by):
        dx = bx - ax
        dy = by - ay
        len_sq = dx * dx + dy * dy
        if len_sq == 0:
            return np.sqrt((px - ax)**2 + (py - ay)**2)
        t = max(0, min(1, ((px - ax) * dx + (py - ay) * dy) / len_sq))
        proj_x = ax + t * dx
        proj_y = ay + t * dy
        return np.sqrt((px - proj_x)**2 + (py - proj_y)**2)
    
    d1 = point_segment_distance(x1, y1, x3, y3, x4, y4)
    d2 = point_segment_distance(x2, y2, x3, y3, x4, y4)
    d3 = point_segment_distance(x3, y3, x1, y1, x2, y2)
    d4 = point_segment_distance(x4, y4, x1, y1, x2, y2)
    
    # Check for intersection
    # Cross product to check orientation
    def cross(ax, ay, bx, by, cx, cy):
        return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    
    d = cross(x1, y1, x2, y2, x3, y3)
    d2_ = cross(x1, y1, x2, y2, x4, y4)
    d3 = cross(x3, y3, x4, y4, x1, y1)
    d4_ = cross(x3, y3, x4, y4, x2, y2)
    
    if ((d > 0 and d2_ < 0) or (d < 0 and d2_ > 0)) and ((d3 > 0 and d4_ < 0) or (d3 < 0 and d4_ > 0)):
        return 0.0
    
    return min(d1, d2, d3, d4)

def objective(params, N_inner):
    """Objective function: maximize minimum clearance (minimize negative clearance).
    params: [x1, y1, a1, x2, y2, a2, ..., xN, yN, aN, outer_side, outer_angle]
    """
    penalty_weight = 1000.0
    min_clearance = np.inf
    total_penalty = 0.0
    
    # Extract parameters
    inner_params = params[:3 * N_inner]
    outer_side = params[3 * N_inner]
    outer_angle = params[3 * N_inner + 1]
    
    # Compute inner hexagon vertices
    inner_verts = []
    for i in range(N_inner):
        x = inner_params[3 * i]
        y = inner_params[3 * i + 1]
        angle = inner_params[3 * i + 2]
        verts = hexagon_vertices(x, y, 1.0, angle)
        inner_verts.append(verts)
    
    # Compute outer hexagon vertices
    outer_verts = hexagon_vertices(0.0, 0.0, outer_side, outer_angle)
    
    # Check each inner hexagon against outer boundary
    for i in range(N_inner):
        for v in inner_verts[i]:
            d = distance_point_to_polygon(v[0], v[1], outer_verts)
            if d < 0:
                total_penalty += penalty_weight * (-d)
            min_clearance = min(min_clearance, max(0, d))
    
    # Check pairs of inner hexagons
    for i in range(N_inner):
        for j in range(i + 1, N_inner):
            d = min_distance_between_hexagons(inner_verts[i], inner_verts[j])
            if d < 0:
                total_penalty += penalty_weight * (-d)
            min_clearance = min(min_clearance, max(0, d))
    
    # We want to maximize min_clearance, so minimize -min_clearance + penalty
    if np.isinf(min_clearance):
        min_clearance = 0.0
    
    return -min_clearance + total_penalty

def construct():
    """Construct packing of N unit hexagons inside a regular hexagon."""
    N_inner = N
    
    # Initial guess: place hexagons in a grid pattern
    # Side length 1 hexagons have width sqrt(3) and height 2
    # Place them in rows
    rows = 3
    cols = 4  # 12 hexagons, but we need 11
    # Adjust to fit 11
    # Let's try a different layout
    
    # Simple initial layout: spread out
    x_positions = []
    y_positions = []
    for i in range(N_inner):
        # Place in a rough grid
        row = i // 4
        col = i % 4
        x = (col - 1.5) * 1.8
        y = (row - 1) * 1.8
        x_positions.append(x)
        y_positions.append(y)
    
    # Initial parameters
    x0 = []
    for i in range(N_inner):
        x0.append(x_positions[i])
        x0.append(y_positions[i])
        x0.append(0.0)  # angle
    x0.append(5.0)   # outer side length
    x0.append(0.0)   # outer angle
    
    # Try multiple random restarts
    best_result = None
    best_score = np.inf
    
    np.random.seed(42)
    
    for restart in range(5):
        if restart == 0:
            init = np.array(x0)
        else:
            # Random perturbation
            init = np.array(x0)
            init[:3*N_inner] += np.random.randn(3*N_inner) * 2.0
            init[3*N_inner] = np.random.uniform(4.0, 8.0)
            init[3*N_inner + 1] = np.random.uniform(0, 60)
        
        # Optimize
        try:
            result = minimize(
                objective,
                init,
                args=(N_inner,),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-6}
            )
            
            if result.success or result.fun < best_score:
                best_score = result.fun
                best_result = result.x
        except Exception as e:
            continue
    
    if best_result is None:
        # Fallback: return a simple valid configuration
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract results
    inner = []
    for i in range(N_inner):
        x = best_result[3 * i]
        y = best_result[3 * i + 1]
        angle = best_result[3 * i + 2]
        inner.append([x, y, angle])
    
    outer_side = best_result[3 * N_inner]
    outer_angle = best_result[3 * N_inner + 1]
    
    return inner, (0.0, 0.0), outer_side, outer_angle

# EVOLVE-BLOCK-START
# The construct function is defined above
# EVOLVE-BLOCK-END
