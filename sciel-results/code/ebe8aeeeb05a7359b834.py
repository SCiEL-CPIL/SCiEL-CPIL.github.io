"""Packing 11 unit hexagons inside a regular hexagon using continuous optimization."""
import math
import time
import random
import numpy as np

N_HEX = 11
DEADLINE_SECONDS = 115.0  # Safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _edge_intersection(p1, p2, p3, p4):
    """Return intersection point of segments p1-p2 and p3-p4, or None."""
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4
    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
    if abs(denom) < 1e-12:
        return None
    t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
    u = ((x1 - x3) * (y1 - y2) - (y1 - y3) * (x1 - x2)) / denom
    if -1e-9 <= t <= 1 + 1e-9 and -1e-9 <= u <= 1 + 1e-9:
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))
    return None

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # Regular hexagon with flat top: vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with circumradius = side, the inradius = side * sqrt(3)/2
    # Using the standard form: |lx| <= side, |ly| <= side*sqrt(3)/2, and |lx + ly/sqrt(3)| <= side, etc.
    # Actually, let's use the 6 edge inequalities directly
    r = side
    # Edges of regular hexagon (pointy-top in local frame after rotation by a):
    # Vertex i is at angle i*pi/3 from center
    # Edge i connects vertex i and vertex i+1
    # The inward normal for edge i points toward center
    # For a regular hexagon, the distance from center to each edge is r*sqrt(3)/2
    inradius = r * math.sqrt(3) / 2.0
    # The 6 supporting lines: for k=0..5, the line is at distance inradius from center,
    # normal direction at angle k*pi/3 + pi/6 (perpendicular to the edge midpoint direction)
    # Actually, edge midpoint is at angle (i+0.5)*pi/3 = i*pi/3 + pi/6
    for k in range(6):
        theta = k * math.pi / 3.0 + math.pi / 6.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > inradius + 1e-9:
            return False
    return True

def _hexagon_overlap_area(v1, v2):
    """Compute approximate overlap area using Sutherland-Hodgman polygon clipping."""
    # Clip v1 by each edge of v2
    def clip_polygon(subject, clip_edge_a, clip_edge_b):
        if not subject:
            return []
        output = []
        # Determine inside/outside for the half-plane defined by clip_edge_a -> clip_edge_b
        # The interior is to the left of the directed edge (for CCW polygon)
        ax, ay = clip_edge_a
        bx, by = clip_edge_b
        ex = bx - ax
        ey = by - ay
        # Normal pointing inward (left of edge)
        # For CCW polygon, interior is left of each edge
        def inside(px, py):
            # Cross product (b-a) x (p-a) >= 0 means left
            return ex * (py - ay) - ey * (px - ax) >= -1e-9
        
        for i in range(len(subject)):
            cur = subject[i]
            nxt = subject[(i + 1) % len(subject)]
            cur_in = inside(cur[0], cur[1])
            nxt_in = inside(nxt[0], nxt[1])
            if cur_in:
                output.append(cur)
                if not nxt_in:
                    # Exiting: add intersection
                    inter = _edge_intersection(cur, nxt, clip_edge_a, clip_edge_b)
                    if inter:
                        output.append(inter)
            else:
                if nxt_in:
                    # Entering: add intersection
                    inter = _edge_intersection(cur, nxt, clip_edge_a, clip_edge_b)
                    if inter:
                        output.append(inter)
                    output.append(nxt)
        return output

    # Ensure both polygons are CCW
    def is_ccw(poly):
        area = 0
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i+1) % len(poly)]
            area += x1*y2 - x2*y1
        return area > 0

    poly1 = list(v1)
    poly2 = list(v2)
    if not is_ccw(poly1):
        poly1.reverse()
    if not is_ccw(poly2):
        poly2.reverse()

    # Clip poly1 by each edge of poly2
    clipped = poly1
    for i in range(len(poly2)):
        a = poly2[i]
        b = poly2[(i+1) % len(poly2)]
        clipped = clip_polygon(clipped, a, b)
        if not clipped:
            return 0.0

    if len(clipped) < 3:
        return 0.0

    # Compute area of clipped polygon
    area = 0
    for i in range(len(clipped)):
        x1, y1 = clipped[i]
        x2, y2 = clipped[(i+1) % len(clipped)]
        area += x1*y2 - x2*y1
    return abs(area) / 2.0

def _compute_outer_side(inner_hexes, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon that contains all inner hexagons.
    
    inner_hexes: list of (x, y, angle_deg)
    outer_center: (cx, cy)
    outer_angle: rotation of outer hexagon in degrees
    """
    cx, cy = outer_center
    oa = math.radians(outer_angle)
    
    # The outer hexagon with center (cx,cy), side s, rotation oa has 6 edges.
    # For each inner hexagon, we need all its vertices to be inside the outer hexagon.
    # The constraint for a vertex (px,py) to be inside the outer hexagon of side s:
    # For each of the 6 edges of the outer hexagon, the vertex must be on the interior side.
    # The distance from center to each edge is s*sqrt(3)/2.
    # The normal for edge k points outward at angle oa + k*pi/3 + pi/6.
    
    # For each inner hexagon vertex, compute the required side length.
    max_ratio = 0.0
    for (ix, iy, ia) in inner_hexes:
        verts = _hex_vertices(ix, iy, 1.0, ia)
        for (px, py) in verts:
            dx = px - cx
            dy = py - cy
            for k in range(6):
                theta = oa + k * math.pi / 3.0 + math.pi / 6.0
                nx = math.cos(theta)
                ny = math.sin(theta)
                dist = dx * nx + dy * ny
                if dist > max_ratio:
                    max_ratio = dist
    
    # The side length s must satisfy: s * sqrt(3)/2 >= max_ratio
    # So s >= max_ratio * 2 / sqrt(3)
    if max_ratio <= 0:
        return 1.0  # Degenerate case
    return max_ratio * 2.0 / math.sqrt(3.0)

def _objective(params):
    """Compute the outer side length given parameters.
    
    params: flat array of length N_HEX*3 + 3
    First N_HEX*3 elements: (x, y, angle_deg) for each inner hexagon
    Last 3 elements: (outer_cx, outer_cy, outer_angle_deg)
    """
    inner = []
    for i in range(N_HEX):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner.append((x, y, a))
    outer_cx = params[N_HEX*3]
    outer_cy = params[N_HEX*3+1]
    outer_angle = params[N_HEX*3+2]
    
    side = _compute_outer_side(inner, (outer_cx, outer_cy), outer_angle)
    
    # Add penalty for overlaps between inner hexagons
    overlap_penalty = 0.0
    for i in range(N_HEX):
        for j in range(i+1, N_HEX):
            v1 = _hex_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = _hex_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            # Quick bounding box check
            min_x1 = min(p[0] for p in v1)
            max_x1 = max(p[0] for p in v1)
            min_y1 = min(p[1] for p in v1)
            max_y1 = max(p[1] for p in v1)
            min_x2 = min(p[0] for p in v2)
            max_x2 = max(p[0] for p in v2)
            min_y2 = min(p[1] for p in v2)
            max_y2 = max(p[1] for p in v2)
            if min_x1 < max_x2 and min_x2 < max_x1 and min_y1 < max_y2 and min_y2 < max_y1:
                ov = _hexagon_overlap_area(v1, v2)
                if ov > 1e-6:
                    overlap_penalty += ov
    
    return side + 100.0 * overlap_penalty

def _initial_params():
    """Create initial parameters: hexagons in a row, outer centered."""
    params = np.zeros(N_HEX*3 + 3)
    # Place hexagons in a horizontal line with spacing 2.0 (touching)
    start_x = -(N_HEX-1) * 1.0
    for i in range(N_HEX):
        params[3*i] = start_x + i * 2.0
        params[3*i+1] = 0.0
        params[3*i+2] = 0.0
    # Outer hexagon centered at origin, no rotation
    params[N_HEX*3] = 0.0
    params[N_HEX*3+1] = 0.0
    params[N_HEX*3+2] = 0.0
    return params

def construct():
    start_time = time.time()
    
    # Try multiple random restarts with gradient-free optimization
    best_params = None
    best_score = float('inf')
    
    # First, try the initial configuration
    params = _initial_params()
    score = _objective(params)
    best_params = params.copy()
    best_score = score
    
    # Use Nelder-Mead simplex optimization from scipy
    from scipy.optimize import minimize
    
    n_restarts = 30
    max_iter = 2000
    
    for restart in range(n_restarts):
        if time.time() - start_time > DEADLINE_SECONDS - 5:
            break
        
        if restart == 0:
            p0 = _initial_params()
        else:
            # Random perturbation of the best solution found so far
            p0 = best_params.copy()
            # Add random noise
            noise_scale = 2.0 / (restart + 1)
            p0[:N_HEX*3] += np.random.normal(0, noise_scale, N_HEX*3)
            p0[N_HEX*3:] += np.random.normal(0, noise_scale, 3)
        
        try:
            result = minimize(
                _objective,
                p0,
                method='Nelder-Mead',
                options={
                    'maxiter': max_iter,
                    'xatol': 1e-6,
                    'fatol': 1e-8,
                    'adaptive': True
                }
            )
            if result.fun < best_score:
                best_score = result.fun
                best_params = result.x.copy()
        except Exception:
            pass
        
        if time.time() - start_time > DEADLINE_SECONDS - 10:
            break
    
    # Extract the best solution
    inner = []
    for i in range(N_HEX):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([float(x), float(y), float(a)])
    
    outer_cx = float(best_params[N_HEX*3])
    outer_cy = float(best_params[N_HEX*3+1])
    outer_angle = float(best_params[N_HEX*3+2])
    
    # Compute the actual side length (without overlap penalty)
    actual_side = _compute_outer_side(inner, (outer_cx, outer_cy), outer_angle)
    
    return inner, (outer_cx, outer_cy), actual_side, outer_angle

# EVOLVE-BLOCK-START
# (construct is defined above)
# EVOLVE-BLOCK-END
