"""Unit regular hexagons inside a regular hexagon: packing N=11 unit hexagons to minimize outer side length."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def hex_vertices(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx,cy) with given side length and rotation."""
    theta = math.radians(angle_deg)
    verts = []
    for i in range(6):
        a = theta + i * math.pi / 3.0
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(verts)

def point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    theta = -math.radians(angle_deg)
    lx = dx * math.cos(theta) - dy * math.sin(theta)
    ly = dx * math.sin(theta) + dy * math.cos(theta)
    
    # A regular hexagon with 'flat top' (angle 0) has vertices at angles 0, 60, 120, 180, 240, 300
    # The hexagon is the intersection of 6 half-planes.
    # For a hexagon with vertices at (s*cos(k*60), s*sin(k*60)), the edges are at distance s*sqrt(3)/2 from center.
    # Actually, for a regular hexagon with side s, the apothem is s*sqrt(3)/2.
    # The 6 edges are perpendicular to directions 30, 90, 150, 210, 270, 330 degrees (for flat-top) 
    # or 0, 60, 120, 180, 240, 300 (for pointy-top).
    
    # Let's use the standard: vertices at angles 0, 60, ... for angle_deg=0.
    # Edge normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Distance from center to each edge is s * sqrt(3) / 2.
    apothem = side * math.sqrt(3) / 2.0
    
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def hexagon_distance_to_edge(px, py, cx, cy, side, angle_deg):
    """Minimum signed distance from point to hexagon boundary (negative if inside)."""
    dx = px - cx
    dy = py - cy
    theta = -math.radians(angle_deg)
    lx = dx * math.cos(theta) - dy * math.sin(theta)
    ly = dx * math.sin(theta) + dy * math.cos(theta)
    
    apothem = side * math.sqrt(3) / 2.0
    min_dist = float('inf')
    
    for k in range(6):
        normal_angle = math.radians(30 + k * 60)
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        dist = lx * nx + ly * ny - apothem
        min_dist = min(min_dist, dist)
    return min_dist

def check_inner_disjoint(inner, tol=1e-9):
    """Check if inner hexagons are disjoint (no overlap)."""
    n = len(inner)
    for i in range(n):
        x1, y1, a1 = inner[i]
        for j in range(i + 1, n):
            x2, y2, a2 = inner[j]
            # Check if any vertex of hex i is inside hex j, or vice versa
            verts1 = hex_vertices(x1, y1, 1.0, a1)
            verts2 = hex_vertices(x2, y2, 1.0, a2)
            
            # Check vertices of 1 in 2
            for v in verts1:
                if point_in_hex(v[0], v[1], x2, y2, 1.0, a2):
                    return False
            # Check vertices of 2 in 1
            for v in verts2:
                if point_in_hex(v[0], v[1], x1, y1, 1.0, a1):
                    return False
            # Check edge intersections (simplified: check midpoints of edges)
            # For regular hexagons, if no vertex is inside the other and they're not too far, 
            # we should also check edge crossings. But for simplicity in initial check,
            # vertex containment is a good first filter.
    return True

def compute_outer_side(inner, outer_center, outer_angle):
    """Compute minimum side length of outer hexagon to contain all inner hexagons."""
    cx, cy = outer_center
    outer_verts = hex_vertices(cx, cy, 1.0, outer_angle)
    
    min_side = 0.0
    for x, y, a in inner:
        inner_verts = hex_vertices(x, y, 1.0, a)
        for v in inner_verts:
            # Find the smallest scale s such that v is inside hex centered at (cx,cy) with side s
            # Transform v to outer hex local frame
            dx = v[0] - cx
            dy = v[1] - cy
            theta = -math.radians(outer_angle)
            lx = dx * math.cos(theta) - dy * math.sin(theta)
            ly = dx * math.sin(theta) + dy * math.cos(theta)
            
            # In local frame, hex with side s has apothem s*sqrt(3)/2
            # Point (lx,ly) is inside if for all 6 directions:
            # lx*cos(30+k*60) + ly*sin(30+k*60) <= s*sqrt(3)/2
            # So s >= (lx*cos(30+k*60) + ly*sin(30+k*60)) / (sqrt(3)/2) for all k
            # s >= max over k of (lx*cos(30+k*60) + ly*sin(30+k*60)) * 2/sqrt(3)
            max_proj = 0.0
            for k in range(6):
                normal_angle = math.radians(30 + k * 60)
                proj = lx * math.cos(normal_angle) + ly * math.sin(normal_angle)
                max_proj = max(max_proj, proj)
            required_side = max_proj * 2.0 / math.sqrt(3.0)
            min_side = max(min_side, required_side)
    
    return max(min_side, 0.0)

def objective(params, inner_fixed, outer_center, outer_angle):
    """Objective: outer side length given inner positions/angles."""
    # params contains only the variables we're optimizing
    # For simplicity, let's optimize inner positions and outer side simultaneously
    pass

def construct():
    start_time = time.time()
    
    # Initial configuration: 11 hexagons in a hexagonal pattern
    # Center + 6 around + 4 more? Let's try a compact arrangement.
    # Standard hex packing: center at origin, 6 at distance 2*sqrt(3) in 60-degree directions
    # Actually for unit hexagons, adjacent centers are at distance 2*sqrt(3) for edge-to-edge
    # Wait, for regular hexagons with side 1, the distance between centers of adjacent hexagons 
    # in a honeycomb is 2 * apothem = 2 * sqrt(3)/2 * ... 
    # Apothem = sqrt(3)/2. Distance between centers of touching hexagons = 2 * apothem = sqrt(3)? No.
    # For two regular hexagons with side 1 to be edge-to-edge, their centers are separated by 2 * apothem = 2*(sqrt(3)/2) = sqrt(3)? 
    # Actually, the distance between centers of two adjacent hexagons in a honeycomb is 2 * (sqrt(3)/2) * 2 / 2... 
    # Let me recalculate: apothem = s*sqrt(3)/2 = sqrt(3)/2 for s=1.
    # Two hexagons sharing an edge: center-to-center distance = 2 * apothem = sqrt(3).
    
    # Let's place them in a hexagonal cluster:
    # (0,0), and 6 at distance sqrt(3) in directions 0,60,120,180,240,300
    # That's 7. Need 4 more.
    
    # Better: use a known good starting point
    s3 = math.sqrt(3)
    inner = []
    # Center
    inner.append([0.0, 0.0, 0.0])
    # Ring 1: 6 hexagons
    for i in range(6):
        a = i * math.pi / 3.0
        inner.append([s3 * math.cos(a), s3 * math.sin(a), 0.0])
    # Ring 2 partial: 4 more at distance 2*sqrt(3)
    for i in [0, 1, 4, 5]:
        a = i * math.pi / 3.0
        inner.append([2 * s3 * math.cos(a), 2 * s3 * math.sin(a), 0.0])
    
    outer_center = (0.0, 0.0)
    outer_angle = 0.0
    
    # Compute initial outer side
    current_side = compute_outer_side(inner, outer_center, outer_angle)
    best_side = current_side
    best_inner = [list(x) for x in inner]
    best_outer_center = outer_center
    best_outer_angle = outer_angle
    
    # Optimization: minimize outer side by adjusting inner positions and outer parameters
    # Use scipy minimize with a smooth objective
    
    def smooth_objective(params):
        # params: [x1,y1,a1, x2,y2,a2, ..., xn,yn,an, cx, cy, outer_angle]
        n_inner = N
        inner_params = params[:3*n_inner]
        cx = params[3*n_inner]
        cy = params[3*n_inner+1]
        outer_ang = params[3*n_inner+2]
        
        # Reconstruct inner
        curr_inner = []
        for i in range(n_inner):
            x = inner_params[3*i]
            y = inner_params[3*i+1]
            a = inner_params[3*i+2]
            curr_inner.append([x, y, a])
        
        # Compute outer side needed
        side = compute_outer_side(curr_inner, (cx, cy), outer_ang)
        
        # Penalty for overlaps (smooth)
        penalty = 0.0
        for i in range(n_inner):
            x1, y1, a1 = curr_inner[i]
            for j in range(i+1, n_inner):
                x2, y2, a2 = curr_inner[j]
                # Approximate distance between centers
                dx = x1 - x2
                dy = y1 - y2
                dist = math.sqrt(dx*dx + dy*dy)
                min_dist = 2 * s3  # Minimum center distance for non-overlap (approximate)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2
        
        return side + penalty * 10.0
    
    # Set up initial params
    init_params = []
    for x, y, a in inner:
        init_params.extend([x, y, a])
    init_params.extend([outer_center[0], outer_center[1], outer_angle])
    
    # Optimize
    result = minimize(smooth_objective, init_params, method='L-BFGS-B', 
                      options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
    
    if result.success:
        opt_params = result.x
        opt_inner = []
        for i in range(N):
            opt_inner.append([opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]])
        opt_cx = opt_params[3*N]
        opt_cy = opt_params[3*N+1]
        opt_ang = opt_params[3*N+2]
        
        opt_side = compute_outer_side(opt_inner, (opt_cx, opt_cy), opt_ang)
        if opt_side < best_side:
            best_side = opt_side
            best_inner = opt_inner
            best_outer_center = (opt_cx, opt_cy)
            best_outer_angle = opt_ang
    
    # Try different outer angles
    for ang in [0, 30, 60, 15, 45]:
        curr_side = compute_outer_side(best_inner, best_outer_center, ang)
        if curr_side < best_side:
            best_side = curr_side
            best_outer_angle = ang
    
    elapsed = time.time() - start_time
    if elapsed > DEADLINE:
        return best_inner, best_outer_center, best_side, best_outer_angle
    
    return best_inner, best_outer_center, best_side, best_outer_angle
