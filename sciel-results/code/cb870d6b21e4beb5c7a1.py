"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import time
import math
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given angle."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return np.array(verts)

def _polygon_area(p):
    """Shoelace formula for polygon area."""
    n = len(p)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += p[i][0] * p[j][1] - p[j][0] * p[i][1]
    return abs(area) / 2.0

def _point_in_hexagon(px, py, cx, cy, angle_deg, side=1.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # Regular hexagon with side s: |x| <= s, |y| <= s*sqrt(3)/2, |y| <= sqrt(3)*(s - |x|/2)
    s = side
    if abs(lx) > s:
        return False
    if abs(ly) > s * math.sqrt(3) / 2:
        return False
    # Check the slanted edges
    if abs(ly) > math.sqrt(3) * (s - abs(lx) / 2):
        return False
    return True

def _hex_overlap_check(hex1, hex2, eps=1e-6):
    """
    Check if two hexagons overlap.
    hex1, hex2: (cx, cy, angle_deg)
    Returns True if they overlap (including touching with negative margin).
    """
    cx1, cy1, a1 = hex1
    cx2, cy2, a2 = hex2
    
    # Quick bounding box check
    dx = abs(cx1 - cx2)
    dy = abs(cy1 - cy2)
    if dx > 2.0 + eps or dy > 2.0 + eps:
        return False
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    v1 = _hex_vertices(cx1, cy1, a1)
    v2 = _hex_vertices(cx2, cy2, a2)
    
    for vx, vy in v1:
        if _point_in_hexagon(vx, vy, cx2, cy2, a2):
            return True
    for vx, vy in v2:
        if _point_in_hexagon(vx, vy, cx1, cy1, a1):
            return True
    
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def ccw(a, b, c):
            return (c[1]-a[1])*(b[0]-a[0]) - (b[1]-a[1])*(c[0]-a[0])
        
        d1 = ccw(p3, p4, p1)
        d2 = ccw(p3, p4, p2)
        d3 = ccw(p1, p2, p3)
        d4 = ccw(p1, p2, p4)
        
        if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
           ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
            return True
        return False
    
    for i in range(6):
        j = (i + 1) % 6
        for k in range(6):
            l = (k + 1) % 6
            if seg_intersect(v1[i], v1[j], v2[k], v2[l]):
                return True
    
    return False

def _outer_side_length(inner_hexes, center, angle_deg):
    """
    Compute the minimum side length of a regular hexagon centered at 'center'
    with given angle that contains all inner hexagons.
    """
    cx, cy = center
    max_dist = 0.0
    for hx, hy, ha in inner_hexes:
        verts = _hex_vertices(hx, hy, ha)
        for vx, vy in verts:
            # Transform to outer hexagon's local frame
            dx = vx - cx
            dy = vy - cy
            a = math.radians(angle_deg)
            cos_a = math.cos(a)
            sin_a = math.sin(a)
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            # For a regular hexagon with side s, the support function in direction (lx,ly)
            # The hexagon contains point (lx,ly) iff max(|lx|/s, |ly|/(s*sqrt(3)/2), 
            #   (|lx|/2 + |ly|/sqrt(3))/s) <= 1
            # So s >= max(|lx|, |ly|*2/sqrt(3), |lx|/2 + |ly|/sqrt(3))
            # Actually, let me think more carefully.
            # A regular hexagon with side s has vertices at distance s from center.
            # The support function h(n) for direction n is the max of n·v over vertices v.
            # For the hexagon to contain the point, we need |n·p| <= h(n) for all n.
            # The minimal s such that the hexagon contains all points is the max over all points
            # of the minimal s for that point.
            
            # For a point at local coords (lx, ly), the minimal hexagon side s that contains it:
            # The hexagon boundary in local coords: |x| <= s, |y| <= s*sqrt(3)/2, 
            #   |y| <= sqrt(3)*(s - |x|/2)
            # So s must satisfy:
            #   s >= |lx|
            #   s >= 2|ly|/sqrt(3)
            #   s >= |lx|/2 + |ly|/sqrt(3)
            
            s_req = max(abs(lx), 2*abs(ly)/math.sqrt(3), abs(lx)/2 + abs(ly)/math.sqrt(3))
            max_dist = max(max_dist, s_req)
    
    return max_dist

def _objective(params, inner_count, time_limit):
    """
    Objective function: minimize the outer hexagon side length.
    params: flat array of [x0,y0,a0, x1,y1,a1, ..., center_x, center_y, outer_angle]
    """
    if time.time() > time_limit:
        return 1e10
    
    x = params[:inner_count*3]
    inner = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(inner_count)]
    cx, cy, outer_angle = params[-3:]
    
    # Check overlaps
    for i in range(inner_count):
        for j in range(i+1, inner_count):
            if _hex_overlap_check(inner[i], inner[j]):
                return 1e10
    
    return _outer_side_length(inner, (cx, cy), outer_angle)

def construct():
    start_time = time.time()
    time_limit = start_time + 110.0  # 10 second safety margin
    
    # Initial placement: hexagonal lattice
    # For N hexagons, arrange in a hexagonal pattern
    # Number of hexagons in a hexagonal arrangement with radius r: 3r(r+1)+1
    # r=0: 1, r=1: 7, r=2: 19, r=3: 37
    # For N=11, use a custom arrangement
    
    # Start with a linear arrangement and optimize
    inner_init = []
    for k in range(-(N // 2), N - N // 2):
        inner_init.append([4.0 * k, 0.0, 0.0])
    
    inner_count = len(inner_init)
    # Parameters: 3 per inner hex + 3 for outer (cx, cy, angle)
    param_count = inner_count * 3 + 3
    
    x0 = []
    for hx, hy, ha in inner_init:
        x0.extend([hx, hy, ha])
    x0.extend([0.0, 0.0, 0.0])
    x0 = np.array(x0)
    
    best_result = None
    best_score = 1e10
    
    # Try multiple initializations
    inits = [x0]
    
    # Add a hexagonal lattice init
    # Arrange N hexagons in a hexagonal pattern
    # For N=11: center + 6 around + 4 more
    hex_init = [(0, 0, 0)]
    # 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        hex_init.append((math.sqrt(3) * math.cos(angle), math.sqrt(3) * math.sin(angle), 0))
    # 4 more at radius 2
    for i in [0, 2, 4, 5]:
        angle = i * math.pi / 3
        hex_init.append((2*math.sqrt(3) * math.cos(angle), 2*math.sqrt(3) * math.sin(angle), 0))
    
    x0_hex = []
    for hx, hy, ha in hex_init[:inner_count]:
        x0_hex.extend([hx, hy, ha])
    x0_hex.extend([0.0, 0.0, 0.0])
    inits.append(np.array(x0_hex))
    
    for init in inits:
        if time.time() > time_limit:
            break
        
        try:
            result = minimize(
                _objective,
                init,
                args=(inner_count, time_limit),
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-8, 'gtol': 1e-6}
            )
            
            if time.time() > time_limit:
                break
            
            score = result.fun
            if score < best_score and score < 1e9:
                best_score = score
                x_best = result.x
                best_result = x_best
        except Exception:
            pass
    
    if best_result is None:
        # Fallback: return the initial linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    # Extract result
    x = best_result
    inner = [(x[i*3], x[i*3+1], x[i*3+2]) for i in range(inner_count)]
    cx, cy, outer_angle = x[-3:]
    
    # Verify no overlaps
    valid = True
    for i in range(inner_count):
        for j in range(i+1, inner_count):
            if _hex_overlap_check(inner[i], inner[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Return a safe fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    side = _outer_side_length(inner, (cx, cy), outer_angle)
    
    # Ensure side is reasonable
    if side < 1.0 or side > 50.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return inner, (cx, cy), side, outer_angle
# EVOLVE-BLOCK-END
