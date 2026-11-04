"""Improved program: continuous optimization for hexagon packing."""
import numpy as np
import time
from scipy.optimize import minimize

N = 12
DEADLINE = 120.0
START_TIME = time.time()

def _elapsed():
    return time.time() - START_TIME

def _remaining():
    return DEADLINE - _elapsed() - 5.0  # 5s safety margin

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation."""
    angle = np.radians(angle_deg)
    # Unit hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
    base = np.array([[np.cos(np.radians(60*i)), np.sin(np.radians(60*i))] for i in range(6)])
    rot = np.array([[np.cos(angle), -np.sin(angle)],
                    [np.sin(angle), np.cos(angle)]])
    verts = base @ rot.T * side + np.array([cx, cy])
    return verts

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    angle = np.radians(angle_deg)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    lx = (dx * cos_a + dy * sin_a) / side
    ly = (-dx * sin_a + dy * cos_a) / side
    # Regular hexagon with flat top: |x| <= 1, |y| <= sqrt(3)/2, and |x| + |y|/sqrt(3) <= 1 for the constraint
    # Actually for a regular hexagon with vertices at 0,60,...,300 degrees:
    # The hexagon is the intersection of 6 half-planes.
    # For unit hexagon (side=1), vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
    # The 6 constraints (normal pointing inward):
    # cos(60i)*x + sin(60i)*y <= 1 for i=0..5  (since apothem = sqrt(3)/2, and for unit side, apothem = sqrt(3)/2)
    # Wait: for a regular hexagon with side s, the apothem is s*sqrt(3)/2.
    # The 6 supporting lines have normals at angles 30, 90, 150, 210, 270, 330 degrees (perpendicular to edges)
    # Actually the vertices are at 0,60,120,180,240,300. The edges are between consecutive vertices.
    # Edge from vertex 0 to vertex 1: midpoint at angle 30, distance from center = apothem = sqrt(3)/2 for unit hex.
    # So constraint: dot((lx,ly), (cos(30+60i), sin(30+60i))) <= sqrt(3)/2 for i=0..5
    # For unit hex (side=1): apothem = sqrt(3)/2
    apothem = np.sqrt(3) / 2
    for i in range(6):
        theta = np.radians(30 + 60 * i)
        val = lx * np.cos(theta) + ly * np.sin(theta)
        if val > apothem + 1e-9:
            return False
    return True

def _hexagon_intersection_area(v1, v2):
    """Approximate intersection area of two convex polygons (6-gons each) using Sutherland-Hodgman or sampling.
    For speed, use a simple polygon clipping or just check if they overlap via vertex containment."""
    # Simple check: if any vertex of v1 is inside v2 or any vertex of v2 is inside v1, they overlap
    # This is not sufficient but gives a quick test
    for i in range(6):
        if _point_in_hexagon(v1[i][0], v1[i][1], 0, 0, 1, 0):  # v2 is at origin in local frame
            return True
    for i in range(6):
        if _point_in_hexagon(v2[i][0], v2[i][1], 0, 0, 1, 0):
            return True
    return False

def _build_outer_hexagon(center, side, angle_deg):
    """Build the outer hexagon vertices."""
    return _hex_vertices(center[0], center[1], side, angle_deg)

def _is_inside_outer(px, py, outer_verts):
    """Check if point is inside the outer hexagon (convex polygon)."""
    # Use winding number or cross product method for convex polygon
    n = len(outer_verts)
    for i in range(n):
        x1, y1 = outer_verts[i]
        x2, y2 = outer_verts[(i+1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if cross < -1e-9:
            return False
    return True

def _all_inside_outer(inner_hexes, outer_verts):
    """Check all vertices of all inner hexagons are inside outer hexagon."""
    for hx, hy, ha in inner_hexes:
        verts = _hex_vertices(hx, hy, 1.0, ha)
        for vx, vy in verts:
            if not _is_inside_outer(vx, vy, outer_verts):
                return False
    return True

def _no_overlap(inner_hexes):
    """Check no two inner hexagons overlap."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i+1, n):
            hx1, hy1, ha1 = inner_hexes[i]
            hx2, hy2, ha2 = inner_hexes[j]
            v1 = _hex_vertices(hx1, hy1, 1.0, ha1)
            v2 = _hex_vertices(hx2, hy2, 1.0, ha2)
            # Check if they overlap
            # Quick rejection: distance between centers
            dx, dy = hx2-hx1, hy2-hy1
            dist = np.sqrt(dx*dx + dy*dy)
            if dist > 4.0:  # max distance between two unit hexagon centers that can touch
                continue
            # Check vertex containment
            for vx, vy in v1:
                if _point_in_hexagon(vx, vy, hx2, hy2, 1.0, ha2):
                    return False
            for vx, vy in v2:
                if _point_in_hexagon(vx, vy, hx1, hy1, 1.0, ha1):
                    return False
    return True

def _is_valid(inner_hexes, center, side, angle_deg):
    """Check if the configuration is valid."""
    outer_verts = _build_outer_hexagon(center, side, angle_deg)
    if not _all_inside_outer(inner_hexes, outer_verts):
        return False
    if not _no_overlap(inner_hexes):
        return False
    return True

def _initial_config():
    """Generate a reasonable initial configuration."""
    # Place hexagons in a honeycomb-like pattern
    positions = []
    angles = [0.0] * N
    # Honeycomb lattice: spacing sqrt(3) * 2 for non-overlapping
    # Actually for unit hexagons, center-to-center distance must be >= 2*apothem = sqrt(3) for edge-to-edge
    # But with rotation, we need more space. Let's use a grid with spacing 3.5
    dx = 3.5
    dy = 3.0
    rows = 4
    for r in range(rows):
        for c in range(3):
            if len(positions) >= N:
                break
            x = (c - 1) * dx + (0.5 * dx if r % 2 else 0)
            y = (r - 1.5) * dy
            positions.append((x, y))
    while len(positions) < N:
        positions.append((0.0, 0.0))
    
    inner = [[x, y, 0.0] for x, y in positions[:N]]
    # Estimate required side length
    # Find bounding box
    all_pts = []
    for hx, hy, ha in inner:
        verts = _hex_vertices(hx, hy, 1.0, ha)
        all_pts.extend(verts)
    all_pts = np.array(all_pts)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)
    width = max_x - min_x
    height = max_y - min_y
    # For a hexagon containing this, side length roughly max(width, height) / sqrt(3) + 1
    side = max(width, height) / np.sqrt(3) + 2.0
    center = ((min_x + max_x)/2, (min_y + max_y)/2)
    return inner, center, side, 0.0

def construct():
    """Main entry point: find minimal outer hexagon side length for 12 unit hexagons."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with a reasonable configuration
    inner, center, side, angle_deg = _initial_config()
    
    # Try to shrink the side by optimization
    # Parameters: positions (2N), angles (N), center (2), side (1), outer angle (1)
    # Total: 2N + N + 2 + 1 + 1 = 3N + 4 = 40 parameters for N=12
    
    def _objective(params):
        """Objective: side length + penalty for violations."""
        n = N
        # Unpack
        pos = params[0:2*n].reshape(n, 2)
        angles = params[2*n:3*n]
        cx, cy = params[3*n:3*n+2]
        s = params[3*n+2]
        outer_angle = params[3*n+3]
        
        if s < 2.0:
            s = 2.0
        
        # Build inner hexagons
        inner_hexes = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
        
        # Check containment
        outer_verts = _build_outer_hexagon((cx, cy), s, outer_angle)
        penalty = 0.0
        for hx, hy, ha in inner_hexes:
            verts = _hex_vertices(hx, hy, 1.0, ha)
            for vx, vy in verts:
                if not _is_inside_outer(vx, vy, outer_verts):
                    # Penalty: how far outside
                    # Find distance to nearest edge
                    penalty += 10.0
        
        # Check overlap
        for i in range(n):
            for j in range(i+1, n):
                hx1, hy1, ha1 = inner_hexes[i]
                hx2, hy2, ha2 = inner_hexes[j]
                dx, dy = hx2-hx1, hy2-hy1
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 3.8:
                    v1 = _hex_vertices(hx1, hy1, 1.0, ha1)
                    v2 = _hex_vertices(hx2, hy2, 1.0, ha2)
                    for vx, vy in v1:
                        if _point_in_hexagon(vx, vy, hx2, hy2, 1.0, ha2):
                            penalty += 50.0
                            break
                    for vx, vy in v2:
                        if _point_in_hexagon(vx, vy, hx1, hy1, 1.0, ha1):
                            penalty += 50.0
                            break
        
        return s + penalty
    
    # Initial parameters
    x0 = np.array([
        [inner[i][0], inner[i][1]] for i in range(N)
    ]).flatten()
    angles0 = np.array([inner[i][2] for i in range(N)])
    x0 = np.concatenate([x0, angles0, [center[0], center[1], side, angle_deg]])
    
    best_result = (inner, center, side, angle_deg)
    best_score = side
    
    # Use L-BFGS-B
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 200, 'ftol': 1e-6, 'gtol': 1e-4})
        
        if _remaining() > 10.0:
            # Extract improved solution
            params = result.x
            n = N
            pos = params[0:2*n].reshape(n, 2)
            angles = params[2*n:3*n]
            cx, cy = params[3*n:3*n+2]
            s = params[3*n+2]
            outer_angle = params[3*n+3]
            
            inner_new = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
            
            # Validate
            if _is_valid(inner_new, (cx, cy), s, outer_angle):
                if s < best_score:
                    best_result = (inner_new, (cx, cy), s, outer_angle)
                    best_score = s
                return inner_new, (cx, cy), s, outer_angle
            
            # If not valid, try to fix by adjusting
            # Just return best valid so far
    except Exception:
        pass
    
    # If optimization didn't improve, try a simple iterative shrinking
    if _remaining() > 20.0:
        # Try to shrink the outer hexagon while maintaining validity
        current_inner, current_center, current_side, current_angle = best_result
        for _ in range(50):
            if _remaining() < 10.0:
                break
            new_side = current_side * 0.98
            if new_side < 3.0:
                break
            if _is_valid(current_inner, current_center, new_side, current_angle):
                current_side = new_side
            else:
                # Try moving inner hexagons slightly toward center
                # Just break
                break
        best_result = (current_inner, current_center, current_side, current_angle)
        best_score = current_side
    
    return best_result

# Keep the original interface
def construct():
    return construct_impl()

def construct_impl():
    """Actual implementation."""
    global START_TIME
    START_TIME = time.time()
    
    # Start with a reasonable configuration
    inner, center, side, angle_deg = _initial_config()
    
    best_inner = inner
    best_center = center
    best_side = side
    best_angle = angle_deg
    
    # Optimization
    def _objective(params):
        n = N
        pos = params[0:2*n].reshape(n, 2)
        angles = params[2*n:3*n]
        cx, cy = params[3*n:3*n+2]
        s = params[3*n+2]
        outer_angle = params[3*n+3]
        
        if s < 2.0:
            s = 2.0
        
        inner_hexes = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
        
        outer_verts = _build_outer_hexagon((cx, cy), s, outer_angle)
        penalty = 0.0
        for hx, hy, ha in inner_hexes:
            verts = _hex_vertices(hx, hy, 1.0, ha)
            for vx, vy in verts:
                if not _is_inside_outer(vx, vy, outer_verts):
                    penalty += 10.0
        
        for i in range(n):
            for j in range(i+1, n):
                hx1, hy1, ha1 = inner_hexes[i]
                hx2, hy2, ha2 = inner_hexes[j]
                dx, dy = hx2-hx1, hy2-hy1
                dist = np.sqrt(dx*dx + dy*dy)
                if dist < 3.8:
                    v1 = _hex_vertices(hx1, hy1, 1.0, ha1)
                    v2 = _hex_vertices(hx2, hy2, 1.0, ha2)
                    for vx, vy in v1:
                        if _point_in_hexagon(vx, vy, hx2, hy2, 1.0, ha2):
                            penalty += 50.0
                            break
                    for vx, vy in v2:
                        if _point_in_hexagon(vx, vy, hx1, hy1, 1.0, ha1):
                            penalty += 50.0
                            break
        
        return s + penalty
    
    x0 = np.array([[inner[i][0], inner[i][1]] for i in range(N)]).flatten()
    angles0 = np.array([inner[i][2] for i in range(N)])
    x0 = np.concatenate([x0, angles0, [center[0], center[1], side, angle_deg]])
    
    try:
        result = minimize(_objective, x0, method='L-BFGS-B', 
                         options={'maxiter': 150, 'ftol': 1e-5, 'gtol': 1e-3})
        
        if _remaining() > 10.0:
            params = result.x
            n = N
            pos = params[0:2*n].reshape(n, 2)
            angles = params[2*n:3*n]
            cx, cy = params[3*n:3*n+2]
            s = max(params[3*n+2], 2.0)
            outer_angle = params[3*n+3]
            
            inner_new = [[pos[i][0], pos[i][1], angles[i]] for i in range(n)]
            
            if _is_valid(inner_new, (cx, cy), s, outer_angle):
                if s < best_side:
                    best_inner, best_center, best_side, best_angle = inner_new, (cx, cy), s, outer_angle
            
            # Try iterative shrinking
            if _remaining() > 15.0:
                cur_inner, cur_center, cur_side, cur_angle = best_inner, best_center, best_side, best_angle
                for _ in range(100):
                    if _remaining() < 10.0:
                        break
                    new_side = cur_side * 0.97
                    if new_side < 3.0:
                        break
                    if _is_valid(cur_inner, cur_center, new_side, cur_angle):
                        cur_side = new_side
                    else:
                        break
                if cur_side < best_side:
                    best_inner, best_center, best_side, best_angle = cur_inner, cur_center, cur_side, cur_angle
    
    except Exception:
        pass
    
    return best_inner, best_center, best_side, best_angle
