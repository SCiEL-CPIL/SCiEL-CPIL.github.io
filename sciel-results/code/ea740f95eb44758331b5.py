"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize
import math

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    """Return the 6 vertices of a regular hexagon with center (cx,cy), side length, and rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside the hexagon."""
    angle_rad = math.radians(angle_deg)
    # Transform point to hexagon's local frame
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    cos_a = math.cos(-angle_rad)
    sin_a = math.sin(-angle_rad)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    # For a regular hexagon with side s, the distance from center to vertex is s
    # The hexagon can be described by |lx| <= s*sqrt(3)/2 in certain directions... 
    # Actually, simpler: use the support function or check against edges
    # A regular hexagon with "radius" (center to vertex) = side length s
    # The half-width in the x-direction (flat-top) is s*sqrt(3)/2, in y-direction is s
    # But it depends on orientation. Let's just check if point is inside the polygon.
    verts = _hex_vertices(cx, cy, side, angle_deg)
    # Ray casting
    inside = False
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        if ((y1 > py) != (y2 > py)) and (px < (x2 - x1) * (py - y1) / (y2 - y1) + x1):
            inside = not inside
    return inside

def _hex_separation(h1, h2):
    """
    h1, h2: (cx, cy, angle_deg) for unit hexagons
    Returns minimum distance between the two hexagons (0 if overlapping/touching)
    We approximate by checking vertex-edge distances.
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    verts1 = _hex_vertices(cx1, cy1, 1.0, a1)
    verts2 = _hex_vertices(cx2, cy2, 1.0, a2)
    
    min_dist = float('inf')
    
    # Check all vertex-edge pairs
    for i in range(6):
        # Vertex of h1 to edge of h2
        vx, vy = verts1[i]
        ex1, ey1 = verts2[i]
        ex2, ey2 = verts2[(i + 1) % 6]
        # Distance from point to line segment
        dx = ex2 - ex1
        dy = ey2 - ey1
        if dx == 0 and dy == 0:
            d = math.hypot(vx - ex1, vy - ey1)
        else:
            t = ((vx - ex1) * dx + (vy - ey1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            px = ex1 + t * dx
            py = ey1 + t * dy
            d = math.hypot(vx - px, vy - py)
        min_dist = min(min_dist, d)
        
        # Vertex of h2 to edge of h1
        vx, vy = verts2[i]
        ex1, ey1 = verts1[i]
        ex2, ey2 = verts1[(i + 1) % 6]
        dx = ex2 - ex1
        dy = ey2 - ey1
        if dx == 0 and dy == 0:
            d = math.hypot(vx - ex1, vy - ey1)
        else:
            t = ((vx - ex1) * dx + (vy - ey1) * dy) / (dx * dx + dy * dy)
            t = max(0, min(1, t))
            px = ex1 + t * dx
            py = ey1 + t * dy
            d = math.hypot(vx - px, vy - py)
        min_dist = min(min_dist, d)
    
    return min_dist

def _objective(params, outer_side, outer_center, outer_angle):
    """
    params: 36 values: for each of 12 hexagons: (x, y, angle_deg)
    Returns penalty: sum of overlap violations + containment violations
    """
    ocx, ocy = outer_center
    penalty = 0.0
    inner_hexes = []
    
    for i in range(N):
        x = params[3*i]
        y = params[3*i+1]
        a = params[3*i+2]
        inner_hexes.append((x, y, a))
        
        # Containment check: all 6 vertices must be inside outer hexagon
        verts = _hex_vertices(x, y, 1.0, a)
        for vx, vy in verts:
            if not _point_in_hex(vx, vy, ocx, ocy, outer_side, outer_angle):
                # Penalize by how far outside
                # Simple penalty: distance to boundary approximation
                penalty += 1.0
    
    # Disjointness check
    for i in range(N):
        for j in range(i+1, N):
            sep = _hex_separation(inner_hexes[i], inner_hexes[j])
            if sep < 0.01:
                penalty += max(0, 0.01 - sep) * 100
    
    return penalty

def construct():
    # Try a grid-based initial guess with optimization
    # Outer hexagon: we want to minimize side length
    # Start with a reasonable guess
    
    best_side = 40.0
    best_params = None
    
    # Try different outer side lengths
    for side in [4.5, 4.8, 5.0, 5.2, 5.5, 5.8, 6.0]:
        # Initial guess: place hexagons in a grid-like pattern
        # For a hexagonal packing, centers should be ~sqrt(3) apart in x, ~3/2 in y for staggered rows
        # But with rotations, we can do better
        
        # Try a simple symmetric arrangement
        # 3 rows of 4, or 4 rows of 3
        # Let's try 3x4 grid
        init_params = []
        spacing_x = 1.8
        spacing_y = 1.7
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * spacing_x
                y = (row - 1) * spacing_y
                a = 0.0
                init_params.extend([x, y, a])
        
        # Optimize
        result = minimize(
            _objective,
            init_params,
            args=(side, (0.0, 0.0), 0.0),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.01}
        )
        
        if result.fun < 1e-6:  # Valid packing found
            best_side = side
            best_params = result.x
            break
    
    # If no valid packing found with the above, use a more conservative approach
    if best_params is None:
        # Use a very safe but larger packing
        inner = []
        for i in range(N):
            # Place in a circle with large radius
            angle = 2 * math.pi * i / N
            r = 3.0
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 10.0, 0.0
    
    # Extract inner hexagons
    inner = []
    for i in range(N):
        x = best_params[3*i]
        y = best_params[3*i+1]
        a = best_params[3*i+2]
        inner.append([x, y, a])
    
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
