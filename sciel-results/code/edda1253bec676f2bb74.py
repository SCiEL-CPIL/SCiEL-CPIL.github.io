"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time
import math

N = 12

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_rad):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    angles = np.linspace(0, 2 * np.pi, 7)[:-1] + angle_rad
    return cx + side * np.cos(angles), cy + side * np.sin(angles)

def _point_in_hexagon(px, py, cx, cy, side, angle_rad):
    """Check if point (px, py) is inside or on the boundary of the hexagon."""
    # Transform point to hexagon's local coordinate system
    dx = px - cx
    dy = py - cy
    # Rotate by -angle
    ca, sa = np.cos(-angle_rad), np.sin(-angle_rad)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # For a regular hexagon with flat-top orientation (angle=0), the vertices are at:
    # (s, 0), (s/2, s*sqrt(3)/2), (-s/2, s*sqrt(3)/2), (-s, 0), (-s/2, -s*sqrt(3)/2), (s/2, -s*sqrt(3)/2)
    # The hexagon can be defined by |x| <= s and |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2 ... 
    # Actually, for a regular hexagon with vertices at angles 0, 60, 120, 180, 240, 300:
    # The constraints are:
    # |x| <= s
    # |x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2  =>  |x + y*sqrt(3)| <= s*sqrt(3)
    # |-x/2 + y*sqrt(3)/2| <= s*sqrt(3)/2  =>  |-x + y*sqrt(3)| <= s*sqrt(3)
    # Wait, let's use the standard form. A regular hexagon centered at origin with side s,
    # with one vertex on the positive x-axis, has the following half-plane constraints:
    # The normals to the edges are at angles 30, 90, 150, 210, 270, 330 degrees.
    # The distance from center to each edge (apothem) is s * sqrt(3) / 2.
    # So the constraints are: n_i . p <= apothem for each edge normal n_i.
    # Edge normals (unit vectors) for a hexagon with vertex at angle 0:
    # The edges are between vertices at angles 0-60, 60-120, etc.
    # The normal to the edge between vertices at angle 0 and 60 is at angle 30.
    # So normals are at 30, 90, 150, 210, 270, 330 degrees.
    # Apothem = s * sqrt(3) / 2
    apothem = side * math.sqrt(3) / 2
    normals = []
    for i in range(6):
        ang = np.deg2rad(30 + 60 * i)
        normals.append((np.cos(ang), np.sin(ang)))
    
    for nx, ny in normals:
        if nx * lx + ny * ly > apothem + 1e-10:
            return False
    return True

def _hexagon_in_hexagon(inner_vertices, outer_vertices):
    """Check if inner hexagon is completely inside outer hexagon."""
    # For each vertex of inner, check if it's inside outer
    # Also need to check edges don't cross, but for convex polygons, 
    # if all vertices of inner are inside outer, and no edges cross, it's inside.
    # Actually, for convex polygons, if all vertices of A are inside B, A is inside B.
    for vx, vy in zip(inner_vertices[0], inner_vertices[1]):
        if not _point_in_hexagon(vx, vy, *outer_vertices[:4]):
            return False
    return True

def _construct_honeycomb():
    """Construct a honeycomb-like arrangement of 12 hexagons."""
    s = 1.0  # inner hexagon side length
    # In a honeycomb, adjacent hexagons are separated by distance s * sqrt(3)
    # Let's place them in a pattern that fits well in a hexagonal container
    # A common arrangement is a 3x4 grid or similar
    
    positions = []
    angles = []
    
    # Let's try a 3x4 grid rotated appropriately
    # Hexagon centers in a honeycomb lattice:
    # dx = s * sqrt(3)
    # dy = s * 1.5
    dx = s * math.sqrt(3)
    dy = s * 1.5
    
    # Place 12 hexagons in a 3x4 arrangement
    for i in range(4):
        for j in range(3):
            x = i * dx + (j % 2) * dx / 2
            y = j * dy
            positions.append([x, y])
            angles.append(0.0)
    
    # Center the arrangement
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2
    
    for p in positions:
        p[0] -= cx
        p[1] -= cy
    
    # Now find the minimum enclosing hexagon
    # We'll use a simple approach: try different rotations and find the minimum side length
    best_side = float('inf')
    best_angle = 0.0
    
    # The outer hexagon should be concentric with the arrangement
    # For a given rotation, we need to find the minimum side length such that
    # all inner hexagons fit inside
    
    # For each inner hexagon vertex, compute the required outer side length
    # for a given outer rotation
    
    for outer_angle_deg in np.linspace(0, 60, 61):
        outer_angle = np.deg2rad(outer_angle_deg)
        max_required = 0.0
        
        for pos, ang in zip(positions, angles):
            # Get vertices of this inner hexagon
            ivx, ivy = _hex_vertices(pos[0], pos[1], s, np.deg2rad(ang))
            
            # For each vertex, compute the maximum distance from center in the direction
            # of the outer hexagon's edge normals
            # The outer hexagon with side S and angle a has apothem S * sqrt(3) / 2
            # A point (x, y) is inside if for all 6 edge normals n_i:
            # n_i . (x - center) <= apothem
            # So apothem >= max_i |n_i . (x - center)|
            # And S = apothem * 2 / sqrt(3)
            
            for vx, vy in zip(ivx, ivy):
                # Transform to outer hexagon's coordinate system
                dx_pt = vx
                dy_pt = vy
                ca, sa = np.cos(-outer_angle), np.sin(-outer_angle)
                lx = ca * dx_pt - sa * dy_pt
                ly = sa * dx_pt + ca * dy_pt
                
                # Compute the maximum over all 6 edge normals
                for i in range(6):
                    n_ang = np.deg2rad(30 + 60 * i)
                    nx = np.cos(n_ang)
                    ny = np.sin(n_ang)
                    val = abs(nx * lx + ny * ly)
                    if val > max_required:
                        max_required = val
        
        side = max_required * 2 / math.sqrt(3)
        if side < best_side:
            best_side = side
            best_angle = outer_angle_deg
    
    inner = [[p[0], p[1], a] for p, a in zip(positions, angles)]
    return inner, (0.0, 0.0), best_side, best_angle

def _try_optimization(initial_inner, initial_center, initial_side, initial_angle, time_budget):
    """Try to optimize the arrangement using a simple local search."""
    start_time = time.time()
    
    inner = [list(p) for p in initial_inner]
    center = list(initial_center)
    side = initial_side
    angle = initial_angle
    
    best_side = side
    best_inner = [list(p) for p in inner]
    best_center = list(center)
    best_angle = angle
    
    def evaluate(arrangement):
        """Evaluate the arrangement and return the required outer side length."""
        max_required = 0.0
        for pos, ang in arrangement:
            ivx, ivy = _hex_vertices(pos[0], pos[1], 1.0, np.deg2rad(ang))
            for vx, vy in zip(ivx, ivy):
                dx_pt = vx - center[0]
                dy_pt = vy - center[1]
                ca, sa = np.cos(-np.deg2rad(angle)), np.sin(-np.deg2rad(angle))
                lx = ca * dx_pt - sa * dy_pt
                ly = sa * dx_pt + ca * dx_pt
                
                for i in range(6):
                    n_ang = np.deg2rad(30 + 60 * i)
                    nx = np.cos(n_ang)
                    ny = np.sin(n_ang)
                    val = abs(nx * lx + ny * ly)
                    if val > max_required:
                        max_required = val
        
        return max_required * 2 / math.sqrt(3)
    
    # Simple local search: try small perturbations
    step_size = 0.1
    for _ in range(1000):
        if time.time() - start_time > time_budget:
            break
        
        # Try perturbing one position
        idx = np.random.randint(N)
        old_x, old_y, old_a = inner[idx]
        inner[idx][0] += np.random.randn() * step_size
        inner[idx][1] += np.random.randn() * step_size
        inner[idx][2] += np.random.randn() * step_size * 5
        
        new_side = evaluate(inner)
        if new_side < best_side:
            best_side = new_side
            best_inner = [list(p) for p in inner]
            best_center = list(center)
            best_angle = angle
        else:
            inner[idx] = [old_x, old_y, old_a]
    
    return best_inner, best_center, best_side, best_angle

def construct():
    """Construct 12 unit hexagons inside a regular hexagon, minimizing the outer side length."""
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin
    
    # First, get a good initial arrangement from the honeycomb construction
    inner, center, side, angle = _construct_honeycomb()
    
    best_inner = [list(p) for p in inner]
    best_center = list(center)
    best_side = side
    best_angle = angle
    
    # Try local optimization if we have time
    if time.time() < deadline - 30:
        time_budget = min(deadline - time.time() - 10, 60)
        opt_inner, opt_center, opt_side, opt_angle = _try_optimization(
            inner, center, side, angle, time_budget
        )
        if opt_side < best_side:
            best_inner = opt_inner
            best_center = opt_center
            best_side = opt_side
            best_angle = opt_angle
    
    # Return the best result found
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
