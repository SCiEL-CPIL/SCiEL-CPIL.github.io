"""
Packing unit regular hexagons inside a regular hexagon.
Minimize the outer side length.
"""
import math
import time
import random
import numpy as np

N = 11
DEADLINE = 140.0  # seconds, safety margin before 150s kill
START_TIME = time.time()

def time_left():
    return DEADLINE - (time.time() - START_TIME)

def hex_vertices(cx, cy, side, angle_deg):
    """Return vertices of a regular hexagon centered at (cx, cy) with given side length and rotation."""
    angle = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return verts

def point_in_hexagon(px, py, cx, cy, side, angle_deg):
    """Check if point (px, py) is inside or on the boundary of a regular hexagon."""
    # Transform to local coordinates
    dx = px - cx
    dy = py - cy
    angle = math.radians(angle_deg)
    cos_a = math.cos(angle)
    sin_a = math.sin(angle)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    
    # In local coords, hexagon vertices are at angles 0, 60, 120, 180, 240, 300 degrees
    # The hexagon is the intersection of 6 half-planes
    # For a regular hexagon with side length s, the distance from center to each side (apothem) is s * sqrt(3)/2
    apothem = side * math.sqrt(3) / 2
    
    # The 6 normals point to the midpoints of the sides
    # Side midpoints are at angles 30, 90, 150, 210, 270, 330 degrees in local coords
    for i in range(6):
        theta = math.radians(30 + i * 60)
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > apothem:
            return False
    return True

def hex_hex_overlap(h1, h2):
    """
    Check if two regular hexagons overlap (interiors intersect).
    h1, h2: (cx, cy, side, angle_deg)
    Uses SAT (Separating Axis Theorem) for convex polygons.
    """
    c1x, c1y, s1, a1 = h1
    c2x, c2y, s2, a2 = h2
    
    v1 = hex_vertices(c1x, c1y, s1, a1)
    v2 = hex_vertices(c2x, c2y, s2, a2)
    
    # Collect all edge normals from both polygons
    edges = []
    for i in range(6):
        # Edge from v1[i] to v1[(i+1)%6]
        dx = v1[(i+1)%6][0] - v1[i][0]
        dy = v1[(i+1)%6][1] - v1[i][1]
        # Normal
        length = math.sqrt(dx*dx + dy*dy)
        if length > 1e-12:
            edges.append((-dy/length, dx/length))
        dx = v2[(i+1)%6][0] - v2[i][0]
        dy = v2[(i+1)%6][1] - v2[i][1]
        length = math.sqrt(dx*dx + dy*dy)
        if length > 1e-12:
            edges.append((-dy/length, dx/length))
    
    for nx, ny in edges:
        # Project both polygons onto this axis
        min1 = max1 = None
        min2 = max2 = None
        for vx, vy in v1:
            proj = vx * nx + vy * ny
            if min1 is None or proj < min1: min1 = proj
            if max1 is None or proj > max1: max1 = proj
        for vx, vy in v2:
            proj = vx * nx + vy * ny
            if min2 is None or proj < min2: min2 = proj
            if max2 is None or proj > max2: max2 = proj
        
        # Check for separation
        if max1 < min2 - 1e-10 or max2 < min1 - 1e-10:
            return False
    
    return True

def is_valid_packing(inner, outer_center, outer_side, outer_angle):
    """
    Check if the packing is valid:
    1. All inner hexagons are inside the outer hexagon
    2. No two inner hexagons overlap
    """
    ocx, ocy = outer_center
    
    # Check all inner hexagons are inside outer
    for ix, iy, iang in inner:
        # Check all 6 vertices of inner hex are inside outer
        v = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in v:
            if not point_in_hexagon(vx, vy, ocx, ocy, outer_side, outer_angle):
                return False
    
    # Check no overlaps between inner hexagons
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            if hex_hex_overlap((inner[i][0], inner[i][1], 1.0, inner[i][2]), 
                              (inner[j][0], inner[j][1], 1.0, inner[j][2])):
                return False
    
    return True

def compute_max_side(inner, outer_center, outer_angle):
    """
    Given fixed inner positions and outer center/angle, find the minimum outer side length
    such that all inner hexagons fit inside.
    """
    ocx, ocy = outer_center
    
    # For each vertex of each inner hexagon, compute the required side length
    max_side = 0.0
    for ix, iy, iang in inner:
        v = hex_vertices(ix, iy, 1.0, iang)
        for vx, vy in v:
            # Find minimum side length s such that (vx, vy) is inside hexagon at (ocx, ocy) with angle outer_angle
            # Transform to local coords of outer hexagon
            dx = vx - ocx
            dy = vy - ocy
            angle = math.radians(outer_angle)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            
            # The hexagon with side s has apothem s*sqrt(3)/2
            # For each of the 6 sides, the constraint is: lx*nx + ly*ny <= s*sqrt(3)/2
            # So s >= (lx*nx + ly*ny) / (sqrt(3)/2) for the tightest constraint
            # Actually, we need s such that for ALL 6 sides: lx*nx_i + ly*ny_i <= s*sqrt(3)/2
            # So s >= max over i of (lx*nx_i + ly*ny_i) / (sqrt(3)/2)
            # But only if the projection is positive; if negative, it's automatically satisfied for any s > 0
            
            max_proj = 0.0
            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
            
            required_side = max_proj / (math.sqrt(3) / 2)
            if required_side > max_side:
                max_side = required_side
    
    return max_side

def greedy_packing(n, max_iter=5000):
    """
    Greedy placement of n unit hexagons to minimize outer side length.
    """
    best_inner = None
    best_side = float('inf')
    
    for attempt in range(3):
        if time_left() < 5:
            break
        
        random.seed(42 + attempt)
        
        # Start with a reasonable initial configuration
        # Try a few different initial layouts
        if attempt == 0:
            # Linear arrangement
            inner = [[2.0 * k, 0.0, 0.0] for k in range(-(n//2), n - n//2)]
        elif attempt == 1:
            # Hexagonal cluster
            inner = []
            # Center
            inner.append([0.0, 0.0, 0.0])
            # Ring 1: 6 hexagons
            for i in range(6):
                angle = i * 60
                x = 2.0 * math.cos(math.radians(angle))
                y = 2.0 * math.sin(math.radians(angle))
                inner.append([x, y, 0.0])
            # Ring 2: add more if needed
            count = 1
            ring = 2
            while count < n:
                for i in range(6 * ring):
                    if count >= n:
                        break
                    angle = i * (360.0 / (6 * ring))
                    dist = 2.0 * ring * math.cos(math.radians(30))
                    # Actually, for hexagonal packing, the distance between centers in ring r is 2*r*sqrt(3)/2 = r*sqrt(3)
                    # Let me use proper hexagonal coordinates
                    pass
                break
            # Simpler: just use a grid-like pattern
            inner = []
            # Use axial coordinates for hex grid
            # Convert to pixel coordinates
            # For pointy-top hexagons with side 1, center distance is sqrt(3)
            # For flat-top, center distance is 2
            # Let's use flat-top: centers at (2*q, sqrt(3)*(r + q/2))
            q = 0
            r = 0
            count = 0
            while count < n:
                inner.append([2.0 * q, math.sqrt(3) * (r + q/2.0), 0.0])
                count += 1
                r += 1
                if r >= ring or r <= -ring:
                    r = 0
                    q += 1
                    ring += 1
        else:
            # Random initial positions
            inner = []
            for i in range(n):
                x = random.uniform(-5, 5)
                y = random.uniform(-5, 5)
                inner.append([x, y, 0.0])
        
        # Optimize: try to minimize the outer side length
        # Use a simple iterative improvement
        current_side = compute_max_side(inner, (0.0, 0.0), 0.0)
        
        for iteration in range(max_iter):
            if time_left() < 5:
                break
            
            # Try moving each hexagon slightly
            improved = False
            for i in range(n):
                if time_left() < 5:
                    break
                
                # Try small perturbations
                for dx in [-0.5, -0.1, 0.0, 0.1, 0.5]:
                    for dy in [-0.5, -0.1, 0.0, 0.1, 0.5]:
                        old_x, old_y, old_a = inner[i]
                        inner[i] = [old_x + dx, old_y + dy, old_a]
                        new_side = compute_max_side(inner, (0.0, 0.0), 0.0)
                        if new_side < current_side - 1e-8:
                            current_side = new_side
                            improved = True
                        else:
                            inner[i] = [old_x, old_y, old_a]
            
            if not improved:
                break
        
        # Try different outer angles
        for oang in [0.0, 30.0]:
            side = compute_max_side(inner, (0.0, 0.0), oang)
            if side < best_side:
                best_side = side
                best_inner = [list(x) for x in inner]
        
        # Also try optimizing outer center
        # Simple: try a few center positions
        for cx in [-1, 0, 1]:
            for cy in [-1, 0, 1]:
                side = compute_max_side(inner, (cx, cy), 0.0)
                if side < best_side:
                    best_side = side
                    best_inner = [list(x) for x in inner]
    
    if best_inner is None:
        best_inner = [[2.0 * k, 0.0, 0.0] for k in range(-(n//2), n - n//2)]
        best_side = compute_max_side(best_inner, (0.0, 0.0), 0.0)
    
    return best_inner, (0.0, 0.0), best_side, 0.0

def construct():
    """
    Construct the packing of N unit hexagons inside a regular hexagon.
    Returns: (inner, center, side, angle)
    """
    if time_left() < 10:
        # Emergency fallback
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        side = compute_max_side(inner, (0.0, 0.0), 0.0)
        return inner, (0.0, 0.0), side, 0.0
    
    # Try multiple strategies
    best_result = None
    best_side = float('inf')
    
    # Strategy 1: Greedy with linear initial
    inner1, center1, side1, angle1 = greedy_packing(N)
    if side1 < best_side:
        best_side = side1
        best_result = (inner1, center1, side1, angle1)
    
    if time_left() < 10:
        return best_result
    
    # Strategy 2: Try hexagonal cluster arrangement
    inner2 = []
    # Axial coordinate to pixel for flat-top hexagons
    # Center at (0,0)
    # Neighbors at distance 2 (for flat-top, side 1, center-to-center is 2)
    # Actually for regular hexagons of side 1, the distance between centers of adjacent hexagons is 2*1*sqrt(3)/2... 
    # No, for regular hexagons, the distance between centers of two adjacent (sharing an edge) hexagons is 2*apothem = 2*(sqrt(3)/2)*1 = sqrt(3)
    # Wait, let me reconsider. A regular hexagon with side s has:
    # - Circumradius = s
    # - Apothem = s*sqrt(3)/2
    # Two hexagons sharing an edge: the distance between centers is 2*apothem = s*sqrt(3)
    
    # So for side 1, center distance for edge-sharing is sqrt(3)
    
    # Hexagonal arrangement using axial coordinates
    # For pointy-top: x = sqrt(3)*(q + r/2), y = 3*r/2
    # For flat-top: x = 3*q/2, y = sqrt(3)*(r + q/2)
    
    # Let's use pointy-top with center distance sqrt(3)
    # Axial to pixel (pointy-top): x = sqrt(3)*(q + r/2), y = 3*r/2
    # But this gives center-to-center distance of sqrt(3) for adjacent hexes
    
    hex_coords = []
    # Generate hexagonal coordinates in order of distance from center
    max_ring = 0
    while len(hex_coords) < N:
        if max_ring == 0:
            hex_coords.append((0, 0))
        else:
            # Generate ring max_ring
            # Start at (max_ring, 0) and go around
            q, r = max_ring, 0
            for i in range(6):
                for j in range(max_ring):
                    hex_coords.append((q, r))
                    # Move in direction i
                    dq = [1, 0, -1, -1, 0, 1][i]
                    dr = [0, -1, -1, 0, 1, 1][i]
                    q += dq
                    r += dr
        if len(hex_coords) < N:
            max_ring += 1
    
    for q, r in hex_coords[:N]:
        x = math.sqrt(3) * (q + r/2.0)
        y = 1.5 * r
        inner2.append([x, y, 0.0])
    
    side2 = compute_max_side(inner2, (0.0, 0.0), 0.0)
    if side2 < best_side:
        best_side = side2
        best_result = (inner2, (0.0, 0.0), side2, 0.0)
    
    if time_left() < 10:
        return best_result
    
    # Strategy 3: Try with 30-degree rotation of inner hexagons
    inner3 = [list(x) for x in inner2]
    for i in range(len(inner3)):
        inner3[i][2] = 30.0
    side3 = compute_max_side(inner3, (0.0, 0.0), 0.0)
    if side3 < best_side:
        best_side = side3
        best_result = (inner3, (0.0, 0.0), side3, 0.0)
    
    # Strategy 4: Try outer hexagon at 30 degrees
    side4 = compute_max_side(inner2, (0.0, 0.0), 30.0)
    if side4 < best_side:
        best_side = side4
        best_result = (inner2, (0.0, 0.0), side4, 30.0)
    
    # Final polish: try to optimize center position for the best configuration
    if best_result is not None:
        inner_best, center_best, side_best, angle_best = best_result
        # Try small center offsets
        for dcx in [-0.5, -0.1, 0.0, 0.1, 0.5]:
            for dcy in [-0.5, -0.1, 0.0, 0.1, 0.5]:
                new_side = compute_max_side(inner_best, (center_best[0] + dcx, center_best[1] + dcy), angle_best)
                if new_side < side_best - 1e-10:
                    side_best = new_side
                    center_best = (center_best[0] + dcx, center_best[1] + dcy)
        best_result = (inner_best, center_best, side_best, angle_best)
    
    if best_result is None:
        inner = [[2.0 * k, 0.0, 0.0] for k in range(-(N//2), N - N//2)]
        side = compute_max_side(inner, (0.0, 0.0), 0.0)
        best_result = (inner, (0.0, 0.0), side, 0.0)
    
    return best_result
