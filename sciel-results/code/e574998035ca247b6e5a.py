"""Improved packing of 11 unit hexagons inside a regular hexagon."""
import time
import math
import random
import numpy as np

N = 11
DEADLINE = 145.0  # seconds; leave 5s safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
    # Transform to hexagon's local frame
    dx = px - cx
    dy = py - cy
    a = math.radians(angle_deg)
    ca = math.cos(-a)
    sa = math.sin(-a)
    lx = ca * dx - sa * dy
    ly = sa * dx + ca * dy
    # In local frame, hexagon has vertices at angles 0,60,...,300
    # The hexagon is the intersection of 6 half-planes
    # For a flat-top hexagon (vertex at angle 0), the constraints are:
    # |ly| <= side * sqrt(3)/2
    # |lx| <= side
    # |lx| + |ly|/sqrt(3) <= side  ... actually let's use the standard form
    # A regular hexagon with circumradius R=side has inradius r = side*sqrt(3)/2
    # The 6 sides are at distance r from center
    # Side normals are at angles 30, 90, 150, 210, 270, 330 degrees (for pointy-top)
    # For vertex-at-0 hex: normals at 0, 60, 120, 180, 240, 300
    r = side * math.sqrt(3) / 2.0
    for i in range(6):
        theta = i * math.pi / 3.0
        nx = math.cos(theta)
        ny = math.sin(theta)
        if lx * nx + ly * ny > r:
            return False
    return True

def _hexagons_overlap(h1, h2, eps=1e-9):
    """Check if two hexagons overlap. h = (cx, cy, angle_deg), side=1."""
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    dx = cx2 - cx1
    dy = cy2 - cy1
    dist = math.sqrt(dx*dx + dy*dy)
    if dist > 2.0 + eps:
        return False
    if dist < 1e-6:
        # Same center - overlap unless same orientation and we consider them touching
        # Actually if same center they definitely overlap
        return True
    # Check if any vertex of h1 is inside h2, or vice versa
    v1 = _hex_vertices(cx1, cy1, 1.0, a1)
    v2 = _hex_vertices(cx2, cy2, 1.0, a2)
    for vx, vy in v1:
        if _point_in_hex(vx, vy, cx2, cy2, 1.0, a2):
            return True
    for vx, vy in v2:
        if _point_in_hex(vx, vy, cx1, cy1, 1.0, a1):
            return True
    # Check edge intersections
    def seg_intersect(p1, p2, p3, p4):
        """Check if segment p1-p2 intersects segment p3-p4."""
        def cross(o, a, b):
            return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
        d1 = cross(p3, p4, p1)
        d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3)
        d4 = cross(p1, p2, p4)
        if ((d1 > eps and d2 < -eps) or (d1 < -eps and d2 > eps)) and \
           ((d3 > eps and d4 < -eps) or (d3 < -eps and d4 > eps)):
            return True
        return False
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def _required_outer_side(inner, outer_center, outer_angle):
    """Compute the minimum side length of outer hexagon to contain all inner hexagons."""
    ocx, ocy = outer_center
    oa = math.radians(outer_angle)
    # For each inner hexagon, find max distance from outer center in each of 6 normal directions
    # The outer hexagon with side S centered at (ocx,ocy) rotated by oa contains a point p
    # iff for all 6 directions k: dot(p - c, n_k) <= S*sqrt(3)/2 where n_k are normals
    # Normals for vertex-at-angle-outer_angle hexagon: at angles outer_angle, outer_angle+60, ...
    max_proj = 0.0
    for (cx, cy, angle_deg) in inner:
        # Check all 6 vertices of this inner hexagon
        verts = _hex_vertices(cx, cy, 1.0, angle_deg)
        for (vx, vy) in verts:
            dx = vx - ocx
            dy = vy - ocy
            for k in range(6):
                theta = oa + k * math.pi / 3.0
                proj = dx * math.cos(theta) + dy * math.sin(theta)
                if proj > max_proj:
                    max_proj = proj
    # S * sqrt(3)/2 >= max_proj  =>  S >= max_proj * 2/sqrt(3)
    return max_proj * 2.0 / math.sqrt(3.0)

def _is_valid(inner, outer_center, outer_side, outer_angle, eps=1e-8):
    """Check if all inner hexagons are inside outer and don't overlap each other."""
    # Check containment
    ocx, ocy = outer_center
    oa = math.radians(outer_angle)
    r_outer = outer_side * math.sqrt(3) / 2.0
    for (cx, cy, angle_deg) in inner:
        verts = _hex_vertices(cx, cy, 1.0, angle_deg)
        for (vx, vy) in verts:
            dx = vx - ocx
            dy = vy - ocy
            for k in range(6):
                theta = oa + k * math.pi / 3.0
                proj = dx * math.cos(theta) + dy * math.sin(theta)
                if proj > r_outer + eps:
                    return False
    # Check pairwise non-overlap
    n = len(inner)
    for i in range(n):
        for j in range(i+1, n):
            if _hexagons_overlap(inner[i], inner[j], eps):
                return False
    return True

def _score(inner, outer_center, outer_side, outer_angle):
    """Compute the actual required outer side (tighter than given)."""
    return _required_outer_side(inner, outer_center, outer_angle)

def construct():
    start_time = time.time()
    best_score = float('inf')
    best_result = None
    
    # Initial good configurations to try
    
    # Strategy 1: Hexagonal lattice arrangement
    # Place hexagons in a compact hexagonal cluster
    # For 11 hexagons: center + 6 around + 4 more? Or 1+6+4=11
    # Actually a hexagonal number: 1, 7, 19... 11 is between 7 and 19
    # Let's try: center, 6 neighbors, then 4 more at the "corners" of the hex ring
    
    # Standard hex lattice positions (side=1, so centers are at distance sqrt(3) for adjacent)
    # Actually for unit hexagons (side 1), adjacent centers are at distance 2*inradius = sqrt(3)
    # Wait: inradius = sqrt(3)/2, so adjacent centers distance = 2 * sqrt(3)/2 = sqrt(3)
    
    # Let me use a different approach: try various structured arrangements
    
    arrangements = []
    
    # Arrangement 1: Linear row (like seed but optimized)
    # 11 in a row: length = 10*sqrt(3) + 2 = ~19.32, so outer side ~ 19.32/sqrt(3) + something
    # Actually for a row of hexagons, the bounding hexagon...
    
    # Arrangement 2: 2 rows
    # Row 1: 6 hexagons, Row 2: 5 hexagons, offset
    # This is like a hexagonal close packing
    
    # Arrangement 3: Compact cluster
    # Center + 6 + 4
    
    # Let's generate structured arrangements
    
    # Hex lattice basis vectors
    # For pointy-top hexagons, neighbors are at:
    # (0, sqrt(3)), (sqrt(3)/2, 3/2), (-sqrt(3)/2, 3/2), etc.
    # Actually let me use the standard axial coordinates
    
    # Let's try a few specific arrangements
    
    # Arrangement A: 3-4-3-1 or similar compact
    # Let's just do a local search from multiple starting points
    
    # Starting point 1: Hexagonal close packing
    # Centers in a hex lattice
    s3 = math.sqrt(3)
    
    # Generate positions for hex lattice
    def hex_lattice_positions(count):
        """Generate positions for 'count' hexagons in a hexagonal close-packed arrangement."""
        positions = [(0.0, 0.0)]
        ring = 1
        while len(positions) < count:
            # Add ring of 6*ring hexagons
            for i in range(6 * ring):
                if len(positions) >= count:
                    break
                # Position on ring
                # Ring i has 6*i hexagons
                # They go around the hexagon
                t = i / (6.0 * ring)
                angle = 2 * math.pi * t
                # For hex lattice, ring positions are at distance ring * s3 from center
                # But it's not circular, it's hexagonal
                # Let me use axial coordinates
                pass
            ring += 1
        return positions[:count]
    
    # Let me use a simpler approach: just place them in a grid-like pattern
    
    # For 11 hexagons, a good compact arrangement:
    # Row 0: 5 hexagons
    # Row 1: 3 hexagons (offset)
    # Row 2: 3 hexagons (offset)
    # Total: 11
    
    # Hex lattice: rows are offset by s3/2 in x, and s3*0.75... 
    # Actually for flat-top hexagons:
    # Horizontal spacing: 3/2 * side = 1.5
    # Vertical spacing: sqrt(3) * side = sqrt(3)
    # Odd rows offset by 0.75 in x
    
    # For pointy-top:
    # Horizontal: sqrt(3)
    # Vertical: 3/2
    # Odd rows offset by sqrt(3)/2
    
    # Let's use pointy-top (vertex at top)
    # Adjacent centers: distance sqrt(3)
    
    # Generate a compact cluster
    def generate_cluster(n):
        """Generate n positions in a hexagonal close-packed cluster."""
        positions = [(0.0, 0.0)]
        # Add rings
        ring = 1
        while len(positions) < n:
            # Ring 'ring' has 6*ring positions
            for i in range(6 * ring):
                if len(positions) >= n:
                    break
                # Axial coordinates for ring
                # For ring r, positions are at:
                # (r, 0), (r, -1), (r, -2), ..., (0, -r), (-1, -r+1), ...
                # Let me use cube coordinates
                # For ring r, the 6r positions are:
                # Start at (r, -r, 0) and go around
                q = 0
                r_coord = 0
                s = 0
                # The 6 corners of ring r are:
                # (r, -r, 0), (r, 0, -r), (0, r, -r), (-r, r, 0), (-r, 0, r), (0, -r, r)
                # Between corners, we interpolate
                corner_idx = i // ring  # 0-5
                pos_in_corner = i % ring
                corners = [
                    (r, -r, 0),
                    (r, 0, -r),
                    (0, r, -r),
                    (-r, r, 0),
                    (-r, 0, r),
                    (0, -r, r)
                ]
                c1 = corners[corner_idx]
                c2 = corners[(corner_idx + 1) % 6]
                # Interpolate
                q = c1[0] + (c2[0] - c1[0]) * pos_in_corner / ring
                r_c = c1[1] + (c2[1] - c1[1]) * pos_in_corner / ring
                s_c = c1[2] + (c2[2] - c1[2]) * pos_in_corner / ring
                # Convert to pixel (pointy-top)
                x = s3 * (q + r_c / 2.0)
                y = 1.5 * r_c
                positions.append((x, y))
            ring += 1
        return positions[:n]
    
    cluster_pos = generate_cluster(N)
    
    # Convert to inner hexagons with angle 0
    inner_arrangements = []
    
    # Arrangement 1: All pointy-top (angle 90 degrees for vertex at top, but let's use 0 for flat-top)
    # Actually angle 0 means vertex at right (flat-top if we consider standard)
    # Let's use angle 0 for all
    inner1 = [[x, y, 0.0] for (x, y) in cluster_pos]
    inner_arrangements.append(inner1)
    
    # Arrangement 2: All rotated 30 degrees
    inner2 = [[x, y, 30.0] for (x, y) in cluster_pos]
    inner_arrangements.append(inner2)
    
    # Arrangement 3: Mixed orientations
    inner3 = []
    for i, (x, y) in enumerate(cluster_pos):
        angle = 0.0 if i % 2 == 0 else 30.0
        inner3.append([x, y, angle])
    inner_arrangements.append(inner3)
    
    # Arrangement 4: Linear row
    row_pos = [(i * s3, 0.0) for i in range(N)]
    inner4 = [[x, y, 0.0] for (x, y) in row_pos]
    inner_arrangements.append(inner4)
    
    # Arrangement 5: Two rows
    row2_pos = []
    # Row 0: 6 hexagons
    for i in range(6):
        row2_pos.append((i * s3, 0.0))
    # Row 1: 5 hexagons, offset
    for i in range(5):
        row2_pos.append((i * s3 + s3/2, 1.5))
    inner5 = [[x, y, 0.0] for (x, y) in row2_pos]
    inner_arrangements.append(inner5)
    
    # Evaluate initial arrangements
    for inner in inner_arrangements:
        if time.time() - start_time > DEADLINE - 10:
            break
        # Center at centroid
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        # Try a few outer angles
        for oa in [0.0, 30.0, 60.0, 90.0, 15.0, 45.0]:
            side = _required_outer_side(inner, (cx, cy), oa)
            if _is_valid(inner, (cx, cy), side, oa):
                if side < best_score:
                    best_score = side
                    best_result = (inner, (cx, cy), side, oa)
    
    # Local search: perturb positions and angles
    if best_result is not None:
        best_inner, best_center, best_side, best_angle = best_result
    else:
        # Fallback: use first arrangement
        best_inner = inner_arrangements[0]
        cx = sum(p[0] for p in best_inner) / N
        cy = sum(p[1] for p in best_inner) / N
        best_center = (cx, cy)
        best_side = _required_outer_side(best_inner, best_center, 0.0)
        best_angle = 0.0
        best_score = best_side
    
    # Local optimization
    current_inner = [list(p) for p in best_inner]
    current_center = list(best_center)
    current_angle = best_angle
    current_score = best_score
    
    # Simulated annealing / random restart local search
    random.seed(42)
    
    for restart in range(50):
        if time.time() - start_time > DEADLINE - 15:
            break
        
        # Perturb
        perturbed_inner = []
        for p in current_inner:
            nx = p[0] + random.gauss(0, 0.3)
            ny = p[1] + random.gauss(0, 0.3)
            na = p[2] + random.gauss(0, 10)
            na %= 360
            perturbed_inner.append([nx, ny, na])
        
        # Also perturb center and outer angle slightly
        perturbed_center = [current_center[0] + random.gauss(0, 0.2), 
                           current_center[1] + random.gauss(0, 0.2)]
        perturbed_angle = (current_angle + random.gauss(0, 5)) % 360
        
        # Evaluate
        side = _required_outer_side(perturbed_inner, tuple(perturbed_center), perturbed_angle)
        
        if _is_valid(perturbed_inner, tuple(perturbed_center), side, perturbed_angle):
            if side < current_score:
                current_inner = perturbed_inner
                current_center = perturbed_center
                current_angle = perturbed_angle
                current_score = side
                if side < best_score:
                    best_score = side
                    best_result = (current_inner, tuple(current_center), side, current_angle)
    
    # More aggressive local search with smaller steps
    for step in range(200):
        if time.time() - start_time > DEADLINE - 10:
            break
        
        # Pick a random hexagon to move
        idx = random.randint(0, N-1)
        new_inner = [list(p) for p in current_inner]
        new_inner[idx][0] += random.gauss(0, 0.1)
        new_inner[idx][1] += random.gauss(0, 0.1)
        new_inner[idx][2] = (new_inner[idx][2] + random.gauss(0, 3)) % 360
        
        side = _required_outer_side(new_inner, tuple(current_center), current_angle)
        if _is_valid(new_inner, tuple(current_center), side, current_angle):
            if side < current_score:
                current_inner = new_inner
                current_score = side
                if side < best_score:
                    best_score = side
                    best_result = (current_inner, tuple(current_center), side, current_angle)
    
    # Try different outer angles with best inner
    if time.time() - start_time < DEADLINE - 5:
        for oa in np.linspace(0, 60, 25):
            if time.time() - start_time > DEADLINE - 5:
                break
            side = _required_outer_side(best_result[0], best_result[1], oa)
            if _is_valid(best_result[0], best_result[1], side, oa):
                if side < best_score:
                    best_score = side
                    best_result = (best_result[0], best_result[1], side, float(oa))
    
    # Final check
    if best_result is None:
        # Should not happen, but fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    inner, center, side, angle = best_result
    return inner, center, side, angle
