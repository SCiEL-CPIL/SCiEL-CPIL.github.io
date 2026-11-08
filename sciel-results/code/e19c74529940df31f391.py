"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, s, angle_deg):
    """Vertices of a regular hexagon with center (cx,cy), side s, rotation angle_deg."""
    a = math.radians(angle_deg)
    verts = []
    for i in range(6):
        theta = a + i * math.pi / 3.0
        verts.append((cx + s * math.cos(theta), cy + s * math.sin(theta)))
    return verts

def _polygon_area_signed(verts):
    n = len(verts)
    area = 0.0
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        area += x1 * y2 - x2 * y1
    return 0.5 * area

def _point_in_polygon(pt, poly):
    x, y = pt
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            x_int = (x2 - x1) * (y - y1) / (y2 - y1) + x1
            if x < x_int:
                inside = not inside
    return inside

def _separating_axis(a, b):
    """Return True if convex polygons a and b (lists of (x,y)) are disjoint."""
    for poly in (a, b):
        for i in range(len(poly)):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % len(poly)]
            # normal
            nx = -(y2 - y1)
            ny = x2 - x1
            norm = math.hypot(nx, ny)
            if norm == 0:
                continue
            nx /= norm
            ny /= norm
            # project
            min_a = max_a = None
            min_b = max_b = None
            for px, py in a:
                d = px * nx + py * ny
                if min_a is None or d < min_a:
                    min_a = d
                if max_a is None or d > max_a:
                    max_a = d
            for px, py in b:
                d = px * nx + py * ny
                if min_b is None or d < min_b:
                    min_b = d
                if max_b is None or d > max_b:
                    max_b = d
            # if intervals don't overlap
            if max_a <= min_b or max_b <= min_a:
                return True
    return False

def _overlap(a, b):
    # a and b are lists of (x,y)
    if _separating_axis(a, b):
        return False
    # check one point of a inside b, or one point of b inside a
    if _point_in_polygon(a[0], b):
        return True
    if _point_in_polygon(b[0], a):
        return True
    return False

def _build_cluster(n_hex):
    """Generate a cluster of n_hex unit hexagons on a hex lattice.
    Returns list of (x, y, angle_deg).
    """
    positions = []
    # center
    positions.append((0.0, 0.0))
    count = 1
    if count >= n_hex:
        return positions
    # shells
    for shell in range(1, 50):
        # add 6*shell positions, but we need to be careful about which ones
        # We'll add all 6*shell and then truncate
        for k in range(6 * shell):
            # generate shell positions
            # The 6*shell positions are arranged in a ring
            # Start at (shell, 0) and go around
            # Actually, for a hex grid, shell s has 6*s points
            # Let's generate them properly
            pass
        # Proper generation for shell s:
        # The vertices of the shell are at:
        # (s, 0), (s-1, s), (-1, s), (-s, s-1), (-s, -1), (-s+1, -s) ... wait no.
        # Let's use the axial coordinate generation
        # A hex at distance s from center
        # Directions: 0, 60, 120, 180, 240, 300
        # We step along the 6 edges of the shell
        for i in range(6):
            # starting point
            # vertex i is at angle i*60
            # but we need 6*s points
            # Let's just generate all axial coords (q,r) with max(|q|,|r|,|q+r|) == s
            pass
        
        # Simpler: generate all (q,r) in a box and filter
        # but that's slow for large shells.
        # Let's stick to the ring generation
        # For shell s, the points are:
        # Edge 1: (s, -s) to (s, 0)  -> q=s, r from -s to 0
        # Edge 2: (s, 0) to (0, s)   -> q from s to 0, r from 0 to s (q+r=s)
        # Edge 3: (0, s) to (-s, s)  -> q from 0 to -s, r=s
        # Edge 4: (-s, s) to (-s, 0) -> q=-s, r from s to 0
        # Edge 5: (-s, 0) to (0, -s) -> q from -s to 0, r from 0 to -s (q+r=-s)
        # Edge 6: (0, -s) to (s, -s) -> q=0... wait (0,-s) to (s,-s) is r=-s, q from 0 to s
        # Let's just iterate
        # Actually, it's easier to just add points and break when we have enough
        # But we need unique points.
        
        # Let's generate the 6*s points for shell s
        points_in_shell = []
        for i in range(6):
            # corner i
            if i == 0:
                # (s, -s) to (s, 0)
                for r in range(-s, 1):
                    points_in_shell.append((s, r))
            elif i == 1:
                # (s, 0) to (0, s)
                for q in range(s-1, -1, -1):
                    points_in_shell.append((q, s - q))
            elif i == 2:
                # (0, s) to (-s, s)
                for q in range(-1, -s-1, -1):
                    points_in_shell.append((q, s))
            elif i == 3:
                # (-s, s) to (-s, 0)
                for r in range(s-1, -1, -1):
                    points_in_shell.append((-s, r))
            elif i == 4:
                # (-s, 0) to (0, -s)
                for q in range(-s+1, 1):
                    points_in_shell.append((q, -s - q))
            elif i == 5:
                # (0, -s) to (s, -s)
                for q in range(1, s+1):
                    points_in_shell.append((q, -s))
        
        # Convert axial to pixel
        # x = s * (sqrt(3) * q + sqrt(3)/2 * r)
        # y = s * (3/2 * r)
        # For unit hexagons, the distance between centers is 2 (since side=1, distance between adjacent centers is 2)
        # So scale factor is 2
        for q, r in points_in_shell:
            x = 2.0 * (math.sqrt(3) * q + 0.5 * math.sqrt(3) * r)
            y = 2.0 * (1.5 * r)
            positions.append((x, y))
            count += 1
            if count >= n_hex:
                break
        if count >= n_hex:
            break
        # remove duplicates? The ring generation above might have duplicates at corners.
        # (s,0) is in edge 1 and edge 2.
        # Let's fix the ranges
        # Edge 1: r from -s to 0 -> (s, -s) ... (s, 0)
        # Edge 2: q from s-1 to 0 -> (s-1, 1) ... (0, s)  [avoids (s,0)]
        # Edge 3: q from -1 to -s -> (-1, s) ... (-s, s)   [avoids (0,s)]
        # Edge 4: r from s-1 to 0 -> (-s, s-1) ... (-s, 0) [avoids (-s,s)]
        # Edge 5: q from -s+1 to 0 -> (-s+1, -s+1) ... (0, -s) [avoids (-s,0)]
        # Edge 6: q from 1 to s -> (1, -s) ... (s, -s)     [avoids (0,-s)]
        # Total: (s+1) + s + s + s + s + s = 6s + 1? No.
        # Edge 1: s+1 points
        # Edge 2: s points
        # Edge 3: s points
        # Edge 4: s points
        # Edge 5: s points
        # Edge 6: s points
        # Total 6s+1. One duplicate (s,-s) is not there, (s,0) is only in edge 1.
        # Wait, (s,-s) is in edge 1. Is it in edge 6? Edge 6 ends at (s,-s).
        # So (s,-s) is duplicate.
        # Let's just use a set.
        
        # Re-do with set
        points_in_shell_set = set()
        for i in range(6):
            if i == 0:
                for r in range(-s, 1):
                    points_in_shell_set.add((s, r))
            elif i == 1:
                for q in range(s-1, -1, -1):
                    points_in_shell_set.add((q, s - q))
            elif i == 2:
                for q in range(-1, -s-1, -1):
                    points_in_shell_set.add((q, s))
            elif i == 3:
                for r in range(s-1, -1, -1):
                    points_in_shell_set.add((-s, r))
            elif i == 4:
                for q in range(-s+1, 1):
                    points_in_shell_set.add((q, -s - q))
            elif i == 5:
                for q in range(1, s+1):
                    points_in_shell_set.add((q, -s))
        
        for q, r in points_in_shell_set:
            if count >= n_hex:
                break
            x = 2.0 * (math.sqrt(3) * q + 0.5 * math.sqrt(3) * r)
            y = 2.0 * (1.5 * r)
            # check if already added
            # (we only added center before)
            positions.append((x, y))
            count += 1
        if count >= n_hex:
            break
    return positions

def _compute_min_side(positions):
    """Compute the minimum outer hexagon side length to contain all unit hexagons at given positions."""
    # The outer hexagon must contain all inner hexagons.
    # An inner hexagon at (cx, cy) with rotation 0 has vertices at distance 1 from center.
    # So we need the outer hexagon to contain all vertices of all inner hexagons.
    # But wait, the inner hexagons can rotate. Here we assume rotation 0.
    # Actually, for a hexagonal cluster, rotation 0 is optimal.
    # The outer hexagon is also rotation 0.
    # The condition is that all inner hexagon vertices must be inside the outer hexagon.
    # Or, more simply, the Minkowski sum of the inner hexagons and the outer hexagon...
    # Actually, it's easier to find the maximum distance from the origin to any vertex of any inner hexagon,
    # and then set the outer hexagon apothem to be at least that distance.
    # But the outer hexagon is not a circle.
    # Let's find the minimum S such that all inner vertices are inside the outer hexagon of side S.
    # For a fixed rotation of the outer hexagon (0), the constraint is:
    # For each inner vertex (vx, vy), it must satisfy the 6 half-plane inequalities of the outer hexagon.
    # The outer hexagon of side S, center (0,0), rotation 0 has:
    # Vertices at (S,0), (S/2, S*sqrt(3)/2), etc.
    # The half-planes are defined by the sides.
    # Side 1: from (S,0) to (S/2, S*sqrt(3)/2). Normal points inward.
    # Actually, let's just use the support function.
    # The outer hexagon contains a point p if and only if for all 6 normals n_i, dot(n_i, p) <= apothem.
    # Apothem = S * sqrt(3) / 2.
    # The 6 normals are at angles 30, 90, 150, 210, 270, 330 degrees (for a hexagon with vertices at 0, 60, ...).
    # Wait, if vertices are at 0, 60, ..., the sides are at 30, 90, ...
    # So normals are at 30, 90, 150, 210, 270, 330.
    
    apothem_required = 0.0
    for cx, cy in positions:
        # Inner hexagon vertices (rotation 0)
        for i in range(6):
            theta = i * math.pi / 3.0
            vx = cx + math.cos(theta)
            vy = cy + math.sin(theta)
            # Check against all 6 normals
            for j in range(6):
                alpha = math.pi / 6 + j * math.pi / 3.0
                nx = math.cos(alpha)
                ny = math.sin(alpha)
                d = vx * nx + vy * ny
                if d > apothem_required:
                    apothem_required = d
    S = apothem_required * 2.0 / math.sqrt(3)
    return S

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Generate cluster
    positions = _build_cluster(N)
    # Ensure we have exactly N positions
    positions = positions[:N]
    
    # Compute minimum side
    best_side = _compute_min_side(positions)
    
    # Build inner list
    inner = []
    for x, y in positions:
        inner.append([x, y, 0.0])
    
    # Return
    return inner, (0.0, 0.0), best_side, 0.0
# EVOLVE-BLOCK-END
