"""Improved packing: fixed 3-4-3 hex cluster, optimize outer hex side via 1D rotation search."""
import math
import time

N = 11
DEADLINE = 140.0  # safety margin before 150s kill

def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy) with given side and rotation."""
    verts = []
    for i in range(6):
        a = math.radians(angle_deg + i * 60.0)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts

def _support_function(outer_verts, direction):
    """Max dot product of outer hex vertices with direction vector."""
    dx, dy = direction
    return max(vx * dx + vy * dy for vx, vy in outer_verts)

def _inner_max_extent(inner_positions, angle_deg, outer_rotation_deg):
    """
    Compute the minimum outer side length needed to contain all inner hexagons,
    given their positions (relative to outer center at origin) and the outer hex rotation.
    
    For a regular hexagon of side S centered at origin with rotation phi, the support
    function in direction theta is S * max_i cos(theta - phi - i*pi/3) ... 
    Actually for a regular hexagon with side S (circumradius = S), the support in direction u
    is S * max over vertices. But we can use: the hexagon contains point p iff
    for all 6 face normals n_k, n_k . p <= S * cos(pi/6) = S * sqrt(3)/2.
    
    Wait, let me reconsider. A regular hexagon with side length S has:
    - Circumradius (center to vertex) = S
    - Inradius (center to edge midpoint) = S * sqrt(3)/2
    
    The hexagon centered at origin with one vertex at angle 0 (flat-top vs pointy-top matters).
    Let's define: vertices at angles 0, 60, 120, 180, 240, 300 degrees (rotated by outer_rotation).
    
    The constraint is: for each inner hex vertex v (in global coords, relative to outer center),
    v must be inside the outer hex. The outer hex has 6 half-plane constraints.
    
    For outer hex with rotation phi, the 6 outward normals are at angles phi + k*60 + 30 (for flat-top)
    or phi + k*60 (depending on convention). Let me use: vertices at phi + k*60, so the edge
    between vertex k and k+1 has outward normal at angle phi + (k+0.5)*60 = phi + k*60 + 30.
    
    The half-plane constraint: for each k, n_k . p <= inradius = S * sqrt(3)/2
    where n_k is the unit outward normal.
    
    So S >= max over all inner hex vertices v and all 6 normals n_k of (n_k . v) / (sqrt(3)/2)
    = (2/sqrt(3)) * max_k (n_k . v)
    
    But we also need to account for the inner hexagons' own size. Each inner hex has 6 vertices
    at distance 1 from its center (since side=1, circumradius=1).
    """
    phi = math.radians(outer_rotation_deg)
    sqrt3 = math.sqrt(3.0)
    
    max_val = 0.0
    for (ix, iy, iangle) in inner_positions:
        # Compute 6 vertices of this inner hex
        iang_rad = math.radians(iangle)
        for j in range(6):
            va = iang_rad + j * math.pi / 3.0
            vx = ix + math.cos(va)
            vy = iy + math.sin(va)
            # Check against 6 half-planes of outer hex
            for k in range(6):
                na = phi + k * math.pi / 3.0 + math.pi / 6.0  # outward normal direction
                nx = math.cos(na)
                ny = math.sin(na)
                val = nx * vx + ny * vy
                if val > max_val:
                    max_val = val
    
    # S >= max_val / (sqrt(3)/2) = 2*max_val/sqrt(3)
    return 2.0 * max_val / sqrt3

def _build_343_cluster():
    """
    Build 11 unit hexagons in a 3-4-3 triangular arrangement.
    In a hex lattice with side-1 hexagons, adjacent centers are distance sqrt(3) apart
    (since inradius = sqrt(3)/2, two touching hexes have center distance = 2*inradius = sqrt(3)).
    
    Let's use a coordinate system where hex centers form a triangular lattice.
    For flat-top hexagons (one vertex pointing right), the lattice vectors are:
    e1 = (sqrt(3), 0)
    e2 = (sqrt(3)/2, 3/2)
    
    3-4-3 arrangement:
    Row 0 (3 hexes): positions along e1
    Row 1 (4 hexes): offset by e2
    Row 2 (3 hexes): offset by 2*e2
    
    Actually, let me think more carefully. A "3-4-3" cluster of hexagons:
    - 3 in top row
    - 4 in middle row  
    - 3 in bottom row
    
    Using axial coordinates for hex grid:
    Let's place them so the cluster is compact.
    
    For pointy-top hexagons, neighbors are at 60-degree intervals.
    Center distance = sqrt(3) for side-1 hexes.
    
    Let me use a simple layout:
    Middle row (4 hexes): x = 0, sqrt(3), 2*sqrt(3), 3*sqrt(3), y = 0
    Top row (3 hexes): x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, y = 3/2
    Bottom row (3 hexes): x = sqrt(3)/2, 3*sqrt(3)/2, 5*sqrt(3)/2, y = -3/2
    """
    s3 = math.sqrt(3.0)
    positions = []
    # Middle row: 4 hexes
    for i in range(4):
        positions.append((i * s3, 0.0, 0.0))
    # Top row: 3 hexes
    for i in range(3):
        positions.append((s3/2.0 + i * s3, 1.5, 0.0))
    # Bottom row: 3 hexes
    for i in range(3):
        positions.append((s3/2.0 + i * s3, -1.5, 0.0))
    
    # Center the cluster at origin
    cx = sum(p[0] for p in positions) / len(positions)
    cy = sum(p[1] for p in positions) / len(positions)
    positions = [(p[0] - cx, p[1] - cy, p[2]) for p in positions]
    
    return positions

def construct():
    start_time = time.time()
    
    # Build the 3-4-3 cluster
    inner = _build_343_cluster()
    
    # Try different outer rotations to find the one that minimizes required side
    # The outer hex rotation relative to inner hexes matters.
    # Inner hexes are all at angle 0 (vertices at 0, 60, 120, ... degrees)
    # Try outer rotations from 0 to 60 degrees
    
    best_side = float('inf')
    best_rotation = 0.0
    
    # Coarse search
    for deg in range(0, 61, 2):
        if time.time() - start_time > DEADLINE:
            break
        side = _inner_max_extent(inner, 0.0, float(deg))
        if side < best_side:
            best_side = side
            best_rotation = float(deg)
    
    # Fine search around best
    if time.time() - start_time < DEADLINE:
        for deg in [best_rotation - 2 + i * 0.1 for i in range(41)]:
            if deg < 0 or deg > 60:
                continue
            if time.time() - start_time > DEADLINE:
                break
            side = _inner_max_extent(inner, 0.0, deg)
            if side < best_side:
                best_side = side
                best_rotation = deg
    
    # Even finer
    if time.time() - start_time < DEADLINE:
        for deg in [best_rotation - 0.1 + i * 0.01 for i in range(21)]:
            if deg < 0 or deg > 60:
                continue
            if time.time() - start_time > DEADLINE:
                break
            side = _inner_max_extent(inner, 0.0, deg)
            if side < best_side:
                best_side = side
                best_rotation = deg
    
    # Also try rotating inner hexes relative to the cluster
    # Try a few inner rotation angles
    best_total = best_side
    best_inner_angle = 0.0
    best_outer_rot = best_rotation
    
    if time.time() - start_time < DEADLINE:
        for inner_ang in [0.0, 15.0, 30.0, 45.0, 60.0]:
            if time.time() - start_time > DEADLINE:
                break
            # Rebuild inner with rotation
            inner_rot = [(p[0], p[1], inner_ang) for p in inner]
            for outer_rot in [best_outer_rot, best_outer_rot + 15.0, best_outer_rot + 30.0]:
                if outer_rot > 60:
                    outer_rot -= 60
                if time.time() - start_time > DEADLINE:
                    break
                side = _inner_max_extent(inner_rot, 0.0, outer_rot)
                if side < best_total:
                    best_total = side
                    best_inner_angle = inner_ang
                    best_outer_rot = outer_rot
                    # Refine
                    for deg in [outer_rot - 0.5 + i * 0.05 for i in range(21)]:
                        if deg < 0 or deg > 60:
                            continue
                        if time.time() - start_time > DEADLINE:
                            break
                        s2 = _inner_max_extent(inner_rot, 0.0, deg)
                        if s2 < best_total:
                            best_total = s2
                            best_outer_rot = deg
    
    # Final result
    inner_final = [(p[0], p[1], best_inner_angle) for p in inner]
    
    return inner_final, (0.0, 0.0), best_total, best_outer_rot

# EVOLVE-BLOCK-START
# (The construct function is defined above; this block is kept for interface compatibility)
# EVOLVE-BLOCK-END
