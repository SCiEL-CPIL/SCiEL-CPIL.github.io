import math
import time
import numpy as np

# EVOLVE-BLOCK-START
N = 11
MAX_TIME = 110.0

# Precompute hexagon geometry (unit side length 1)
# Vertices of a unit regular hexagon (pointy-top vs flat-top)
# For a regular hexagon with side 1, vertices at angles 0, 60, 120, 180, 240, 300
# Using flat-top: vertices at (1,0), (0.5, sqrt(3)/2), (-0.5, sqrt(3)/2), (-1,0), (-0.5,-sqrt(3)/2), (0.5,-sqrt(3)/2)
# Actually let's use the standard: for side s=1, the hexagon has 6 vertices
# We'll use a representation where angle=0 means a vertex points right

_HEX_VERTS = np.array([
    [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
])
_HEX_VERTS = _HEX_VERTS * 1.0  # side length 1

def _hex_vertices(cx, cy, angle_rad):
    """Return 6 vertices of a unit hexagon centered at (cx,cy) rotated by angle_rad."""
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    R = np.array([[c, -s], [s, c]])
    return R @ _HEX_VERTS.T + np.array([cx, cy])

def _signed_dist_point_to_hex(px, py, cx, cy, angle_rad):
    """
    Approximate signed distance from point (px,py) to a unit hexagon.
    Positive = outside, negative = inside.
    Uses distance to hexagon boundary.
    """
    # Transform point to hexagon's local frame
    dx, dy = px - cx, py - cy
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    lx = c * dx + s * dy
    ly = -s * dx + c * dy
    
    # For a regular hexagon with "circumradius" R=1 (distance center to vertex = 1),
    # the inradius (apothem) = sqrt(3)/2
    # The hexagon is the intersection of 6 half-planes.
    # Distance to boundary = min over 6 sides of (signed distance to that side's line)
    # Each side is at distance apothem = sqrt(3)/2 from center, normal pointing outward
    
    apothem = math.sqrt(3) / 2.0
    # Normals to each side (outward), at angles 30, 90, 150, 210, 270, 330 degrees
    # Side i has normal at angle (i+0.5)*60 degrees for i=0..5
    # Actually for a hexagon with vertex at angle 0, the sides are between vertices
    # Side between vertex 0 (angle 0) and vertex 1 (angle 60): normal at angle 30
    # Side between vertex 1 (60) and vertex 2 (120): normal at angle 90
    # etc.
    
    min_dist = float('inf')
    for i in range(6):
        normal_angle = (i + 0.5) * math.pi / 3.0  # 30, 90, 150, 210, 270, 330
        nx = math.cos(normal_angle)
        ny = math.sin(normal_angle)
        # Signed distance from point to the side line (positive = outside)
        # The side line: nx*x + ny*y = apothem
        dist = nx * lx + ny * ly - apothem
        if dist < min_dist:
            min_dist = dist
    return min_dist

def _hex_hex_overlap_distance(h1, h2):
    """
    h1, h2: (cx, cy, angle_rad)
    Returns a measure of overlap (positive if overlapping, 0 if touching/separated).
    Uses sampling: check if vertices of each hex are inside the other.
    For a smoother measure, we sample points.
    """
    cx1, cy1, a1 = h1
    cx2, cy2, a2 = h2
    
    # Check vertices of hex1 inside hex2 and vice versa
    max_penetration = 0.0
    
    v1 = _hex_vertices(cx1, cy1, a1)
    for i in range(6):
        px, py = v1[i]
        d = _signed_dist_point_to_hex(px, py, cx2, cy2, a2)
        if d < 0:
            max_penetration = max(max_penetration, -d)
    
    v2 = _hex_vertices(cx2, cy2, a2)
    for i in range(6):
        px, py = v2[i]
        d = _signed_dist_point_to_hex(px, py, cx1, cy1, a1)
        if d < 0:
            max_penetration = max(max_penetration, -d)
    
    return max_penetration

def _check_validity(inner, center, side, angle_deg):
    """
    Check if the packing is valid: all inner hexagons are inside the outer hexagon
    and no two inner hexagons overlap.
    Returns (is_valid, max_violation)
    """
    cx, cy = center
    outer_angle = math.radians(angle_deg)
    
    max_violation = 0.0
    
    # Check each inner hex is inside the outer hex
    for (ix, iy, iang) in inner:
        iang_rad = math.radians(iang)
        v = _hex_vertices(ix, iy, iang_rad)
        for j in range(6):
            px, py = v[j]
            d = _signed_dist_point_to_hex(px, py, cx, cy, outer_angle)
            # For the outer hex, side length is `side`, so we need to scale
            # The signed dist function assumes side=1. Scale: dist_outside = d * side
            # Actually, the hexagon with side `s` is a scaled version. 
            # A point is inside the outer hex (side s) iff scaled_point is inside unit hex.
            # Let's recompute: distance from center for outer hex
            dx, dy = px - cx, py - cy
            c, s = math.cos(outer_angle), math.sin(outer_angle)
            lx = c * dx + s * dy
            ly = -s * dx + c * dy
            apothem_outer = side * math.sqrt(3) / 2.0
            for k in range(6):
                normal_angle = (k + 0.5) * math.pi / 3.0
                nx = math.cos(normal_angle)
                ny = math.sin(normal_angle)
                dist = nx * lx + ny * ly - apothem_outer
                if dist > max_violation:
                    max_violation = dist
    
    # Check pairwise overlaps
    for i in range(len(inner)):
        for j in range(i+1, len(inner)):
            overlap = _hex_hex_overlap_distance(
                (inner[i][0], inner[i][1], math.radians(inner[i][2])),
                (inner[j][0], inner[j][1], math.radians(inner[j][2]))
            )
            if overlap > max_violation:
                max_violation = overlap
    
    return (max_violation < 1e-6), max_violation

def _compute_outer_side_for_packing(inner, center):
    """
    Given inner hexagons and outer center, find the minimum side length of the outer hex
    (with same rotation as... we need to also optimize rotation) such that all inner hexes fit.
    """
    cx, cy = center
    # For each inner hex vertex, compute the required "radius" in each of 6 normal directions
    # The outer hex must contain all points. The minimum enclosing hexagon...
    # Simplify: for a given outer rotation, compute the required side length.
    # We'll try a few rotations and pick the best.
    
    best_side = float('inf')
    for outer_angle_deg in range(0, 30, 5):  # try rotations 0, 5, 10, 15, 20, 25
        outer_angle = math.radians(outer_angle_deg)
        required = 0.0
        for (ix, iy, iang) in inner:
            iang_rad = math.radians(iang)
            v = _hex_vertices(ix, iy, iang_rad)
            for j in range(6):
                px, py = v[j]
                dx, dy = px - cx, py - cy
                c, s = math.cos(outer_angle), math.sin(outer_angle)
                lx = c * dx + s * dy
                ly = -s * dx + c * dy
                for k in range(6):
                    normal_angle = (k + 0.5) * math.pi / 3.0
                    nx = math.cos(normal_angle)
                    ny = math.sin(normal_angle)
                    dist = nx * lx + ny * ly
                    if dist > required:
                        required = dist
        side = required / (math.sqrt(3) / 2.0)
        if side < best_side:
            best_side = side
    
    return best_side, outer_angle_deg

def construct():
    t_start = time.time()
    
    # Initialize: place hexagons in a hexagonal lattice pattern
    # For N=11, a good initial guess is a hexagonal arrangement
    # Let's use a centered hexagonal pattern
    
    # Hexagonal lattice positions for N hexagons
    # A hexagon with "radius" r (in hex units) has 3r(r+1)+1 hexagons
    # For r=1: 7 hexagons, r=2: 19 hexagons
    # For N=11, we'll use a custom arrangement
    
    positions = []
    angles = [0.0] * N
    
    # Start with a hexagonal cluster
    # Center + 6 around + 4 more
    hex_spacing = 2.0 * math.sqrt(3) / 2.0 * 1.5  # spacing to avoid overlap initially
    # Actually for unit hexagons (side 1), the distance between centers of adjacent hexes
    # in a hexagonal packing is 2 * apothem = sqrt(3) ≈ 1.732
    # But they can be closer if rotated. Let's start with sqrt(3) spacing.
    
    base = math.sqrt(3)
    
    # Generate hexagonal lattice points
    lattice = [(0, 0)]
    for ring in range(1, 4):
        for i in range(6 * ring):
            angle = i * math.pi / (3 * ring)
            # Hexagonal ring positions
            # This is getting complex. Let's just use a simple grid.
            pass
    
    # Simpler: use a rectangular grid with hexagonal offset
    # For N=11, use 4x3 grid minus one
    cols, rows = 4, 3
    count = 0
    for r in range(rows):
        for c in range(cols):
            if count >= N:
                break
            x = c * base * 1.1
            y = r * base * 0.95
            if r % 2 == 1:
                x += base * 0.55
            positions.append([x, y])
            count += 1
        if count >= N:
            break
    
    # If we don't have enough, add more
    while len(positions) < N:
        x = len(positions) * 0.5
        y = 0.0
        positions.append([x, y])
    
    inner = [[p[0], p[1], 0.0] for p in positions]
    center = [0.0, 0.0]
    
    # Physics-based relaxation
    # State: positions (2N) and angles (N)
    # We'll optimize to minimize overlaps and then minimize outer side length
    
    dt = 0.05
    damping = 0.95
    repulsion_strength = 5.0
    attraction_strength = 1.0
    
    best_score = float('inf')
    best_inner = None
    best_center = None
    best_side = None
    best_angle = None
    
    n_iter = 0
    while time.time() - t_start < MAX_TIME:
        t_iter_start = time.time()
        
        # Compute forces
        forces = np.zeros((N, 3))  # fx, fy, torque
        
        # Repulsion between overlapping hexagons
        for i in range(N):
            for j in range(i+1, N):
                overlap = _hex_hex_overlap_distance(
                    (inner[i][0], inner[i][1], math.radians(inner[i][2])),
                    (inner[j][0], inner[j][1], math.radians(inner[j][2]))
                )
                if overlap > 0:
                    # Push apart
                    dx = inner[j][0] - inner[i][0]
                    dy = inner[j][1] - inner[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-8:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist
                    
                    f = repulsion_strength * overlap
                    forces[i, 0] -= dx * f
                    forces[i, 1] -= dy * f
                    forces[j, 0] += dx * f
                    forces[j, 1] += dy * f
                    
                    # Torque to reduce overlap (try rotating)
                    forces[i, 2] += f * 0.1
                    forces[j, 2] -= f * 0.1
        
        # Attraction toward center (to compact the packing)
        cx, cy = center
        for i in range(N):
            dx = cx - inner[i][0]
            dy = cy - inner[i][1]
            dist = math.sqrt(dx*dx + dy*dy)
            if dist > 1e-8:
                f = attraction_strength * dist * 0.1
                forces[i, 0] += dx / dist * f
                forces[i, 1] += dy / dist * f
        
        # Update positions
        for i in range(N):
            inner[i][0] += forces[i, 0] * dt
            inner[i][1] += forces[i, 1] * dt
            inner[i][2] += forces[i, 2] * dt
            # Damping is implicit in the force model
        
        # Recompute outer side length
        side, angle_deg = _compute_outer_side_for_packing(inner, center)
        
        if side < best_score:
            best_score = side
            best_inner = [list(x) for x in inner]
            best_center = [cx, cy]
            best_side = side
            best_angle = angle_deg
        
        n_iter += 1
        # Check time per iteration
        if time.time() - t_iter_start > 5.0 and n_iter > 10:
            break
    
    # Final validity check
    if best_inner is not None:
        is_valid, violation = _check_validity(best_inner, best_center, best_side, best_angle)
        if is_valid:
            return best_inner, tuple(best_center), best_side, best_angle
        else:
            # If invalid, try to fix by increasing side length
            # Or return a safe fallback
            pass
    
    # Fallback: return a valid (but possibly suboptimal) packing
    # Use the seed program's approach as fallback
    fallback_inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
    return fallback_inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END
