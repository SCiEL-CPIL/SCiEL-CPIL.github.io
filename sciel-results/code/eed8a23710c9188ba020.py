"""Packing unit regular hexagons inside a regular hexagon via hexagonal lattice + local search."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, angle_deg=0.0):
    """Vertices of a regular hexagon centered at (cx,cy) with given rotation (deg)."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = math.cos(theta)
        y = math.sin(theta)
        # rotate
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        verts.append((cx + rx, cy + ry))
    return verts

def _outer_hex_vertices(cx, cy, side, angle_deg=0.0):
    """Vertices of outer regular hexagon."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        rx = x * cos_a - y * sin_a
        ry = x * sin_a + y * cos_a
        verts.append((cx + rx, cy + ry))
    return verts

def _point_in_hex(px, py, cx, cy, side, angle_deg=0.0):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with given side and rotation."""
    a = math.radians(angle_deg)
    cos_a, sin_a = math.cos(a), math.sin(a)
    # transform to local coords
    dx = px - cx
    dy = py - cy
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    # regular hexagon with flat-top orientation: |x| <= side, |y| <= side*sqrt(3)/2, etc.
    # Actually for a hexagon with vertices at angles 0,60,...: 
    # the hexagon is defined by 6 half-planes. For a hexagon of "radius" R (center to vertex),
    # here side = R for regular hexagon.
    # The hexagon with vertices at R*cos(k*60), R*sin(k*60) can be described as:
    # For each of the 6 directions, the projection must be <= R*cos(30) = R*sqrt(3)/2
    # Actually, let's use the standard: a regular hexagon with side s has inradius = s*sqrt(3)/2
    # and the 6 supporting lines are at distance s*sqrt(3)/2 from center in 6 directions (30,90,150,...)
    R = side
    inradius = R * math.sqrt(3) / 2
    for k in range(6):
        theta = math.radians(30 + 60 * k)
        # normal direction
        nx = math.cos(theta)
        ny = math.sin(theta)
        proj = lx * nx + ly * ny
        if proj > inradius + 1e-9:
            return False
    return True

def _hex_in_outer(cx, cy, angle_deg, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if unit hexagon at (cx,cy) with rotation angle_deg is fully inside outer hexagon."""
    verts = _hex_vertices(cx, cy, angle_deg)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, outer_cx, outer_cy, outer_side, outer_angle_deg):
            return False
    return True

def _hexes_disjoint(hex1_cx, hex1_cy, hex1_angle, hex2_cx, hex2_cy, hex2_angle):
    """Check if two unit hexagons are disjoint (non-overlapping interiors)."""
    # For two regular hexagons of side 1, they are disjoint if the distance between centers
    # is at least the minimum distance between a vertex of one and the boundary of the other.
    # Conservative check: compute all vertex-edge distances.
    v1 = _hex_vertices(hex1_cx, hex1_cy, hex1_angle)
    v2 = _hex_vertices(hex2_cx, hex2_cy, hex2_angle)
    
    # Check if any vertex of hex1 is inside hex2 or vice versa
    for vx, vy in v1:
        if _point_in_hex(vx, vy, hex2_cx, hex2_cy, 1.0, hex2_angle):
            return False
    for vx, vy in v2:
        if _point_in_hex(vx, vy, hex1_cx, hex1_cy, 1.0, hex1_angle):
            return False
    
    # Check edge-edge intersections
    def seg_intersect(p1, p2, p3, p4):
        # Check if segment p1-p2 intersects segment p3-p4
        def orient(a, b, c):
            val = (b[1]-a[1])*(c[0]-a[0]) - (b[0]-a[0])*(c[1]-a[1])
            if abs(val) < 1e-10:
                return 0
            return 1 if val > 0 else -1
        def on_segment(a, b, c):
            # Check if b is on segment a-c
            if (min(a[0],c[0])-1e-10 <= b[0] <= max(a[0],c[0])+1e-10 and
                min(a[1],c[1])-1e-10 <= b[1] <= max(a[1],c[1])+1e-10):
                return True
            return False
        o1 = orient(p1, p2, p3)
        o2 = orient(p1, p2, p4)
        o3 = orient(p3, p4, p1)
        o4 = orient(p3, p4, p2)
        if o1 != o2 and o3 != o4:
            return True
        if o1 == 0 and on_segment(p1, p3, p2):
            return True
        if o2 == 0 and on_segment(p1, p4, p2):
            return True
        if o3 == 0 and on_segment(p3, p1, p4):
            return True
        if o4 == 0 and on_segment(p3, p2, p4):
            return True
        return False
    
    for i in range(6):
        for j in range(6):
            if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return False
    return True

def _is_valid_placement(positions, angles, outer_cx, outer_cy, outer_side, outer_angle_deg):
    """Check if all hexagons are inside outer and mutually disjoint."""
    for i, (cx, cy) in enumerate(positions):
        if not _hex_in_outer(cx, cy, angles[i], outer_cx, outer_cy, outer_side, outer_angle_deg):
            return False
    for i in range(len(positions)):
        for j in range(i+1, len(positions)):
            if not _hexes_disjoint(positions[i][0], positions[i][1], angles[i],
                                   positions[j][0], positions[j][1], angles[j]):
                return False
    return True

def _hexagonal_lattice_positions(n, spacing, angle_deg=0.0):
    """Generate positions for n hexagons on a hexagonal lattice.
    spacing is the distance between adjacent centers.
    Returns list of (x, y) positions."""
    # Hexagonal lattice: rows offset
    # For a hexagonal packing of unit hexagons, the minimum center-to-center distance
    # is 2*sqrt(3)/2 * 1 = sqrt(3) for aligned hexagons (flat-to-flat).
    # Actually, two unit hexagons side by side: center distance = 2 * inradius = 2 * sqrt(3)/2 = sqrt(3)
    # But if rotated, the required distance changes.
    # Let's use spacing as parameter.
    
    # Arrange in a roughly hexagonal cluster
    # Use axial coordinates
    positions = []
    # Generate enough points and take first n
    max_r = 10
    candidates = []
    for q in range(-max_r, max_r+1):
        for r in range(-max_r, max_r+1):
            # Convert axial to cartesian
            x = spacing * (q + r/2)
            y = spacing * (math.sqrt(3)/2 * r)
            candidates.append((x, y, q*q + r*r + q*r))
    candidates.sort(key=lambda c: c[2])
    for x, y, _ in candidates[:n]:
        positions.append((x, y))
    return positions

def _optimize_with_lattice(deadline):
    """Try hexagonal lattice configurations and optimize."""
    best_side = 40.0
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0
    
    # The minimum center-to-center distance for two unit hexagons depends on relative rotation.
    # For aligned hexagons (same orientation), min distance = sqrt(3) ≈ 1.732
    # Let's try various spacings and outer rotations.
    
    for outer_angle in [0.0, 30.0]:
        for spacing in [1.73, 1.74, 1.75, 1.8, 1.85, 1.9, 2.0]:
            positions = _hexagonal_lattice_positions(N, spacing)
            angles = [0.0] * N
            
            # Center the cluster
            cx_avg = sum(p[0] for p in positions) / N
            cy_avg = sum(p[1] for p in positions) / N
            positions = [(p[0]-cx_avg, p[1]-cy_avg) for p in positions]
            
            # Compute bounding hexagon side needed
            # The outer hexagon must contain all vertices of all inner hexagons.
            # For a given outer rotation, find the minimum side length.
            # The side length is determined by the maximum projection of any vertex onto the 6 normal directions.
            
            all_verts = []
            for i, (cx, cy) in enumerate(positions):
                verts = _hex_vertices(cx, cy, angles[i])
                all_verts.extend(verts)
            
            # Find minimum outer hexagon side centered at origin with given rotation
            inradius_max = 0.0
            for vx, vy in all_verts:
                for k in range(6):
                    theta = math.radians(30 + 60 * k + outer_angle)
                    nx = math.cos(theta)
                    ny = math.sin(theta)
                    proj = vx * nx + vy * ny
                    inradius_max = max(inradius_max, abs(proj))
            
            side = inradius_max / (math.sqrt(3)/2)
            
            if side < best_side:
                # Verify validity
                if _is_valid_placement(positions, angles, 0.0, 0.0, side, outer_angle):
                    best_side = side
                    best_positions = positions
                    best_angles = angles
                    best_outer_angle = outer_angle
    
    return best_side, best_positions, best_angles, best_outer_angle

def _local_search(positions, angles, outer_cx, outer_cy, outer_side, outer_angle_deg, deadline):
    """Local search to improve the packing."""
    best_side = outer_side
    best_positions = [list(p) for p in positions]
    best_angles = list(angles)
    
    # Try to shrink the outer hexagon by moving hexagons
    for iteration in range(200):
        if time.time() > deadline:
            break
        improved = False
        for i in range(N):
            for dx in [-0.1, 0, 0.1]:
                for dy in [-0.1, 0, 0.1]:
                    new_positions = [list(p) for p in best_positions]
                    new_positions[i] = [new_positions[i][0] + dx, new_positions[i][1] + dy]
                    
                    # Check if still valid with slightly smaller outer side
                    new_side = best_side - 0.01
                    if new_side < 1.0:
                        continue
                    if _is_valid_placement(new_positions, best_angles, outer_cx, outer_cy, new_side, outer_angle_deg):
                        best_side = new_side
                        best_positions = new_positions
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break
        if not improved:
            break
    
    return best_side, best_positions, best_angles, outer_angle_deg

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    # Try lattice-based approach
    side, positions, angles, outer_angle = _optimize_with_lattice(deadline)
    
    if positions is not None and time.time() < deadline - 10:
        # Local search to improve
        side, positions, angles, outer_angle = _local_search(
            positions, angles, 0.0, 0.0, side, outer_angle, deadline
        )
    
    # Format positions as (x, y, angle_degrees)
    inner = [[positions[i][0], positions[i][1], angles[i]] for i in range(N)]
    
    return inner, (0.0, 0.0), side, outer_angle
# EVOLVE-BLOCK-END
