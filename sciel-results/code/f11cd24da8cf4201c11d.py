import time
import math
import numpy as np
from itertools import product

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # 115s to leave safety margin before 120s/150s kill

    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon with center (cx,cy), side length, rotation in degrees."""
        angle = math.radians(angle_deg)
        # Regular hexagon vertices at angles 0, 60, 120, 180, 240, 300 degrees
        verts = []
        for k in range(6):
            theta = angle + k * math.pi / 3.0
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            verts.append((x, y))
        return verts

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside or on boundary of hexagon."""
        # Transform to hexagon's local frame
        angle = math.radians(angle_deg)
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)
        dx = px - cx
        dy = py - cy
        lx = dx * cos_a + dy * sin_a
        ly = -dx * sin_a + dy * cos_a
        # In local frame, hexagon with side s has vertices at angles 0,60,...
        # A regular hexagon can be described as intersection of 6 half-planes
        # The distance from center to each edge is s * sqrt(3) / 2
        # For a flat-top or pointy-top? Let's use the standard: vertices at 0,60,120,180,240,300
        # The edges are perpendicular to directions 30,90,150,210,270,330
        # Actually, let's just check against the 6 edges
        verts = [(side * math.cos(math.radians(60 * k)), side * math.sin(math.radians(60 * k))) for k in range(6)]
        for i in range(6):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % 6]
            # Cross product to check if point is on the correct side
            ex = x2 - x1
            ey = y2 - y1
            px_local = lx - x1
            py_local = ly - y1
            cross = ex * py_local - ey * px_local
            # For CCW polygon, interior is to the left, so cross >= 0
            if cross < -1e-9:
                return False
        return True

    def hexagon_overlap(v1, v2):
        """Check if two convex hexagons overlap using separating axis theorem."""
        axes = []
        for verts in [v1, v2]:
            for i in range(6):
                x1, y1 = verts[i]
                x2, y2 = verts[(i + 1) % 6]
                ex = x2 - x1
                ey = y2 - y1
                length = math.sqrt(ex * ex + ey * ey)
                if length > 1e-12:
                    axes.append((-ey / length, ex / length))
        
        for ax, ay in axes:
            min1 = min(v[0] * ax + v[1] * ay for v in v1)
            max1 = max(v[0] * ax + v[1] * ay for v in v1)
            min2 = min(v[0] * ax + v[1] * ay for v in v2)
            max2 = max(v[0] * ax + v[1] * ay for v in v2)
            if max1 < min2 or max2 < min1:
                return False
        return True

    def is_valid_packing(inner, outer_center, outer_side, outer_angle):
        """Check if all inner hexagons fit inside outer and don't overlap each other."""
        outer_verts = hex_vertices(outer_center[0], outer_center[1], outer_side, outer_angle)
        inner_verts_list = []
        for x, y, ang in inner:
            verts = hex_vertices(x, y, 1.0, ang)
            inner_verts_list.append(verts)
            # Check if inner hexagon is inside outer
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        # Check pairwise overlaps
        for i in range(len(inner_verts_list)):
            for j in range(i + 1, len(inner_verts_list)):
                if hexagon_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return False
        return True

    # Generate candidate configurations
    # Strategy: Try different arrangements of N=11 unit hexagons
    # Common packings: honeycomb-like, rows, clusters
    
    candidates = []
    
    # Configuration 1: Simple row
    for spacing in [1.7, 1.8, 1.9, 2.0, 2.1, 2.2]:
        inner = []
        for k in range(-(N // 2), N - N // 2):
            inner.append([spacing * k, 0.0, 0.0])
        candidates.append(inner)
    
    # Configuration 2: Two rows (honeycomb style)
    # Row 1: 6 hexagons, Row 2: 5 hexagons offset
    for dx in [1.7, 1.8, 1.9, 2.0, 2.1]:
        for dy in [1.5, 1.6, 1.7, 1.8]:
            inner = []
            # Row 1
            for k in range(-2, 4):
                inner.append([dx * k, dy, 0.0])
            # Row 2
            for k in range(-3, 3):
                inner.append([dx * (k + 0.5), -dy, 0.0])
            if len(inner) == N:
                candidates.append(inner)
    
    # Configuration 3: Three rows
    for dx in [1.7, 1.8, 1.9, 2.0]:
        for dy in [1.5, 1.6, 1.7]:
            inner = []
            # Row 1: 4 hexagons
            for k in range(-1, 3):
                inner.append([dx * k, 2 * dy, 0.0])
            # Row 2: 4 hexagons
            for k in range(-1, 3):
                inner.append([dx * (k + 0.5), dy, 0.0])
            # Row 3: 3 hexagons
            for k in range(-1, 2):
                inner.append([dx * k, 0.0, 0.0])
            # Row 4: 0? We need 11 total. 4+4+3=11
            if len(inner) == N:
                candidates.append(inner)
    
    # Configuration 4: Cluster (center + 6 around + 4 more)
    # Center at (0,0), 6 neighbors at distance sqrt(3) ~ 1.732
    d = math.sqrt(3)
    for scale in [0.95, 1.0, 1.05, 1.1]:
        inner = [[0.0, 0.0, 0.0]]
        for k in range(6):
            ang = k * 60.0
            x = d * scale * math.cos(math.radians(ang))
            y = d * scale * math.sin(math.radians(ang))
            inner.append([x, y, 0.0])
        # Add 4 more in second ring
        for k in range(4):
            ang = k * 90.0 + 45.0
            x = 2 * d * scale * math.cos(math.radians(ang))
            y = 2 * d * scale * math.sin(math.radians(ang))
            inner.append([x, y, 0.0])
        if len(inner) == N:
            candidates.append(inner)
    
    # Configuration 5: Linear with slight zigzag
    for base_dx in [1.8, 1.9, 2.0]:
        for zig in [0.0, 0.5, 1.0]:
            inner = []
            for k in range(-(N // 2), N - N // 2):
                y = zig * (1 if k % 2 == 0 else -1)
                inner.append([base_dx * k, y, 0.0])
            candidates.append(inner)

    best_score = float('inf')
    best_result = None
    
    for inner in candidates:
        if time.time() > deadline:
            break
        
        # Try to find minimum outer side length for this configuration
        # The outer hexagon must contain all inner hexagons
        # We can compute the minimum enclosing hexagon
        
        # Simple approach: try different outer sizes and rotations
        # The outer hexagon center should be near the centroid of inner hexagons
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        
        # Try various outer side lengths
        # Estimate minimum size
        max_dist = 0
        for x, y, ang in inner:
            # Approximate extent
            for k in range(6):
                theta = math.radians(ang + 60 * k)
                vx = x + math.cos(theta)
                vy = y + math.sin(theta)
                dist = math.sqrt((vx - cx)**2 + (vy - cy)**2)
                max_dist = max(max_dist, dist)
        
        # Minimum outer side is at least max_dist / (sqrt(3)/2) for a hexagon
        min_side_estimate = max_dist * 2 / math.sqrt(3)
        
        for outer_side in np.linspace(min_side_estimate * 0.9, min_side_estimate * 1.5, 50):
            if time.time() > deadline:
                break
            
            for outer_angle in [0, 30, 60, 90, 120, 150]:
                if is_valid_packing(inner, (cx, cy), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner, (cx, cy), outer_side, outer_angle)
                    break  # Found valid for this side, no need to try larger
        
        # Also try centered at origin
        for outer_side in np.linspace(min_side_estimate * 0.9, min_side_estimate * 1.5, 30):
            if time.time() > deadline:
                break
            for outer_angle in [0, 30, 60, 90, 120, 150]:
                if is_valid_packing(inner, (0.0, 0.0), outer_side, outer_angle):
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (inner, (0.0, 0.0), outer_side, outer_angle)
                    break
    
    if best_result is None:
        # Fallback: return a valid but loose packing
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_result
# EVOLVE-BLOCK-END
