"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def construct():
    t0 = time.time()
    deadline = 150.0 - 10.0  # 10s safety margin
    
    # Gap factor to ensure strict disjointness
    gap = 1.001
    spacing = 2.0 * gap  # center-to-center distance for same-orientation hexagons
    
    # Hexagonal lattice arrangement: 3-4-3 pattern
    # Flat-top hexagons, all with angle 0
    positions = []
    # Row 0 (top): 3 hexagons
    for i in range(3):
        x = (i - 1) * spacing
        y = spacing * np.sqrt(3) / 2
        positions.append([x, y, 0.0])
    # Row 1 (middle): 4 hexagons
    for i in range(4):
        x = (i - 1.5) * spacing
        y = 0.0
        positions.append([x, y, 0.0])
    # Row 2 (bottom): 3 hexagons
    for i in range(3):
        x = (i - 1) * spacing
        y = -spacing * np.sqrt(3) / 2
        positions.append([x, y, 0.0])
    
    inner = positions
    center = (0.0, 0.0)
    
    # Compute minimum outer hexagon side length
    # A regular hexagon with center (cx, cy), side s, rotation angle a
    # has vertices at (cx + s*cos(a + k*60°), cy + s*sin(a + k*60°)) for k=0..5
    # The hexagon contains a point (px, py) if for all 6 edges, the point is on the inside
    
    def hexagon_contains_vertex(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside or on the boundary of the outer hexagon."""
        a = np.radians(angle_deg)
        # For a regular hexagon centered at (cx,cy) with circumradius s and rotation a,
        # the distance from center to each side (inradius) is s * cos(30°) = s * √3/2
        # The hexagon is the intersection of 6 half-planes.
        # Edge k is perpendicular to direction (a + k*60°) and at distance s*√3/2 from center.
        # A point is inside if its projection onto each edge normal is <= s*√3/2
        inradius = s * np.sqrt(3) / 2.0
        for k in range(6):
            theta = a + k * np.pi / 3.0
            # Normal direction
            nx = np.cos(theta)
            ny = np.sin(theta)
            # Distance from center to point along normal
            dist = (px - cx) * nx + (py - cy) * ny
            if dist > inradius + 1e-9:
                return False
        return True
    
    def all_inner_vertices_in_outer(cx, cy, s, outer_angle_deg):
        """Check all vertices of all inner hexagons are inside the outer hexagon."""
        for (ix, iy, iangle) in inner:
            a = np.radians(iangle)
            for k in range(6):
                theta = a + k * np.pi / 3.0
                vx = ix + np.cos(theta)
                vy = iy + np.sin(theta)
                if not hexagon_contains_vertex(vx, vy, cx, cy, s, outer_angle_deg):
                    return False
        return True
    
    def required_side_length(cx, cy, outer_angle_deg):
        """Binary search for minimum side length such that all inner vertices are inside."""
        lo = 0.0
        hi = 100.0
        for _ in range(60):
            mid = (lo + hi) / 2.0
            if all_inner_vertices_in_outer(cx, cy, mid, outer_angle_deg):
                hi = mid
            else:
                lo = mid
        return hi
    
    # Try different outer hexagon orientations to find the minimum side length
    best_side = float('inf')
    best_angle = 0.0
    
    # Try a range of outer angles
    for outer_angle in np.arange(0, 60, 5.0):
        if time.time() > deadline:
            break
        s = required_side_length(0.0, 0.0, outer_angle)
        if s < best_side:
            best_side = s
            best_angle = outer_angle
    
    # Refine around the best angle
    if time.time() < deadline:
        for outer_angle in np.arange(best_angle - 5, best_angle + 5, 0.5):
            if time.time() > deadline:
                break
            s = required_side_length(0.0, 0.0, outer_angle)
            if s < best_side:
                best_side = s
                best_angle = outer_angle
    
    # Final fine-tune
    if time.time() < deadline:
        for outer_angle in np.arange(best_angle - 0.5, best_angle + 0.5, 0.05):
            if time.time() > deadline:
                break
            s = required_side_length(0.0, 0.0, outer_angle)
            if s < best_side:
                best_side = s
                best_angle = outer_angle
    
    # Add a tiny buffer to ensure validity
    best_side *= 1.0001
    
    return inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END

