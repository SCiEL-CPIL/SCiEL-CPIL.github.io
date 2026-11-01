"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    def hex_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle = np.radians(angle_deg)
        verts = []
        for i in range(6):
            theta = angle + i * np.pi / 3
            verts.append([cx + side * np.cos(theta), cy + side * np.sin(theta)])
        return np.array(verts)
    
    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy)."""
        angle = np.radians(angle_deg)
        # Transform point to hexagon's local frame
        dx = px - cx
        dy = py - cy
        # Rotate by -angle
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        # In local frame, hexagon has vertices at angles 0, 60, 120, 180, 240, 300
        # The hexagon contains points where max over the 6 edge normals of (n_i . p) <= inradius
        # Edge normals point at 30, 90, 150, 210, 270, 330 degrees
        inradius = side * np.sqrt(3) / 2
        for i in range(6):
            theta = np.radians(30 + i * 60)
            nx = np.cos(theta)
            ny = np.sin(theta)
            if lx * nx + ly * ny > inradius:
                return False
        return True
    
    def hexagon_in_hexagon(inner_cx, inner_cy, inner_angle, outer_cx, outer_cy, outer_side, outer_angle):
        """Check if inner unit hexagon is contained in outer hexagon."""
        # Check all 6 vertices of inner hexagon are inside outer
        inner_verts = hex_vertices(inner_cx, inner_cy, 1.0, inner_angle)
        for v in inner_verts:
            if not point_in_hexagon(v[0], v[1], outer_cx, outer_cy, outer_side, outer_angle):
                return False
        return True
    
    def cluster_contains(inner_positions, inner_angles, outer_cx, outer_cy, outer_side, outer_angle):
        """Check all inner hexagons are contained in outer."""
        for i in range(len(inner_positions)):
            if not hexagon_in_hexagon(inner_positions[i][0], inner_positions[i][1], 
                                       inner_angles[i], outer_cx, outer_cy, outer_side, outer_angle):
                return False
        return True
    
    def min_outer_side_for_rotation(inner_positions, inner_angles, outer_angle, outer_cx=0.0, outer_cy=0.0):
        """Binary search for minimum outer side length for given rotation."""
        # Lower bound: must contain at least one hexagon
        lo = 1.0
        # Upper bound: large enough
        hi = 20.0
        
        # First verify hi works
        for _ in range(30):
            mid = (lo + hi) / 2
            if cluster_contains(inner_positions, inner_angles, outer_cx, outer_cy, mid, outer_angle):
                hi = mid
            else:
                lo = mid
            if time.time() > deadline:
                break
        return hi
    
    def try_configuration(inner_positions, inner_angles, max_time=10.0):
        """Try to find optimal outer hexagon for given inner configuration."""
        start = time.time()
        best_side = 20.0
        best_angle = 0.0
        
        # Try different outer rotations
        n_angles = 60
        for i in range(n_angles):
            if time.time() > deadline or time.time() > start + max_time:
                break
            angle = i * 180.0 / n_angles  # 0 to 180 degrees
            side = min_outer_side_for_rotation(inner_positions, inner_angles, angle)
            if side < best_side:
                best_side = side
                best_angle = angle
        
        # Refine around best angle
        if time.time() < deadline - 5:
            a_lo = best_angle - 3.0
            a_hi = best_angle + 3.0
            for _ in range(20):
                a1 = a_lo + (a_hi - a_lo) * 0.333
                a2 = a_lo + (a_hi - a_lo) * 0.667
                s1 = min_outer_side_for_rotation(inner_positions, inner_angles, a1)
                s2 = min_outer_side_for_rotation(inner_positions, inner_angles, a2)
                if s1 < s2:
                    a_hi = a2
                else:
                    a_lo = a1
                if time.time() > deadline - 2:
                    break
            side = min_outer_side_for_rotation(inner_positions, inner_angles, (a_lo + a_hi) / 2)
            if side < best_side:
                best_side = side
                best_angle = (a_lo + a_hi) / 2
        
        return best_side, best_angle
    
    # Generate candidate honeycomb cluster configurations
    # Hexagon centers in a honeycomb grid: spacing between adjacent centers = sqrt(3)
    sqrt3 = np.sqrt(3)
    
    candidates = []
    
    # Configuration 1: 1 center + 6 surrounding + 5 more (roughly hexagonal)
    # Honeycomb coordinates: use axial coordinates
    # Center at (0,0), neighbors at distance sqrt3
    config1 = [[0, 0]]
    for i in range(6):
        angle = i * 60
        config1.append([sqrt3 * np.cos(np.radians(angle)), sqrt3 * np.sin(np.radians(angle))])
    # Add 5 more: extend one direction
    for i in range(1, 6):
        angle = i * 60
        config1.append([2 * sqrt3 * np.cos(np.radians(angle)), 2 * sqrt3 * np.sin(np.radians(angle))])
    # That's 12 total
    if len(config1) == 12:
        candidates.append((config1, [0.0] * 12))
    
    # Configuration 2: 2x6 rectangular arrangement
    config2 = []
    for row in range(2):
        for col in range(6):
            x = col * sqrt3
            y = row * 1.5 * 2  # vertical spacing for honeycomb rows
            # Actually for hexagons side 1, adjacent in same row: dx=sqrt3, dy=0
            # Next row offset: dx=sqrt3/2, dy=1.5
            y = row * 1.5
            x = col * sqrt3 + (row % 2) * sqrt3 / 2
            config2.append([x, y])
    if len(config2) == 12:
        candidates.append((config2, [0.0] * 12))
    
    # Configuration 3: Compact hexagonal shape (rows of 3,4,4,3 = 14, too many)
    # Try rows of 2,3,4,3 = 12
    config3 = []
    rows = [2, 3, 4, 3]
    for r, count in enumerate(rows):
        y = r * 1.5
        for c in range(count):
            x = c * sqrt3
            # Center the row
            x_offset = -(count - 1) * sqrt3 / 2
            config3.append([x + x_offset + (r % 2) * sqrt3 / 2, y])
    if len(config3) == 12:
        candidates.append((config3, [0.0] * 12))
    
    # Configuration 4: Rows of 3, 3, 3, 3
    config4 = []
    for r in range(4):
        y = r * 1.5
        for c in range(3):
            x = c * sqrt3 - sqrt3
            config4.append([x + (r % 2) * sqrt3 / 2, y])
    if len(config4) == 12:
        candidates.append((config4, [0.0] * 12))
    
    # Configuration 5: Flower with extended arms
    config5 = [[0, 0]]
    for i in range(6):
        angle = i * 60
        config5.append([sqrt3 * np.cos(np.radians(angle)), sqrt3 * np.sin(np.radians(angle))])
    # Add 5 more at distance 2*sqrt3 in 5 directions
    for i in range(5):
        angle = i * 72
        config5.append([2 * sqrt3 * np.cos(np.radians(angle)), 2 * sqrt3 * np.sin(np.radians(angle))])
    if len(config5) == 12:
        candidates.append((config5, [0.0] * 12))
    
    # Configuration 6: 3x4 grid-like
    config6 = []
    for r in range(3):
        for c in range(4):
            x = c * sqrt3
            y = r * 1.5
            x += (r % 2) * sqrt3 / 2
            config6.append([x - 1.5 * sqrt3, y - 1.5])
    if len(config6) == 12:
        candidates.append((config6, [0.0] * 12))
    
    # Try each configuration
    best_result = None
    best_side = 20.0
    
    for idx, (positions, angles) in enumerate(candidates):
        if time.time() > deadline - 10:
            break
        side, angle = try_configuration(positions, angles, max_time=12.0)
        if side < best_side:
            best_side = side
            best_result = (positions, angles, angle)
    
    # If we found a good result, use it
    if best_result is not None:
        positions, angles, outer_angle = best_result
        inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
        return inner, (0.0, 0.0), best_side, outer_angle
    else:
        # Fallback: simple linear arrangement
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
# EVOLVE-BLOCK-END

