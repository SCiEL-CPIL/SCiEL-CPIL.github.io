import math
import time

def construct():
    start_time = time.time()
    deadline = start_time + 100.0
    
    sqrt3 = math.sqrt(3.0)
    
    # Unit hexagon vertices (side 1), centered at origin, vertex at angle 0 degrees
    hex_verts = [(math.cos(math.radians(60*j)), math.sin(math.radians(60*j))) for j in range(6)]
    
    def get_hex_points(cx, cy, angle_deg=0.0):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + x*ca - y*sa, cy + x*sa + y*ca) for x, y in hex_verts]
    
    def min_side_at_rotation(points, center, rot_deg):
        """Minimal side of regular hexagon centered at 'center', rotated by rot_deg,
        that contains all points. The hexagon has vertices at rot_deg, rot_deg+60, ..."""
        s = 0.0
        for k in range(3):
            angle = math.radians(30 + rot_deg + 60*k)
            ux, uy = math.cos(angle), math.sin(angle)
            projs = [(px - center[0])*ux + (py - center[1])*uy for px, py in points]
            spread = max(projs) - min(projs)
            s = max(s, spread / sqrt3)
        return s
    
    def find_best_for_cluster(positions, inner_rot=0.0):
        """Find best outer hexagon for given hexagon positions."""
        all_points = []
        for cx, cy in positions:
            all_points.extend(get_hex_points(cx, cy, inner_rot))
        
        # Centroid of all vertices
        cx_m = sum(p[0] for p in all_points) / len(all_points)
        cy_m = sum(p[1] for p in all_points) / len(all_points)
        
        best_s = float('inf')
        best_rot = 0
        for rot in range(0, 30, 1):
            s = min_side_at_rotation(all_points, (cx_m, cy_m), rot)
            if s < best_s:
                best_s = s
                best_rot = rot
        
        # Also try center at centroid of hexagon centers
        cx_c = sum(p[0] for p in positions) / len(positions)
        cy_c = sum(p[1] for p in positions) / len(positions)
        for rot in range(0, 30, 1):
            s = min_side_at_rotation(all_points, (cx_c, cy_c), rot)
            if s < best_s:
                best_s = s
                best_rot = rot
                cx_m, cy_m = cx_c, cy_c
        
        return best_s, (cx_m, cy_m), best_rot
    
    # Configuration 1: Line of 11 hexagons
    line_positions = [(k * sqrt3, 0.0) for k in range(-5, 6)]
    s_line, c_line, r_line = find_best_for_cluster(line_positions)
    
    best_side = s_line
    best_positions = line_positions
    best_center = c_line
    best_outer_rot = r_line
    
    if time.time() > deadline - 30:
        inner = [[p[0], p[1], 0.0] for p in best_positions]
        return inner, (best_center[0], best_center[1]), best_side * 1.001, float(best_outer_rot)
    
    # Configuration 2: Hexagonal grid (center + 6 + 4)
    hex_dirs = [(math.cos(math.radians(60*j)), math.sin(math.radians(60*j))) for j in range(6)]
    hex_positions = [(0.0, 0.0)]
    for dx, dy in hex_dirs:
        hex_positions.append((sqrt3*dx, sqrt3*dy))
    # Add 4 more from ring 2
    ring2 = []
    for j in range(6):
        ring2.append((2*sqrt3*hex_dirs[j][0], 2*sqrt3*hex_dirs[j][1]))
        j1 = (j+1) % 6
        ix = sqrt3*hex_dirs[j][0] + sqrt3*hex_dirs[j1][0]
        iy = sqrt3*hex_dirs[j][1] + sqrt3*hex_dirs[j1][1]
        ring2.append((ix, iy))
    for pos in ring2:
        if len(hex_positions) < 11:
            hex_positions.append(pos)
    
    s_hex, c_hex, r_hex = find_best_for_cluster(hex_positions)
    if s_hex < best_side:
        best_side = s_hex
        best_positions = hex_positions
        best_center = c_hex
        best_outer_rot = r_hex
    
    if time.time() > deadline - 30:
        inner = [[p[0], p[1], 0.0] for p in best_positions]
        return inner, (best_center[0], best_center[1]), best_side * 1.001, float(best_outer_rot)
    
    # Configuration 3: 3-5-3 pattern
    def make_353():
        positions = []
        # Use hex grid: rows offset
        # Row y=0: 5 hexagons at x = -2*sqrt3, -sqrt3, 0, sqrt3, 2*sqrt3
        for k in range(-2, 3):
            positions.append((k*sqrt3, 0.0))
        # Row y=1.5: 3 hexagons at x = -1.5*sqrt3, 0, 1.5*sqrt3
        for k in [-1, 0, 1]:
            positions.append((k*1.5*sqrt3, 1.5))
        # Row y=-1.5: 3 hexagons
        for k in [-1, 0, 1]:
            positions.append((k*1.5*sqrt3, -1.5))
        return positions
    
    pos_353 = make_353()
    s_353, c_353, r_353 = find_best_for_cluster(pos_353)
    if s_353 < best_side:
        best_side = s_353
        best_positions = pos_353
        best_center = c_353
        best_outer_rot = r_353
    
    if time.time() > deadline - 30:
        inner = [[p[0], p[1], 0.0] for p in best_positions]
        return inner, (best_center[0], best_center[1]), best_side * 1.001, float(best_outer_rot)
    
    # Configuration 4: Try rotating inner hexagons for better packing
    # For the line configuration, try rotating all inner hexagons by 30 degrees
    s_line_rot, c_line_rot, r_line_rot = find_best_for_cluster(line_positions, inner_rot=30.0)
    if s_line_rot < best_side:
        best_side = s_line_rot
        best_positions = line_positions
        best_center = c_line_rot
        best_outer_rot = r_line_rot
    
    # Build final result
    inner = [[p[0], p[1], 0.0] for p in best_positions]
    final_side = best_side * 1.001  # Small margin
    
    return inner, (best_center[0], best_center[1]), final_side, float(best_outer_rot)

