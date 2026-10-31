import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    sqrt3 = np.sqrt(3.0)
    
    def hex_vertices(cx, cy, angle_deg=0.0):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.array([0, 60, 120, 180, 240, 300], dtype=float) * np.pi / 180.0
        return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])
    
    def min_enclosing_hex_side(positions, center, outer_angle_deg):
        cx, cy = center
        outer_angle_rad = np.radians(outer_angle_deg)
        cos_a = np.cos(-outer_angle_rad)
        sin_a = np.sin(-outer_angle_rad)
        
        max_s = 0.0
        for (ix, iy, iangle) in positions:
            v = hex_vertices(ix, iy, iangle)
            for (vx, vy) in v:
                dx = vx - cx
                dy = vy - cy
                lx = dx * cos_a - dy * sin_a
                ly = dx * sin_a + dy * cos_a
                s_req = max(
                    lx + ly/sqrt3,
                    2.0*ly/sqrt3,
                    -lx + ly/sqrt3,
                    -lx - ly/sqrt3,
                    -2.0*ly/sqrt3,
                    lx - ly/sqrt3
                )
                if s_req > max_s:
                    max_s = s_req
        return max_s
    
    # Honeycomb placement of 11 hexagons
    hex_grid = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, -1), (1, -1), (-1, -1),
        (2, 0), (2, -1)
    ]
    
    positions = []
    for (col, row) in hex_grid:
        x = sqrt3 * (col + row/2.0)
        y = 1.5 * row
        positions.append([x, y, 0.0])
    
    # Find best enclosing hexagon
    best_s = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    for angle in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
        all_pts = []
        for (x, y, a) in positions:
            v = hex_vertices(x, y, a)
            all_pts.extend(v.tolist())
        all_pts = np.array(all_pts)
        
        for cx in [0.0, (all_pts[:,0].min()+all_pts[:,0].max())/2.0]:
            for cy in [0.0, (all_pts[:,1].min()+all_pts[:,1].max())/2.0]:
                s = min_enclosing_hex_side(positions, (cx, cy), angle)
                if s < best_s:
                    best_s = s
                    best_center = (cx, cy)
                    best_angle = angle
    
    inner = [[p[0], p[1], p[2]] for p in positions]
    return inner, (best_center[0], best_center[1]), best_s, best_angle
# EVOLVE-BLOCK-END
