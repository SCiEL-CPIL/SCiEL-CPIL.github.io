import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # safety margin before 150s kill

    def hex_vertices(cx, cy, s, angle_deg):
        """Return vertices of a regular hexagon centered at (cx,cy) with side s and rotation angle_deg."""
        angle = np.radians(angle_deg)
        angles = angle + np.pi / 3 * np.arange(6)
        xs = cx + s * np.cos(angles)
        ys = cy + s * np.sin(angles)
        return np.column_stack([xs, ys])

    def point_in_hex(px, py, cx, cy, s, angle_deg):
        """Check if point (px,py) is inside hexagon centered at (cx,cy) with side s and rotation angle_deg."""
        dx = px - cx
        dy = py - cy
        angle = np.radians(angle_deg)
        cos_a = np.cos(-angle)
        sin_a = np.sin(-angle)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        
        a = s * np.sqrt(3) / 2
        d1 = abs(lx * np.sqrt(3)/2 + ly * 0.5)
        d2 = abs(ly)
        d3 = abs(-lx * np.sqrt(3)/2 + ly * 0.5)
        return d1 <= a + 1e-9 and d2 <= a + 1e-9 and d3 <= a + 1e-9

    def check_disjoint(i, j, pos, angles, s=1.0):
        """Check if two unit hexagons at positions with given angles are disjoint."""
        cx1, cy1 = pos[i]
        cx2, cy2 = pos[j]
        a1 = angles[i]
        a2 = angles[j]
        
        v1 = hex_vertices(cx1, cy1, s, a1)
        v2 = hex_vertices(cx2, cy2, s, a2)
        
        # Check if any vertex of v1 is inside v2 or vice versa
        for k in range(6):
            px, py = v1[k]
            if point_in_hex(px, py, cx2, cy2, s, a2):
                return False
        for k in range(6):
            px, py = v2[k]
            if point_in_hex(px, py, cx1, cy1, s, a1):
                return False
        
        # Check edge intersections
        def seg_intersect(p1, p2, p3, p4):
            def cross(o, a, b):
                return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
            d1 = cross(p3, p4, p1)
            d2 = cross(p3, p4, p2)
            d3 = cross(p1, p2, p3)
            d4 = cross(p1, p2, p4)
            if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
               ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
                return True
            return False
        
        for i in range(6):
            for j in range(6):
                if seg_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                    return False
        
        return True

    def is_valid_configuration(pos, angles, outer_cx, outer_cy, outer_s, outer_angle):
        """Check if configuration is valid: all hexagons inside outer, all disjoint."""
        for i in range(N):
            cx, cy = pos[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, 1.0, a)
            for k in range(6):
                px, py = verts[k]
                if not point_in_hex(px, py, outer_cx, outer_cy, outer_s, outer_angle):
                    return False
        
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint(i, j, pos, angles):
                    return False
        
        return True

    # Hexagonal lattice configuration
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1),
    ]
    
    pos = []
    for q, r in axial_coords:
        x = 1.5 * q
        y = np.sqrt(3) * (r + q/2)
        pos.append([x, y])
    pos = np.array(pos)
    angles = [0.0] * 12
    
    centroid = pos.mean(axis=0)
    
    # Try different outer hexagon angles and find minimum side
    best_side = float('inf')
    best_outer_angle = 0.0
    
    for outer_angle in [0.0, 30.0, 60.0]:
        if time.time() > deadline - 20:
            break
        
        def is_valid_side(side):
            return is_valid_configuration(pos, angles, centroid[0], centroid[1], side, outer_angle)
        
        # Binary search on outer side length
        lo = 2.0
        hi = 15.0
        
        if not is_valid_side(hi):
            continue
        
        for _ in range(30):
            if time.time() > deadline - 5:
                break
            mid = (lo + hi) / 2
            if is_valid_side(mid):
                hi = mid
            else:
                lo = mid
        
        if hi < best_side:
            best_side = hi
            best_outer_angle = outer_angle
    
    if best_side < float('inf'):
        inner = [[pos[i][0], pos[i][1], angles[i]] for i in range(N)]
        return inner, (centroid[0], centroid[1]), best_side, best_outer_angle
    
    # Fallback
    inner = [[2.0 * i, 0.0, 0.0] for i in range(N)]
    return inner, (0.0, 0.0), 20.0, 0.0
# EVOLVE-BLOCK-END
