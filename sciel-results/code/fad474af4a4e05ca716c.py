"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import time

N = 11

# EVOLVE-BLOCK-START
def _hex_vertices(cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    angles = angle + np.pi * np.arange(6) / 3.0
    return np.column_stack([
        cx + side * np.cos(angles),
        cy + side * np.sin(angles)
    ])

def _point_in_hexagon(px, py, cx, cy, side, angle_deg):
    angle = np.radians(angle_deg)
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle)
    sin_a = np.sin(-angle)
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    inradius = side * np.sqrt(3) / 2
    for k in range(6):
        normal_angle = angle + np.radians(30 + 60 * k)
        nx = np.cos(normal_angle)
        ny = np.sin(normal_angle)
        if lx * nx + ly * ny > inradius:
            return False
    return True

def _segments_intersect(p1, p2, p3, p4):
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

def _hexagon_overlap(c1x, c1y, c1_angle, c2x, c2y, c2_angle, side1=1.0, side2=1.0):
    v1 = _hex_vertices(c1x, c1y, side1, c1_angle)
    v2 = _hex_vertices(c2x, c2y, side2, c2_angle)
    for i in range(6):
        if _point_in_hexagon(v1[i, 0], v1[i, 1], c2x, c2y, side2, c2_angle):
            return True
        if _point_in_hexagon(v2[i, 0], v2[i, 1], c1x, c1y, side1, c1_angle):
            return True
    for i in range(6):
        for j in range(6):
            if _segments_intersect(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6]):
                return True
    return False

def construct():
    start_time = time.time()
    deadline = 145.0
    
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = float('inf')
    best_angle = 0.0
    
    sqrt3 = np.sqrt(3)
    
    # Generate hexagonal lattice positions
    max_ring = 3
    positions = []
    for q in range(-max_ring, max_ring + 1):
        for r in range(-max_ring, max_ring + 1):
            if abs(q + r) <= max_ring:
                x = sqrt3 * (q + r / 2.0)
                y = 1.5 * r
                positions.append((x, y))
    
    positions.sort(key=lambda p: p[0]**2 + p[1]**2)
    
    for num_hex in range(1, N + 1):
        if time.time() > deadline:
            break
        if num_hex > len(positions):
            continue
        
        candidate_positions = positions[:num_hex]
        
        for angle_deg in [0.0, 30.0, 60.0, 90.0, 120.0, 150.0]:
            if time.time() > deadline:
                break
            
            valid = True
            for i in range(num_hex):
                for j in range(i + 1, num_hex):
                    x1, y1 = candidate_positions[i]
                    x2, y2 = candidate_positions[j]
                    if _hexagon_overlap(x1, y1, angle_deg, x2, y2, angle_deg):
                        valid = False
                        break
                if not valid:
                    break
            
            if not valid:
                continue
            
            all_vertices = []
            for x, y in candidate_positions:
                verts = _hex_vertices(x, y, 1.0, angle_deg)
                all_vertices.append(verts)
            all_vertices = np.vstack(all_vertices)
            
            centroid = np.mean(all_vertices, axis=0)
            cx, cy = centroid[0], centroid[1]
            
            best_s_for_theta = float('inf')
            for theta_deg in np.arange(0, 180, 5.0):
                if time.time() > deadline:
                    break
                
                theta = np.radians(theta_deg)
                inradius_needed = 0
                for k in range(6):
                    normal_angle = theta + np.radians(30 + 60 * k)
                    nx = np.cos(normal_angle)
                    ny = np.sin(normal_angle)
                    projections = (all_vertices[:, 0] - cx) * nx + (all_vertices[:, 1] - cy) * ny
                    max_proj = np.max(projections)
                    if max_proj > inradius_needed:
                        inradius_needed = max_proj
                
                s = 2.0 * inradius_needed / sqrt3
                if s < best_s_for_theta:
                    best_s_for_theta = s
            
            if best_s_for_theta < best_side:
                best_side = best_s_for_theta
                best_center = (cx, cy)
                best_angle = angle_deg
                best_inner = [[x, y, angle_deg] for x, y in candidate_positions]
    
    if best_inner is None:
        best_inner = [[i * sqrt3, 0.0, 0.0] for i in range(N)]
        best_center = ((N - 1) * sqrt3 / 2.0, 0.0)
        best_side = N * sqrt3 + 2.0
        best_angle = 0.0
    
    return best_inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
