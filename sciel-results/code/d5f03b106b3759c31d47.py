import math
import time
import numpy as np

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 120.0
SAFETY_MARGIN = 3.0

def min_enclosing_hexagon_side_fast(pts, angle_deg=0.0):
    """
    Fast computation of minimum side length of enclosing regular hexagon.
    For a fixed orientation, S = max_k(width_k) / sqrt(3)
    """
    a = math.radians(angle_deg)
    normals = []
    for k in range(6):
        theta = a + math.pi / 6.0 + k * math.pi / 3.0
        normals.append((math.cos(theta), math.sin(theta)))
    
    max_S = 0.0
    center_x = 0.0
    center_y = 0.0
    
    for nx, ny in normals:
        projections = [px * nx + py * ny for px, py in pts]
        w = max(projections) - min(projections)
        S_k = w / math.sqrt(3)
        if S_k > max_S:
            max_S = S_k
        center_x += (max(projections) + min(projections)) / 2.0 * nx
        center_y += (max(projections) + min(projections)) / 2.0 * ny
    
    center_x /= 6
    center_y /= 6
    
    return max_S, center_x, center_y

def check_no_overlap(pts, tolerance=0.001):
    """Check if unit hexagons (same orientation) don't overlap."""
    n = len(pts)
    for i in range(n):
        for j in range(i + 1, n):
            dx = pts[i][0] - pts[j][0]
            dy = pts[i][1] - pts[j][1]
            d = math.sqrt(dx*dx + dy*dy)
            if d < 2 * APOTHEM - tolerance:
                return False
    return True

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    sqrt3 = math.sqrt(3)
    
    # Initial hexagonal grid arrangement
    positions = [(0.0, 0.0)]
    for k in range(6):
        theta = k * math.pi / 3.0
        positions.append((sqrt3 * math.cos(theta), sqrt3 * math.sin(theta)))
    for k in range(4):
        theta = math.pi/6 + k * math.pi / 3.0
        positions.append((2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)))
    
    pos = np.array(positions, dtype=float)
    n = len(pos)
    
    # Force-directed relaxation
    dt = 0.02
    best_S = float('inf')
    best_pos = pos.copy()
    
    for it in range(5000):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        forces = np.zeros_like(pos)
        
        # Repulsion between pairs
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = math.sqrt(dx*dx + dy*dy)
                
                if dist < 1e-10:
                    dx = (np.random.random() - 0.5) * 0.01
                    dy = (np.random.random() - 0.5) * 0.01
                    dist = math.sqrt(dx*dx + dy*dy)
                
                min_dist = 2 * APOTHEM
                if dist < min_dist + 0.3:
                    force_mag = 3.0 * (min_dist + 0.3 - dist) / (dist + 1e-10)
                    fx = force_mag * dx / dist
                    fy = force_mag * dy / dist
                    forces[i, 0] += fx
                    forces[i, 1] += fy
                    forces[j, 0] -= fx
                    forces[j, 1] -= fy
        
        # Attraction toward centroid
        centroid = np.mean(pos, axis=0)
        for i in range(n):
            forces[i, 0] += 0.05 * (centroid[0] - pos[i, 0])
            forces[i, 1] += 0.05 * (centroid[1] - pos[i, 1])
        
        pos += dt * forces
        dt *= 0.999
        
        # Check and record best
        if check_no_overlap(pos):
            S, _, _ = min_enclosing_hexagon_side_fast(pos, 0.0)
            if S < best_S:
                best_S = S
                best_pos = pos.copy()
    
    # Try different outer orientations
    final_best_S = float('inf')
    final_best_angle = 0.0
    final_center = (0.0, 0.0)
    
    for angle_deg in np.linspace(0, 60, 61):
        S, cx, cy = min_enclosing_hexagon_side_fast(best_pos, angle_deg)
        if S < final_best_S:
            final_best_S = S
            final_best_angle = angle_deg
            final_center = (cx, cy)
    
    # Scale down to ensure no overlap
    SCALE = 0.999
    cx, cy = final_center
    pts_scaled = []
    for x, y in best_pos:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    S_final, cx_final, cy_final = min_enclosing_hexagon_side_fast(np.array(pts_scaled), final_best_angle)
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(final_best_angle)])
    
    center = [float(cx_final), float(cy_final)]
    side = float(S_final)
    rotation = float(final_best_angle)
    
    return inner, center, side, rotation
