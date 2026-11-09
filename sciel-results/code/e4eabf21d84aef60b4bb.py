"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
import random

N = 12

def _hexagon_vertices(cx, cy, side, angle_deg=0.0):
    angle_rad = np.radians(angle_deg)
    vertices = []
    for i in range(6):
        a = angle_rad + i * np.pi / 3
        x = cx + side * np.cos(a)
        y = cy + side * np.sin(a)
        vertices.append((x, y))
    return np.array(vertices)

def _point_in_hexagon(px, py, cx, cy, side, angle_deg=0.0):
    dx = px - cx
    dy = py - cy
    angle_rad = np.radians(angle_deg)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    lx = dx * cos_a + dy * sin_a
    ly = -dx * sin_a + dy * cos_a
    proj0 = lx
    proj1 = lx * np.cos(np.pi/3) + ly * np.sin(np.pi/3)
    proj2 = lx * np.cos(2*np.pi/3) + ly * np.sin(2*np.pi/3)
    return max(abs(proj0), abs(proj1), abs(proj2)) <= side

def _hexagons_disjoint(v1, a1, v2, a2, side=1.0, eps=1e-9):
    for vx, vy in v1:
        if _point_in_hexagon(vx, vy, v2[0][0], v2[0][1], side, a2):
            return False
    for vx, vy in v2:
        if _point_in_hexagon(vx, vy, v1[0][0], v1[0][1], side, a1):
            return False
    return True

def _all_disjoint(positions, angles, side=1.0):
    n = len(positions)
    verts = []
    for i in range(n):
        verts.append(_hexagon_vertices(positions[i][0], positions[i][1], side, angles[i]))
    for i in range(n):
        for j in range(i+1, n):
            if not _hexagons_disjoint(verts[i], angles[i], verts[j], angles[j], side):
                return False
    return True

def _min_enclosing_hexagon_side(positions, angles, side=1.0):
    all_verts = []
    for i in range(len(positions)):
        v = _hexagon_vertices(positions[i][0], positions[i][1], side, angles[i])
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    
    best_s = float('inf')
    cx = np.mean(all_verts[:, 0])
    cy = np.mean(all_verts[:, 1])
    
    for rot_deg in np.linspace(0, 60, 13):
        rot_rad = np.radians(rot_deg)
        cos_r = np.cos(rot_rad)
        sin_r = np.sin(rot_rad)
        dx = all_verts[:, 0] - cx
        dy = all_verts[:, 1] - cy
        rx = dx * cos_r - dy * sin_r
        ry = dx * sin_r + dy * cos_r
        
        proj0 = np.abs(rx)
        proj1 = np.abs(rx * np.cos(np.pi/3) + ry * np.sin(np.pi/3))
        proj2 = np.abs(rx * np.cos(2*np.pi/3) + ry * np.sin(2*np.pi/3))
        s_needed = np.max(np.maximum(proj0, np.maximum(proj1, proj2)))
        if s_needed < best_s:
            best_s = s_needed
    
    return best_s

def _objective(positions, angles, side=1.0):
    if not _all_disjoint(positions, angles, side):
        return float('inf')
    return _min_enclosing_hexagon_side(positions, angles, side)

# EVOLVE-BLOCK-START
def construct():
    random.seed(42)
    np.random.seed(42)
    
    sqrt3 = np.sqrt(3)
    
    # Initial honeycomb configuration: 3 rows of 4
    positions = []
    angles = []
    for row in range(3):
        y = (row - 1) * 1.5
        x_offset = (sqrt3 / 2) * (row % 2)
        for col in range(4):
            x = col * sqrt3 - 1.5 * sqrt3 + x_offset
            positions.append([x, y])
            angles.append(0.0)
    
    current_pos = [list(p) for p in positions]
    current_ang = angles[:]
    current_obj = _objective(current_pos, current_ang)
    
    best_pos = [list(p) for p in current_pos]
    best_ang = current_ang[:]
    best_obj = current_obj
    
    T = 5.0
    T_min = 0.01
    cooling = 0.995
    max_iters = 3000
    
    for it in range(max_iters):
        if T < T_min:
            break
        idx = random.randint(0, N-1)
        dx = random.uniform(-0.5, 0.5)
        dy = random.uniform(-0.5, 0.5)
        da = random.uniform(-10, 10)
        
        new_pos = [list(p) for p in current_pos]
        new_ang = current_ang[:]
        new_pos[idx][0] += dx
        new_pos[idx][1] += dy
        new_ang[idx] = (current_ang[idx] + da) % 360
        
        new_obj = _objective(new_pos, new_ang)
        
        if new_obj < current_obj:
            current_pos = new_pos
            current_ang = new_ang
            current_obj = new_obj
            if current_obj < best_obj:
                best_pos = [list(p) for p in current_pos]
                best_ang = current_ang[:]
                best_obj = current_obj
        else:
            if current_obj == float('inf'):
                current_pos = new_pos
                current_ang = new_ang
                current_obj = new_obj
                if current_obj < best_obj:
                    best_pos = [list(p) for p in current_pos]
                    best_ang = current_ang[:]
                    best_obj = current_obj
            else:
                delta = new_obj - current_obj
                if delta < 0 or random.random() < np.exp(-delta / T):
                    current_pos = new_pos
                    current_ang = new_ang
                    current_obj = new_obj
                    if current_obj < best_obj:
                        best_pos = [list(p) for p in current_pos]
                        best_ang = current_ang[:]
                        best_obj = current_obj
        
        T *= cooling
    
    inner = [[p[0], p[1], a] for p, a in zip(best_pos, best_ang)]
    cx = float(np.mean([p[0] for p in best_pos]))
    cy = float(np.mean([p[1] for p in best_pos]))
    side = float(best_obj)
    
    return inner, (cx, cy), side, 0.0
# EVOLVE-BLOCK-END

