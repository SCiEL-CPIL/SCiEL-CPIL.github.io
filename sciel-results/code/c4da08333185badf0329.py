import numpy as np
import time

N = 12

def hex_vertices(cx, cy, angle_rad):
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack((cx + np.cos(angles), cy + np.sin(angles)))

def min_container_side(positions, angles, container_angle):
    max_val = 0.0
    normal_angles = container_angle + np.pi/6 + np.arange(6) * np.pi / 3
    normals = np.column_stack((np.cos(normal_angles), np.sin(normal_angles)))
    
    for i in range(N):
        verts = hex_vertices(positions[i][0], positions[i][1], angles[i])
        dots = verts @ normals.T
        max_dot = np.max(dots)
        if max_dot > max_val:
            max_val = max_dot
            
    S = max_val * 2.0 / np.sqrt(3.0)
    return S

def is_inside_hex(px, py, cx, cy, angle_rad):
    dx = px - cx
    dy = py - cy
    cos_a = np.cos(-angle_rad)
    sin_a = np.sin(-angle_rad)
    rx = dx * cos_a - dy * sin_a
    ry = dx * sin_a + dy * cos_a
    
    normal_angles = np.pi/6 + np.arange(6) * np.pi / 3
    normals_x = np.cos(normal_angles)
    normals_y = np.sin(normal_angles)
    
    dots = rx * normals_x + ry * normals_y
    return np.max(dots) <= np.sqrt(3)/2 + 1e-9

def construct():
    start_time = time.time()
    deadline = start_time + 110.0
    
    d = np.sqrt(3.0)
    
    pos = [[0.0, 0.0]]
    ang = [0.0]
    for k in range(6):
        a = k * np.pi / 3
        pos.append([d * np.cos(a), d * np.sin(a)])
        ang.append(0.0)
    for k in range(5):
        a = k * 2 * np.pi / 5 + np.pi/5
        pos.append([2*d * np.cos(a), 2*d * np.sin(a)])
        ang.append(0.0)
        
    cont_angle = 0.0
    best_S = min_container_side(pos, ang, cont_angle)
    best_pos = [p[:] for p in pos]
    best_ang = ang[:]
    best_cont_angle = cont_angle
    
    T = 0.5
    T_min = 0.001
    
    current_verts = [hex_vertices(pos[i][0], pos[i][1], ang[i]) for i in range(N)]
    
    while time.time() < deadline:
        T = max(T * 0.999, T_min)
        
        current_S = min_container_side(pos, ang, cont_angle)
        
        move_type = np.random.rand()
        new_pos = [p[:] for p in pos]
        new_ang = ang[:]
        new_cont_angle = cont_angle
        
        if move_type < 0.7:
            idx = np.random.randint(N)
            new_pos[idx][0] += np.random.randn() * 0.5
            new_pos[idx][1] += np.random.randn() * 0.5
        elif move_type < 0.9:
            idx = np.random.randint(N)
            new_ang[idx] += np.random.randn() * 0.5
        else:
            new_cont_angle += np.random.randn() * 0.1
            
        overlap = False
        if move_type < 0.9:
            new_verts = hex_vertices(new_pos[idx][0], new_pos[idx][1], new_ang[idx])
            for j in range(N):
                if j == idx: continue
                for k in range(6):
                    if is_inside_hex(new_verts[k,0], new_verts[k,1], pos[j][0], pos[j][1], ang[j]):
                        overlap = True
                        break
                if overlap: break
                for k in range(6):
                    if is_inside_hex(current_verts[j][k,0], current_verts[j][k,1], new_pos[idx][0], new_pos[idx][1], new_ang[idx]):
                        overlap = True
                        break
                if overlap: break
        else:
            pass

        if overlap:
            continue
            
        new_S = min_container_side(new_pos, new_ang, new_cont_angle)
        
        delta = new_S - current_S
        
        if delta < 0 or np.random.rand() < np.exp(-delta / T):
            pos = new_pos
            ang = new_ang
            cont_angle = new_cont_angle
            current_S = new_S
            if move_type < 0.9:
                current_verts[idx] = new_verts
            if current_S < best_S:
                best_S = current_S
                best_pos = [p[:] for p in pos]
                best_ang = ang[:]
                best_cont_angle = cont_angle
                
    inner = []
    for i in range(N):
        inner.append([best_pos[i][0], best_pos[i][1], np.degrees(best_ang[i])])
        
    return inner, (0.0, 0.0), best_S, np.degrees(best_cont_angle)
