import numpy as np
import time
import math

N = 12

def get_hex_vertices(cx, cy, angle, side=1.0):
    verts = []
    for i in range(6):
        theta = angle + i * math.pi / 3.0
        x = cx + side * math.cos(theta)
        y = cy + side * math.sin(theta)
        verts.append((x, y))
    return np.array(verts)

def project_points(points, axis):
    return np.dot(points, axis)

def get_axes(vertices):
    axes = []
    n = len(vertices)
    for i in range(n):
        p1 = vertices[i]
        p2 = vertices[(i + 1) % n]
        edge = p2 - p1
        normal = np.array([-edge[1], edge[0]])
        norm = np.linalg.norm(normal)
        if norm > 1e-9:
            normal = normal / norm
        else:
            normal = np.array([1.0, 0.0])
        axes.append(normal)
    return np.array(axes)

def sat_overlap_depth(verts1, verts2):
    axes1 = get_axes(verts1)
    axes2 = get_axes(verts2)
    
    min_overlap = float('inf')
    
    for axis in np.vstack([axes1, axes2]):
        proj1 = project_points(verts1, axis)
        proj2 = project_points(verts2, axis)
        
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 < min2 or max2 < min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_overlap:
            min_overlap = overlap
            
    return min_overlap

def point_in_hex_penalty(x, y, S):
    apothem = S * math.sqrt(3) / 2.0
    angles = [math.pi/6, math.pi/2, 5*math.pi/6, 7*math.pi/6, 3*math.pi/2, 11*math.pi/6]
    penalty = 0.0
    for a in angles:
        nx = math.cos(a)
        ny = math.sin(a)
        dist = nx * x + ny * y
        if dist > apothem:
            penalty += (dist - apothem)
    return penalty

def total_penalty(state, S):
    overlap_pen = 0.0
    contain_pen = 0.0
    verts_list = []
    for x, y, ang in state:
        v = get_hex_vertices(x, y, ang)
        verts_list.append(v)
        for px, py in v:
            contain_pen += point_in_hex_penalty(px, py, S)
            
    for i in range(N):
        for j in range(i + 1, N):
            depth = sat_overlap_depth(verts_list[i], verts_list[j])
            if depth > 0:
                overlap_pen += depth * 10.0
    return overlap_pen, contain_pen

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    # Initialize with a compact grid-like structure to help convergence
    state = []
    # 12 hexagons: 3 rows of 4?
    # Hex grid spacing: dx = sqrt(3), dy = 1.5
    # Centered at 0
    positions = []
    for r in range(-1, 2):
        for c in range(-1, 3):
            if len(positions) >= 12: break
            x = c * math.sqrt(3) + (0.5 * math.sqrt(3) if r % 2 != 0 else 0)
            y = r * 1.5
            # Center the grid roughly
            x -= 1.5 * math.sqrt(3)
            y -= 0.75
            positions.append((x, y))
            
    for i in range(N):
        if i < len(positions):
            x, y = positions[i]
        else:
            x = np.random.uniform(-2, 2)
            y = np.random.uniform(-2, 2)
        ang = 0.0
        state.append([x, y, ang])
        
    current_S = 4.5
    current_state = [list(s) for s in state]
    
    op, cp = total_penalty(current_state, current_S)
    current_cost = current_S + 100.0 * (op + cp)
    
    best_S = current_S
    best_state = [list(s) for s in current_state]
    best_cost = current_cost
    
    T = 1.0
    cooling = 0.9995
    
    while time.time() < deadline:
        new_state = [list(s) for s in current_state]
        new_S = current_S
        
        move_type = np.random.rand()
        
        if move_type < 0.7:
            idx = np.random.randint(N)
            dx = np.random.normal(0, 0.5)
            dy = np.random.normal(0, 0.5)
            dth = np.random.normal(0, 0.5)
            new_state[idx][0] += dx
            new_state[idx][1] += dy
            new_state[idx][2] += dth
        elif move_type < 0.9:
            idx1 = np.random.randint(N)
            idx2 = np.random.randint(N)
            while idx2 == idx1:
                idx2 = np.random.randint(N)
            for idx in [idx1, idx2]:
                dx = np.random.normal(0, 0.3)
                dy = np.random.normal(0, 0.3)
                dth = np.random.normal(0, 0.3)
                new_state[idx][0] += dx
                new_state[idx][1] += dy
                new_state[idx][2] += dth
        else:
            new_S = current_S + np.random.normal(0, 0.1)
            if new_S < 3.0: new_S = 3.0
            
        op, cp = total_penalty(new_state, new_S)
        new_cost = new_S + 100.0 * (op + cp)
        
        if new_cost < current_cost:
            current_state = new_state
            current_S = new_S
            current_cost = new_cost
            
            if new_cost < best_cost:
                best_cost = new_cost
                best_S = new_S
                best_state = [list(s) for s in new_state]
        else:
            delta = new_cost - current_cost
            if delta < 0: delta = 0
            prob = math.exp(-delta / T)
            if np.random.rand() < prob:
                current_state = new_state
                current_S = new_S
                current_cost = new_cost
                
        T *= cooling
        if T < 1e-6:
            T = 1e-6
            
    inner = []
    for x, y, ang in best_state:
        inner.append([x, y, math.degrees(ang) % 360.0])
        
    return inner, (0.0, 0.0), best_S, 0.0
# EVOLVE-BLOCK-END
