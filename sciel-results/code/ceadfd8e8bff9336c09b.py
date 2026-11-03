import numpy as np
import time

N = 12
START_TIME = time.time()
TIME_LIMIT = 140.0

def _hex_vertices(cx, cy, s, angle_deg):
    a = np.radians(angle_deg)
    verts = []
    for k in range(6):
        theta = a + k * np.pi / 3
        verts.append((cx + s * np.cos(theta), cy + s * np.sin(theta)))
    return np.array(verts)

def _max_projection(verts, cx, cy, angle_deg):
    a = np.radians(angle_deg)
    max_val = 0.0
    for k in range(6):
        theta = a + k * np.pi / 3
        dx = np.cos(theta)
        dy = np.sin(theta)
        proj = np.abs((verts[:, 0] - cx) * dx + (verts[:, 1] - cy) * dy)
        m = np.max(proj)
        if m > max_val:
            max_val = m
    return max_val

def _side_length_for_config(positions, angle_deg, cx, cy):
    all_verts = []
    for (px, py) in positions:
        v = _hex_vertices(px, py, 1.0, angle_deg)
        all_verts.append(v)
    all_verts = np.vstack(all_verts)
    max_proj = _max_projection(all_verts, cx, cy, angle_deg)
    return max_proj * 2.0 / np.sqrt(3.0)

def _check_no_overlap(positions, eps=1e-6):
    n = len(positions)
    for i in range(n):
        for j in range(i+1, n):
            dx = positions[i][0] - positions[j][0]
            dy = positions[i][1] - positions[j][1]
            dist = np.sqrt(dx*dx + dy*dy)
            if dist < np.sqrt(3) - eps:
                return False
    return True

def _generate_lattice_config(row_counts):
    positions = []
    for row_idx, count in enumerate(row_counts):
        for col_idx in range(count):
            x = (col_idx - (count - 1) / 2.0) * np.sqrt(3)
            y = row_idx * 1.5
            if row_idx % 2 == 1:
                x += np.sqrt(3) / 2
            positions.append((x, y))
    return positions

# EVOLVE-BLOCK-START
def construct():
    global START_TIME
    if time.time() - START_TIME > TIME_LIMIT:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_config = None
    best_angle = 0.0
    best_center = (0.0, 0.0)
    
    row_configs = [
        [2, 3, 3, 2, 2], [3, 4, 3, 2], [2, 4, 4, 2], [3, 3, 3, 3],
        [4, 4, 4], [2, 3, 4, 3], [3, 4, 4, 1], [1, 4, 4, 3],
        [2, 2, 4, 4], [4, 3, 4, 1], [1, 3, 4, 4], [2, 4, 3, 3],
        [3, 3, 4, 2], [4, 2, 4, 2], [2, 4, 2, 4], [3, 2, 3, 4],
        [4, 1, 3, 4], [1, 4, 3, 4], [2, 3, 2, 5], [5, 2, 3, 2],
        [3, 5, 2, 2], [2, 5, 3, 2], [2, 2, 5, 3], [3, 2, 5, 2],
        [2, 5, 2, 3], [5, 3, 2, 2], [2, 3, 5, 2], [3, 5, 4],
        [5, 4, 3], [4, 5, 3], [3, 4, 5], [5, 3, 4], [4, 3, 5],
        [2, 6, 4], [4, 6, 2], [6, 4, 2], [2, 4, 6], [4, 2, 6],
        [6, 2, 4], [3, 6, 3], [6, 3, 3], [3, 3, 6], [1, 5, 6],
        [6, 5, 1], [1, 6, 5], [5, 6, 1], [6, 1, 5], [5, 1, 6],
        [2, 7, 3], [3, 7, 2], [7, 3, 2], [2, 3, 7], [3, 2, 7],
        [7, 2, 3], [1, 7, 4], [4, 7, 1], [7, 4, 1], [1, 4, 7],
        [4, 1, 7], [2, 8, 2], [8, 2, 2], [2, 2, 8], [3, 8, 1],
        [1, 8, 3], [8, 1, 3], [3, 1, 8], [1, 3, 8], [8, 3, 1],
        [4, 8, 0], [0, 8, 4], [8, 4, 0], [4, 0, 8], [0, 4, 8],
        [5, 7, 0], [7, 5, 0], [5, 0, 7], [0, 5, 7], [7, 0, 5],
        [0, 7, 5], [3, 3, 2, 4], [4, 2, 3, 3], [2, 3, 3, 4],
    ]
    
    row_configs = [c for c in row_configs if sum(c) == N and all(x >= 0 for x in c)]
    # Remove duplicates
    seen = set()
    unique_configs = []
    for c in row_configs:
        key = tuple(c)
        if key not in seen:
            seen.add(key)
            unique_configs.append(c)
    row_configs = unique_configs
    
    angle_range = np.arange(0, 60, 1.0)
    
    for config in row_configs:
        if time.time() - START_TIME > TIME_LIMIT - 5:
            break
        
        positions = _generate_lattice_config(config)
        
        if not _check_no_overlap(positions):
            continue
        
        cx_init = np.mean([p[0] for p in positions])
        cy_init = np.mean([p[1] for p in positions])
        
        for angle in angle_range:
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            
            for dx in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                for dy in [-1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5]:
                    cx = cx_init + dx
                    cy = cy_init + dy
                    
                    side = _side_length_for_config(positions, angle, cx, cy)
                    
                    if side < best_side:
                        best_side = side
                        best_config = positions[:]
                        best_angle = angle
                        best_center = (cx, cy)
    
    if best_config is not None and time.time() - START_TIME < TIME_LIMIT - 10:
        for angle in np.arange(best_angle - 3, best_angle + 3, 0.2):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            cx = best_center[0]
            cy = best_center[1]
            side = _side_length_for_config(best_config, angle, cx, cy)
            if side < best_side:
                best_side = side
                best_angle = angle
        
        for _ in range(100):
            if time.time() - START_TIME > TIME_LIMIT - 5:
                break
            improved = False
            for dx in [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]:
                for dy in [-0.5, -0.2, -0.1, 0.0, 0.1, 0.2, 0.5]:
                    cx = best_center[0] + dx
                    cy = best_center[1] + dy
                    side = _side_length_for_config(best_config, best_angle, cx, cy)
                    if side < best_side:
                        best_side = side
                        best_center = (cx, cy)
                        improved = True
            if not improved:
                break
    
    inner = []
    for (px, py) in best_config:
        inner.append([px, py, best_angle])
    
    return inner, best_center, best_side, best_angle
# EVOLVE-BLOCK-END
