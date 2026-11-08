import numpy as np
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def axial_to_cartesian(q, r):
        x = np.sqrt(3) * (q + r/2)
        y = 1.5 * r
        return np.array([x, y])

    def get_side_length(positions, outer_center, outer_angle):
        cos_a = np.cos(-outer_angle)
        sin_a = np.sin(-outer_angle)
        rot_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        local_pos = (positions - outer_center) @ rot_matrix.T
        
        thetas = np.deg2rad([30, 90, 150, 210, 270, 330])
        max_proj = 0
        for theta in thetas:
            n = np.array([np.cos(theta), np.sin(theta)])
            projs = local_pos @ n
            m = np.max(np.abs(projs))
            if m > max_proj:
                max_proj = m
        
        apothem = max_proj
        side = 2 * apothem / np.sqrt(3)
        return side

    # Generate candidate initial configurations
    # 1. 11 closest to origin in hex grid
    coords = []
    for q in range(-3, 4):
        for r in range(-3, 4):
            dist = abs(q) + abs(r) + abs(q+r)
            coords.append((dist, q, r))
    coords.sort()
    
    initial_configs = []
    # Config 1: 11 closest
    initial_configs.append([axial_to_cartesian(q, r) for q, r in [c[1:] for c in coords[:N]]])
    
    # Config 2: Try a different shape, e.g. remove corners from the 19-cluster
    # 19 cluster is side 2 hexagon.
    # Remove 8 hexagons from the "corners" to make it more compact?
    # The 11 closest is already quite compact.
    
    best_side = float('inf')
    best_result = None
    
    for config in initial_configs:
        positions = np.array(config)
        outer_center = np.mean(positions, axis=0)
        outer_angle = 0.0
        current_side = get_side_length(positions, outer_center, outer_angle)
        
        if current_side < best_side:
            best_side = current_side
            best_positions = positions.copy()
            best_center = outer_center.copy()
        
        # Local Search
        step_size = 0.2
        while time.time() < deadline:
            improved = False
            for i in range(N):
                for dx in [-step_size, step_size]:
                    for dy in [-step_size, step_size]:
                        new_pos = positions.copy()
                        new_pos[i] += np.array([dx, dy])
                        new_center = np.mean(new_pos, axis=0)
                        new_side = get_side_length(new_pos, new_center, outer_angle)
                        
                        if new_side < current_side:
                            positions = new_pos
                            outer_center = new_center
                            current_side = new_side
                            improved = True
                            
                            if current_side < best_side:
                                best_side = current_side
                                best_positions = positions.copy()
                                best_center = outer_center.copy()
            
            if not improved:
                step_size *= 0.7
                if step_size < 0.01:
                    break

    inner = []
    for i in range(N):
        inner.append([best_positions[i][0], best_positions[i][1], 0.0])
    
    return inner, (best_center[0], best_center[1]), best_side, 0.0

# EVOLVE-BLOCK-END
