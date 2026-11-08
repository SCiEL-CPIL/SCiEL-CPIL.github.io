import time
import numpy as np
from scipy.optimize import minimize

N = 12
DEADLINE = 145.0  # seconds, safety margin before 150s kill

def _hex_verts(cx, cy, s, angle_deg):
    """Return vertices of a regular hexagon centered at (cx,cy), side s, rotated angle_deg."""
    a = np.radians(angle_deg)
    angles = a + np.arange(6) * np.pi / 3.0
    xs = cx + s * np.cos(angles)
    ys = cy + s * np.sin(angles)
    return np.column_stack([xs, ys])

def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    v1, v2 are (6, 2) arrays of vertices.
    """
    def get_normals(verts):
        normals = []
        n = len(verts)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i+1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            # Ensure normal points outward
            mid = (p1 + p2) / 2
            center = np.mean(verts, axis=0)
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)
    
    n1 = get_normals(v1)
    n2 = get_normals(v2)
    all_normals = np.vstack([n1, n2])
    
    min_overlap = float('inf')
    
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)
    
    if min_overlap == float('inf'):
        return 0.0
    
    return min_overlap

def _min_enclosing_hexagon_side(positions, angles, outer_angle_deg):
    """
    Compute the minimum side length of a regular hexagon with the given outer angle
    that contains all the inner hexagons (with unit side length).
    
    positions: list of (x, y) centers
    angles: list of rotation angles in degrees
    outer_angle_deg: rotation of the outer hexagon in degrees
    """
    outer_angle_rad = np.radians(outer_angle_deg)
    normal_angles = outer_angle_rad + (np.pi / 6.0) + np.arange(6) * np.pi / 3.0
    cos_normals = np.cos(normal_angles)
    sin_normals = np.sin(normal_angles)
    
    max_proj = 0.0
    for i in range(N):
        v = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
        projs = v[:, 0] * cos_normals + v[:, 1] * sin_normals
        col_max = np.max(projs, axis=0)
        global_max = np.max(col_max)
        if global_max > max_proj:
            max_proj = global_max
            
    # For a regular hexagon with side length s, apothem = s * sqrt(3) / 2
    # So s = 2 * apothem / sqrt(3)
    side = max_proj * 2.0 / np.sqrt(3.0)
    return side

def _objective(params):
    """
    Objective function for the optimizer.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12, outer_angle]
    Returns the minimum enclosing hexagon side length.
    """
    positions = []
    angles = []
    for i in range(N):
        x = params[i*3]
        y = params[i*3+1]
        a = params[i*3+2]
        positions.append((x, y))
        angles.append(a)
    
    outer_angle = params[N*3]
    
    # Compute minimum enclosing hexagon side
    side = _min_enclosing_hexagon_side(positions, angles, outer_angle)
    
    # Add penalty for overlaps
    penalty = 0.0
    for i in range(N):
        v1 = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
        for j in range(i+1, N):
            v2 = _hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                penalty += depth * depth * 100.0  # Heavy penalty for overlaps
    
    return side + penalty

def _try_pack(time_budget=5.0):
    """
    Try to find a valid packing.
    Returns (success, best_params, best_side).
    """
    start_time = time.time()
    best_params = None
    best_side = float('inf')
    
    def get_initial_config():
        # Random scatter
        config = []
        for i in range(N):
            x = np.random.uniform(-3, 3)
            y = np.random.uniform(-3, 3)
            a = np.random.uniform(0, 60)
            config.extend([x, y, a])
        config.append(np.random.uniform(0, 60)) # Outer angle
        return np.array(config)
        
    def get_grid_config(rows, cols, spacing_x, spacing_y):
        config = []
        count = 0
        for r in range(rows):
            for c in range(cols):
                if count >= N:
                    break
                x = c * spacing_x + (r % 2) * spacing_x / 2
                y = r * spacing_y
                x -= (cols-1) * spacing_x / 2
                y -= (rows-1) * spacing_y / 2
                config.extend([x, y, 0.0])
                count += 1
        while len(config) < N*3:
             config.extend([np.random.uniform(-3,3), np.random.uniform(-3,3), 0.0])
        config.append(0.0) # Outer angle
        return np.array(config[:N*3+1])

    # Try multiple initializations
    initial_configs = [
        get_grid_config(3, 4, 2.1, 1.8),
        get_grid_config(4, 3, 2.1, 1.8),
        get_grid_config(2, 6, 2.1, 1.8),
        get_initial_config(),
        get_initial_config(),
        get_initial_config()
    ]
    
    for init in initial_configs:
        if time.time() - start_time > time_budget:
            break
            
        try:
            res = minimize(
                _objective, 
                init, 
                method='Nelder-Mead',
                options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-6}
            )
            
            # Check if the result is valid (no overlaps)
            positions = []
            angles = []
            for i in range(N):
                x = res.x[i*3]
                y = res.x[i*3+1]
                a = res.x[i*3+2]
                positions.append((x, y))
                angles.append(a)
            
            outer_angle = res.x[N*3]
            
            # Check for overlaps
            has_overlap = False
            for i in range(N):
                v1 = _hex_verts(positions[i][0], positions[i][1], 1.0, angles[i])
                for j in range(i+1, N):
                    v2 = _hex_verts(positions[j][0], positions[j][1], 1.0, angles[j])
                    depth = _sat_overlap_depth(v1, v2)
                    if depth > 1e-6:
                        has_overlap = True
                        break
                if has_overlap:
                    break
            
            if not has_overlap:
                side = _min_enclosing_hexagon_side(positions, angles, outer_angle)
                if side < best_side:
                    best_side = side
                    best_params = res.x.copy()
                    
        except Exception:
            continue
            
    return best_params is not None, best_params, best_side

def construct():
    start_time = time.time()
    
    best_params = None
    best_side = float('inf')
    
    # Try packing with different time budgets
    for time_budget in [10.0, 15.0, 20.0]:
        if time.time() - start_time > DEADLINE - 15:
            break
            
        success, params, side = _try_pack(time_budget=time_budget)
        
        if success and side < best_side:
            best_side = side
            best_params = params
            
            # If we find a good solution, we can stop early
            if best_side < 4.5:
                break
    
    # If no valid packing found, return a fallback
    if best_params is None:
        # Fallback: return a large hexagon with a valid grid
        inner = []
        for r in range(3):
            for c in range(4):
                x = c * 2.1 + (r % 2) * 1.05
                y = r * 1.8
                x -= 3 * 2.1 / 2
                y -= 2 * 1.8 / 2
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0
    
    # Extract final result from best_params
    inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        inner.append([x, y, a])
        
    outer_angle = best_params[N*3]
    
    # Recompute the side length for safety
    positions = [(inner[i][0], inner[i][1]) for i in range(N)]
    angles = [inner[i][2] for i in range(N)]
    final_side = _min_enclosing_hexagon_side(positions, angles, outer_angle)
    
    return inner, (0.0, 0.0), final_side, outer_angle

if __name__ == "__main__":
    inner, center, side, angle = construct()
    print(f"Side: {side}, Angle: {angle}")
