import numpy as np
import time
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin

    def hexagon_vertices(cx, cy, side, angle_rad):
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def max_overlap_depth(v1, v2):
        axes = []
        for v in [v1, v2]:
            for i in range(6):
                j = (i + 1) % 6
                edge = v[j] - v[i]
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm > 1e-12:
                    axes.append(normal / norm)
        
        max_overlap = 0.0
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            
            if max1 < min2 or max2 < min1:
                return 0.0
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap > max_overlap:
                max_overlap = overlap
        
        return max_overlap

    def calc_min_outer_side(all_verts, theta):
        angles = theta + np.radians(np.arange(0, 360, 60))
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        dots = all_verts @ normals.T
        widths = dots.max(axis=0) - dots.min(axis=0)
        max_width = np.max(widths)
        apothem = max_width / 2.0
        side = 2 * apothem / np.sqrt(3)
        return side

    def find_best_outer_orientation(all_verts):
        from scipy.optimize import minimize_scalar
        res = minimize_scalar(lambda th: calc_min_outer_side(all_verts, th), 
                              bounds=(0, np.pi/3), method='bounded', 
                              options={'xatol': 1e-4})
        return res.fun, res.x

    def objective(params):
        if time.time() > deadline:
            return 1e10
            
        n = 12
        positions = np.array([[params[i*3], params[i*3+1]] for i in range(n)])
        angles = np.array([params[i*3+2] for i in range(n)])
        
        verts_list = []
        for i in range(n):
            v = hexagon_vertices(positions[i, 0], positions[i, 1], 1.0, angles[i])
            verts_list.append(v)
            
        penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                depth = max_overlap_depth(verts_list[i], verts_list[j])
                if depth > 0:
                    penalty += depth * depth
        
        if penalty > 1e-6:
            return 1000.0 + penalty
        
        all_verts = np.vstack(verts_list)
        side, _ = find_best_outer_orientation(all_verts)
        return side

    # Generate initial honeycomb configuration
    d_x = np.sqrt(3) # Horizontal spacing
    d_y = 1.5        # Vertical spacing
    
    positions = []
    angles = []
    
    # 3 rows of 4
    for i in range(3):
        for j in range(4):
            x = d_x * (j + 0.5 * (i % 2))
            y = d_y * i
            positions.append([x, y])
            angles.append(0.0)
            
    positions = np.array(positions)
    angles = np.array(angles)
    
    # Center the configuration
    pos_center = positions.mean(axis=0)
    positions = positions - pos_center
    
    best_params = np.concatenate([positions.flatten(), angles])
    best_obj = objective(best_params)
    
    # Optimize
    bounds = [(-5.0, 5.0), (-5.0, 5.0), (0.0, np.pi)] * 12
    
    try:
        result = minimize(
            objective,
            best_params,
            method='Nelder-Mead',
            bounds=bounds,
            options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 2000, 'maxfev': 2000}
        )
        if result.fun < best_obj:
            best_obj = result.fun
            best_params = result.x
    except Exception as e:
        pass
    
    # If time allows, try a second start with rotated initial config
    if time.time() < deadline - 30.0:
        theta_rot = np.pi / 6
        rot_mat = np.array([[np.cos(theta_rot), -np.sin(theta_rot)], [np.sin(theta_rot), np.cos(theta_rot)]])
        pos2 = positions @ rot_mat.T
        angles2 = angles + theta_rot
        params2 = np.concatenate([pos2.flatten(), angles2])
        
        try:
            result2 = minimize(
                objective,
                params2,
                method='Nelder-Mead',
                bounds=bounds,
                options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 1000, 'maxfev': 1000}
            )
            if result2.fun < best_obj:
                best_obj = result2.fun
                best_params = result2.x
        except Exception as e:
            pass

    # Extract best solution
    if best_params is not None and len(best_params) == 36:
        best_positions = np.array([[best_params[i*3], best_params[i*3+1]] for i in range(12)])
        best_angles = np.array([best_params[i*3+2] for i in range(12)])
        
        valid = True
        verts_list = []
        for i in range(12):
            v = hexagon_vertices(best_positions[i, 0], best_positions[i, 1], 1.0, best_angles[i])
            verts_list.append(v)
            for j in range(i+1, 12):
                if max_overlap_depth(verts_list[i], verts_list[j]) > 1e-6:
                    valid = False
                    break
            if not valid:
                break
        
        if valid:
            all_verts = np.vstack(verts_list)
            outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
            
            angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            mins = dots.min(axis=0)
            maxs = dots.max(axis=0)
            mids = (mins + maxs) / 2.0
            c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
            outer_center = tuple(c)
            outer_angle_deg = np.degrees(outer_angle_rad)
        else:
            # Fallback
            all_verts = np.vstack([hexagon_vertices(p[0], p[1], 1.0, a) for p, a in zip(positions, angles)])
            outer_side, outer_angle_rad = find_best_outer_orientation(all_verts)
            angles = outer_angle_rad + np.radians(np.arange(0, 360, 60))
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            dots = all_verts @ normals.T
            mins = dots.min(axis=0)
            maxs = dots.max(axis=0)
            mids = (mins + maxs) / 2.0
            c, _, _, _ = np.linalg.lstsq(normals.T, mids, rcond=None)
            outer_center = tuple(c)
            outer_angle_deg = np.degrees(outer_angle_rad)
            best_positions = positions
            best_angles = angles
    else:
        outer_center = (0.0, 0.0)
        outer_side = 10.0
        outer_angle_deg = 0.0
        best_positions = positions
        best_angles = angles

    inner = [[p[0], p[1], np.degrees(a)] for p, a in zip(best_positions, best_angles)]
    return inner, outer_center, outer_side, outer_angle_deg

# EVOLVE-BLOCK-END
