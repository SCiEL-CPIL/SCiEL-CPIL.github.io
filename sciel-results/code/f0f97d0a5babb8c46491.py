"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # 10s safety margin
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        
        for na in normal_angles:
            nx = np.cos(na)
            ny = np.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        """
        axes = []
        for i in range(len(v1)):
            p1 = v1[i]
            p2 = v1[(i + 1) % len(v1)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for i in range(len(v2)):
            p1 = v2[i]
            p2 = v2[(i + 1) % len(v2)]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            if np.linalg.norm(normal) > 1e-9:
                axes.append(normal / np.linalg.norm(normal))
        
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            # Check for gap
            if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                return True
        return False

    def hexagons_disjoint(v1, s1, a1, v2, s2, a2):
        verts1 = hexagon_vertices(v1[0], v1[1], s1, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], s2, a2)
        return polygons_disjoint_sat(verts1, verts2)

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        for i in range(len(inner_positions)):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        for i in range(len(inner_positions)):
            for j in range(i+1, len(inner_positions)):
                ci = (inner_positions[i][0], inner_positions[i][1])
                cj = (inner_positions[j][0], inner_positions[j][1])
                ai = inner_positions[i][2]
                aj = inner_positions[j][2]
                if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                    return False
        return True

    def objective(params):
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            x = params[3*i]
            y = params[3*i+1]
            a = params[3*i+2]
            inner_positions.append((x, y, a))
        
        ox = params[3*n_inner]
        oy = params[3*n_inner+1]
        os = params[3*n_inner+2]
        oa = params[3*n_inner+3]
        
        outer_center = (ox, oy)
        outer_side = max(os, 0.1)
        outer_angle = oa
        
        inside_ok = all_inside_outer(inner_positions, outer_center, outer_side, outer_angle)
        disjoint_ok = pairwise_disjoint(inner_positions)
        
        if inside_ok and disjoint_ok:
            return outer_side
        
        penalty = 0.0
        if not inside_ok:
            # Calculate how much is outside
            for i in range(n_inner):
                cx, cy, angle = inner_positions[i]
                verts = hexagon_vertices(cx, cy, 1.0, angle)
                for vx, vy in verts:
                    if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                        penalty += 1.0
        if not disjoint_ok:
            for i in range(n_inner):
                for j in range(i+1, n_inner):
                    ci = (inner_positions[i][0], inner_positions[i][1])
                    cj = (inner_positions[j][0], inner_positions[j][1])
                    ai = inner_positions[i][2]
                    aj = inner_positions[j][2]
                    if not hexagons_disjoint(ci, 1.0, ai, cj, 1.0, aj):
                        penalty += 1.0
                        
        return outer_side + penalty * 10.0

    # Initial valid configuration: Grid
    # Spacing x: 2.1 (safe for width 2.0)
    # Spacing y: 1.8 (safe for height 1.732)
    # 4 cols, 3 rows
    initial_positions = []
    x_coords = [-3.0, -1.0, 1.0, 3.0]
    y_coords = [-1.8, 0.0, 1.8]
    
    for y in y_coords:
        for x in x_coords:
            initial_positions.append([x, y, 0.0])
            
    # Outer hexagon
    # Bounding box approx: x [-4.1, 4.1], y [-3.06, 3.06]
    # Try outer side 5.0, angle 30 deg (flat top)
    outer_side_init = 5.0
    outer_center_init = [0.0, 0.0]
    outer_angle_init = 30.0
    
    params = []
    for pos in initial_positions:
        params.extend([pos[0], pos[1], pos[2]])
    params.extend([outer_center_init[0], outer_center_init[1], outer_side_init, outer_angle_init])
    
    # Verify initial is valid
    inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
    if not (all_inside_outer(inner_init, outer_center_init, outer_side_init, outer_angle_init) and pairwise_disjoint(inner_init)):
        # If invalid, expand outer
        outer_side_init = 6.0
        params[3*N+2] = outer_side_init
        inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
        if not (all_inside_outer(inner_init, outer_center_init, outer_side_init, outer_angle_init) and pairwise_disjoint(inner_init)):
             # Last resort: just return this, evaluator will penalize, but we want to try optimize
             pass

    best_params = params.copy()
    best_score = objective(params)
    
    # Optimize
    if time.time() < deadline - 10:
        try:
            # Use Nelder-Mead with adaptive
            res = minimize(objective, params, method='Nelder-Mead', options={'maxiter': 500, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    # Second round if time permits
    if time.time() < deadline - 5:
        try:
            res2 = minimize(objective, best_params, method='Nelder-Mead', options={'maxiter': 200, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass

    final_params = best_params
    
    inner = []
    for i in range(N):
        x = final_params[3*i]
        y = final_params[3*i+1]
        a = final_params[3*i+2]
        inner.append([x, y, a])
    
    ox = final_params[3*N]
    oy = final_params[3*N+1]
    os = max(final_params[3*N+2], 0.1)
    oa = final_params[3*N+3]
    
    # Final validation check
    if not (all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner)):
        # Fallback to initial valid guess
        inner = [[p[0], p[1], p[2]] for p in initial_positions]
        ox, oy = outer_center_init
        os = outer_side_init
        oa = outer_angle_init
        # Ensure this fallback is actually valid, if not, expand os
        if not (all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner)):
             # Brute force expand
             while not all_inside_outer(inner, (ox, oy), os, oa):
                 os += 0.1
                 if os > 10: break

    return inner, (ox, oy), os, oa
# EVOLVE-BLOCK-END
