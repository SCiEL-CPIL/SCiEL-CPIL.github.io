"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import numpy as np
from scipy.optimize import minimize, differential_evolution
import time

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # 5s safety margin before 150s kill
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        angle_rad = np.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """
        Vectorized check if a single point (px, py) is inside a hexagon.
        Returns True if inside, False otherwise.
        """
        dx = px - cx
        dy = py - cy
        angle_rad = np.radians(angle_deg)
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        
        apothem = side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        dists = local_x * nx + local_y * ny
        return np.all(dists <= apothem + 1e-9)

    def polygons_disjoint_sat(v1, v2):
        """
        Check if two convex polygons are disjoint using Separating Axis Theorem.
        v1, v2: Nx2 arrays of vertices.
        Optimized for 6-vertex polygons.
        """
        # Precompute edges and normals for v1
        edges1 = np.roll(v1, -1, axis=0) - v1
        norms1 = np.column_stack([-edges1[:, 1], edges1[:, 0]])
        norms1 /= (np.linalg.norm(norms1, axis=1, keepdims=True) + 1e-9)
        
        # Precompute edges and normals for v2
        edges2 = np.roll(v2, -1, axis=0) - v2
        norms2 = np.column_stack([-edges2[:, 1], edges2[:, 0]])
        norms2 /= (np.linalg.norm(norms2, axis=1, keepdims=True) + 1e-9)
        
        # Combine all axes
        all_axes = np.vstack([norms1, norms2])
        
        # Project v1 and v2 onto all axes
        proj1 = np.dot(v1, all_axes.T)
        proj2 = np.dot(v2, all_axes.T)
        
        min1 = np.min(proj1, axis=0)
        max1 = np.max(proj1, axis=0)
        min2 = np.min(proj2, axis=0)
        max2 = np.max(proj2, axis=0)
        
        # Check for separation on any axis
        separated = (max1 < min2 - 1e-9) | (max2 < min1 - 1e-9)
        return np.any(separated)

    def hexagons_disjoint(v1, s1, a1, v2, s2, a2):
        verts1 = hexagon_vertices(v1[0], v1[1], s1, a1)
        verts2 = hexagon_vertices(v2[0], v2[1], s2, a2)
        return polygons_disjoint_sat(verts1, verts2)

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        # Vectorize inner vertices calculation? 
        # inner_positions is a list of tuples. 
        # We can batch the point-in-polygon checks if we stack the points.
        # But point_in_hexagon above is for a single point. 
        # Let's create a batched version for speed.
        
        n_inner = len(inner_positions)
        if n_inner == 0: return True
        
        # Extract centers and angles
        centers = np.array([(p[0], p[1]) for p in inner_positions])
        angles = np.array([p[2] for p in inner_positions])
        
        # Generate vertices for all inner hexagons at once?
        # Each hex has 6 verts.
        # Vertices shape: (n_inner, 6, 2)
        angle_rad = np.radians(angles)
        base_angles = np.radians(np.arange(0, 360, 60))
        
        # Rotation matrices for each hexagon
        cos_a = np.cos(-angle_rad)
        sin_a = np.sin(-angle_rad)
        
        # Local vertices of unit hexagon
        local_verts = np.column_stack([
            np.cos(base_angles),
            np.sin(base_angles)
        ])
        
        # Rotate local vertices for each hexagon
        # local_verts: (6, 2)
        # We need to rotate each hex's vertices by its angle.
        # Rotated local verts shape: (n_inner, 6, 2)
        # local_verts[:, np.newaxis, :] -> (6, 1, 2)
        # cos_a[:, np.newaxis] -> (n_inner, 1)
        
        rx = local_verts[:, 0]
        ry = local_verts[:, 1]
        
        # Rotation: x' = x cos - y sin, y' = x sin + y cos
        # For each inner hex i, angle a_i
        # x'_i = x * cos(a_i) - y * sin(a_i)
        
        rotated_x = rx * cos_a[:, np.newaxis] - ry * sin_a[:, np.newaxis]
        rotated_y = rx * sin_a[:, np.newaxis] + ry * cos_a[:, np.newaxis]
        
        # Translate to centers
        # centers: (n_inner, 2)
        verts_x = rotated_x + centers[:, 0:1]
        verts_y = rotated_y + centers[:, 1:2]
        
        # Stack to (n_inner, 6, 2)
        all_verts = np.stack([verts_x, verts_y], axis=-1)
        
        # Check if all vertices are inside outer hexagon
        # Outer center, side, angle
        ocx, ocy = outer_center
        # We can reuse the logic from point_in_hexagon but vectorized for all points at once.
        
        # Points: all_verts (n_inner, 6, 2)
        # Flatten to (n_inner*6, 2)
        flat_verts = all_verts.reshape(-1, 2)
        
        # Calculate local coordinates relative to outer center
        dx = flat_verts[:, 0] - ocx
        dy = flat_verts[:, 1] - ocy
        
        outer_angle_rad = np.radians(outer_angle)
        cos_o = np.cos(-outer_angle_rad)
        sin_o = np.sin(-outer_angle_rad)
        
        local_x = dx * cos_o - dy * sin_o
        local_y = dx * sin_o + dy * cos_o
        
        apothem = outer_side * np.sqrt(3) / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        
        # dists shape: (n_inner*6, 6)
        dists = local_x[:, np.newaxis] * nx[np.newaxis, :] + local_y[:, np.newaxis] * ny[np.newaxis, :]
        
        # Check if any point violates any constraint
        # If any dist > apothem, it's outside
        if np.any(dists > apothem + 1e-9):
            return False
            
        return True

    def pairwise_disjoint(inner_positions):
        n_inner = len(inner_positions)
        for i in range(n_inner):
            for j in range(i+1, n_inner):
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

    # Initial valid configuration: Hexagonal cluster
    # Center at (0,0).
    # Ring 0: (0,0)
    # Ring 1: 6 hexes at distance 2*sqrt(3)/2 * 2 ? No, distance between centers of touching hexes is 2*apothem = sqrt(3).
    # Wait, side length 1. Apothem = sqrt(3)/2. Distance between centers = 2 * sqrt(3)/2 = sqrt(3).
    # Let's place them in a hex grid.
    # (0,0)
    # (sqrt(3), 0), (-sqrt(3), 0)
    # (sqrt(3)/2, 1.5), (-sqrt(3)/2, 1.5), (sqrt(3)/2, -1.5), (-sqrt(3)/2, -1.5)
    # That's 7 hexes.
    # We need 12. Add 5 more from Ring 2.
    # Ring 2 distances are larger.
    # Let's just use a tight grid initially and let the optimizer fix it.
    # A 3x4 grid with spacing sqrt(3) in x and 1.5 in y?
    # Hex width = 2. Height = sqrt(3).
    # If aligned flat-top:
    # x spacing: 1.5 (for pointy top) or 2 (for flat top)?
    # Standard hex grid:
    # Flat top: width 2, height sqrt(3).
    # Horizontal distance: 1.5 * side? No.
    # Let's stick to the previous grid but tighter.
    # x spacing: 1.5 * 2 = 3? No.
    
    # Let's use the 4x3 grid but with optimal spacing for flat-top hexes.
    # Width = 2. Height = sqrt(3) ~ 1.732.
    # If we align them:
    # Row 0: y=0, x = -3, -1, 1, 3 (spacing 2, touch at vertices)
    # Row 1: y=sqrt(3), x = -2, 0, 2 (spacing 2, offset by 1)
    # Row 2: y=2*sqrt(3), x = -3, -1, 1, 3
    # Total 11. Add one more?
    # This is getting complicated. Let's just use the 4x3 grid with spacing 2.0 and 1.732.
    # It might overlap at vertices if not careful.
    # Distance between (0,0) and (1, sqrt(3)) is sqrt(1+3)=2.
    # Min distance between centers of disjoint unit hexes is 2 (touching).
    # So we can place them at distance 2.
    
    initial_positions = []
    # 4x3 grid
    for i in range(3):
        y = i * 1.732
        for j in range(4):
            x = (j - 1.5) * 2.0
            initial_positions.append([x, y, 0.0])
            
    outer_side_init = 5.0
    outer_center_init = [0.0, 1.732]
    outer_angle_init = 0.0
    
    params = []
    for pos in initial_positions:
        params.extend([pos[0], pos[1], pos[2]])
    params.extend([outer_center_init[0], outer_center_init[1], outer_side_init, outer_angle_init])
    
    # Verify initial is valid, expand outer if needed
    inner_init = [(p[0], p[1], p[2]) for p in initial_positions]
    max_attempts = 20
    for _ in range(max_attempts):
        if all_inside_outer(inner_init, outer_center_init, outer_side_init, outer_angle_init) and pairwise_disjoint(inner_init):
            break
        outer_side_init += 0.1
        params[3*N+2] = outer_side_init
        if outer_side_init > 10.0:
            break

    best_params = params.copy()
    best_score = objective(params)
    
    # Optimize with multiple rounds
    if time.time() < deadline - 15:
        try:
            # First pass: coarser optimization
            res = minimize(objective, params, method='Nelder-Mead', 
                         options={'maxiter': 1000, 'xatol': 0.01, 'fatol': 0.01, 'adaptive': True,
                                  'maxfev': 3000})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass
            
    if time.time() < deadline - 10:
        try:
            # Second pass: finer optimization
            res2 = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True,
                                   'maxfev': 1500})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x
        except Exception:
            pass
            
    if time.time() < deadline - 5:
        try:
            # Third pass: very fine
            res3 = minimize(objective, best_params, method='Nelder-Mead', 
                          options={'maxiter': 200, 'xatol': 0.001, 'fatol': 0.001, 'adaptive': True,
                                   'maxfev': 500})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x
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
        for _ in range(50):
            if all_inside_outer(inner, (ox, oy), os, oa) and pairwise_disjoint(inner):
                break
            os += 0.1
            if os > 12.0:
                break

    return inner, (ox, oy), os, oa
# EVOLVE-BLOCK-END
