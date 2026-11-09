"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin (evaluator limit 150s, keep 10s buffer)
    
    SQRT3 = math.sqrt(3)
    
    # Helper: Get vertices of a unit hexagon centered at (cx, cy) with rotation angle
    def get_hex_vertices(cx, cy, angle_deg):
        verts = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            x = cx + math.cos(theta)
            y = cy + math.sin(theta)
            verts.append((x, y))
        return verts

    # Helper: Separating Axis Theorem for convex polygons
    # Returns (min_penetration_depth, contact_normal, contact_axis)
    # If disjoint, depth is 0.
    def sat_check(poly1, poly2):
        # Get axes (normals to edges) from both polygons
        axes = []
        for poly in (poly1, poly2):
            n = len(poly)
            for i in range(n):
                p1 = poly[i]
                p2 = poly[(i + 1) % n]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                norm = math.sqrt(dx*dx + dy*dy)
                if norm == 0:
                    continue
                nx = -dy / norm
                ny = dx / norm
                axes.append((nx, ny))
        
        min_pen = float('inf')
        best_axis = None
        
        for axis in axes:
            # Project poly1
            min1 = float('inf')
            max1 = -float('inf')
            for p in poly1:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min1: min1 = proj
                if proj > max1: max1 = proj
                
            # Project poly2
            min2 = float('inf')
            max2 = -float('inf')
            for p in poly2:
                proj = p[0]*axis[0] + p[1]*axis[1]
                if proj < min2: min2 = proj
                if proj > max2: max2 = proj
                
            # Check for separation
            if max1 < min2 or max2 < min1:
                return 0.0, None, None # Disjoint
                
            # Calculate overlap on this axis
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
                best_axis = axis
        
        # Determine direction: push poly1 away from poly2
        # Vector from centroid of poly2 to centroid of poly1
        c1x = sum(p[0] for p in poly1) / 6
        c1y = sum(p[1] for p in poly1) / 6
        c2x = sum(p[0] for p in poly2) / 6
        c2y = sum(p[1] for p in poly2) / 6
        
        vec_x = c1x - c2x
        vec_y = c1y - c2y
        
        # Project vector onto axis to determine sign
        proj_vec = vec_x * best_axis[0] + vec_y * best_axis[1]
        if proj_vec < 0:
            best_axis = (-best_axis[0], -best_axis[1])
            
        return min_pen, best_axis, best_axis

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # Outer hexagon is centered at (0,0) with rotation angle outer_angle_deg
    def calc_outer_side(hexagons, outer_angle_deg):
        # The condition for a point (x,y) to be inside a regular hexagon of side S
        # centered at origin with "flat-top" orientation (vertices at 0, 60, ...) rotated by alpha:
        # The hexagon is the intersection of 6 half-planes.
        # The distance from origin to each side (apothem) is A = S * SQRT3 / 2.
        # The normal vectors to the sides are at angles 30, 90, 150, 210, 270, 330 + alpha.
        
        base_angles = [30, 90, 150, 210, 270, 330]
        max_proj = 0.0
        
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = vx * nx + vy * ny
                    if proj > max_proj:
                        max_proj = proj
        
        # S = 2 * A / SQRT3
        return 2.0 * max_proj / SQRT3

    # Generate initial compact lattice configuration
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    
    def axial_to_cartesian(q, r, spacing=SQRT3 + 0.2):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial positions
    init_positions = []
    for q, r in axial_coords:
        x, y = axial_to_cartesian(q, r)
        init_positions.append([x, y, 0.0])
        
    # Center the initial configuration
    cx_avg = sum(p[0] for p in init_positions) / N
    cy_avg = sum(p[1] for p in init_positions) / N
    init_positions = [[p[0] - cx_avg, p[1] - cy_avg, p[2]] for p in init_positions]

    best_score = float('inf')
    best_config = None
    
    # Deterministic Iterative Refinement
    # 1. Resolve collisions using SAT normals
    # 2. Scale to fit outer hexagon
    # 3. Repeat with decreasing step size
    
    current_config = [p[:] for p in init_positions]
    
    # We will try a few random restarts of the deterministic solver to escape local minima
    for restart in range(20):
        if time.time() > deadline - 5.0:
            break
            
        # Perturb initial config slightly for restarts > 0
        config = [p[:] for p in init_positions]
        if restart > 0:
            for i in range(N):
                config[i][0] += random.gauss(0, 0.5)
                config[i][1] += random.gauss(0, 0.5)
                config[i][2] += random.gauss(0, 10)
            # Recenter
            cx_avg = sum(p[0] for p in config) / N
            cy_avg = sum(p[1] for p in config) / N
            for i in range(N):
                config[i][0] -= cx_avg
                config[i][1] -= cy_avg

        # Iterative solver
        # Step size for position updates
        step_size = 1.0
        
        for iteration in range(500):
            if time.time() > deadline - 5.0:
                break
                
            verts_list = [get_hex_vertices(config[i][0], config[i][1], config[i][2]) for i in range(N)]
            
            # 1. Collision Resolution
            # Accumulate forces for each hexagon
            forces = [[0.0, 0.0] for _ in range(N)]
            
            has_collision = False
            for i in range(N):
                for j in range(i+1, N):
                    pen, normal, _ = sat_check(verts_list[i], verts_list[j])
                    if pen > 1e-9:
                        has_collision = True
                        # Push apart along normal
                        # Force magnitude proportional to penetration
                        f = pen * step_size
                        forces[i][0] += normal[0] * f
                        forces[i][1] += normal[1] * f
                        forces[j][0] -= normal[0] * f
                        forces[j][1] -= normal[1] * f
            
            # Apply forces
            if has_collision:
                for i in range(N):
                    config[i][0] += forces[i][0]
                    config[i][1] += forces[i][1]
                
                # Recenter to keep origin fixed (removes global translation DOF)
                cx_avg = sum(p[0] for p in config) / N
                cy_avg = sum(p[1] for p in config) / N
                for i in range(N):
                    config[i][0] -= cx_avg
                    config[i][1] -= cy_avg
                    
            # 2. Scaling to minimize outer side length
            # Calculate current side length for a few candidate angles
            # To be fast, we check angle 0 and angle 30 (or just optimize angle later)
            # For now, fix outer angle to 0 during iteration, optimize at the end.
            # Actually, the "tightest" fit depends on the cluster shape.
            # Let's find the min side length over a coarse set of angles for the current config
            # to determine the optimal scale factor? No, scaling is uniform.
            # The side length S is linear with respect to the max projection.
            # S = 2 * max_proj / SQRT3.
            # If we scale the cluster by factor 'k', max_proj scales by 'k', so S scales by 'k'.
            # We want to find the largest 'k' such that the scaled cluster fits in a unit outer hexagon?
            # No, we want to minimize S. So we want to shrink the cluster as much as possible?
            # Wait, the outer hexagon size is the score. We can make it as small as possible
            # as long as it contains the inner hexagons.
            # So, for a fixed configuration of inner hexagons (relative positions),
            # the minimal outer side length is determined by the "width" of the cluster.
            # Scaling the cluster down reduces the required outer side length.
            # BUT, the inner hexagons have a FIXED size (side 1).
            # So we cannot scale the inner hexagons.
            # We can only translate and rotate them.
            # Therefore, the "Scaling" step in the previous thought was incorrect.
            # The objective is purely the side length of the bounding hexagon.
            # We need to move hexagons to reduce the bounding box.
            
            # So, how do we reduce the bounding box?
            # We need to pull hexagons that are far from the center towards the center,
            # provided they don't collide.
            
            # Let's add a "centering force" or "shrinkage force"
            # Force towards origin, scaled by distance?
            # Or simply: if no collisions, try to move each hexagon towards the origin by a small amount.
            
            # Better approach: Gradient descent on the objective function.
            # Objective: S(outer_angle)
            # This is non-smooth (max of linear functions).
            # We can approximate the gradient by moving each hexagon slightly and checking delta S.
            # But that's expensive.
            
            # Alternative: "Compressive" local search.
            # If no collisions, try to shrink the cluster.
            # How? Move each hexagon towards the centroid of the cluster?
            # Or towards the origin?
            # Since we recentered, the centroid is at the origin.
            # Moving towards the origin reduces the max projection if the hexagon is on the "outside".
            
            # Let's try a simple heuristic:
            # If no collisions, move each hexagon towards the origin by a small step.
            # This effectively "shrink-wraps" the cluster.
            
            if not has_collision:
                # Move towards origin
                shrink_step = 0.05 * step_size
                for i in range(N):
                    x, y, a = config[i]
                    dist = math.sqrt(x*x + y*y)
                    if dist > 1e-6:
                        # Normalize vector to origin
                        nx = -x / dist
                        ny = -y / dist
                        config[i][0] += nx * shrink_step
                        config[i][1] += ny * shrink_step
                        
                # Recenter
                cx_avg = sum(p[0] for p in config) / N
                cy_avg = sum(p[1] for p in config) / N
                for i in range(N):
                    config[i][0] -= cx_avg
                    config[i][1] -= cy_avg

            # Decrease step size (annealing-like but deterministic)
            step_size *= 0.95
            
            # Evaluate current best
            # Calculate side length for best angle
            verts_list_final = [get_hex_vertices(config[i][0], config[i][1], config[i][2]) for i in range(N)]
            best_side_local = float('inf')
            best_oa_local = 0.0
            for oa in range(0, 60, 10):
                s = calc_outer_side(verts_list_final, oa)
                if s < best_side_local:
                    best_side_local = s
                    best_oa_local = oa
                    
            if best_side_local < best_score:
                best_score = best_side_local
                best_config = [p[:] for p in config]
                best_oa = best_oa_local

    # Final Refinement: Fine-tune the best config found
    if best_config is not None:
        final_config = [p[:] for p in best_config]
        
        # 1. Find best outer angle
        verts_list = [get_hex_vertices(final_config[i][0], final_config[i][1], final_config[i][2]) for i in range(N)]
        best_side = float('inf')
        best_oa = 0.0
        for oa in range(0, 60, 1):
            s = calc_outer_side(verts_list, oa)
            if s < best_side:
                best_side = s
                best_oa = oa
                
        # 2. Try to squeeze further with small random perturbations (Local Search)
        # If we have time
        if time.time() < deadline - 5.0:
            current_best = best_side
            for _ in range(200):
                if time.time() > deadline - 5.0:
                    break
                # Perturb one hexagon
                idx = random.randint(0, N-1)
                orig = [p[:] for p in final_config]
                final_config[idx][0] += random.gauss(0, 0.1)
                final_config[idx][1] += random.gauss(0, 0.1)
                final_config[idx][2] += random.gauss(0, 5)
                
                # Check validity
                verts_list_new = [get_hex_vertices(final_config[i][0], final_config[i][1], final_config[i][2]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        pen, _, _ = sat_check(verts_list_new[i], verts_list_new[j])
                        if pen > 1e-9:
                            valid = False
                            break
                    if not valid: break
                
                if valid:
                    # Check side length
                    s_new = float('inf')
                    for oa in range(0, 60, 5):
                        s = calc_outer_side(verts_list_new, oa)
                        if s < s_new:
                            s_new = s
                            
                    if s_new < current_best:
                        current_best = s_new
                        best_side = s_new
                        # Keep this config
                        # Note: we need to update best_oa if we found a better angle
                        # But we only checked coarse angles here. We'll do fine search at the end.
                    else:
                        # Revert
                        final_config = orig
                else:
                    # Revert
                    final_config = orig

        # Final fine angle search
        verts_list = [get_hex_vertices(final_config[i][0], final_config[i][1], final_config[i][2]) for i in range(N)]
        best_side = float('inf')
        best_oa = 0.0
        for oa in np.arange(0, 60, 0.5):
            s = calc_outer_side(verts_list, oa)
            if s < best_side:
                best_side = s
                best_oa = oa

        inner_list = []
        for i in range(N):
            inner_list.append([final_config[i][0], final_config[i][1], final_config[i][2]])
            
        return (inner_list, (0.0, 0.0), best_side, best_oa)
        
    # Fallback
    verts_list = [get_hex_vertices(p[0], p[1], p[2]) for p in init_positions]
    side = calc_outer_side(verts_list, 0.0)
    inner_list = [[p[0], p[1], p[2]] for p in init_positions]
    return (inner_list, (0.0, 0.0), side, 0.0)

# EVOLVE-BLOCK-END
