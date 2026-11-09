import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 115.0  # Safety margin (evaluator limit 120s)
    
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

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(angle_deg):
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in vertices:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2, d_sq):
        """
        Fast disjoint check.
        """
        if d_sq > 4.0:
            return True
        if d_sq < 3.0:
            return False
            
        # SAT check
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for nx, ny in axis_dirs:
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    def evaluate_hard(p, outer_angle_deg=0.0):
        """
        Hard evaluation: returns penalty + side length.
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                    overlap_penalty += 1000.0
        
        side = calc_outer_side(hexagons, outer_angle_deg)
        return overlap_penalty + side

    def evaluate_smooth(p, outer_angle_deg=0.0, alpha=0.1):
        """
        Smooth evaluation for Nelder-Mead: uses log-sum-exp for outer side
        and smooth penalty for overlaps.
        """
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        cx_avg = sum(c[0] for c in centers) / N
        cy_avg = sum(c[1] for c in centers) / N
        shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
        
        hexagons = []
        hex_axes = []
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
            hex_axes.append(get_hex_axes(a))
        
        # Smooth overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                dx = shifted_centers[i][0] - shifted_centers[j][0]
                dy = shifted_centers[i][1] - shifted_centers[j][1]
                d_sq = dx*dx + dy*dy
                if d_sq < 3.0:
                    # Smooth penalty: grows as distance decreases
                    # Use a soft plus or similar
                    # d_min for disjoint is sqrt(3) ~ 1.732
                    # d_sq_min = 3.0
                    # Penalty ~ (3.0 - d_sq)^2 if d_sq < 3.0
                    diff = 3.0 - d_sq
                    if diff > 0:
                        overlap_penalty += diff * diff * 100.0
        
        # Smooth outer side
        # We want to minimize max projection.
        # Let P_k be the projection of vertex k onto axis a.
        # We want min S such that S * SQRT3 / 2 >= max |P_k|
        # Equivalent to minimizing max |P_k|.
        # Use log-sum-exp: log(sum(exp(|P_k| / alpha))) * alpha
        # But we have 6 axes. We need max over all axes and vertices.
        
        base_angles = [30, 90, 150, 210, 270, 330]
        axis_dirs = []
        for a in base_angles:
            rad = math.radians(a + outer_angle_deg)
            axis_dirs.append((math.cos(rad), math.sin(rad)))
        
        # Collect all absolute projections
        max_proj_smooth = 0.0
        for nx, ny in axis_dirs:
            projs = []
            for hex_verts in hexagons:
                for vx, vy in hex_verts:
                    proj = abs(vx * nx + vy * ny)
                    projs.append(proj)
            
            if projs:
                max_p = max(projs)
                # Log-sum-exp
                # To avoid overflow, subtract max
                if max_p > 0:
                    lse = max_p + alpha * math.log(sum(math.exp((p - max_p) / alpha) for p in projs))
                else:
                    lse = 0.0
                if lse > max_proj_smooth:
                    max_proj_smooth = lse
        
        side = 2.0 * max_proj_smooth / SQRT3
        return overlap_penalty + side

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    coords4 = [(-1,-1),(0,-1),(1,-1),(-1,0),(0,0),(1,0),(-1,1),(0,1),(1,1),(2,0),(2,1)]
    coords5 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,1), (-2,1), (-2,0)]
    
    configs = [coords1, coords2, coords3, coords4, coords5]
    
    best_params = None
    best_score = float('inf')

    def soft_body_relax(params, steps=500):
        """
        Physics-based relaxation.
        Forces:
        1. Repulsion between overlapping hexagons (or close ones).
        2. Attraction to center (to keep them compact).
        3. Damping.
        """
        pos = np.array(params, dtype=float)
        # pos: [x1, y1, a1, x2, y2, a2, ...]
        
        # We will move only x, y for simplicity in physics, then polish angles with NM
        # Actually, let's allow angle jitter too, but mainly x,y.
        
        vel = np.zeros_like(pos)
        
        for step in range(steps):
            if time.time() > deadline - 10.0:
                break
            
            forces = np.zeros_like(pos)
            
            # Extract centers and angles
            centers = [(pos[i*3], pos[i*3+1]) for i in range(N)]
            angles = [pos[i*3+2] for i in range(N)]
            
            # Calculate pairwise forces
            for i in range(N):
                for j in range(i+1, N):
                    dx = centers[i][0] - centers[j][0]
                    dy = centers[i][1] - centers[j][1]
                    d_sq = dx*dx + dy*dy
                    d = math.sqrt(d_sq)
                    
                    # Min distance for disjoint is sqrt(3) ~ 1.732
                    # We want to push them apart if d < 1.8
                    min_d = 1.8
                    if d < min_d and d > 1e-6:
                        # Repulsive force
                        # F ~ k * (min_d - d)
                        k = 10.0
                        f_mag = k * (min_d - d)
                        fx = (dx / d) * f_mag
                        fy = (dy / d) * f_mag
                        
                        forces[i*3] += fx
                        forces[i*3+1] += fy
                        forces[j*3] -= fx
                        forces[j*3+1] -= fy
                        
                        # Torque? Maybe not necessary for initial relaxation
                        # But slight angle adjustment might help
                        # Skip for now to keep it simple and stable
            
            # Attraction to center (0,0)
            # This helps keep the cluster compact
            k_attr = 0.1
            for i in range(N):
                cx, cy = centers[i]
                forces[i*3] -= cx * k_attr
                forces[i*3+1] -= cy * k_attr
            
            # Angle damping (try to align to 0 or 60? No, let them float)
            # Just dampen angle velocity
            for i in range(N):
                forces[i*3+2] = 0.0 # No force on angle in this simple model
            
            # Update velocities and positions
            dt = 0.05
            damping = 0.9
            for i in range(len(pos)):
                vel[i] = (vel[i] + forces[i] * dt) * damping
                pos[i] += vel[i] * dt
            
            # Optional: Random perturbation to escape local minima
            if step % 50 == 0:
                for i in range(N):
                    pos[i*3] += random.gauss(0, 0.1)
                    pos[i*3+1] += random.gauss(0, 0.1)
                    pos[i*3+2] += random.gauss(0, 5.0)
        
        return pos

    # Run soft-body relaxation from each initial configuration
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        
        # Relax
        relaxed_params = soft_body_relax(params, steps=300)
        
        # Polish with Nelder-Mead using smooth objective
        # We need to optimize x, y, angle for all N hexagons
        # Total 3N variables.
        
        def objective_nm(p):
            return evaluate_smooth(p, 0.0, alpha=0.1)
        
        try:
            result = minimize(objective_nm, relaxed_params, method='Nelder-Mead', 
                               options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-5, 'adaptive': True})
            polished_params = result.x
            smooth_score = result.fun
            
            # Check hard score
            hard_score = evaluate_hard(polished_params, 0.0)
            
            if hard_score < best_score:
                best_score = hard_score
                best_params = polished_params.copy()
                
        except Exception as e:
            # If NM fails, use relaxed params
            hard_score = evaluate_hard(relaxed_params, 0.0)
            if hard_score < best_score:
                best_score = hard_score
                best_params = relaxed_params.copy()

    # If no improvement found, fallback to best initial config
    if best_params is None:
        config = configs[0]
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)
        best_score = evaluate_hard(best_params, 0.0)

    # Final Validation and Outer Angle Optimization
    final_params = best_params.copy()
    
    centers = []
    angles = []
    for i in range(N):
        centers.append((final_params[i*3], final_params[i*3+1]))
        angles.append(final_params[i*3+2])
        
    cx_avg = sum(c[0] for c in centers) / N
    cy_avg = sum(c[1] for c in centers) / N
    shifted_centers = [(x - cx_avg, y - cy_avg) for x, y in centers]
    
    hexagons = []
    hex_axes = []
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        hex_axes.append(get_hex_axes(a))
        
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            dx = shifted_centers[i][0] - shifted_centers[j][0]
            dy = shifted_centers[i][1] - shifted_centers[j][1]
            d_sq = dx*dx + dy*dy
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j], d_sq):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # Fallback
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg_fb = sum(p[0] for p in inner_list) / N
        cy_avg_fb = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg_fb, y - cy_avg_fb, a] for x, y, a in inner_list]
        hexagons_fb = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons_fb, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Optimize outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
