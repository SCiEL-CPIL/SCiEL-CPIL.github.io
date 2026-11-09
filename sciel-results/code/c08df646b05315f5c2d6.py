"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin (evaluator limit 150s, keep 15s buffer)
    
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

    # Helper: Get the 6 unique normal axes for a hexagon
    def get_hex_axes(angle_deg):
        """Get the 6 unique normal axes for a hexagon with given rotation."""
        axes = []
        for i in range(6):
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    # Fast disjoint check using SAT
    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        """Check if two hexagons are disjoint using Separating Axis Theorem."""
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = min(vx * nx + vy * ny for vx, vy in hex1_verts)
            max1 = max(vx * nx + vy * ny for vx, vy in hex1_verts)
            min2 = min(vx * nx + vy * ny for vx, vy in hex2_verts)
            max2 = max(vx * nx + vy * ny for vx, vy in hex2_verts)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
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
        
        return 2.0 * max_proj / SQRT3

    # Axial to Cartesian conversion
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Multiple initial configurations
    # Config 1: Compact 3-5-3 arrangement
    coords1 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # Config 2: Flower + extensions
    coords2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # Config 3: Different compact shape
    coords3 = [(0,0), (1,0), (-1,0), (0,1), (1,1), (-1,1), (0,-1), (1,-1), (-1,-1), (2,0), (-2,0)]

    def make_initial_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        return np.array(params)

    # Evaluate a configuration: return (overlap_penalty, best_side)
    def evaluate_config(params, num_outer_samples=12):
        centers = []
        angles = []
        for i in range(N):
            centers.append((params[i*3], params[i*3+1]))
            angles.append(params[i*3+2])
        
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
        
        # Overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penetration = max(0, SQRT3 - dist)
                    overlap_penalty += 100.0 * penetration * penetration + 20.0
        
        # Find best outer angle
        best_side = float('inf')
        for k in range(num_outer_samples):
            oa_deg = k * 60.0 / num_outer_samples
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
        
        return overlap_penalty + best_side

    # Main optimization: Multi-start SA with Nelder-Mead refinement
    best_params = None
    best_score = float('inf')
    
    configs = [coords1, coords2, coords3]
    
    for config in configs:
        if time.time() > deadline - 30.0:
            break
        
        params = make_initial_params(config)
        
        # Simulated Annealing
        T = 5.0
        T_min = 0.01
        cooling_rate = 0.998
        current_score = evaluate_config(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        step = 0
        while T > T_min and time.time() < deadline - 25.0:
            new_params = params.copy()
            idx = random.randint(0, N-1)
            
            # Perturbation
            step_scale = 0.2 + 0.1 * (T / 5.0)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            da = random.gauss(0, 10.0)
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate_config(new_params)
            
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % 20 == 0:
                T *= cooling_rate
        
        # Nelder-Mead refinement
        if time.time() < deadline - 15.0:
            def objective_nm(p):
                return evaluate_config(p, num_outer_samples=6)
            
            try:
                res = minimize(objective_nm, local_best_params, method='Nelder-Mead',
                             options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
                if res.fun < local_best_score:
                    local_best_params = res.x
                    local_best_score = res.fun
            except Exception:
                pass
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
    
    # Final validation and extraction
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
        
    # Check disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break
    
    if not valid:
        # Fallback: use initial lattice
        for config in configs:
            inner_list = []
            for q, r in config:
                x, y = axial_to_cartesian(q, r, spacing=1.0)
                inner_list.append([x, y, 0.0])
            cx_avg = sum(p[0] for p in inner_list) / N
            cy_avg = sum(p[1] for p in inner_list) / N
            inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
            test_hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
            test_axes = [get_hex_axes(p[2]) for p in inner_list]
            test_valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint(test_hexagons[i], test_hexagons[j], test_axes[i], test_axes[j]):
                        test_valid = False
                        break
                if not test_valid:
                    break
            if test_valid:
                best_side = calc_outer_side(test_hexagons, 0.0)
                return (inner_list, (0.0, 0.0), best_side, 0.0)
        
        # Ultimate fallback
        inner_list = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Find best outer angle with fine resolution
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
