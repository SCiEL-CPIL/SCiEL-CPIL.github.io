import math
import time
import random
import numpy as np
from scipy.optimize import minimize

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

    # Helper: Get the 6 unique normal axes for a hexagon at given center and rotation
    def get_hex_axes(cx, cy, angle_deg):
        axes = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(theta)
            ny = math.sin(theta)
            axes.append((nx, ny))
        return axes

    def project_centered(hex_verts, axis):
        nx, ny = axis
        min_val = float('inf')
        max_val = float('-inf')
        for vx, vy in hex_verts:
            proj = vx * nx + vy * ny
            if proj < min_val:
                min_val = proj
            if proj > max_val:
                max_val = proj
        return min_val, max_val

    def is_disjoint_fast(hex1_verts, hex2_verts, axes1, axes2):
        all_axes = axes1 + axes2
        for axis in all_axes:
            nx, ny = axis
            min1 = max1 = None
            min2 = max2 = None
            for vx, vy in hex1_verts:
                proj = vx * nx + vy * ny
                if min1 is None or proj < min1:
                    min1 = proj
                if max1 is None or proj > max1:
                    max1 = proj
            for vx, vy in hex2_verts:
                proj = vx * nx + vy * ny
                if min2 is None or proj < min2:
                    min2 = proj
                if max2 is None or proj > max2:
                    max2 = proj
            if max1 < min2 or max2 < min1:
                return True
        return False

    def calc_outer_side_fast(hexagons, outer_angle_deg):
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

    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Differentiable Objective Function for L-BFGS-B
    # Uses smooth penetration depth approximation for overlap penalty
    def objective(p):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration
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
            hex_axes.append(get_hex_axes(cx, cy, a))
        
        # Smooth overlap penalty
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # Smooth penalty based on penetration
                    # Touching distance is sqrt(3)
                    penetration = max(0, SQRT3 - dist)
                    # Quadratic penalty with high weight to enforce disjointness
                    overlap_penalty += 100.0 * penetration * penetration + 5.0
        
        # Outer side length
        # Use a fixed outer angle for the gradient step to maintain differentiability
        # We will refine the angle later
        side = calc_outer_side_fast(hexagons, 0.0)
        
        return overlap_penalty + side

    # Initial configurations
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    
    def make_initial_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        return np.array(params)
    
    best_params = None
    best_score = float('inf')
    
    configs = [coords1, coords2]
    
    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        params = make_initial_params(config)
        
        # Run L-BFGS-B
        try:
            # Use finite difference gradients initially, or provide analytic if possible
            # L-BFGS-B with finite differences is robust for this small problem
            res = minimize(objective, params, method='L-BFGS-B', 
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6, 'maxls': 50})
            
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x
        except Exception:
            pass

    if best_params is None:
        # Fallback to initial config 1
        best_params = make_initial_params(coords1)
        best_score = objective(best_params)

    # Refinement phase: 
    # 1. Try to reduce overlaps further with a stricter penalty or local search
    # 2. Optimize the outer hexagon angle
    
    # Local search to fix any remaining overlaps
    def local_fix(params):
        p = params.copy()
        improved = True
        iters = 0
        while improved and iters < 100 and time.time() < deadline - 5.0:
            improved = False
            iters += 1
            current_score = objective(p)
            
            # Try small perturbations
            for _ in range(10):
                idx = random.randint(0, N-1)
                new_p = p.copy()
                step = 0.05
                new_p[idx*3] += random.gauss(0, step)
                new_p[idx*3+1] += random.gauss(0, step)
                new_p[idx*3+2] += random.gauss(0, 5.0)
                
                new_score = objective(new_p)
                if new_score < current_score:
                    p = new_p
                    current_score = new_score
                    improved = True
                    break
        return p

    # Apply local fix
    if time.time() < deadline - 10.0:
        best_params = local_fix(best_params)
        best_score = objective(best_params)

    # Final validation and angle optimization
    centers = []
    angles = []
    for i in range(N):
        centers.append((best_params[i*3], best_params[i*3+1]))
        angles.append(best_params[i*3+2])
        
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
        hex_axes.append(get_hex_axes(cx, cy, a))
        
    # Check disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid: break
    
    if not valid:
        # Fallback: Use the best valid configuration from initial configs
        # This is a safety net; L-BFGS-B should have found a valid one
        for config in [coords1, coords2]:
            inner_list = []
            for q, r in config:
                x, y = axial_to_cartesian(q, r, spacing=1.0)
                inner_list.append([x, y, 0.0])
            cx_avg = sum(p[0] for p in inner_list) / N
            cy_avg = sum(p[1] for p in inner_list) / N
            inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
            test_hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
            test_axes = [get_hex_axes(p[0], p[1], p[2]) for p in inner_list]
            test_valid = True
            for i in range(N):
                for j in range(i+1, N):
                    if not is_disjoint_fast(test_hexagons[i], test_hexagons[j], test_axes[i], test_axes[j]):
                        test_valid = False
                        break
                if not test_valid: break
            if test_valid:
                best_side = calc_outer_side_fast(test_hexagons, 0.0)
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
        best_side = calc_outer_side_fast(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Find best outer angle (finer search)
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 600):  # 0 to 59.9 in 0.1 degree steps
        oa = oa_deg * 0.1
        side = calc_outer_side_fast(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
