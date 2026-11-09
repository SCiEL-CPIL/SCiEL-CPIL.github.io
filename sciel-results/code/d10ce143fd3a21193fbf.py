"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
import math
import time
import random
import numpy as np

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 110.0  # Safety margin (evaluator limit 120s, keep 10s buffer)
    
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

    # Helper: Check if two hexagons overlap using Separating Axis Theorem (SAT)
    # Optimized: precompute axes once per hexagon, use vectorized projections
    def get_hex_axes(cx, cy, angle_deg):
        """Get the 6 unique normal axes for a hexagon at given center and rotation."""
        axes = []
        for i in range(6):
            theta = math.radians(angle_deg + i * 60)
            # Normal to edge i is perpendicular to the edge direction
            # Edge i goes from vertex i to vertex i+1
            # The outward normal direction for a regular hexagon with vertex at angle theta
            # is at angle theta + 30 degrees (since apothem direction is 30 deg from vertex)
            # Actually, for a hexagon with vertices at 0,60,120,..., the edge between 
            # vertex at 0 and 60 has normal pointing outward at 30 degrees.
            # More carefully: the normal to the edge between vertices at angles a and b
            # (where b = a+60) points in direction (a+b)/2 = a+30
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_centered(hex_verts, axis):
        """Project hexagon vertices onto axis, return (min, max)."""
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
        """Fast disjoint check using precomputed axes."""
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

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # Optimized: precompute cos/sin for outer hexagon axes
    def calc_outer_side_fast(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        # Precompute the 6 axis directions
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
        # Max projection must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Objective function for Simulated Annealing
    # Uses smooth overlap penalty based on penetration depth for better SA landscape
    
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Multiple initial configurations to try
    # Config 1: Flower + 4 extensions (original)
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # Config 2: More compact - try a 2x2x2-like arrangement
    # In hex grid, a compact 11-hexagon shape could be:
    # Row r=-1: q=-1,0,1
    # Row r=0: q=-2,-1,0,1,2
    # Row r=1: q=-1,0,1
    # That's 3+5+3=11
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    
    def make_initial_params(coords):
        params = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        return np.array(params)
    
    # Fast evaluation function for SA
    def evaluate_fast(p):
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
        
        # Smooth overlap penalty: use minimum distance between polygons
        # For convex polygons, if they overlap, compute penetration depth
        # Simplified: if not disjoint, add penalty proportional to how much they overlap
        # Use a simple metric: for each overlapping pair, add penalty based on 
        # the maximum penetration along separating axes
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    # They overlap - add penalty
                    # Compute a rough penetration depth
                    # Use center-to-center distance as proxy
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    # For touching hexagons, center distance is sqrt(3) ≈ 1.732
                    # If dist < sqrt(3), they're overlapping
                    penetration = max(0, SQRT3 - dist)
                    overlap_penalty += 50.0 * penetration * penetration + 10.0
        
        # Find best outer angle (coarse, 6 samples)
        best_side = float('inf')
        for oa_deg in [0, 10, 20, 30, 40, 50]:
            side = calc_outer_side_fast(hexagons, oa_deg)
            if side < best_side:
                best_side = side
        
        return overlap_penalty + best_side

    # Multi-start SA
    best_params = None
    best_score = float('inf')
    
    configs = [coords1, coords2]
    
    for config in configs:
        if time.time() > deadline - 15.0:
            break
        
        params = make_initial_params(config)
        
        # State for SA
        current_score = evaluate_fast(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        # Simulated Annealing Parameters
        T = 8.0
        T_min = 0.05
        cooling_rate = 0.997
        steps_per_temp = 30
        
        step = 0
        while T > T_min and time.time() < deadline - 10.0:
            # Generate perturbation - larger steps for exploration
            new_params = params.copy()
            idx = random.randint(0, N-1)
            
            # Scale step size with temperature
            step_scale = 0.15 + 0.1 * (T / 8.0)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            da = random.gauss(0, 8.0)  # degrees
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate_fast(new_params)
            
            # Acceptance probability
            delta_e = new_score - current_score
            if delta_e < 0 or random.random() < math.exp(-delta_e / T):
                params = new_params
                current_score = new_score
                if current_score < local_best_score:
                    local_best_score = current_score
                    local_best_params = params.copy()
            
            step += 1
            if step % steps_per_temp == 0:
                T *= cooling_rate
        
        if local_best_score < best_score:
            best_score = local_best_score
            best_params = local_best_params.copy()
            
    # Final validation and refinement of best_params
    final_params = best_params.copy()
    
    # Quick local refinement: Nelder-Mead on the best result found by SA
    from scipy.optimize import minimize
    
    def objective_nm(p):
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
            hex_axes.append(get_hex_axes(cx, cy, a))
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint_fast(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    dx = shifted_centers[i][0] - shifted_centers[j][0]
                    dy = shifted_centers[i][1] - shifted_centers[j][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    penetration = max(0, SQRT3 - dist)
                    overlap_penalty += 50.0 * penetration * penetration + 10.0
        
        best_side = float('inf')
        for oa_deg in range(0, 60, 5):
            side = calc_outer_side_fast(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                
        return overlap_penalty + best_side

    try:
        res = minimize(objective_nm, final_params, method='Nelder-Mead', 
                       options={'maxiter': 150, 'xatol': 1e-5, 'fatol': 1e-5})
        if res.fun < best_score:
            final_params = res.x
            best_score = res.fun
    except Exception:
        pass

    # Extract final result
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
        # Fallback: try to fix by using the best valid configuration from SA
        # If that fails, use initial lattice
        # Try each initial config as fallback
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

    # Find best outer angle (finer search with 0.5 degree resolution)
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
