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

    # Helper: Get axes for SAT (normals to edges)
    def get_hex_axes(cx, cy, angle_deg):
        """
        For a regular hexagon with vertices at angles angle_deg + i*60,
        the edge between vertex i and i+1 has a normal direction.
        The outward normal for the edge between vertices at angles a and b (b=a+60)
        is at angle (a+b)/2 = a+30.
        """
        axes = []
        for i in range(6):
            # Normal angle is offset by 30 degrees from the vertex angle
            normal_angle = math.radians(angle_deg + i * 60 + 30)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            axes.append((nx, ny))
        return axes

    def project_poly(vertices, axis):
        """Project polygon vertices onto axis, return (min, max)."""
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

    def is_disjoint(hex1_verts, hex2_verts, axes1, axes2):
        """
        Check if two hexagons are disjoint using SAT.
        Returns True if disjoint (no overlap), False if overlapping.
        """
        all_axes = axes1 + axes2
        for axis in all_axes:
            min1, max1 = project_poly(hex1_verts, axis)
            min2, max2 = project_poly(hex2_verts, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
        """
        Calculate the minimum side length S of a regular hexagon centered at (0,0)
        with rotation outer_angle_deg that contains all given hexagons.
        """
        base_angles = [30, 90, 150, 210, 270, 330]
        # Precompute axis directions for the outer hexagon
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
        # Max projection from center must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Objective function for Simulated Annealing
    # Uses HARD constraint: if any overlap, return large penalty + side length
    # This ensures the search landscape clearly distinguishes valid/invalid states
    
    def evaluate(p, outer_angle_deg=0.0):
        centers = []
        angles = []
        for i in range(N):
            centers.append((p[i*3], p[i*3+1]))
            angles.append(p[i*3+2])
        
        # Center the configuration to (0,0) to minimize outer side length
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
        
        # Check for overlaps (Hard Constraint)
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                    overlap_penalty += 1000.0
        
        # Calculate outer side length
        # For SA, we use a fixed outer angle (0.0) for speed, 
        # but in final refinement we optimize it.
        # However, to guide SA better, we can use a coarse best angle.
        # Let's stick to 0.0 for speed in the inner loop, 
        # as the relative positions matter more than the global rotation for the shape.
        # Actually, the shape's orientation relative to the outer hexagon matters.
        # But since we can rotate the whole cluster effectively by rotating individual hexagons 
        # relative to the global frame, or just optimize the outer angle at the end,
        # using a fixed outer angle here is a valid heuristic.
        
        side = calc_outer_side(hexagons, 0.0)
        
        return overlap_penalty + side

    # Axial to Cartesian for initial configurations
    def axial_to_cartesian(q, r, spacing=1.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Define multiple initial configurations to escape local minima
    # Config 1: Flower + 4 extensions
    coords1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    # Config 2: Compact 3-5-3 block
    coords2 = [(-1,-1),(0,-1),(1,-1),(-2,0),(-1,0),(0,0),(1,0),(2,0),(-1,1),(0,1),(1,1)]
    # Config 3: Linear-ish or different topology
    coords3 = [(0,0), (1,0), (2,0), (3,0), (4,0), (-1,0), (-2,0), (0,1), (1,1), (2,1), (3,1)]
    
    configs = [coords1, coords2, coords3]
    
    best_params = None
    best_score = float('inf')
    
    for config in configs:
        if time.time() > deadline - 20.0:
            break
            
        # Initialize params
        params = []
        for q, r in config:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        params = np.array(params)
        
        # Simulated Annealing
        current_score = evaluate(params)
        local_best_params = params.copy()
        local_best_score = current_score
        
        T = 15.0  # High initial temp to jump over hard constraints
        T_min = 0.01
        cooling_rate = 0.995
        steps_per_temp = 50
        
        step = 0
        while T > T_min and time.time() < deadline - 10.0:
            # Generate perturbation
            new_params = params.copy()
            idx = random.randint(0, N-1)
            
            # Move position and rotate
            # Step size scales with temperature
            step_scale = 0.2 + 0.2 * (T / 15.0)
            dx = random.gauss(0, step_scale)
            dy = random.gauss(0, step_scale)
            da = random.gauss(0, 10.0)  # degrees
            
            new_params[idx*3] += dx
            new_params[idx*3+1] += dy
            new_params[idx*3+2] += da
            
            new_score = evaluate(new_params)
            
            # Acceptance
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
            
    if best_params is None:
        # Fallback to first config
        params = []
        for q, r in coords1:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            params.extend([x, y, 0.0])
        best_params = np.array(params)

    # Local Refinement using Nelder-Mead
    # We use the same objective function. Since it has a hard penalty,
    # Nelder-Mead might struggle if it steps into invalid regions,
    # but starting from a valid SA result helps.
    # We will run it for a limited number of iterations.
    
    def objective_nm(p):
        return evaluate(p, outer_angle_deg=0.0)
        
    try:
        # Use a tighter tolerance
        res = minimize(objective_nm, best_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4})
        if res.fun < best_score:
            best_score = res.fun
            best_params = res.x
    except Exception:
        pass

    # Final Validation and Optimization of Outer Angle
    final_params = best_params.copy()
    
    # Extract centers and angles
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
        
    # Verify validity one last time
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j], hex_axes[i], hex_axes[j]):
                valid = False
                break
        if not valid:
            break
            
    if not valid:
        # If invalid, try to recover by using the initial config which is known valid
        # (assuming the initial lattice is valid, which it is for spacing 1.0)
        print("Warning: Final result invalid, falling back to initial config")
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

    # Optimize outer angle with fine resolution
    best_side = float('inf')
    best_oa = 0.0
    # Search 0 to 60 degrees with 0.1 degree steps
    for oa_deg in range(0, 600):
        oa = oa_deg * 0.1
        side = calc_outer_side(hexagons, oa)
        if side < best_side:
            best_side = side
            best_oa = oa
            
    # Construct final inner list
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
