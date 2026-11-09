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
    def get_axes(poly):
        axes = []
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            norm = math.sqrt(dx*dx + dy*dy)
            if norm == 0: continue
            nx = -dy / norm
            ny = dx / norm
            axes.append((nx, ny))
        return axes

    def project(poly, axis):
        dots = [p[0]*axis[0] + p[1]*axis[1] for p in poly]
        return min(dots), max(dots)

    def is_disjoint(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
        return False

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    def calc_outer_side(hexagons, outer_angle_deg):
        base_angles = [30, 90, 150, 210, 270, 330]
        
        max_proj = 0.0
        for hex_verts in hexagons:
            for vx, vy in hex_verts:
                for a in base_angles:
                    rad = math.radians(a + outer_angle_deg)
                    nx = math.cos(rad)
                    ny = math.sin(rad)
                    proj = abs(vx * nx + vy * ny)
                    if proj > max_proj:
                        max_proj = proj
        
        # Apothem of outer hex is S * SQRT3 / 2
        # Max projection must be <= Apothem
        # S >= 2 * max_proj / SQRT3
        return 2.0 * max_proj / SQRT3

    # Objective function for Simulated Annealing
    # We treat the cluster as rigid relative to the first hexagon, but we optimize 
    # the positions of all hexagons individually to allow for "breathing" and rotation 
    # effects, while penalizing overlaps heavily but smoothly.
    
    # To make SA effective, we will optimize the local configurations.
    # Initial state: A hexagonal lattice of 11 hexagons.
    # Axial coordinates for 11 hexagons (center + 6 ring + 4 more? No, 1+6=7. 
    # Let's use a compact shape: 1 center, 6 around, 4 more attached to the ring.
    # Actually, a 2x2x2 block or similar? 
    # Let's stick to the "flower" + 4 extensions used in the seed, but start tighter.
    
    def axial_to_cartesian(q, r, spacing=2.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Base 7 hexagons + 4 more
    # 0,0
    # 1,0, 0,1, -1,1, -1,0, 0,-1, 1,-1
    # 2,0, 2,-1, 2,1, 1,1
    coords = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (2,-1), (2,1), (1,1)]
    
    # Initial positions: Tight lattice (spacing = 2.0 is touching for flat-top? 
    # Distance between centers in hex lattice with side 1 is 2*side*sqrt(3)/2? 
    # No, distance between centers of adjacent flat-top hexagons is sqrt(3) * side.
    # Wait, apothem is sqrt(3)/2. Distance between parallel sides is sqrt(3).
    # So center-to-center distance for touching is sqrt(3).
    # The axial_to_cartesian formula with spacing=1 gives distance sqrt(3)?
    # x = sqrt(3)*(q+r/2), y=1.5*r.
    # Distance (0,0) to (1,0): sqrt(3). Correct.
    # So spacing=1.0 in axial_to_cartesian gives touching hexagons.
    
    initial_params = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=1.0)
        initial_params.extend([x, y, 0.0])
        
    params = np.array(initial_params)
    
    # State for SA
    best_params = params.copy()
    best_score = float('inf')
    current_score = float('inf')
    
    # Calculate initial score
    def evaluate(p):
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
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
        
        # Overlap penalty
        # Sum of penetration depths? Or just count?
        # For SA, a smooth penalty is better. 
        # Let's use a large constant penalty for any overlap, plus the side length.
        # This makes it a "hard" constraint in the landscape, but SA can jump over barriers.
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j]):
                    overlap_penalty += 1000.0
        
        # Find best outer angle (coarse)
        best_side = float('inf')
        for oa_deg in range(0, 60, 10):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
        
        return overlap_penalty + best_side

    current_score = evaluate(params)
    if current_score < best_score:
        best_score = current_score
        best_params = params.copy()

    # Simulated Annealing Parameters
    T = 10.0
    T_min = 0.01
    cooling_rate = 0.995
    steps_per_temp = 50
    
    step = 0
    while T > T_min and time.time() < deadline - 5.0:
        if step % 100 == 0 and time.time() > deadline - 10.0:
            break
            
        # Generate perturbation
        new_params = params.copy()
        # Move a random hexagon
        idx = random.randint(0, N-1)
        dx = random.gauss(0, 0.1)
        dy = random.gauss(0, 0.1)
        da = random.gauss(0, 5.0) # degrees
        
        new_params[idx*3] += dx
        new_params[idx*3+1] += dy
        new_params[idx*3+2] += da
        
        # Occasionally move two hexagons or rotate cluster?
        # For now, single hexagon moves are sufficient for local search.
        
        new_score = evaluate(new_params)
        
        # Acceptance probability
        delta_e = new_score - current_score
        if delta_e < 0 or random.random() < math.exp(-delta_e / T):
            params = new_params
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_params = params.copy()
        
        step += 1
        if step % steps_per_temp == 0:
            T *= cooling_rate
            
    # Final validation and refinement of best_params
    # The best_params found by SA might have tiny overlaps due to floating point or 
    # the "hard" penalty not being perfectly smooth. 
    # We will try to "fix" overlaps by slightly pushing apart overlapping hexagons.
    
    final_params = best_params.copy()
    
    # Quick local refinement: Nelder-Mead on the best result found by SA
    # This can polish the solution.
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
        for i in range(N):
            cx, cy = shifted_centers[i]
            a = angles[i]
            hex_verts = get_hex_vertices(cx, cy, a)
            hexagons.append(hex_verts)
        
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                if not is_disjoint(hexagons[i], hexagons[j]):
                    overlap_penalty += 1000.0
        
        best_side = float('inf')
        for oa_deg in range(0, 60, 5):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                
        return overlap_penalty + best_side

    try:
        res = minimize(objective_nm, final_params, method='Nelder-Mead', 
                       options={'maxiter': 200, 'xatol': 1e-5, 'fatol': 1e-5})
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
    for i in range(N):
        cx, cy = shifted_centers[i]
        a = angles[i]
        hex_verts = get_hex_vertices(cx, cy, a)
        hexagons.append(hex_verts)
        
    # Check disjointness
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not is_disjoint(hexagons[i], hexagons[j]):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to initial lattice if SA/NM failed to find valid config
        # This should rarely happen if we start from a valid lattice
        inner_list = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=1.0)
            inner_list.append([x, y, 0.0])
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        best_side = calc_outer_side(hexagons, 0.0)
        return (inner_list, (0.0, 0.0), best_side, 0.0)

    # Find best outer angle (finer search)
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 60, 1):
        side = calc_outer_side(hexagons, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
            
    inner_list = []
    for i in range(N):
        inner_list.append([shifted_centers[i][0], shifted_centers[i][1], angles[i]])
        
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
