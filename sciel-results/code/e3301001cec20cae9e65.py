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

    # Helper: Calculate minimum outer hexagon side length containing all given hexagons
    # The outer hexagon is centered at the origin.
    def calc_outer_side(hexagons, outer_angle_deg):
        # Outer hexagon normals (flat-top, rotated by outer_angle_deg)
        # Sides are perpendicular to normals at 30, 90, 150, 210, 270, 330 + outer_angle
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

    # Helper: Calculate penetration depth between two convex polygons using SAT
    # Returns 0 if disjoint, > 0 if overlapping
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

    def penetration_depth(poly1, poly2):
        axes = get_axes(poly1) + get_axes(poly2)
        min_pen = float('inf')
        for axis in axes:
            min1, max1 = project(poly1, axis)
            min2, max2 = project(poly2, axis)
            if max1 < min2 or max2 < min1:
                return 0.0
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_pen:
                min_pen = overlap
        return min_pen

    # Objective function for Nelder-Mead polish
    # Params: [x, y, theta_deg]
    # theta_deg is the rotation of the entire cluster
    def objective_nm(params):
        x, y, theta_deg = params
        theta_rad = math.radians(theta_deg)
        
        # Generate hexagons with cluster rotation and translation
        hexagons = []
        for i in range(N):
            # Local position and angle
            lx, ly, la_deg = local_positions[i]
            # Rotate local position by theta_rad
            rx = lx * math.cos(theta_rad) - ly * math.sin(theta_rad)
            ry = lx * math.sin(theta_rad) + ly * math.cos(theta_rad)
            # Translate
            cx = rx + x
            cy = ry + y
            # Rotate local angle
            la_deg_total = la_deg + theta_deg
            hex_verts = get_hex_vertices(cx, cy, la_deg_total)
            hexagons.append(hex_verts)
            
        # Check overlaps
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen = penetration_depth(hexagons[i], hexagons[j])
                if pen > 1e-6:
                    overlap_penalty += pen * 100.0
                    if overlap_penalty > 1000: # Early exit for large overlaps
                        break
            if overlap_penalty > 1000:
                break
                
        if overlap_penalty > 0:
            return overlap_penalty + calc_outer_side(hexagons, 0.0)
            
        # Find best outer angle (coarse search)
        best_side = float('inf')
        for oa_deg in range(0, 60, 5):
            side = calc_outer_side(hexagons, oa_deg)
            if side < best_side:
                best_side = side
                
        return best_side

    # Simulated Annealing on the rigid cluster
    # State: [x, y, theta_deg]
    # Note: We treat the cluster as rigid. The local positions are fixed from a good initial guess.
    
    # Initial setup: Start with a loose lattice configuration
    # Use axial coordinates to generate initial positions
    def axial_to_cartesian(q, r, spacing=2.0):
        x = SQRT3 * spacing * (q + r / 2.0)
        y = 1.5 * spacing * r
        return x, y

    # Base 7 hexagons + 4 more
    base_7 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    extra = [(2,0), (2,-1), (2,1), (1,1)]
    coords = base_7 + extra
    
    # Local positions relative to cluster center
    local_positions = []
    for q, r in coords:
        x, y = axial_to_cartesian(q, r, spacing=2.0)
        local_positions.append((x, y, 0.0)) # Initial angle 0
        
    # Center the local positions
    cx_avg = sum(p[0] for p in local_positions) / N
    cy_avg = sum(p[1] for p in local_positions) / N
    local_positions = [(x - cx_avg, y - cy_avg, a) for x, y, a in local_positions]

    # Simulated Annealing
    best_state = [0.0, 0.0, 0.0] # x, y, theta
    best_score = float('inf')
    
    T = 10.0
    T_min = 1e-5
    alpha = 0.995
    num_steps = 5000
    
    current_state = [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-30, 30)]
    
    def get_score(state):
        return objective_nm(state)
    
    current_score = get_score(current_state)
    if current_score < best_score:
        best_score = current_score
        best_state = current_state[:]
        
    for i in range(num_steps):
        if time.time() > deadline - 20.0:
            break
            
        # Generate neighbor
        new_state = current_state[:]
        # Random walk in x, y, theta
        new_state[0] += random.gauss(0, 0.5)
        new_state[1] += random.gauss(0, 0.5)
        new_state[2] += random.gauss(0, 5.0)
        
        new_score = get_score(new_state)
        
        # Accept if better, or with probability exp(-(delta)/T)
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_state = new_state
            current_score = new_score
            if current_score < best_score:
                best_score = current_score
                best_state = current_state[:]
                
        T *= alpha
        if T < T_min:
            T = T_min

    # Polish with Nelder-Mead
    from scipy.optimize import minimize
    
    # Try polishing from the best SA state
    try:
        res = minimize(objective_nm, best_state, method='Nelder-Mead', 
                       options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6})
        
        if res.fun < best_score:
            best_state = res.x
            best_score = res.fun
    except Exception as e:
        pass

    # Final validation and construction
    x, y, theta_deg = best_state
    theta_rad = math.radians(theta_deg)
    
    inner_list = []
    hexagons = []
    for i in range(N):
        lx, ly, la_deg = local_positions[i]
        rx = lx * math.cos(theta_rad) - ly * math.sin(theta_rad)
        ry = lx * math.sin(theta_rad) + ly * math.cos(theta_rad)
        cx = rx + x
        cy = ry + y
        la_deg_total = la_deg + theta_deg
        
        inner_list.append([cx, cy, la_deg_total])
        hex_verts = get_hex_vertices(cx, cy, la_deg_total)
        hexagons.append(hex_verts)
        
    # Check disjointness again
    valid = True
    for i in range(N):
        for j in range(i+1, N):
            if not (penetration_depth(hexagons[i], hexagons[j]) < 1e-6):
                valid = False
                break
        if not valid: break
        
    if not valid:
        # Fallback to loose lattice if SA failed to find a valid packing
        inner_list = []
        hexagons = []
        for q, r in coords:
            x, y = axial_to_cartesian(q, r, spacing=2.0)
            inner_list.append([x, y, 0.0])
            hex_verts = get_hex_vertices(x, y, 0.0)
            hexagons.append(hex_verts)
            
        # Center
        cx_avg = sum(p[0] for p in inner_list) / N
        cy_avg = sum(p[1] for p in inner_list) / N
        inner_list = [[x - cx_avg, y - cy_avg, a] for x, y, a in inner_list]
        hexagons = [get_hex_vertices(p[0], p[1], p[2]) for p in inner_list]
        
    # Find best outer angle
    best_side = float('inf')
    best_oa = 0.0
    for oa_deg in range(0, 60, 1):
        side = calc_outer_side(hexagons, oa_deg)
        if side < best_side:
            best_side = side
            best_oa = oa_deg
            
    return (inner_list, (0.0, 0.0), best_side, best_oa)

# EVOLVE-BLOCK-END
