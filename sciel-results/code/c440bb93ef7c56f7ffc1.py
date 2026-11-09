# EVOLVE-BLOCK-START
import numpy as np
import math
import time
import random
from scipy.optimize import minimize

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin
    
    sqrt3 = math.sqrt(3)
    
    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of regular hexagon with given center, side, rotation."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            theta = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(theta), cy + side * math.sin(theta)))
        return verts

    def compute_min_side(inner_positions, outer_center, outer_angle):
        """Compute minimum outer side to contain all inner hexagons."""
        cx, cy = outer_center
        max_side = 0.0
        angle_rad = math.radians(outer_angle)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        
        # Pre-compute normals for the outer hexagon in local coordinates
        normals = []
        for k in range(6):
            theta = math.radians(30 + k * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        for ix, iy, iang in inner_positions:
            inner_verts = hexagon_vertices(ix, iy, 1.0, iang)
            for vx, vy in inner_verts:
                dx = vx - cx
                dy = vy - cy
                # Rotate to align with outer hexagon
                rx = dx * cos_a - dy * sin_a
                ry = dx * sin_a + dy * cos_a
                
                # Find maximum projection on each of 6 normals
                for nx, ny in normals:
                    proj = rx * nx + ry * ny
                    # Required side: proj / (sqrt(3)/2)
                    required_side = proj / (sqrt3 / 2)
                    if required_side > max_side:
                        max_side = required_side
        return max_side + 1e-10

    def check_overlap_exact(pos1, pos2):
        """
        Exact overlap check using SAT. Returns 0 if disjoint, positive penalty if overlapping.
        """
        # Bounding box check
        dx = pos1[0] - pos2[0]
        dy = pos1[1] - pos2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.0:
            return 0.0
        if dist_sq < 0.01:
            return 1.0 # Significant overlap
            
        v1 = hexagon_vertices(pos1[0], pos1[1], 1.0, pos1[2])
        v2 = hexagon_vertices(pos2[0], pos2[1], 1.0, pos2[2])
        
        # Collect all normals
        normals = []
        for i in range(6):
            # Edge 1
            ex = v1[(i+1)%6][0] - v1[i][0]
            ey = v1[(i+1)%6][1] - v1[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
            
            # Edge 2
            ex = v2[(i+1)%6][0] - v2[i][0]
            ey = v2[(i+1)%6][1] - v2[i][1]
            nx = -ey
            ny = ex
            length = math.sqrt(nx*nx + ny*ny)
            if length > 1e-9:
                normals.append((nx/length, ny/length))
        
        # Check for separation along each normal
        min_overlap = float('inf')
        for nx, ny in normals:
            # Project both hexagons onto this normal
            proj1 = [v[0]*nx + v[1]*ny for v in v1]
            proj2 = [v[0]*nx + v[1]*ny for v in v2]
            
            min1, max1 = min(proj1), max(proj1)
            min2, max2 = min(proj2), max(proj2)
            
            # If separated, no overlap
            if max1 <= min2 or max2 <= min1:
                return 0.0
            
            # Calculate overlap in this direction
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < min_overlap:
                min_overlap = overlap
        
        # If we get here, they overlap
        return min_overlap

    def objective(params):
        """
        Objective function for Nelder-Mead.
        params: array of size 3*N = [x1, y1, a1, x2, y2, a2, ...]
        """
        positions = []
        for i in range(N):
            x = params[i*3]
            y = params[i*3+1]
            a = params[i*3+2]
            positions.append([x, y, a])
        
        # Calculate side length (assuming outer angle 0 for now)
        side = compute_min_side(positions, (0.0, 0.0), 0.0)
        
        # Calculate overlap penalty
        pen = 0.0
        for i in range(N):
            for j in range(i+1, N):
                pen += check_overlap_exact(positions[i], positions[j])
        
        return side + 1000.0 * pen

    def nelder_mead_optimize(initial_params):
        """
        Run Nelder-Mead optimization on the given initial parameters.
        """
        try:
            res = minimize(objective, initial_params, method='Nelder-Mead', 
                           options={'maxiter': 300, 'xatol': 1e-4, 'fatol': 1e-4})
            return res.x
        except Exception:
            return initial_params

    # Generate initial configurations
    initial_configs = []
    
    # Config 1: Hexagonal lattice (central 11)
    coords = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if max(abs(q), abs(r), abs(q+r)) <= 2:
                coords.append((q, r))
    
    def dist_to_center(coord):
        q, r = coord
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        return x*x + y*y
        
    coords.sort(key=dist_to_center)
    selected_coords = coords[:N]
    
    params1 = []
    for q, r in selected_coords:
        x = sqrt3 * (q + r/2.0)
        y = 1.5 * r
        params1.extend([x, y, 0.0])
    initial_configs.append(params1)
    
    # Config 2: Random perturbation of lattice
    random.seed(42)
    params2 = []
    for q, r in selected_coords:
        x = sqrt3 * (q + r/2.0) + random.gauss(0, 0.1)
        y = 1.5 * r + random.gauss(0, 0.1)
        a = random.gauss(0, 5.0)
        params2.extend([x, y, a])
    initial_configs.append(params2)
    
    # Config 3: Random positions
    params3 = []
    for i in range(N):
        x = random.uniform(-3, 3)
        y = random.uniform(-3, 3)
        a = random.uniform(0, 360)
        params3.extend([x, y, a])
    initial_configs.append(params3)
    
    best_params = initial_configs[0]
    best_score = objective(best_params)
    
    for i, init in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
            
        # Run Nelder-Mead
        opt_params = nelder_mead_optimize(init)
        score = objective(opt_params)
        
        if score < best_score:
            best_score = score
            best_params = opt_params

    # Extract best inner positions
    best_inner = []
    for i in range(N):
        x = best_params[i*3]
        y = best_params[i*3+1]
        a = best_params[i*3+2]
        best_inner.append([x, y, a])
    
    # Simple SA Refinement
    current_temp = 0.1
    cooling_rate = 0.99
    min_temp = 0.01
    current_positions = [p[:] for p in best_inner]
    current_score = objective(best_params)
    best_inner_sa = [p[:] for p in best_inner]
    best_score_sa = current_score
    
    steps = 0
    max_steps = 5000
    
    while current_temp > min_temp and time.time() < deadline - 10.0:
        idx = random.randint(0, N-1)
        old_pos = current_positions[idx][:]
        new_pos = old_pos[:]
        
        # Move
        dx = random.gauss(0, 0.05)
        dy = random.gauss(0, 0.05)
        new_pos[0] += dx
        new_pos[1] += dy
        
        # Rotate
        d_angle = random.gauss(0, 2.0)
        new_pos[2] += d_angle
        
        # Apply change
        current_positions[idx] = new_pos
        
        # Construct params for objective
        temp_params = []
        for p in current_positions:
            temp_params.extend(p)
            
        new_score = objective(temp_params)
        
        delta = new_score - current_score
        if delta < 0 or random.random() < math.exp(-delta / current_temp):
            current_score = new_score
            if current_score < best_score_sa:
                best_score_sa = current_score
                best_inner_sa = [p[:] for p in current_positions]
        else:
            # Revert
            current_positions[idx] = old_pos
            
        current_temp *= cooling_rate
        steps += 1
        
        if time.time() > deadline - 10:
            break
            
    if best_score_sa < best_score:
        best_inner = best_inner_sa
        best_score = best_score_sa

    # Optimize outer angle
    best_side = float('inf')
    best_angle = 0.0
    for outer_angle in [0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 10.0, 20.0, 40.0, 50.0]:
        if time.time() > deadline - 5:
            break
        side = compute_min_side(best_inner, (0.0, 0.0), outer_angle)
        if side < best_side:
            best_side = side
            best_angle = outer_angle
            
    # Verify no overlap
    final_penalty = 0.0
    for i in range(N):
        for j in range(i+1, N):
            final_penalty += check_overlap_exact(best_inner[i], best_inner[j])
            
    if final_penalty > 1e-6:
        # Fallback to safe lattice if optimization failed
        safe_inner = []
        for q, r in selected_coords:
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            safe_inner.append([x, y, 0.0])
        best_inner = safe_inner
        best_side = compute_min_side(safe_inner, (0.0, 0.0), 0.0)
        best_angle = 0.0

    return best_inner, (0.0, 0.0), best_side, best_angle
# EVOLVE-BLOCK-END
