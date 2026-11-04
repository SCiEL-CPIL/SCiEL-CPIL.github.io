import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0 (pointy-top, vertex at 0 deg)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    sqrt3 = math.sqrt(3)

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        # For a regular hexagon with side S, the inradius (apothem) is S*sqrt(3)/2.
        # The maximum projection onto edge normals gives the required inradius.
        # So S = max_proj * 2 / sqrt(3)
        return max_proj * 2.0 / math.sqrt(3)

    def soft_min_outer_side(positions, angles, outer_angle=0.0, temperature=0.01):
        """
        Smooth surrogate for calc_min_outer_side using soft-max.
        This makes the objective differentiable and easier to optimize.
        """
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals) # (6, 2)
        
        all_projs = []
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T # (6, 6)
            all_projs.append(projs.flatten())
        
        all_projs = np.concatenate(all_projs)  # (N*6*6,)
        
        # Soft-max to approximate max
        scaled = all_projs / temperature
        max_scaled = np.max(scaled)
        exp_vals = np.exp(scaled - max_scaled)
        soft_max = temperature * (max_scaled + np.log(np.sum(exp_vals)))
        
        return soft_max * 2.0 / math.sqrt(3)

    def sat_gap(poly1, poly2):
        """
        Compute the minimum gap (positive if separated, negative if overlapping)
        between two convex polygons using SAT.
        Returns the maximum separation (minimum gap) over all axes.
        If any axis has positive separation, they are disjoint.
        """
        min_gap = float('inf')
        
        # Axes from poly1
        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap  # Separated
        
        # Axes from poly2
        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            gap = max(min1 - max2, min2 - max1)
            if gap < 0:
                if gap < min_gap:
                    min_gap = gap
            else:
                return gap  # Separated
        
        return min_gap

    def is_valid_config(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    # --- Part 1: Generate diverse initial configurations ---
    
    def generate_honeycomb_config(radius):
        """Generate hexagons in a honeycomb pattern."""
        positions = []
        for q in range(-radius, radius+1):
            for r in range(-radius, radius+1):
                if abs(q + r) <= radius:
                    x = 1.5 * q
                    y = sqrt3 * (r + 0.5 * q)
                    positions.append((x, y))
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(positions[:N])
    
    def generate_ring_config():
        """Generate hexagons in a ring pattern around center."""
        positions = []
        # Center
        positions.append((0, 0))
        # First ring: 6 hexagons at distance sqrt(3)
        for k in range(6):
            ang = k * math.pi / 3
            x = sqrt3 * math.cos(ang)
            y = sqrt3 * math.sin(ang)
            positions.append((x, y))
        # Second ring: 6 more hexagons at distance 2*sqrt(3)
        for k in range(6):
            ang = k * math.pi / 3 + math.pi/6
            x = 2 * sqrt3 * math.cos(ang)
            y = 2 * sqrt3 * math.sin(ang)
            positions.append((x, y))
        return np.array(positions[:N])
    
    def generate_grid_config():
        """Generate hexagons in a grid pattern."""
        positions = []
        for i in range(-2, 3):
            for j in range(-2, 3):
                x = i * 1.5
                y = i * sqrt3 / 2 + j * sqrt3
                positions.append((x, y))
        positions.sort(key=lambda p: p[0]**2 + p[1]**2)
        return np.array(positions[:N])
    
    initial_configs = []
    
    # Honeycomb patterns with different radii
    for radius in [2, 3]:
        pos = generate_honeycomb_config(radius)
        pos -= np.mean(pos, axis=0)
        initial_configs.append((pos, np.zeros(N)))
    
    # Ring pattern
    pos = generate_ring_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))
    
    # Grid pattern
    pos = generate_grid_config()
    pos -= np.mean(pos, axis=0)
    initial_configs.append((pos, np.zeros(N)))
    
    # Random configurations
    rng = np.random.RandomState(42)
    for seed in range(10):
        positions = []
        angles = []
        for i in range(N):
            # Place hexagons in a roughly circular region
            r = rng.uniform(0, 4)
            theta = rng.uniform(0, 2*math.pi)
            x = r * math.cos(theta)
            y = r * math.sin(theta)
            positions.append((x, y))
            angles.append(rng.uniform(0, 2*math.pi))
        
        # Center the configuration
        pos_arr = np.array(positions)
        pos_arr -= np.mean(pos_arr, axis=0)
        initial_configs.append((pos_arr, np.array(angles)))

    best_score = float('inf')
    best_params = None

    # --- Objective function with smooth penalty ---
    
    def objective(params, penalty_weight=100.0):
        if time.time() > deadline - 5:
            return 1e6
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap**2
        
        base_score = soft_min_outer_side(positions, angles, 0.0, temperature=0.05)
        return base_score + penalty

    # --- Part 2: Multi-start optimization ---
    
    for idx, (init_pos, init_ang) in enumerate(initial_configs):
        if time.time() > deadline - 30:
            break
            
        x0 = np.concatenate([init_pos.flatten(), init_ang])
        
        # Quick Nelder-Mead optimization
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead', 
                options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            
            if res.fun < best_score:
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid_config(new_positions, new_angles):
                    # Use exact objective for score comparison
                    exact_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if exact_score < best_score:
                        best_score = exact_score
                        best_params = res.x
        except Exception:
            pass

    # --- Part 3: Local refinement and polish ---
    
    if best_params is not None:
        current_best = best_score
        current_params = best_params
        
        # Multiple restarts with small perturbations
        for _ in range(15):
            if time.time() > deadline - 20:
                break
                
            noise_pos = np.random.normal(0, 0.05, 2*N)
            noise_ang = np.random.normal(0, 3, N) * math.pi / 180
            x_perturbed = current_params.copy()
            x_perturbed[:2*N] += noise_pos
            x_perturbed[2*N:] += noise_ang
            
            try:
                res = minimize(
                    objective, 
                    x_perturbed, 
                    method='Nelder-Mead', 
                    options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10}
                )
                if res.fun < current_best:
                    new_positions = res.x[:2*N].reshape(N, 2)
                    new_angles = res.x[2*N:]
                    if is_valid_config(new_positions, new_angles):
                        exact_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                        if exact_score < current_best:
                            current_best = exact_score
                            current_params = res.x
            except Exception:
                continue

        best_score = current_best
        best_params = current_params
        
        # Final high-precision polish pass with exact objective
        if time.time() < deadline - 15:
            def exact_objective(params):
                positions = params[:2*N].reshape(N, 2)
                angles = params[2*N:]
                return calc_min_outer_side(positions, angles, 0.0)
            
            try:
                res_polish = minimize(
                    exact_objective, 
                    best_params, 
                    method='Nelder-Mead', 
                    options={'maxiter': 10000, 'xatol': 1e-12, 'fatol': 1e-14}
                )
                new_positions = res_polish.x[:2*N].reshape(N, 2)
                new_angles = res_polish.x[2*N:]
                if is_valid_config(new_positions, new_angles):
                    polish_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    if polish_score < best_score:
                        best_score = polish_score
                        best_params = res_polish.x
            except Exception:
                pass

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
