import math
import time
import numpy as np

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    # Precompute unit hexagon vertices for angle 0
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_deg):
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0) 
        with rotation outer_angle that contains all inner hexagons.
        """
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
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
        def project_polygon(poly, axis):
            proj = poly @ axis
            return np.min(proj), np.max(proj)
            
        axes = []
        for i in range(6):
            edge1 = poly1[(i+1)%6] - poly1[i]
            normal1 = np.array([-edge1[1], edge1[0]])
            axes.append(normal1)
            edge2 = poly2[(i+1)%6] - poly2[i]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)
            
        for axis in axes:
            norm = np.linalg.norm(axis)
            if norm < 1e-9: continue
            axis = axis / norm
            min1, max1 = project_polygon(poly1, axis)
            min2, max2 = project_polygon(poly2, axis)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    # Precompute all 12 SAT axis angles (radians) for hexagon-hexagon separation
    # These are the normals to all 12 edges (6 from each hexagon)
    sat_axes = []
    for k in range(6):
        edge1 = unit_verts_0[(k+1)%6] - unit_verts_0[k]
        normal1 = np.array([-edge1[1], edge1[0]])
        sat_axes.append(normal1 / np.linalg.norm(normal1))
    # Since both hexagons are regular and unit, the 12 axes from two hexagons
    # are just rotations of these 6 base normals by the relative angle.
    # We'll compute them on the fly based on the actual angles.

    def min_gap_hexagons(pos1, ang1_deg, pos2, ang2_deg):
        """
        Compute the minimum separation distance between two unit regular hexagons.
        Uses SAT with the 12 relevant axes.
        Returns the minimum gap (positive = disjoint, negative = overlapping).
        """
        # Get vertices
        v1 = get_hex_verts(pos1[0], pos1[1], 1.0, ang1_deg)
        v2 = get_hex_verts(pos2[0], pos2[1], 1.0, ang2_deg)
        
        # Compute all 12 SAT axes
        axes = []
        for k in range(6):
            e1 = v1[(k+1)%6] - v1[k]
            n1 = np.array([-e1[1], e1[0]])
            axes.append(n1 / np.linalg.norm(n1))
            e2 = v2[(k+1)%6] - v2[k]
            n2 = np.array([-e2[1], e2[0]])
            axes.append(n2 / np.linalg.norm(n2))
        
        min_sep = np.inf
        for axis in axes:
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            if max1 < min2:
                gap = min2 - max1
            elif max2 < min1:
                gap = min1 - max2
            else:
                gap = 0.0
            
            if gap < min_sep:
                min_sep = gap
        
        return min_sep

    def check_all_disjoint(positions, angles):
        """Check if all hexagons are disjoint. Returns True if valid."""
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    # Initial guess: Honeycomb lattice
    sqrt3 = math.sqrt(3)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
            
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init_positions = np.array(lattice_points[:12])
    init_positions -= np.mean(init_positions, axis=0)
    init_angles = np.zeros(N)
    
    x0 = np.concatenate([init_positions.flatten(), init_angles])

    best_score = calc_min_outer_side(init_positions, init_angles, 0.0)
    best_params = x0.copy()

    # Simulated Annealing
    # Objective: minimize outer side length, with penalty for overlap
    def objective(params):
        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]
        
        # Check for overlaps - compute min gap over all pairs
        min_gap = np.inf
        for i in range(N):
            for j in range(i+1, N):
                gap = min_gap_hexagons(positions[i], angles[i], positions[j], angles[j])
                if gap < min_gap:
                    min_gap = gap
        
        side = calc_min_outer_side(positions, angles, 0.0)
        
        # Penalty for overlap: if min_gap < 0, add a large penalty
        if min_gap < 0:
            return side + 100.0 * abs(min_gap)
        
        return side

    # Simulated Annealing
    np.random.seed(42)
    
    T_start = 1.0
    T_end = 0.001
    alpha = 0.9995  # Cooling rate
    max_iterations = 50000
    
    current_params = x0.copy()
    current_obj = objective(current_params)
    best_params_sa = current_params.copy()
    best_obj_sa = current_obj
    
    for it in range(max_iterations):
        if time.time() > deadline - 10:
            break
        
        # Update temperature
        T = T_start * (alpha ** it)
        
        # Generate a random perturbation
        new_params = current_params.copy()
        
        # Perturb positions
        pos_noise = np.random.normal(0, 0.3, 2*N)
        new_params[:2*N] += pos_noise
        
        # Perturb angles (in radians)
        ang_noise = np.random.normal(0, 0.2, N)
        new_params[2*N:] += ang_noise
        
        # Evaluate new solution
        new_obj = objective(new_params)
        
        # Accept or reject
        if new_obj < current_obj:
            current_params = new_params
            current_obj = new_obj
            if new_obj < best_obj_sa:
                best_obj_sa = new_obj
                best_params_sa = new_params.copy()
        else:
            # Accept with probability exp(-(new_obj - current_obj) / T)
            delta = new_obj - current_obj
            if delta < 1000:  # Avoid overflow
                prob = math.exp(-delta / max(T, 1e-10))
                if np.random.random() < prob:
                    current_params = new_params
                    current_obj = new_obj
    
    # Update best if SA found something better
    if best_obj_sa < best_score:
        # Verify with hard SAT
        new_positions = best_params_sa[:2*N].reshape(N, 2)
        new_angles = best_params_sa[2*N:]
        polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
        valid = True
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    valid = False
                    break
            if not valid: break
        
        if valid:
            best_score = calc_min_outer_side(new_positions, new_angles, 0.0)
            best_params = best_params_sa

    # Phase 2: Local refinement using Nelder-Mead on the best solution
    if time.time() < deadline - 20:
        # Use a penalty-based objective for NM
        def objective_nm(params):
            if time.time() > deadline - 5:
                return 1e6
            positions = params[:2*N].reshape(N, 2)
            angles = params[2*N:]
            
            # Check for overlaps
            min_gap = np.inf
            for i in range(N):
                for j in range(i+1, N):
                    gap = min_gap_hexagons(positions[i], angles[i], positions[j], angles[j])
                    if gap < min_gap:
                        min_gap = gap
            
            side = calc_min_outer_side(positions, angles, 0.0)
            
            if min_gap < 0:
                return side + 100.0 * abs(min_gap)
            
            return side

        try:
            res_nm = minimize(
                objective_nm, 
                best_params, 
                method='Nelder-Mead',
                options={'maxiter': 5000, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
            )
            
            if res_nm.fun < best_score + 1e-6:
                new_positions = res_nm.x[:2*N].reshape(N, 2)
                new_angles = res_nm.x[2*N:]
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                        if not valid: break
                if valid:
                    best_score = calc_min_outer_side(new_positions, new_angles, 0.0)
                    best_params = res_nm.x
        except Exception:
            pass

    # Phase 3: Multiple random restarts with SA
    if time.time() < deadline - 15:
        current_best = best_score
        current_params = best_params
        
        for restart in range(3):
            if time.time() > deadline - 10:
                break
            
            # Start from a perturbed version of the current best
            x_restart = current_params.copy()
            noise_pos = np.random.normal(0, 0.5, 2*N)
            noise_ang = np.random.normal(0, 0.5, N)
            x_restart[:2*N] += noise_pos
            x_restart[2*N:] += noise_ang
            
            # Run SA from this starting point
            T_start_r = 0.5
            T_end_r = 0.001
            alpha_r = 0.9995
            max_iter_r = 20000
            
            current_params_r = x_restart.copy()
            current_obj_r = objective(current_params_r)
            best_params_r = current_params_r.copy()
            best_obj_r = current_obj_r
            
            for it in range(max_iter_r):
                if time.time() > deadline - 5:
                    break
                
                T_r = T_start_r * (alpha_r ** it)
                
                new_params_r = current_params_r.copy()
                pos_noise_r = np.random.normal(0, 0.2, 2*N)
                new_params_r[:2*N] += pos_noise_r
                ang_noise_r = np.random.normal(0, 0.15, N)
                new_params_r[2*N:] += ang_noise_r
                
                new_obj_r = objective(new_params_r)
                
                if new_obj_r < current_obj_r:
                    current_params_r = new_params_r
                    current_obj_r = new_obj_r
                    if new_obj_r < best_obj_r:
                        best_obj_r = new_obj_r
                        best_params_r = new_params_r.copy()
                else:
                    delta_r = new_obj_r - current_obj_r
                    if delta_r < 1000:
                        prob_r = math.exp(-delta_r / max(T_r, 1e-10))
                        if np.random.random() < prob_r:
                            current_params_r = new_params_r
                            current_obj_r = new_obj_r
            
            if best_obj_r < current_best:
                new_positions = best_params_r[:2*N].reshape(N, 2)
                new_angles = best_params_r[2*N:]
                polys = [get_hex_verts(new_positions[i][0], new_positions[i][1], 1.0, new_angles[i]) for i in range(N)]
                valid = True
                for i in range(N):
                    for j in range(i+1, N):
                        if not check_disjoint_fast(polys[i], polys[j]):
                            valid = False
                            break
                        if not valid: break
                if valid:
                    current_best = calc_min_outer_side(new_positions, new_angles, 0.0)
                    current_params = best_params_r
        
        best_score = current_best
        best_params = current_params

    final_positions = best_params[:2*N].reshape(N, 2)
    final_angles = best_params[2*N:]
    final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    
    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0

# Need to import minimize for Nelder-Mead
from scipy.optimize import minimize
