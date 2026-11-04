import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Precompute unit hexagon vertices for angle 0 (flat-top, vertices at 0, 60, ... degrees)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])
    
    # Precompute edge normals for angle 0 hexagon
    edge_normals_0 = np.zeros((6, 2))
    for i in range(6):
        v1 = unit_verts_0[i]
        v2 = unit_verts_0[(i+1)%6]
        edge = v2 - v1
        # Normal pointing outwards
        normal = np.array([-edge[1], edge[0]])
        # Ensure it points away from origin (dot with midpoint > 0)
        mid = (v1 + v2) / 2
        if np.dot(normal, mid) < 0:
            normal = -normal
        edge_normals_0[i] = normal
    # Normalize
    norms = np.linalg.norm(edge_normals_0, axis=1, keepdims=True)
    edge_normals_0 = edge_normals_0 / norms

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
        normals = np.array(normals)
        
        max_proj = 0.0
        
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        return max_proj * 2.0 / math.sqrt(3)

    def check_disjoint_fast(poly1, poly2):
        """
        Fast check for disjointness using SAT.
        Returns True if disjoint (or touching), False if overlapping.
        """
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
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            if max1 < min2 or max2 < min1:
                return True
                
        return False

    def is_valid(positions, angles):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if not check_disjoint_fast(polys[i], polys[j]):
                    return False
        return True

    # Objective function for fixed angles (0 degrees)
    def obj_fixed_angles(pos_flat):
        if time.time() > deadline - 5:
            return 1000.0
            
        positions = pos_flat.reshape(N, 2)
        angles = np.zeros(N)
        
        # Compute outer side length
        side_len = calc_min_outer_side(positions, angles, 0.0)
        
        # Check for overlaps and add penalty
        # Using SAT with fixed normals is faster
        penalty = 0.0
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, 0.0) for i in range(N)]
        
        for i in range(N):
            for j in range(i+1, N):
                # Check SAT for fixed orientation
                # Axes are edge_normals_0
                # Project both polys onto each axis
                proj1 = polys[i] @ edge_normals_0.T
                proj2 = polys[j] @ edge_normals_0.T
                
                min1 = np.min(proj1, axis=0)
                max1 = np.max(proj1, axis=0)
                min2 = np.min(proj2, axis=0)
                max2 = np.max(proj2, axis=0)
                
                # Gaps: min2 - max1 and min1 - max2
                # If any gap is negative, they overlap
                gaps1 = min2 - max1
                gaps2 = min1 - max2
                
                # Find the maximum overlap depth (most negative gap)
                # Actually, we just need to know if they overlap and by how much
                # Sum of squared negative gaps is a good penalty
                neg_gaps1 = np.maximum(0, -gaps1)
                neg_gaps2 = np.maximum(0, -gaps2)
                
                penalty += np.sum(neg_gaps1**2) + np.sum(neg_gaps2**2)
        
        return side_len + 10.0 * penalty

    best_score = float('inf')
    best_result = None

    # Strategy: Hybrid Discrete-then-Continuous
    # 1. Generate diverse initial guesses from lattice structures
    # 2. Refine each with local optimization
    # 3. Keep the best valid result

    sqrt3 = math.sqrt(3)
    
    # Define a few distinct lattice-based initial configurations
    init_configs = []
    
    # Config 1: Standard hexagonal lattice subset (closest to center)
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * 1.5
            y = i * sqrt3 / 2.0 + j * sqrt3
            lattice_points.append((x, y))
    
    lattice_points.sort(key=lambda p: p[0]**2 + p[1]**2)
    init1 = np.array(lattice_points[:12])
    init1 -= np.mean(init1, axis=0)
    init_configs.append(init1)
    
    # Config 2: Slightly stretched lattice (to encourage tighter packing via optimization)
    # Stretching vertically might help if the optimal shape is not perfectly symmetric
    init2 = np.array([(x, y * 1.05) for x, y in lattice_points[:12]])
    init2 -= np.mean(init2, axis=0)
    init_configs.append(init2)
    
    # Config 3: Slightly compressed lattice
    init3 = np.array([(x * 0.95, y * 0.95) for x, y in lattice_points[:12]])
    init3 -= np.mean(init3, axis=0)
    init_configs.append(init3)
    
    # Config 4: Random perturbation of the standard lattice
    rng = np.random.RandomState(42)
    init4 = init1 + rng.normal(0, 0.1, init1.shape)
    init_configs.append(init4)
    
    # Config 5: Another random perturbation
    rng2 = np.random.RandomState(123)
    init5 = init1 + rng2.normal(0, 0.2, init1.shape)
    init_configs.append(init5)

    # Refine each configuration
    for init_pos in init_configs:
        if time.time() > deadline - 10:
            break
            
        current_best = float('inf')
        current_result = None
        
        # Try Powell first (good for low-dimensional, non-smooth-ish objectives with penalty)
        try:
            res = minimize(
                obj_fixed_angles, 
                init_pos.flatten(), 
                method='Powell',
                options={'maxiter': 2000, 'xtol': 1e-6, 'ftol': 1e-8, 'maxfev': 5000}
            )
            
            if res.fun < current_best:
                current_best = res.fun
                current_result = res.x
        except Exception:
            pass
        
        # If Powell didn't find a good solution, try Nelder-Mead
        if current_result is None or current_best > 5.0:
            try:
                res = minimize(
                    obj_fixed_angles, 
                    init_pos.flatten(), 
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                if res.fun < current_best:
                    current_best = res.fun
                    current_result = res.x
            except Exception:
                pass
        
        # Update global best if this local search found something better
        if current_result is not None and current_best < best_score:
            best_score = current_best
            best_result = current_result

    # Finalize result
    if best_result is not None:
        final_positions = best_result.reshape(N, 2)
        final_angles = np.zeros(N)
        
        # Verify validity
        if not is_valid(final_positions, final_angles):
             # Fallback to a known valid config if invalid
             final_positions = np.array([
                [0, 0], [2.1, 0], [-2.1, 0],
                [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
                [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
                [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
            ])
            
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)
    else:
        # Emergency fallback
        final_positions = np.array([
            [0, 0], [2.1, 0], [-2.1, 0],
            [1.05, 1.81], [-1.05, 1.81], [0, 3.62],
            [1.05, -1.81], [-1.05, -1.81], [0, -3.62],
            [3.15, 1.81], [-3.15, 1.81], [3.15, -1.81]
        ])
        final_angles = np.zeros(N)
        final_side = calc_min_outer_side(final_positions, final_angles, 0.0)

    inner = []
    for i in range(N):
        inner.append([final_positions[i][0], final_positions[i][1], final_angles[i]])
    
    return inner, (0.0, 0.0), final_side, 0.0
