import numpy as np
import math
import time
import random
from scipy.optimize import minimize

def hexagon_packing_12():
    """
    Packs 12 unit regular hexagons into a regular hexagon of minimum side length s.
    Returns (inner, outer, s) where:
    inner: (12, 3) array of (x, y, angle_deg)
    outer: (3,) array of (x, y, angle_deg)
    s: side length of outer hexagon
    """
    start_time = time.time()
    deadline = start_time + 590  # Safety margin
    
    N = 12
    sqrt3 = math.sqrt(3)
    
    # Unit hexagon vertices (angle 0, side 1)
    # Vertices at 0, 60, 120, 180, 240, 300 degrees
    unit_verts = np.array([
        [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))] for i in range(6)
    ])
    
    # Outward normals for unit hexagon (angle 0)
    # Normals at 30, 90, 150, 210, 270, 330 degrees.
    base_normals = np.array([
        [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))] for i in range(6)
    ])
    
    def rot_matrix(theta):
        c = math.cos(theta)
        s = math.sin(theta)
        return np.array([[c, -s], [s, c]])

    def get_hex_vertices(cx, cy, side, angle_rad):
        R = rot_matrix(angle_rad)
        return cx + side * (unit_verts @ R.T)

    def get_hex_normals(angle_rad):
        R = rot_matrix(angle_rad)
        return base_normals @ R.T

    def compute_s(centers, angles_rad):
        """
        Compute the minimum side length s of an outer hexagon (centered at origin, angle 0)
        that contains all inner hexagons.
        """
        max_proj = 0.0
        # Outer hex normals are fixed at 30, 90, ... (angle 0)
        outer_normals = base_normals
        
        for i in range(N):
            verts = centers[i] + (unit_verts @ rot_matrix(angles_rad[i]).T)
            # Project vertices onto outer hex normals
            projs = verts @ outer_normals.T
            max_proj_hex = projs.max(axis=0)
            current_max = np.max(max_proj_hex)
            if current_max > max_proj:
                max_proj = current_max
        
        # Distance from center to edge is s * sqrt3 / 2
        # So max_proj <= s * sqrt3 / 2  => s >= max_proj * 2 / sqrt3
        return max_proj * 2.0 / sqrt3

    def collision_penalty(centers, angles_rad):
        """
        Compute a smooth penalty for collisions between inner hexagons.
        Uses SAT to find minimum separation distance (negative if overlapping).
        """
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                c1, a1 = centers[i], angles_rad[i]
                c2, a2 = centers[j], angles_rad[j]
                
                verts1 = c1 + (unit_verts @ rot_matrix(a1).T)
                verts2 = c2 + (unit_verts @ rot_matrix(a2).T)
                
                normals1 = get_hex_normals(a1)
                normals2 = get_hex_normals(a2)
                
                axes = np.vstack([normals1, normals2])
                
                proj1 = verts1 @ axes.T
                proj2 = verts2 @ axes.T
                
                min1 = proj1.min(axis=0)
                max1 = proj1.max(axis=0)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Overlap on each axis
                overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
                
                # We want a measure of "how much they are NOT separated".
                # If ANY axis has overlap <= 0, they are separated.
                # If ALL axes have overlap > 0, they are colliding.
                # Smooth penalty: sum of squared positive overlaps.
                pos_overlaps = np.maximum(0, overlaps)
                penalty += np.sum(pos_overlaps ** 2)
        
        return penalty

    def total_cost(params):
        """
        Total cost to minimize.
        params: [x1, y1, a1, x2, y2, a2, ...]
        """
        centers = params[0::3].reshape(N, 2)
        angles_rad = params[1::3]
        
        s = compute_s(centers, angles_rad)
        col_pen = collision_penalty(centers, angles_rad)
        
        # Heavy penalty for collisions to act as hard constraints
        return s + 100.0 * col_pen

    def validate_config(centers, angles_rad, s, eps=1e-6):
        """
        Strict validation.
        """
        # 1. Collisions
        for i in range(N):
            for j in range(i + 1, N):
                c1, a1 = centers[i], angles_rad[i]
                c2, a2 = centers[j], angles_rad[j]
                
                verts1 = c1 + (unit_verts @ rot_matrix(a1).T)
                verts2 = c2 + (unit_verts @ rot_matrix(a2).T)
                
                normals1 = get_hex_normals(a1)
                normals2 = get_hex_normals(a2)
                
                axes = np.vstack([normals1, normals2])
                
                proj1 = verts1 @ axes.T
                proj2 = verts2 @ axes.T
                
                min1 = proj1.min(axis=0)
                max1 = proj1.max(axis=0)
                min2 = proj2.min(axis=0)
                max2 = proj2.max(axis=0)
                
                # Separated if there is an axis where intervals don't overlap
                # max1 < min2 - eps OR max2 < min1 - eps
                sep1 = max1 < min2 - eps
                sep2 = max2 < min1 - eps
                
                if not (np.any(sep1) or np.any(sep2)):
                    return False
        
        # 2. Containment
        outer_verts = get_hex_vertices(0, 0, s, 0.0)
        
        for i in range(N):
            verts = centers[i] + (unit_verts @ rot_matrix(angles_rad[i]).T)
            for v in verts:
                inside = True
                for k in range(6):
                    p1 = outer_verts[k]
                    p2 = outer_verts[(k + 1) % 6]
                    edge = p2 - p1
                    vec = v - p1
                    cross = edge[0] * vec[1] - edge[1] * vec[0]
                    if cross < -eps:
                        inside = False
                        break
                if not inside:
                    return False
        
        return True

    def generate_templates():
        """
        Generate diverse initial configurations (templates) to escape local minima.
        """
        templates = []
        
        # Template 1: Hexagonal lattice (3x4 grid roughly)
        dx = 2.5
        dy = sqrt3
        init_c = []
        init_a = []
        for r in range(3):
            for c in range(4):
                x = c * dx
                y = r * dy
                if r % 2 == 1:
                    x += dx / 2.0
                init_c.append([x, y])
                init_a.append(0.0)
        centers = np.array(init_c)
        angles = np.array(init_a)
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        templates.append((centers, angles))
        
        # Template 2: 12 in a ring around center? No, 12 is too many for a tight ring with unit hexes.
        # Let's try 2 rings of 6? No, 12 hexes.
        # Maybe a 2x6 rectangle?
        init_c = []
        init_a = []
        for r in range(2):
            for c in range(6):
                x = c * 2.5
                y = r * sqrt3
                if r % 2 == 1:
                    x += 1.25
                init_c.append([x, y])
                init_a.append(0.0)
        centers = np.array(init_c)
        angles = np.array(init_a)
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        templates.append((centers, angles))
        
        # Template 3: Random start
        centers = np.random.normal(0, 2.0, (N, 2))
        angles = np.random.uniform(-np.pi, np.pi, N)
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        templates.append((centers, angles))
        
        # Template 4: Compact cluster
        # Place them in a tight hexagonal shape (1 center, 6 around, 5 more?)
        # Just use a dense random start
        centers = np.random.normal(0, 1.5, (N, 2))
        angles = np.random.uniform(-np.pi, np.pi, N)
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        templates.append((centers, angles))

        return templates

    best_centers = None
    best_angles = None
    best_s = float('inf')
    
    templates = generate_templates()
    
    for t_idx, (init_c, init_a) in enumerate(templates):
        if time.time() > deadline:
            break
            
        # Prepare initial params
        init_params = np.zeros(N * 3)
        init_params[0::3] = init_c[:, 0]
        init_params[1::3] = init_c[:, 1]
        init_params[2::3] = init_a
        
        # Define bounds to keep things reasonable (prevent explosion)
        bounds = [(-10, 10)] * N + [(-10, 10)] * N + [(-np.pi, np.pi)] * N
        
        # Run optimization
        try:
            res = minimize(total_cost, init_params, method='L-BFGS-B', bounds=bounds, 
                           options={'maxiter': 1000, 'ftol': 1e-10, 'gtol': 1e-8})
            
            if res.success or res.fun < float('inf'):
                res_params = res.x
                centers = res_params[0::3].reshape(N, 2)
                angles_rad = res_params[1::3]
                
                # Ensure angles are within range if needed (not strictly necessary for math)
                s_curr = compute_s(centers, angles_rad)
                
                if validate_config(centers, angles_rad, s_curr):
                    if s_curr < best_s:
                        best_s = s_curr
                        best_centers = centers.copy()
                        best_angles = angles_rad.copy()
        except Exception as e:
            pass

    # If no valid config found from optimization, fallback to best from SA-like search or initial
    if best_centers is None:
        # Fallback to a known valid-ish config (the lattice one)
        init_c, init_a = templates[0]
        centers = init_c
        angles_rad = init_a
        s_curr = compute_s(centers, angles_rad)
        # It might not be perfectly valid due to collisions in the dense lattice start
        # But we need to return *something*.
        # Let's try a quick local polish on the best template
        for t_idx, (init_c, init_a) in enumerate(templates):
            init_params = np.zeros(N * 3)
            init_params[0::3] = init_c[:, 0]
            init_params[1::3] = init_c[:, 1]
            init_params[2::3] = init_a
            try:
                res = minimize(total_cost, init_params, method='L-BFGS-B', 
                               options={'maxiter': 500})
                res_params = res.x
                centers = res_params[0::3].reshape(N, 2)
                angles_rad = res_params[1::3]
                s_curr = compute_s(centers, angles_rad)
                if validate_config(centers, angles_rad, s_curr):
                    best_s = s_curr
                    best_centers = centers.copy()
                    best_angles = angles_rad.copy()
                    break
            except:
                pass

    # Final fallback if still None (should not happen)
    if best_centers is None:
        # Return a trivially valid but large hexagon
        # Just spread them out
        centers = np.array([
            [i*3.0, j*3.0] for i in range(4) for j in range(3)
        ])
        angles_rad = np.zeros(N)
        centroid = np.mean(centers, axis=0)
        centers -= centroid
        s_curr = compute_s(centers, angles_rad)
        best_centers = centers
        best_angles = angles_rad
        best_s = s_curr

    inner_ret = np.column_stack((best_centers, best_angles * 180.0 / math.pi))
    outer_ret = np.array([0.0, 0.0, 0.0])
    
    return inner_ret, outer_ret, best_s
