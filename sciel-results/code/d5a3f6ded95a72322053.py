import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0

    def time_left():
        return deadline - time.time()

    # Unit hexagon vertices (pointy-top: vertices at 0, 60, ... 300 degrees)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts_fast(cx, cy, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        return unit_verts_0 @ rot.T + np.array([cx, cy])

    def calc_outer_side(params, outer_center=(0.0, 0.0), outer_angle=0.0):
        """
        params: [x0..x11, y0..y11, a0..a11] (36 values)
        Returns min side length of outer hexagon containing all inner hexagons.
        """
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        theta_rad = math.radians(outer_angle)
        normals = []
        for k in range(6):
            ang_n = theta_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang_n), math.sin(ang_n)])
        normals = np.array(normals)
        
        max_proj = 0.0
        oc = np.array(outer_center)
        
        for i in range(N):
            cx, cy = pos[i]
            a_rad = ang_rad[i]
            verts = get_hex_verts_fast(cx, cy, a_rad)
            verts_rel = verts - oc
            projs = verts_rel @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i
        
        side = max_proj * 2.0 / SQRT3
        return side

    def sat_overlap_depth(poly1, poly2):
        """
        Returns the minimum penetration depth if poly1 and poly2 overlap.
        Returns 0.0 if they are disjoint or just touching.
        """
        min_overlap = 0.0
        axes = []
        
        for k in range(6):
            edge = poly1[(k+1)%6] - poly1[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
        
        for k in range(6):
            edge = poly2[(k+1)%6] - poly2[k]
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-9:
                axes.append(n/norm)
                
        for axis in axes:
            proj1 = np.dot(poly1, axis)
            proj2 = np.dot(poly2, axis)
            min1, max1 = np.min(proj1), np.max(proj1)
            min2, max2 = np.min(proj2), np.max(proj2)
            
            overlap = min(max1, max2) - max(min1, min2)
            if overlap < 0:
                return 0.0
            if overlap > min_overlap:
                min_overlap = overlap
                
        return min_overlap

    def calc_penalty(params):
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        
        polys = []
        for i in range(N):
            cx, cy = pos[i]
            verts = get_hex_verts_fast(cx, cy, ang_rad[i])
            polys.append(verts)
            
        penalty = 0.0
        for i in range(N):
            p1 = polys[i]
            for j in range(i+1, N):
                p2 = polys[j]
                
                min_x1, min_y1 = np.min(p1, axis=0)
                max_x1, max_y1 = np.max(p1, axis=0)
                min_x2, min_y2 = np.min(p2, axis=0)
                max_x2, max_y2 = np.max(p2, axis=0)
                
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                depth = sat_overlap_depth(p1, p2)
                if depth > 0:
                    penalty += depth * depth * 10.0
                    
        return penalty

    def objective(params):
        side = calc_outer_side(params)
        penalty = calc_penalty(params)
        return side + penalty

    def is_valid(params, tol=1e-4):
        pos = params[:2*N].reshape(N, 2)
        ang_rad = params[2*N:]
        polys = [get_hex_verts_fast(pos[i][0], pos[i][1], ang_rad[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                # Bounding box check first
                min_x1, min_y1 = np.min(polys[i], axis=0)
                max_x1, max_y1 = np.max(polys[i], axis=0)
                min_x2, min_y2 = np.min(polys[j], axis=0)
                max_x2, max_y2 = np.max(polys[j], axis=0)
                if max_x1 < min_x2 or max_x2 < min_x1 or max_y1 < min_y2 or max_y2 < min_y1:
                    continue
                
                depth = sat_overlap_depth(polys[i], polys[j])
                if depth > tol:
                    return False
        return True

    # --- GENERATE INITIAL CONFIGURATIONS ---
    inits = []
    
    # 1. Hexagonal lattice clusters (dense packing)
    # Lattice vectors for touching hexagons
    v1 = np.array([1.5, SQRT3/2])
    v2 = np.array([0, SQRT3])
    
    # Generate a grid of lattice points
    lattice_points = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            p = i * v1 + j * v2
            lattice_points.append(p)
    
    lattice_points.sort(key=lambda p: np.dot(p, p))
    
    # Select different subsets of 12 points from the lattice
    # The first 12 are the most compact cluster
    base_cluster = np.array(lattice_points[:12])
    
    # Try different scales and rotations of this cluster
    for scale in [0.95, 0.98, 1.0, 1.02, 1.05]:
        for rot_deg in [0, 30, 60, 90]:
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)], 
                                [math.sin(rot_rad), math.cos(rot_rad)]])
            
            scaled = base_cluster * scale
            rotated = scaled @ rot_mat.T
            
            # Center
            centroid = np.mean(rotated, axis=0)
            centered = rotated - centroid
            
            # Assign angles (aligned with global rotation or 0)
            ang = np.zeros(N)
            # If we rotate the cluster, we might want to rotate the hexagons too? 
            # No, hexagons can rotate independently. Let's start with 0 for all.
            
            params = np.concatenate([centered.flatten(), ang])
            inits.append(params)
            
            # Also try with hexagons rotated by rot_deg
            ang_rot = np.full(N, math.radians(rot_deg))
            params_rot = np.concatenate([centered.flatten(), ang_rot])
            inits.append(params_rot)

    # 2. Random perturbations of the best cluster
    np.random.seed(42)
    best_cluster = base_cluster.copy()
    centroid = np.mean(best_cluster, axis=0)
    best_cluster -= centroid
    
    for _ in range(10):
        perturbed = best_cluster + np.random.normal(0, 0.1, (N, 2))
        ang = np.random.uniform(-0.1, 0.1, N)
        inits.append(np.concatenate([perturbed.flatten(), ang]))

    # 3. Random starts
    for _ in range(5):
        pos = np.random.uniform(-3, 3, (N, 2))
        ang = np.random.uniform(0, 2*math.pi, N)
        inits.append(np.concatenate([pos.flatten(), ang]))

    # --- OPTIMIZE ---
    best_side = float('inf')
    best_params = None
    best_outer_center = [0.0, 0.0]
    best_outer_angle = 0.0

    # Sort inits by objective value to start with the best
    # This is expensive, so just pick a few promising ones
    # Let's just iterate through them
    
    # Phase 1: Coarse Optimization
    for idx, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        
        try:
            res = minimize(
                objective, 
                x0, 
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
            )
            
            if res.fun < best_side:
                # Verify validity
                if is_valid(res.x, tol=1e-3):
                    best_side = calc_outer_side(res.x)
                    best_params = res.x.copy()
        except:
            pass

    # Phase 2: Fine Refinement
    if best_params is not None:
        try:
            res = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True}
            )
            if res.fun < best_side:
                if is_valid(res.x, tol=1e-4):
                    best_side = calc_outer_side(res.x)
                    best_params = res.x.copy()
        except:
            pass

    # Phase 3: Optimize Outer Hexagon Center and Angle
    if best_params is not None:
        pos = best_params[:2*N].reshape(N, 2)
        
        def outer_obj(outer_params):
            oc = outer_params[:2]
            oa = outer_params[2]
            side = calc_outer_side(best_params, outer_center=oc, outer_angle=np.degrees(oa))
            return side

        centroid = np.mean(pos, axis=0)
        x0_outer = np.array([centroid[0], centroid[1], 0.0])
        
        try:
            res_outer = minimize(
                outer_obj, 
                x0_outer, 
                method='Nelder-Mead',
                options={'maxiter': 600, 'xatol': 1e-7, 'fatol': 1e-9}
            )
            
            if res_outer.fun < best_side:
                best_side = res_outer.fun
                best_outer_center = res_outer.x[:2]
                best_outer_angle = np.degrees(res_outer.x[2])
        except:
            pass

    # Fallback
    if best_params is None:
        sqrt3 = math.sqrt(3)
        coords = [
            (-2, 0), (0, 0), (2, 0),
            (-1, sqrt3), (1, sqrt3), (3, sqrt3),
            (-2, 2*sqrt3), (0, 2*sqrt3), (2, 2*sqrt3),
            (-1, 3*sqrt3), (1, 3*sqrt3), (3, 3*sqrt3)
        ]
        cx = sum(c[0] for c in coords) / N
        cy = sum(c[1] for c in coords) / N
        pos = np.array([[c[0]-cx, c[1]-cy] for c in coords])
        ang = np.zeros(N)
        best_params = np.concatenate([pos.flatten(), ang])
        best_side = calc_outer_side(best_params)
        best_outer_center = [0.0, 0.0]
        best_outer_angle = 0.0

    final_pos = best_params[:2*N].reshape(N, 2)
    final_ang = best_params[2*N:]
    
    inner = []
    for i in range(N):
        inner.append([final_pos[i][0], final_pos[i][1], np.degrees(final_ang[i])])
    
    return inner, tuple(best_outer_center), best_side, best_outer_angle
