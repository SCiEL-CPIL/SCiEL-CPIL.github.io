import numpy as np
from scipy.optimize import differential_evolution
import time

def hexagon_packing_12():
    start_time = time.time()
    deadline = start_time + 600
    
    sqrt3 = np.sqrt(3)
    
    # Inner hexagon vertex offsets (side 1, angle 0)
    inner_angles = np.arange(6) * np.pi / 3
    inner_offsets = np.column_stack([np.cos(inner_angles), np.sin(inner_angles)])
    
    # Outward normals for unit hexagon edges (for SAT)
    sat_normal_angles = (30 + 60 * np.arange(6)) * np.pi / 180
    sat_normals = np.column_stack([np.cos(sat_normal_angles), np.sin(sat_normal_angles)])
    
    def get_hex_vertices(center):
        return center + inner_offsets
    
    def check_separated(c1, c2, tol=1e-6):
        v1 = get_hex_vertices(c1)
        v2 = get_hex_vertices(c2)
        for k in range(6):
            n = sat_normals[k]
            proj1 = v1 @ n
            proj2 = v2 @ n
            if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
                return True
        return False
    
    def all_separated(centers, tol=1e-6):
        for i in range(12):
            for j in range(i + 1, 12):
                if not check_separated(centers[i], centers[j], tol):
                    return False
        return True
    
    def compute_min_s(centers):
        inward_normal_angles = (30 + 60 * np.arange(6)) * np.pi / 180
        inward_normals = np.column_stack([np.cos(inward_normal_angles), np.sin(inward_normal_angles)])
        
        max_required = 0.0
        for i in range(12):
            verts = get_hex_vertices(centers[i])
            for k in range(6):
                n = inward_normals[k]
                proj = verts @ n
                required = proj.max() / (sqrt3 / 2)
                if required > max_required:
                    max_required = required
        return max_required
    
    # Initial honeycomb arrangement
    initial = np.array([
        [0.0, 0.0],
        [-sqrt3, 0.0],
        [sqrt3, 0.0],
        [-sqrt3/2, 1.5],
        [sqrt3/2, 1.5],
        [-sqrt3/2, -1.5],
        [sqrt3/2, -1.5],
        [-3*sqrt3/2, 1.5],
        [3*sqrt3/2, 1.5],
        [-3*sqrt3/2, -1.5],
        [3*sqrt3/2, -1.5],
        [0.0, -3.0]
    ])
    
    best_centers = initial.copy()
    best_s = compute_min_s(best_centers)
    best_valid = all_separated(best_centers)
    
    if not best_valid:
        # Fallback: spread out more
        initial2 = initial * 1.5
        s2 = compute_min_s(initial2)
        if all_separated(initial2):
            best_centers = initial2
            best_s = s2
            best_valid = True
    
    def objective(x):
        centers = x.reshape(12, 2)
        s = compute_min_s(centers)
        if not all_separated(centers):
            return s + 1000.0
        return s
    
    bounds = [(-8.0, 8.0)] * 24
    
    if time.time() < deadline - 60:
        try:
            result = differential_evolution(
                objective, bounds,
                maxiter=200, tol=1e-8,
                seed=42, polish=True,
                popsize=15
            )
            if time.time() < deadline - 30:
                opt_centers = result.x.reshape(12, 2)
                opt_s = compute_min_s(opt_centers)
                if all_separated(opt_centers):
                    if opt_s < best_s:
                        best_centers = opt_centers
                        best_s = opt_s
                        best_valid = True
        except Exception:
            pass
    
    inner_hex_data = np.zeros((12, 3))
    for i in range(12):
        inner_hex_data[i, 0] = best_centers[i, 0]
        inner_hex_data[i, 1] = best_centers[i, 1]
        inner_hex_data[i, 2] = 0.0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(best_s)
