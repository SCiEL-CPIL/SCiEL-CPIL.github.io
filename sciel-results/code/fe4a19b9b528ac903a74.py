import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580

    # Unit hexagon vertices (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # 3 edge normals (perpendicular to edges)
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    def compute_enclosing_side(pts):
        """Compute minimum side length of enclosing hexagon centered at origin."""
        max_proj = 0.0
        for k in range(3):
            n_k = normals[k]
            pts_proj = pts @ n_k
            verts_proj = hex_verts @ n_k
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            if max_abs_sum > max_proj:
                max_proj = max_abs_sum
        return (2.0 / np.sqrt(3.0)) * max_proj

    def check_separation(pts):
        """Check all pairs are separated using 12-normal criterion."""
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                # Project on 3 normals, check if any gap > 1e-6
                proj_v = np.abs(v @ normals.T)
                # Width of unit hex along each normal is 2*sqrt(3)/2 = sqrt(3)
                # Gap = |proj| - sqrt(3)
                gaps = proj_v - np.sqrt(3)
                if np.max(gaps) <= 1e-6:
                    return False
        return True

    def contains_in_hex(pts, side):
        """Check all inner hex vertices are inside outer hex."""
        outer_verts = side * hex_verts
        for i in range(11):
            for v in range(6):
                p = pts[i] + hex_verts[v]
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    def valid(pts, side):
        if not np.all(np.isfinite(pts)) or not np.isfinite(side):
            return False
        if side < np.sqrt(11):
            return False
        if not check_separation(pts):
            return False
        if not contains_in_hex(pts, side):
            return False
        return True

    def objective(x):
        """Objective: minimize enclosing hex side length with overlap penalty."""
        pts = x.reshape(11, 2)
        side = compute_enclosing_side(pts)
        
        # Overlap penalty
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj = np.max(proj_v)
                if max_proj < np.sqrt(3) + 1e-4:
                    depth = np.sqrt(3) + 1e-4 - max_proj
                    penalty += depth * depth
        
        return side + 1000.0 * penalty

    # Initial configuration: center + 6 ring + 4 outer
    d = np.sqrt(3)
    pts = np.array([[0, 0]])
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts = np.vstack([pts, [[d * np.cos(angle), d * np.sin(angle)]]])
    for angle_deg in [0, 120, 180, 300]:
        angle = np.deg2rad(angle_deg)
        pts = np.vstack([pts, [[2 * d * np.cos(angle), 2 * d * np.sin(angle)]]])
    
    x0 = pts.flatten()
    best_x = x0.copy()
    best_side = compute_enclosing_side(best_x.reshape(11, 2))
    best_val = objective(x0)
    
    np.random.seed(42)
    for restart in range(5):
        if time.time() > deadline - 100:
            break
        
        if restart == 0:
            x_start = x0
        else:
            x_start = best_x + np.random.normal(0, 0.1, x0.shape)
        
        res = minimize(objective, x_start, method='Nelder-Mead',
                      options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10,
                              'adaptive': True})
        
        cur_side = compute_enclosing_side(res.x.reshape(11, 2))
        cur_val = objective(res.x)
        
        if cur_val < best_val:
            best_x = res.x.copy()
            best_val = cur_val
            best_side = cur_side
    
    pts_final = best_x.reshape(11, 2)
    
    # Repair: push overlapping hexagons apart
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        all_ok = True
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts_final[i] - pts_final[j]
                proj_v = np.abs(v @ normals.T)
                max_proj = np.max(proj_v)
                if max_proj < np.sqrt(3) + 1e-6:
                    all_ok = False
                    min_idx = np.argmin(proj_v)
                    n = normals[min_idx]
                    dot = np.dot(v, n)
                    if dot > 0:
                        pts_final[i] += 0.01 * n
                        pts_final[j] -= 0.01 * n
                    else:
                        pts_final[i] -= 0.01 * n
                        pts_final[j] += 0.01 * n
        if all_ok:
            break
    
    final_side = compute_enclosing_side(pts_final)
    
    # Ensure boundary containment
    if not contains_in_hex(pts_final, final_side):
        for exp in [1.001, 1.005, 1.01, 1.02, 1.05, 1.1]:
            if contains_in_hex(pts_final, final_side * exp):
                final_side *= exp
                break
    
    # Final validity check
    if not valid(pts_final, final_side):
        # Fallback: use initial configuration
        pts_final = x0.reshape(11, 2)
        final_side = compute_enclosing_side(pts_final)
        for exp in [1.001, 1.005, 1.01, 1.02, 1.05, 1.1]:
            if contains_in_hex(pts_final, final_side * exp):
                final_side *= exp
                break
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_side
