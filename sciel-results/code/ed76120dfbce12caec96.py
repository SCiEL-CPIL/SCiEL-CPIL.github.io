import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # safety margin

    # Unit hexagon vertices (angle 0)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([[np.cos(a), np.sin(a)] for a in v_angles])
    
    # 3 unique edge normals (perpendicular to edges, angles 30, 90, 150)
    n_angles = np.deg2rad([30, 90, 150])
    normals = np.array([[np.cos(a), np.sin(a)] for a in n_angles])
    
    # 12 edge normals (6 edges, each with 2 signs) - but for separation we only need 3 unique directions
    # because |proj| on n and -n give same magnitude
    
    def compute_side(pts):
        """Compute the minimum side length of enclosing regular hexagon centered at origin, angle 0."""
        # The outer hexagon has vertices at side * [cos(60k), sin(60k)] for k=0..5
        # For a point p to be inside, for each edge from v_k to v_{k+1}, 
        # cross(v_{k+1}-v_k, p-v_k) >= 0 (using outward normals)
        # 
        # Alternatively: a point p is inside hexagon of side s iff
        # for each of the 3 normal directions n_k, |p dot n_k| <= s * (hex_verts dot n_k)_max
        # 
        # max projection of unit hex vertices on normal at 30 deg:
        # vertices at 0,60,...: cos(0-30)=cos(30)=sqrt(3)/2, cos(60-30)=cos(30)=sqrt(3)/2, etc.
        # Actually: vertex at angle a, normal at angle b: cos(a-b)
        # For normal at 30: cos(0-30)=cos(30)=0.866, cos(60-30)=0.866, cos(120-30)=cos(90)=0, 
        # cos(180-30)=cos(150)=-0.866, etc.
        # So max |proj| = sqrt(3)/2
        # Thus for hexagon of side s: max |p dot n_k| <= s * sqrt(3)/2
        # So s >= 2 * max_k |p dot n_k| / sqrt(3)
        
        max_proj = 0.0
        for k in range(3):
            n_k = normals[k]
            projs = np.abs(pts @ n_k)
            max_proj_k = np.max(projs)
            if max_proj_k > max_proj:
                max_proj = max_proj_k
        return 2.0 * max_proj / np.sqrt(3.0)

    def check_separation(pts):
        """Check if all pairs of unit hexagons at positions pts are separated by > 1e-6 gap."""
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                # Project on the 3 normal directions
                proj_v = np.abs(v @ normals.T)
                # Width of unit hexagon along each normal: 2 * max|hex_verts dot n|
                # For normal at 30 deg: max|cos(angle_diff)| where angle_diff in {0,60,...}
                # = sqrt(3)/2, so width = 2 * sqrt(3)/2 = sqrt(3)
                # Gap on normal k = proj_v[k] - sqrt(3)
                # We need gap > 1e-6 on at least one normal
                gaps = proj_v - np.sqrt(3)
                if np.max(gaps) <= 1e-6:
                    return False
        return True

    def check_containment(pts, side):
        """Check if all unit hexagons at pts are inside outer hexagon of given side."""
        outer_verts = side * hex_verts
        for i in range(11):
            for v_idx in range(6):
                p = pts[i] + hex_verts[v_idx]
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    def check_validity(pts, side):
        """Full validity check."""
        if not np.all(np.isfinite(pts)):
            return False
        if not np.isfinite(side):
            return False
        if side < np.sqrt(11) - 1e-9:
            return False
        if not check_separation(pts):
            return False
        if not check_containment(pts, side):
            return False
        return True

    def objective(x):
        """Objective: minimize enclosing hexagon side + penalty for overlaps."""
        pts = x.reshape(11, 2)
        s = compute_side(pts)
        
        # Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < np.sqrt(3) + 1e-4:
                    depth = (np.sqrt(3) + 1e-4) - max_proj_v
                    penalty += depth * depth
        
        return s + 100.0 * penalty

    # Initial guess: hexagonal arrangement
    d = np.sqrt(3)
    pts = np.array([
        [0, 0],
        [d, 0], [-d/2, d*np.sqrt(3)/2], [-d/2, -d*np.sqrt(3)/2],
        [d/2, d*np.sqrt(3)/2], [d/2, -d*np.sqrt(3)/2], [-d, 0],
        [2*d, 0], [-d, d*np.sqrt(3)], [-d, -d*np.sqrt(3)],
        [0, d*np.sqrt(3)*2], [0, -d*np.sqrt(3)*2]
    ])
    
    x0 = pts.flatten()
    best_x = x0.copy()
    best_val = objective(x0)
    best_side = compute_side(best_x.reshape(11, 2))
    
    # Try multiple restarts with different initial configurations
    np.random.seed(42)
    
    # Restart 0: original
    if time.time() < deadline - 100:
        res = minimize(objective, x0, method='Powell', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Restart 1: perturbed
    if time.time() < deadline - 80:
        x_start = x0 + np.random.normal(0, 0.1, x0.shape)
        res = minimize(objective, x_start, method='Powell', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Restart 2: different initial config - more spread out
    if time.time() < deadline - 60:
        pts2 = np.array([
            [0, 0],
            [1.7, 0], [-0.85, 1.47], [-0.85, -1.47],
            [0.85, 1.47], [0.85, -1.47], [-1.7, 0],
            [3.0, 0], [-2.55, 2.2], [-2.55, -2.2],
            [0, 3.3], [0, -3.3]
        ])
        x_start = pts2.flatten()
        res = minimize(objective, x_start, method='Powell', 
                      options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Restart 3: Nelder-Mead on best so far
    if time.time() < deadline - 40:
        res = minimize(objective, best_x, method='Nelder-Mead', 
                      options={'maxiter': 3000, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    pts_final = best_x.reshape(11, 2)
    
    # Post-processing: resolve any remaining tiny overlaps
    for _ in range(100):
        if time.time() > deadline - 10:
            break
        valid_sep = True
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts_final[i] - pts_final[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < np.sqrt(3) + 1e-6:
                    valid_sep = False
                    min_idx = np.argmin(proj_v)
                    n = normals[min_idx]
                    dot = np.dot(v, n)
                    push = 0.005
                    if dot > 0:
                        pts_final[i] += push * n
                        pts_final[j] -= push * n
                    else:
                        pts_final[i] -= push * n
                        pts_final[j] += push * n
        
        if valid_sep:
            break
    
    # Compute final side
    final_side = compute_side(pts_final)
    
    # Verify validity; if containment fails, expand side
    if not check_containment(pts_final, final_side):
        for expansion in [1.005, 1.01, 1.02, 1.05]:
            test_side = final_side * expansion
            if check_containment(pts_final, test_side):
                final_side = test_side
                break
    
    # Final separation check
    if not check_separation(pts_final):
        # Try to fix by expanding slightly
        for scale in [1.001, 1.002, 1.005]:
            pts_test = pts_final * scale
            if check_separation(pts_test):
                pts_final = pts_test
                final_side = compute_side(pts_final)
                break
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, final_side
