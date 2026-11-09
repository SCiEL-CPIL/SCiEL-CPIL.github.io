import numpy as np
import time
from scipy.optimize import minimize

def hexagon_packing_11():
    start_time = time.time()
    deadline = start_time + 580  # safety margin
    
    # Normals for the hexagon edges (angles 30, 90, 150)
    angles = np.deg2rad([30, 90, 150])
    normals = np.array([
        [np.cos(a), np.sin(a)] for a in angles
    ])
    
    # Vertices of a unit hexagon (side 1)
    v_angles = np.deg2rad([0, 60, 120, 180, 240, 300])
    hex_verts = np.array([
        [np.cos(a), np.sin(a)] for a in v_angles
    ])
    
    # Threshold for non-overlap
    threshold = np.sqrt(3) + 1e-5

    def compute_side(pts):
        """Compute the minimum side length of enclosing hexagon centered at origin."""
        max_proj_all = 0.0
        for k in range(3):
            n_k = normals[k]
            pts_proj = pts @ n_k
            verts_proj = hex_verts @ n_k
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            if max_abs_sum > max_proj_all:
                max_proj_all = max_abs_sum
        return (2.0 / np.sqrt(3.0)) * max_proj_all

    def objective_and_penalty(x):
        pts = x.reshape(11, 2)
        
        # Compute required side length
        max_proj_all = 0.0
        for k in range(3):
            n_k = normals[k]
            pts_proj = pts @ n_k
            verts_proj = hex_verts @ n_k
            sums = pts_proj[:, np.newaxis] + verts_proj[np.newaxis, :]
            max_abs_sum = np.max(np.abs(sums))
            if max_abs_sum > max_proj_all:
                max_proj_all = max_abs_sum
                
        S_req = (2.0 / np.sqrt(3.0)) * max_proj_all
        
        # Calculate Penalty for overlaps
        penalty = 0.0
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < threshold:
                    depth = threshold - max_proj_v
                    penalty += depth * depth
                    
        return S_req + 100.0 * penalty

    def is_valid(pts):
        """Check if the configuration is valid (no overlaps, all inside)."""
        # Check overlaps
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                proj_v = np.abs(v @ normals.T)
                max_proj_v = np.max(proj_v)
                if max_proj_v < threshold - 1e-6:
                    return False
        
        # Check all inside outer hexagon
        side = compute_side(pts)
        if side < np.sqrt(11) - 1e-6:
            return False
        
        return True

    # Initial guess: symmetric arrangement
    d = np.sqrt(3)
    pts = []
    pts.append([0, 0])
    for k in range(6):
        angle = np.deg2rad(60 * k)
        pts.append([d * np.cos(angle), d * np.sin(angle)])
        
    spike_angles = [0, 120, 180, 300]
    for angle_deg in spike_angles:
        angle = np.deg2rad(angle_deg)
        pts.append([2 * d * np.cos(angle), 2 * d * np.sin(angle)])
        
    x0 = np.array(pts).flatten()
    
    best_x = x0.copy()
    best_val = objective_and_penalty(x0)
    best_side = compute_side(best_x.reshape(11, 2))
    
    # Try multiple optimization runs
    rng = np.random.default_rng(42)
    
    # First run: standard Powell
    if time.time() < deadline - 120:
        res = minimize(objective_and_penalty, x0, method='Powell', 
                      options={'maxiter': 3000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res.fun < best_val:
            best_x = res.x.copy()
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Second run: perturbed start
    if time.time() < deadline - 120:
        x0_pert = x0 + rng.normal(0, 0.05, x0.shape)
        res2 = minimize(objective_and_penalty, x0_pert, method='Powell', 
                       options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res2.fun < best_val:
            best_x = res2.x.copy()
            best_val = res2.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Third run: different perturbation
    if time.time() < deadline - 60:
        x0_pert2 = x0 + rng.normal(0, 0.1, x0.shape)
        res3 = minimize(objective_and_penalty, x0_pert2, method='Powell', 
                       options={'maxiter': 1500, 'xatol': 1e-10, 'fatol': 1e-10})
        if res3.fun < best_val:
            best_x = res3.x.copy()
            best_val = res3.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Fourth run: try Nelder-Mead
    if time.time() < deadline - 60:
        res4 = minimize(objective_and_penalty, best_x, method='Nelder-Mead',
                       options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
        if res4.fun < best_val:
            best_x = res4.x.copy()
            best_val = res4.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    # Final local refinement
    if time.time() < deadline - 30:
        res5 = minimize(objective_and_penalty, best_x, method='Powell',
                       options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
        if res5.fun < best_val:
            best_x = res5.x.copy()
            best_val = res5.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    pts_final = best_x.reshape(11, 2)
    side = compute_side(pts_final)
    
    # Ensure side is at least sqrt(11)
    if side < np.sqrt(11):
        side = np.sqrt(11)
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0.0, 0.0, 0.0])
    
    return inner_hex_data, outer_hex_data, float(side)
