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
    
    # Threshold for non-overlap: separation must be > 1e-6
    # We use a slightly larger threshold for optimization to ensure robustness
    threshold = np.sqrt(3) + 1e-4

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

    def check_validity(pts, side):
        """Check if the packing is valid according to the task contract."""
        # All values finite
        if not np.all(np.isfinite(pts)):
            return False
        if not np.isfinite(side):
            return False
        if side < np.sqrt(11):
            return False
        
        # Check separation between all pairs of inner hexagons
        for i in range(11):
            for j in range(i + 1, 11):
                v = pts[i] - pts[j]
                # Project on all 12 edge normals (3 directions, 2 signs each)
                # Actually, we need to check if projected intervals are apart by > 1e-6
                # Two hexagons are separated if on at least one normal, the gap > 1e-6
                # The gap on normal n is: |proj_i - proj_j| - (width_i/2 + width_j/2)
                # For unit hexagons, width along normal n is 2 * max_v |v dot n|
                # But we need to check if intervals are "apart" meaning non-overlapping with gap > 1e-6
                
                # For hexagons with same orientation (angle=0), the width along any of the 3 normals
                # is 2*sqrt(3) (since max projection of vertices on normal at 30 deg is sqrt(3))
                # Wait, let me reconsider. The hexagon has vertices at angles 0,60,...,300.
                # Projection on normal at angle 30: cos(30)=sqrt(3)/2, so max proj = sqrt(3)
                # So width = 2*sqrt(3)
                # Gap = |proj_i - proj_j| - 2*sqrt(3)
                # We need gap > 1e-6 on at least one of the 3 normals (since the 12 normals come in ± pairs)
                
                proj_v = v @ normals.T
                gaps = np.abs(proj_v) - 2 * np.sqrt(3)
                if np.max(gaps) <= 1e-6:
                    return False
        
        # Check that all inner hexagons are inside the outer hexagon
        # Outer hexagon has side length 'side', centered at origin, angle 0
        # Vertices of outer hex: side * [cos(60k), sin(60k)] for k=0..5
        outer_verts = side * hex_verts
        for i in range(11):
            for v in range(6):
                p = pts[i] + hex_verts[v]
                for e in range(6):
                    p1 = outer_verts[e]
                    p2 = outer_verts[(e + 1) % 6]
                    # Check cross(p2 - p1, p - p1) >= -1e-6
                    cross = (p2[0] - p1[0]) * (p[1] - p1[1]) - (p2[1] - p1[1]) * (p[0] - p1[0])
                    if cross < -1e-6:
                        return False
        return True

    def objective_and_penalty(x):
        pts = x.reshape(11, 2)
        
        # Calculate S_req
        S_req = compute_side(pts)
        
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

    # Initial guess: 1 center + 6 ring + 4 spikes
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
    
    # Optimization with multiple restarts
    np.random.seed(42)
    for restart in range(3):
        if time.time() > deadline - 120:
            break
            
        if restart == 0:
            x_start = x0
        else:
            x_start = x0 + np.random.normal(0, 0.05, x0.shape)
        
        res = minimize(objective_and_penalty, x_start, method='Powell', 
                      options={'maxiter': 1500, 'xatol': 1e-8, 'fatol': 1e-8})
        
        if res.fun < best_val:
            best_x = res.x
            best_val = res.fun
            best_side = compute_side(best_x.reshape(11, 2))
    
    pts_final = best_x.reshape(11, 2)
    
    # Final validation and potential repair
    if not check_validity(pts_final, best_side):
        # Try to repair by slightly expanding the side
        for expansion in [1.001, 1.002, 1.005, 1.01]:
            test_side = best_side * expansion
            if check_validity(pts_final, test_side):
                best_side = test_side
                break
    
    # Recalculate S for final result
    final_side = compute_side(pts_final)
    if final_side > best_side:
        best_side = final_side
    
    inner_hex_data = np.zeros((11, 3))
    inner_hex_data[:, 0] = pts_final[:, 0]
    inner_hex_data[:, 1] = pts_final[:, 1]
    inner_hex_data[:, 2] = 0
    
    outer_hex_data = np.array([0, 0, 0])
    
    return inner_hex_data, outer_hex_data, best_side
