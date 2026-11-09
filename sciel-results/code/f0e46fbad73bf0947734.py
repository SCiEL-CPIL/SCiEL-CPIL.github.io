"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    # Using Flat-Top hexagon grid coordinates
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    n_centers = len(all_centers)  # 19
    
    # Normal directions for flat-top outer hexagon (pointing outward from center)
    # Normals at 30, 90, 150, 210, 270, 330 degrees
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta to fit in outer hex of side L."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def optimal_theta_for_hexagon(cx, cy):
        """Find rotation theta that minimizes the required L for this hexagon."""
        best_L = float('inf')
        best_theta = 0.0
        # Coarse search: 0 to 60 degrees in 1 degree steps
        for t_deg in np.arange(0, 60.5, 1.0):
            t = math.radians(t_deg)
            L = min_L_for_hexagon(cx, cy, t)
            if L < best_L:
                best_L = L
                best_theta = t
        # Refine with finer steps
        for step_deg in [0.1, 0.01, 0.001]:
            step = math.radians(step_deg)
            t_start = best_theta - step * 5
            t_end = best_theta + step * 5 + step
            for t_deg in np.arange(math.degrees(t_start), math.degrees(t_end), step_deg):
                t = math.radians(t_deg)
                L = min_L_for_hexagon(cx, cy, t)
                if L < best_L:
                    best_L = L
                    best_theta = t
        return best_L, best_theta
    
    # Minimal scale to ensure strict disjointness
    SCALE = 1.000001
    
    best_global_L = float('inf')
    best_global_subset = None
    best_global_thetas = None
    best_global_phi = 0.0
    
    # Search over global rotation phi of the cluster
    # Symmetry allows 0 to 60 degrees
    # We use 60 steps of 1 degree, then refine the best 2
    phi_coarse = np.arange(0, 60.5, 1.0)
    
    for phi_deg in phi_coarse:
        if time.time() > deadline - 5.0:
            break
            
        phi = math.radians(phi_deg)
        cos_phi = math.cos(phi)
        sin_phi = math.sin(phi)
        
        rotated_centers = []
        for x, y in all_centers:
            rx = x * cos_phi - y * sin_phi
            ry = x * sin_phi + y * cos_phi
            rotated_centers.append((rx, ry))
            
        # Precompute optimal theta for each rotated center
        center_optimal = []
        for i in range(n_centers):
            x, y = rotated_centers[i]
            L_h, theta_h = optimal_theta_for_hexagon(x, y)
            center_optimal.append((L_h, theta_h))
            
        # Find best subset of 12
        best_local_L = float('inf')
        best_local_subset = None
        best_local_thetas = None
        
        for subset in combinations(range(n_centers), 12):
            L = max(center_optimal[i][0] for i in subset)
            if L < best_local_L:
                best_local_L = L
                best_local_subset = subset
                best_local_thetas = [center_optimal[i][1] for i in subset]
                
        if best_local_L < best_global_L:
            best_global_L = best_local_L
            best_global_subset = best_local_subset
            best_global_thetas = best_local_thetas
            best_global_phi = phi
            
    # Refine the best phi found
    if best_global_phi is not None:
        phi_refine_start = best_global_phi - math.radians(1.0)
        phi_refine_end = best_global_phi + math.radians(1.0) + math.radians(0.1)
        phi_refine = np.arange(math.degrees(phi_refine_start), math.degrees(phi_refine_end), 0.1)
        
        for phi_deg in phi_refine:
            if time.time() > deadline - 2.0:
                break
                
            phi = math.radians(phi_deg)
            cos_phi = math.cos(phi)
            sin_phi = math.sin(phi)
            
            rotated_centers = []
            for x, y in all_centers:
                rx = x * cos_phi - y * sin_phi
                ry = x * sin_phi + y * cos_phi
                rotated_centers.append((rx, ry))
                
            center_optimal = []
            for i in range(n_centers):
                x, y = rotated_centers[i]
                L_h, theta_h = optimal_theta_for_hexagon(x, y)
                center_optimal.append((L_h, theta_h))
                
            best_local_L = float('inf')
            best_local_subset = None
            best_local_thetas = None
            
            for subset in combinations(range(n_centers), 12):
                L = max(center_optimal[i][0] for i in subset)
                if L < best_local_L:
                    best_local_L = L
                    best_local_subset = subset
                    best_local_thetas = [center_optimal[i][1] for i in subset]
                    
            if best_local_L < best_global_L:
                best_global_L = best_local_L
                best_global_subset = best_local_subset
                best_global_thetas = best_local_thetas
                best_global_phi = phi

    if best_global_subset is None:
        # Fallback
        best_global_subset = tuple(range(12))
        best_global_thetas = [0.0] * 12
        best_global_L = 10.0 # Large penalty

    # Construct inner hexagons
    inner = []
    cos_phi = math.cos(best_global_phi)
    sin_phi = math.sin(best_global_phi)
    
    for idx, i in enumerate(best_global_subset):
        x, y = all_centers[i]
        rx = x * cos_phi - y * sin_phi
        ry = x * sin_phi + y * cos_phi
        inner.append([rx * SCALE, ry * SCALE, math.degrees(best_global_thetas[idx])])
    
    final_L = best_global_L * SCALE
    
    return inner, (0.0, 0.0), final_L, 0.0
