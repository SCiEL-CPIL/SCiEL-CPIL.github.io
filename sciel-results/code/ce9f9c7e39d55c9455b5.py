import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0
    
    # Hexagon geometry constants
    d = math.sqrt(3)
    
    # Generate centers for 1, 6, 12 rings
    centers = [(0.0, 0.0)]
    
    # Ring 1: 6 neighbors
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))
        
    # Ring 2: 12 neighbors
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))
                
    # Precompute vertices for a unit hexagon (pointy-top)
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    # Precompute base points for the first 7 centers (center + ring 1)
    base_centers = np.array(centers[:7])
    base_points = (base_centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]).reshape(-1, 2)
    
    # Precompute ring 2 center points
    ring2_centers_arr = np.array(ring2_centers)
    ring2_points = (ring2_centers_arr[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]).reshape(-1, 2)
    
    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    
    # Lattice Search to find top candidates
    all_combinations = list(itertools.combinations(range(12), 4))
    candidates = [] # List of (S, config, theta)
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        combo_indices = np.array(combo)
        combo_points = ring2_points[combo_indices * 6 : (combo_indices + 1) * 6]
        points = np.vstack([base_points, combo_points])
        
        def calc_S(theta):
            angles = theta + np.arange(6) * (math.pi / 3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = points @ normals.T
            max_proj = np.max(np.abs(projs))
            return (2.0 / math.sqrt(3)) * max_proj

        # Coarse sweep
        coarse_thetas = np.linspace(0, math.pi / 3, 100)
        coarse_Ss = [calc_S(t) for t in coarse_thetas]
        min_idx = np.argmin(coarse_Ss)
        best_coarse_theta = coarse_thetas[min_idx]
        best_coarse_S = coarse_Ss[min_idx]
        
        # Local refinement
        try:
            res = minimize(calc_S, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_coarse_S:
                best_local_S = res.fun
                best_local_theta = res.x
            else:
                best_local_S = best_coarse_S
                best_local_theta = best_coarse_theta
        except Exception:
            best_local_S = best_coarse_S
            best_local_theta = best_coarse_theta
            
        candidates.append((best_local_S, combo, best_local_theta))
        
        if best_local_S < best_S:
            best_S = best_local_S
            best_config = combo
            best_theta = best_local_theta

    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Sort candidates by S
    candidates.sort(key=lambda x: x[0])
    
    # Take top 3 candidates for local optimization
    top_candidates = candidates[:3]
    
    best_final_S = best_S
    best_final_centers = None
    best_final_theta = best_theta
    
    # Precompute verts_rel for optimization
    # We assume inner hexagons are rotated by 0.
    
    def objective(v):
        # v: [x0, y0, ..., x10, y10, theta]
        centers_arr = np.array(v[:22]).reshape(11, 2)
        theta = v[22]
        
        # Overlap penalty
        # Min distance between centers must be >= sqrt(3)
        penalty = 0.0
        for i in range(11):
            for j in range(i+1, 11):
                dist = np.linalg.norm(centers_arr[i] - centers_arr[j])
                if dist < math.sqrt(3):
                    penalty += (math.sqrt(3) - dist) ** 2
        
        if penalty > 1e-6:
            return 100.0 + penalty # Return large value if overlapping
            
        # Compute S
        all_points = np.empty((66, 2))
        for i in range(11):
            all_points[i*6:(i+1)*6] = centers_arr[i] + verts_rel
            
        angles = theta + np.arange(6) * (math.pi / 3)
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        projs = all_points @ normals.T
        max_proj = np.max(np.abs(projs))
        S = (2.0 / math.sqrt(3)) * max_proj
        
        return S

    for S_init, combo, theta_init in top_candidates:
        if time.time() > deadline - safety_margin:
            break
            
        # Construct initial centers
        init_centers = []
        for cx, cy in centers[:7]:
            init_centers.append((cx, cy))
        for idx in combo:
            cx, cy = ring2_centers[idx]
            init_centers.append((cx, cy))
            
        v0 = []
        for cx, cy in init_centers:
            v0.append(cx)
            v0.append(cy)
        v0.append(theta_init)
        
        try:
            res = minimize(objective, v0, method='Powell', options={'maxiter': 2000, 'maxfev': 20000, 'xatol': 1e-6, 'fatol': 1e-8})
            if res.success or res.fun < best_final_S:
                if res.fun < best_final_S:
                    best_final_S = res.fun
                    # Extract best centers
                    best_centers_arr = np.array(res.x[:22]).reshape(11, 2)
                    best_final_theta = res.x[22]
                    # Store centers
                    best_final_centers = best_centers_arr
        except Exception:
            pass

    # Format output
    if best_final_centers is not None:
        inner_rows = []
        for i in range(11):
            cx, cy = best_final_centers[i]
            inner_rows.append([float(cx), float(cy), 0.0])
        center = (0.0, 0.0)
        side = float(best_final_S)
        angle_deg = float(math.degrees(best_final_theta))
        return inner_rows, center, side, angle_deg
    else:
        # Fallback to lattice result
        inner_rows = []
        for cx, cy in centers[:7]:
            inner_rows.append([float(cx), float(cy), 0.0])
        for idx in best_config:
            cx, cy = ring2_centers[idx]
            inner_rows.append([float(cx), float(cy), 0.0])
            
        center = (0.0, 0.0)
        side = float(best_S)
        angle_deg = float(math.degrees(best_theta))
        return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
