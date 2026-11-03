import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar

N = 11

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0  # Keep a bit more margin for safety

    # Hexagon geometry constants
    d = math.sqrt(3)  # distance between centers of adjacent unit hexagons

    # Generate centers: center + ring 1 (6) + ring 2 (12)
    centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = d * math.cos(angle)
        y = d * math.sin(angle)
        centers.append((x, y))

    # Ring 2: 12 hexagons
    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = d * (q + r / 2.0)
                y = d * (math.sqrt(3) / 2.0) * r
                ring2_centers.append((x, y))

    # Base centers (center + ring 1)
    base_centers = centers[:7]

    # Precompute vertices relative to center (unit hexagon, pointy-top)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    # Precompute base points (center + ring 1 vertices)
    base_points = []
    for cx, cy in base_centers:
        for vx, vy in verts_rel:
            base_points.append((cx + vx, cy + vy))
    base_points_arr = np.array(base_points)  # shape (42, 2)

    # Precompute ring 2 points for each of the 12 positions
    ring2_points_list = []
    for cx, cy in ring2_centers:
        pts = []
        for vx, vy in verts_rel:
            pts.append((cx + vx, cy + vy))
        ring2_points_list.append(np.array(pts))  # each shape (6, 2)

    # All combinations of 4 from 12 ring 2 positions
    all_combinations = list(itertools.combinations(range(12), 4))

    # Function to compute S for a given points array and theta
    def calc_S(points_arr, theta):
        # points_arr: (N_pts, 2)
        # theta: scalar
        k_angles = np.arange(6) * math.pi / 3
        angles = theta + k_angles
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (6, 2)
        
        # dots: (6, N_pts)
        dots = np.sum(normals[:, np.newaxis, :] * points_arr[np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        # max over points: (6,)
        max_per_normal = np.max(abs_dots, axis=-1)
        # max over normals: scalar
        max_proj = np.max(max_per_normal)
        
        return (2.0 / math.sqrt(3)) * max_proj

    best_S = float('inf')
    best_combo = None
    best_theta = 0.0
    top_candidates = [] # List of (S_coarse, combo)

    # Coarse sweep to identify top candidates
    # We can skip the full 495 if we are time-pressed, but 495 is small.
    n_coarse = 100
    theta_coarse = np.linspace(0, math.pi / 3, n_coarse, endpoint=False)
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
        
        # Build points array for this combination
        combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])  # (66, 2)
        
        # Vectorized coarse sweep
        # theta_coarse: (100,)
        k_angles = np.arange(6) * math.pi / 3
        angles = theta_coarse[:, np.newaxis] + k_angles[np.newaxis, :]
        normals = np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # (100, 6, 2)
        
        # dots: (100, 6, 66)
        dots = np.sum(normals[:, :, np.newaxis, :] * combo_points[np.newaxis, np.newaxis, :, :], axis=-1)
        abs_dots = np.abs(dots)
        
        # max over points: (100, 6)
        max_per_normal = np.max(abs_dots, axis=-1)
        # max over normals: (100,)
        max_proj = np.max(max_per_normal, axis=-1)
        S_vals = (2.0 / math.sqrt(3)) * max_proj
        
        min_idx = np.argmin(S_vals)
        min_S = S_vals[min_idx]
        min_theta = theta_coarse[min_idx]
        
        top_candidates.append((min_S, combo, min_theta))
        
        if min_S < best_S:
            best_S = min_S
            best_combo = combo
            best_theta = min_theta

    # Sort candidates by S
    top_candidates.sort(key=lambda x: x[0])
    
    # Refine the top 10 candidates using scipy.optimize.minimize_scalar
    refined_best_S = float('inf')
    refined_best_combo = None
    refined_best_theta = 0.0
    
    if time.time() < deadline - safety_margin:
        for S_coarse, combo, theta_init in top_candidates[:10]:
            if time.time() > deadline - safety_margin:
                break
                
            combo_points = np.vstack([base_points_arr] + [ring2_points_list[i] for i in combo])
            
            # Define objective function for minimize_scalar
            # It expects a scalar x
            def obj(theta):
                return calc_S(combo_points, theta)
                
            # Use bounded method for reliability
            # The function is periodic with period pi/3
            # We can search in [0, pi/3]
            try:
                res = minimize_scalar(obj, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-6})
                if res.success:
                    S_opt = res.fun
                    theta_opt = res.x
                    if S_opt < refined_best_S:
                        refined_best_S = S_opt
                        refined_best_combo = combo
                        refined_best_theta = theta_opt
            except Exception:
                continue

    # If refinement improved, use it
    if refined_best_combo is not None:
        best_S = refined_best_S
        best_combo = refined_best_combo
        best_theta = refined_best_theta

    # Fallback
    if best_combo is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Build final center list
    final_centers = list(base_centers)
    for idx in best_combo:
        final_centers.append(ring2_centers[idx])

    # Format output
    inner_rows = []
    for cx, cy in final_centers:
        inner_rows.append([cx, cy, 0.0])

    center = (0.0, 0.0)
    side = best_S
    angle_deg = math.degrees(best_theta)

    return inner_rows, center, side, angle_deg
