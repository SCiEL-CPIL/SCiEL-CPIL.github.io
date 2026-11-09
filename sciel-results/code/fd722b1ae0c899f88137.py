"""Packing 12 unit hexagons in minimal regular hexagon."""
import math
import time
import numpy as np
from itertools import combinations

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 130.0  # Safety margin before 150s kill
    
    sqrt3 = math.sqrt(3.0)
    
    # Generate all 19 centers of the side-3 hexagon cluster
    # Axial coordinates (q, r) with max(|q|, |r|, |q+r|) <= 2
    all_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q + r) <= 2:
                x = 1.5 * q
                y = sqrt3 * (r + q / 2.0)
                all_centers.append((x, y))
    
    n_centers = len(all_centers)  # 19
    
    # Normal directions for flat-top outer hexagon
    normal_angles = np.array([math.radians(30 + 60 * j) for j in range(6)])
    normal_cos = np.cos(normal_angles)
    normal_sin = np.sin(normal_angles)
    
    def min_L_for_hexagon(cx, cy, theta):
        """Find the minimum L needed for a unit hexagon at (cx,cy) with rotation theta."""
        vertex_angles = np.array([math.radians(60 * k) + theta for k in range(6)])
        vx = cx + np.cos(vertex_angles)
        vy = cy + np.sin(vertex_angles)
        
        projs = vx[:, np.newaxis] * normal_cos[np.newaxis, :] + vy[:, np.newaxis] * normal_sin[np.newaxis, :]
        max_proj = np.max(projs)
        return max_proj / (sqrt3 / 2.0)
    
    def min_L_for_all_hexagons(centers, thetas):
        """Find the max L needed for all hexagons."""
        max_L = 0.0
        for i, (cx, cy) in enumerate(centers):
            L = min_L_for_hexagon(cx, cy, thetas[i])
            if L > max_L:
                max_L = L
        return max_L
    
    # Small scale to ensure strict non-overlap
    SCALE = 1.000001
    
    # Phase 1: Quick scan of all subsets with default orientation (theta=0)
    # to find the top candidates
    subset_scores = []
    for subset in combinations(range(n_centers), 12):
        if time.time() > deadline:
            break
        centers_sub = [(all_centers[i][0] * SCALE, all_centers[i][1] * SCALE) for i in subset]
        thetas_zero = [0.0] * 12
        L = min_L_for_all_hexagons(centers_sub, thetas_zero)
        subset_scores.append((L, subset))
    
    # Sort by L and take top 50 candidates
    subset_scores.sort(key=lambda x: x[0])
    top_candidates = subset_scores[:50]
    
    if not top_candidates:
        # Fallback
        best_subset = tuple(range(12))
        centers_sub = [(all_centers[i][0] * SCALE, all_centers[i][1] * SCALE) for i in best_subset]
        thetas = [0.0] * 12
        best_L = min_L_for_all_hexagons(centers_sub, thetas)
        inner = [[cx, cy, 0.0] for cx, cy in centers_sub]
        return inner, (0.0, 0.0), best_L, 0.0
    
    # Phase 2: For each top candidate, optimize the rotation angles
    best_L_global = float('inf')
    best_config = None
    
    for rank, (L_init, subset) in enumerate(top_candidates):
        if time.time() > deadline:
            break
        
        centers_sub = [(all_centers[i][0] * SCALE, all_centers[i][1] * SCALE) for i in subset]
        
        # Start with theta=0 for all, then do coordinate descent
        thetas = [0.0] * 12
        current_L = min_L_for_all_hexagons(centers_sub, thetas)
        
        # Coordinate descent: optimize each theta one at a time
        improved = True
        max_iter = 20
        for iteration in range(max_iter):
            if time.time() > deadline:
                break
            improved = False
            for i in range(12):
                # Find best theta for hexagon i, keeping others fixed
                # The L is determined by max over all hexagons
                # We want to minimize max(L_i(theta_i), L_others)
                # Since L_others is fixed, we just need L_i(theta_i) <= current_L
                # But to improve, we want to reduce the max
                
                # Coarse search
                best_theta_i = thetas[i]
                best_L_i = min_L_for_hexagon(centers_sub[i][0], centers_sub[i][1], thetas[i])
                
                # Try a range of angles
                for t_deg in np.arange(0, 60.5, 1.0):
                    t = math.radians(t_deg)
                    L_i = min_L_for_hexagon(centers_sub[i][0], centers_sub[i][1], t)
                    # The overall L is max of all L_j
                    # If L_i is not the bottleneck, changing it doesn't help
                    # But if it is, we want to minimize it
                    # Simple approach: just minimize L_i
                    if L_i < best_L_i:
                        best_L_i = L_i
                        best_theta_i = t
                
                # Refine around best_theta_i
                for step_deg in [0.1, 0.01]:
                    step = math.radians(step_deg)
                    t_start = best_theta_i - step * 5
                    t_end = best_theta_i + step * 5
                    for t in np.linspace(t_start, t_end, 21):
                        L_i = min_L_for_hexagon(centers_sub[i][0], centers_sub[i][1], t)
                        if L_i < best_L_i:
                            best_L_i = L_i
                            best_theta_i = t
                
                if best_theta_i != thetas[i]:
                    thetas[i] = best_theta_i
                    improved = True
            
            # Check if overall L improved
            new_L = min_L_for_all_hexagons(centers_sub, thetas)
            if new_L < current_L:
                current_L = new_L
            else:
                break
            
            if not improved:
                break
        
        if current_L < best_L_global:
            best_L_global = current_L
            best_config = (centers_sub, thetas, subset)
        
        # If we're already well below the initial estimate, we can stop early
        if best_L_global < top_candidates[0][0] * 0.99:
            break
    
    if best_config is None:
        # Fallback to first candidate
        L_init, subset = top_candidates[0]
        centers_sub = [(all_centers[i][0] * SCALE, all_centers[i][1] * SCALE) for i in subset]
        thetas = [0.0] * 12
        best_L_global = min_L_for_all_hexagons(centers_sub, thetas)
        best_config = (centers_sub, thetas, subset)
    
    centers_sub, thetas, subset = best_config
    
    # Phase 3: Final high-precision polish
    # Use a more careful optimization
    for polish_iter in range(5):
        if time.time() > deadline:
            break
        improved = False
        for i in range(12):
            cx, cy = centers_sub[i]
            # Very fine search around current theta
            t0 = thetas[i]
            best_t = t0
            best_L_i = min_L_for_hexagon(cx, cy, t0)
            
            # Local search with decreasing step
            for step_deg in [0.01, 0.001, 0.0001]:
                step = math.radians(step_deg)
                for k in range(-10, 11):
                    t = best_t + k * step
                    L_i = min_L_for_hexagon(cx, cy, t)
                    if L_i < best_L_i:
                        best_L_i = L_i
                        best_t = t
                        improved = True
            
            thetas[i] = best_t
        
        # Recompute overall L
        new_L = min_L_for_all_hexagons(centers_sub, thetas)
        if new_L < best_L_global:
            best_L_global = new_L
    
    # Construct inner hexagons
    inner = []
    for i, (cx, cy) in enumerate(centers_sub):
        inner.append([cx, cy, math.degrees(thetas[i])])
    
    return inner, (0.0, 0.0), best_L_global, 0.0
