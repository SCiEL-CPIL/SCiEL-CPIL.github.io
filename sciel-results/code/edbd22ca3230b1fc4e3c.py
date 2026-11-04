import math
import time

N = 11

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    axial_positions = [
        (-1, 0), (0, 0), (1, 0),
        (-1, 1), (0, 1), (1, 1), (2, 1),
        (-1, 2), (0, 2), (1, 2), (2, 2),
    ]
    
    centers = []
    for q, r in axial_positions:
        x = sqrt3 * (q + r / 2.0)
        y = 1.5 * r
        centers.append((x, y))
    
    best_S = float('inf')
    best_center = (0.0, 0.0)
    best_theta_deg = 0.0
    
    for theta_deg in range(0, 60, 5):
        if time.time() > deadline:
            break
        theta = math.radians(theta_deg)
        phis = [theta + math.radians(30 + 60 * k) for k in range(6)]
        
        xs = [x for x, y in centers]
        ys = [y for x, y in centers]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        max_range = max(x_max - x_min, y_max - y_min)
        
        n_grid = 12
        best_S_local = float('inf')
        best_cx, best_cy = (x_min + x_max) / 2, (y_min + y_max) / 2
        
        for gi in range(n_grid + 1):
            if time.time() > deadline:
                break
            cx = x_min - max_range * 0.3 + (x_max - x_min + 0.6 * max_range) * gi / n_grid
            for gj in range(n_grid + 1):
                cy = y_min - max_range * 0.3 + (y_max - y_min + 0.6 * max_range) * gj / n_grid
                
                max_val = 0.0
                for (px, py) in centers:
                    dx = px - cx
                    dy = py - cy
                    for phi in phis:
                        val = dx * math.cos(phi) + dy * math.sin(phi)
                        if val > max_val:
                            max_val = val
                
                S_local = (2.0 / sqrt3) * max_val
                if S_local < best_S_local:
                    best_S_local = S_local
                    best_cx, best_cy = cx, cy
        
        if best_S_local < best_S:
            best_S = best_S_local
            best_center = (best_cx, best_cy)
            best_theta_deg = float(theta_deg)
    
    # Refine
    theta = math.radians(best_theta_deg)
    phis = [theta + math.radians(30 + 60 * k) for k in range(6)]
    
    cx0, cy0 = best_center
    xs = [x for x, y in centers]
    ys = [y for x, y in centers]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    max_range = max(x_max - x_min, y_max - y_min)
    
    n_fine = 25
    best_S_refined = best_S
    best_cx_f, best_cy_f = cx0, cy0
    
    for gi in range(n_fine + 1):
        if time.time() > deadline:
            break
        cx = cx0 - max_range * 0.2 + max_range * 0.4 * gi / n_fine
        for gj in range(n_fine + 1):
            cy = cy0 - max_range * 0.2 + max_range * 0.4 * gj / n_fine
            
            max_val = 0.0
            for (px, py) in centers:
                dx = px - cx
                dy = py - cy
                for phi in phis:
                    val = dx * math.cos(phi) + dy * math.sin(phi)
                    if val > max_val:
                        max_val = val
            
            S_local = (2.0 / sqrt3) * max_val
            if S_local < best_S_refined:
                best_S_refined = S_local
                best_cx_f, best_cy_f = cx, cy
    
    best_S = best_S_refined
    best_center = (best_cx_f, best_cy_f)
    
    inner = [[x, y, 0.0] for x, y in centers]
    outer_center = best_center
    outer_side = best_S
    outer_rotation = best_theta_deg
    
    return inner, outer_center, outer_side, outer_rotation

