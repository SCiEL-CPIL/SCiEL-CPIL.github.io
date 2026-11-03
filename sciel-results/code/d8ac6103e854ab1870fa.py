import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 140.0
    
    sqrt3 = math.sqrt(3)
    
    # Precompute unit hexagon vertices
    unit_verts = np.array([
        [1.0, 0.0],
        [0.5, sqrt3/2],
        [-0.5, sqrt3/2],
        [-1.0, 0.0],
        [-0.5, -sqrt3/2],
        [0.5, -sqrt3/2]
    ])
    
    def support_unit(n_vec):
        return np.max(unit_verts @ n_vec)
        
    def calc_S(positions, alpha_deg, beta_deg):
        # positions: numpy array of shape (12, 2)
        # alpha_deg: rotation of inner hexagons
        # beta_deg: rotation of outer hexagon
        alpha_rad = math.radians(alpha_deg)
        beta_rad = math.radians(beta_deg)
        
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30) + beta_rad + k * math.radians(60)
            n = np.array([math.cos(normal_angle), math.sin(normal_angle)])
            
            # Rotate n by -alpha
            c = math.cos(-alpha_rad)
            s = math.sin(-alpha_rad)
            n_rot = np.array([c*n[0] - s*n[1], s*n[0] + c*n[1]])
            
            h_unit_val = support_unit(n_rot)
            
            # Project all positions
            projs = positions @ n + h_unit_val
            max_p_proj = np.max(projs)
            
            if max_p_proj > max_proj:
                max_proj = max_p_proj
                
        return max_proj * 2.0 / sqrt3

    def objective(params):
        # params: 24 positions + 1 alpha + 1 beta = 26
        pos = params[:24].reshape(12, 2)
        alpha = params[24]
        beta = params[25]
        
        S = calc_S(pos, alpha, beta)
        
        # Overlap penalty
        # Min center distance for disjoint unit hexagons is sqrt(3)
        # We use a slightly larger bound to be safe or just sqrt(3)
        d_min = sqrt3
        penalty = 0.0
        for i in range(12):
            for j in range(i+1, 12):
                dist = np.linalg.norm(pos[i] - pos[j])
                if dist < d_min:
                    penalty += (d_min - dist) ** 2
                    
        # Large multiplier to enforce non-overlap
        return S + 100.0 * penalty

    # Initial configurations
    configs = []
    
    # Config 1: Hexagonal cluster (center + 6 + 5)
    # Center (0,0)
    # Ring 1: distance sqrt(3)
    # Ring 2: distance 2*sqrt(3) roughly
    hex_cluster = []
    hex_cluster.append((0, 0))
    for k in range(6):
        angle = math.radians(60 * k)
        hex_cluster.append((sqrt3 * math.cos(angle), sqrt3 * math.sin(angle)))
    # Add 5 more around the ring
    for k in range(5):
        angle = math.radians(60 * k + 30)
        # Distance for ring 2 in hex packing is 2*sqrt(3)
        # But we want them to fit in a hexagon.
        # Let's just place them at distance 2*sqrt(3) from center
        hex_cluster.append((2*sqrt3 * math.cos(angle), 2*sqrt3 * math.sin(angle)))
    # Center the cluster
    xs = [p[0] for p in hex_cluster]
    ys = [p[1] for p in hex_cluster]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    hex_cluster = [(x-cx, y-cy) for x, y in hex_cluster]
    configs.append(hex_cluster)
    
    # Config 2: 3x4 Parallelogram
    para = []
    for q in range(3):
        for r in range(4):
            x = sqrt3 * (q + r/2.0)
            y = sqrt3 * (r * 0.5) # Wait, axial to cartesian for hex grid
            # Axial: x = s(q+r/2), y = s(r*sqrt3/2)
            # If s=1, x = q+r/2, y = r*sqrt3/2
            # Distance between neighbors is sqrt( (1)^2 + (sqrt3/2)^2 ) = sqrt(1+3/4) = sqrt(7/4) ? No.
            # Hex grid distance is 1 if side is 1? No, center distance is sqrt(3) for flat-to-flat.
            # Let's use the standard hex grid spacing.
            # Distance between centers of touching hexagons is sqrt(3).
            # Axial coords: q, r.
            # x = sqrt(3) * (q + r/2)
            # y = 1.5 * r  (since 3/2 * 1 = 1.5)
            # Let's re-derive.
            # If side length is 1, width across flats is sqrt(3).
            # So x-step for q is sqrt(3).
            # y-step for r is 1.5.
            # x-offset for r is sqrt(3)/2.
            x = sqrt3 * (q + r/2.0)
            y = 1.5 * r
            para.append((x, y))
    xs = [p[0] for p in para]
    ys = [p[1] for p in para]
    cx = (min(xs) + max(xs)) / 2.0
    cy = (min(ys) + max(ys)) / 2.0
    para = [(x-cx, y-cy) for x, y in para]
    configs.append(para)
    
    best_S = float('inf')
    best_params = None
    
    for pos_init in configs:
        if time.time() > deadline:
            break
            
        # Flatten positions
        pos_flat = np.array(pos_init).flatten()
        # Add alpha, beta
        # Try a few initial angles
        for a0 in [0, 30, 15]:
            for b0 in [0, 30, 15]:
                if time.time() > deadline:
                    break
                x0 = np.append(pos_flat, [a0, b0])
                try:
                    res = minimize(objective, x0, method='Nelder-Mead', options={'maxiter': 2000, 'xatol': 1e-4, 'fatol': 1e-4})
                    if res.fun < best_S:
                        # Check if penalty is small (valid packing)
                        pos_res = res.x[:24].reshape(12, 2)
                        # Quick check for overlaps
                        valid = True
                        for i in range(12):
                            for j in range(i+1, 12):
                                if np.linalg.norm(pos_res[i] - pos_res[j]) < sqrt3 * 0.99:
                                    valid = False
                                    break
                            if not valid: break
                        if valid:
                            best_S = res.fun
                            best_params = res.x
                except:
                    pass

    if best_params is None:
        # Fallback to a simple valid configuration if optimization failed
        pos_init = configs[0]
        pos_flat = np.array(pos_init).flatten()
        best_params = np.append(pos_flat, [0, 0])
        best_S = calc_S(np.array(pos_init), 0, 0)

    # Extract final result
    pos_final = best_params[:24].reshape(12, 2)
    alpha = best_params[24]
    beta = best_params[25]
    
    inner = []
    for i in range(12):
        inner.append([pos_final[i][0], pos_final[i][1], alpha])
        
    return inner, (0.0, 0.0), best_S, beta
