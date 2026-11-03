import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
D = math.sqrt(3) # Min distance between centers of aligned unit hexagons

def construct():
    start_time = time.time()
    deadline = 120.0
    safety_margin = 5.0

    # 1. Generate Lattice Points
    # Ring 0
    pts = [(0.0, 0.0)]
    # Ring 1
    for i in range(6):
        a = i * math.pi / 3
        pts.append((D * math.cos(a), D * math.sin(a)))
    # Ring 2
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                x = D * (q + r / 2.0)
                y = D * (math.sqrt(3) / 2.0) * r
                pts.append((x, y))
    
    pts = np.array(pts)
    num_pts = len(pts)
    
    # Vertices of a unit hexagon (flat-top, side 1)
    # Center (0,0). Vertices at angles 0, 60, ...
    hex_verts = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])
    
    best_S_global = float('inf')
    best_config_global = None
    best_theta_global = 0.0
    top_candidates = [] # List of (S, centers, theta)

    # 2. Iterate over combinations
    # We need 11 points from num_pts (which is 19)
    # To ensure symmetry or just brute force? Brute force is fine.
    # However, many combinations are equivalent by rotation/reflection.
    # We can just brute force all 75582.
    
    for combo in itertools.combinations(range(num_pts), 11):
        if time.time() > deadline - safety_margin:
            break
            
        centers = pts[list(combo)]
        
        # Function to calculate S for given theta
        def calc_S(theta, centers=centers):
            # Normals
            angles = theta + np.arange(6) * (math.pi / 3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
            
            # Vertices of all hexagons
            # centers: (11, 2)
            # hex_verts: (6, 2)
            # We want (11*6, 2)
            all_verts = (centers[:, np.newaxis, :] + hex_verts[np.newaxis, :, :]).reshape(-1, 2)
            
            # Projections
            projs = np.abs(all_verts @ normals.T) # (66, 6)
            max_proj = np.max(projs)
            return (2.0 / math.sqrt(3)) * max_proj

        # Find best theta for this configuration
        # Coarse scan
        coarse_thetas = np.linspace(0, math.pi / 3, 50)
        coarse_Ss = np.array([calc_S(t) for t in coarse_thetas])
        min_idx = np.argmin(coarse_Ss)
        
        # Refine
        try:
            res = minimize_scalar(calc_S, bounds=(coarse_thetas[max(0, min_idx-1)], coarse_thetas[min(49, min_idx+1)]), method='bounded', options={'xatol': 1e-6})
            if res.success:
                S_val = res.fun
                theta_val = res.x
            else:
                S_val = coarse_Ss[min_idx]
                theta_val = coarse_thetas[min_idx]
        except:
            S_val = coarse_Ss[min_idx]
            theta_val = coarse_thetas[min_idx]

        if S_val < best_S_global:
            best_S_global = S_val
            best_config_global = centers.copy()
            best_theta_global = theta_val
            
        top_candidates.append((S_val, centers.copy(), theta_val))

    # 3. Refine Top Candidates
    top_candidates.sort(key=lambda x: x[0])
    
    # Take top 20
    for S_init, centers_init, theta_init in top_candidates[:20]:
        if time.time() > deadline - safety_margin:
            break
            
        # Variables: x1, y1, ..., x11, y11, theta
        # Total 23 variables
        x0 = np.concatenate([centers_init.flatten(), [theta_init]])
        
        def objective(x):
            c = x[:22].reshape(11, 2)
            theta = x[22]
            return calc_S(theta, c)
            
        # Constraints: distance between centers >= sqrt(3)
        # We need 55 constraints (11*10/2)
        cons = []
        for i in range(11):
            for j in range(i+1, 11):
                cons.append({
                    'type': 'ineq',
                    'fun': lambda x, i=i, j=j: np.dot(x[2*i:2*i+2] - x[2*j:2*j+2], x[2*i:2*i+2] - x[2*j:2*j+2]) - 3.0
                })
                
        try:
            res = minimize(objective, x0, method='SLSQP', bounds=[(-10, 10)]*22 + [(0, math.pi/3)], constraints=cons, options={'maxiter': 100, 'ftol': 1e-8})
            if res.success and res.fun < best_S_global:
                best_S_global = res.fun
                best_config_global = res.x[:22].reshape(11, 2)
                best_theta_global = res.x[22]
        except Exception:
            pass

    # Fallback
    if best_config_global is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Format output
    inner_rows = []
    for cx, cy in best_config_global:
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (0.0, 0.0)
    side = float(best_S_global)
    angle_deg = float(math.degrees(best_theta_global))
    
    return inner_rows, center, side, angle_deg
