import math
import itertools
import time
import numpy as np
from scipy.optimize import minimize_scalar, linprog

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
                
    if len(ring2_centers) != 12:
        pass

    # Precompute vertices for a unit hexagon (pointy-top)
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] 
        for i in range(6)
    ])

    best_S = float('inf')
    best_config = None
    best_theta = 0.0
    best_center = (0.0, 0.0)
    
    all_combinations = list(itertools.combinations(range(12), 4))
    
    for combo in all_combinations:
        if time.time() > deadline - safety_margin:
            break
            
        # Construct the 11 centers
        current_centers = np.array(centers[:7])
        combo_centers = np.array([ring2_centers[idx] for idx in combo])
        current_centers = np.vstack([current_centers, combo_centers])
        
        # Get all vertices for these 11 hexagons
        # Shape: (66, 2)
        points = (current_centers[:, np.newaxis, :] + verts_rel[np.newaxis, :, :]).reshape(-1, 2)
        
        # Function to compute minimal S for a given theta using width heuristic
        def calc_S_width(theta):
            angles = theta + np.arange(6) * (math.pi / 3)
            normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
            
            # Projections: (66, 6)
            projs = points @ normals.T
            max_projs = np.max(projs, axis=0)
            min_projs = np.min(projs, axis=0)
            widths = max_projs - min_projs
            max_width = np.max(widths)
            # S = 2/sqrt(3) * R_in, R_in = max_width / 2
            # S = 1/sqrt(3) * max_width
            return (1.0 / math.sqrt(3)) * max_width

        # Coarse sweep to find a good starting point
        coarse_thetas = np.linspace(0, math.pi / 3, 100)
        coarse_Ss = [calc_S_width(t) for t in coarse_thetas]
        min_idx = np.argmin(coarse_Ss)
        best_coarse_theta = coarse_thetas[min_idx]
        best_coarse_S = coarse_Ss[min_idx]
        
        # Local refinement
        try:
            res = minimize_scalar(calc_S_width, bounds=(0, math.pi/3), method='bounded', options={'xatol': 1e-8})
            if res.success and res.fun < best_coarse_S:
                best_local_S = res.fun
                best_local_theta = res.x
            else:
                best_local_S = best_coarse_S
                best_local_theta = best_coarse_theta
        except Exception:
            best_local_S = best_coarse_S
            best_local_theta = best_coarse_theta
            
        if best_local_S < best_S:
            best_S = best_local_S
            best_config = current_centers
            best_theta = best_local_theta

    if best_config is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    # Refine the best candidate with LP to get exact center and minimal S
    # We use the best_theta found
    theta = best_theta
    angles = theta + np.arange(6) * (math.pi / 3)
    normals = np.column_stack([np.cos(angles), np.sin(angles)]) # (6, 2)
    
    # LP formulation:
    # Minimize R
    # Variables: Cx, Cy, R
    # Constraints: -R <= p . n_k - C . n_k <= R
    # p . n_k - Cx * nx - Cy * ny - R <= 0
    # -p . n_k + Cx * nx + Cy * ny - R <= 0
    
    # c = [0, 0, 1]
    c = np.array([0.0, 0.0, 1.0])
    
    # A_ub and b_ub
    # 66 points * 6 normals * 2 inequalities = 792 rows
    # Each row corresponds to one inequality
    
    A_ub = []
    b_ub = []
    
    for k in range(6):
        nx, ny = normals[k]
        for p in points:
            px, py = p
            proj = px * nx + py * ny
            # Inequality 1: proj - nx*cx - ny*cy - R <= 0
            A_ub.append([nx, ny, 1.0])
            b_ub.append(proj)
            
            # Inequality 2: -proj + nx*cx + ny*cy - R <= 0
            A_ub.append([-nx, -ny, 1.0])
            b_ub.append(-proj)
            
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    
    # Bounds: Cx, Cy can be anything, R >= 0
    bounds = [(None, None), (None, None), (0, None)]
    
    try:
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.success:
            cx, cy, R_opt = res.x
            # S = 2/sqrt(3) * R
            S_opt = (2.0 / math.sqrt(3)) * R_opt
            # Update best if this is better (it should be, as width was a heuristic/lower bound? 
            # Actually width heuristic R = max(W)/2. 
            # LP finds minimal R such that there exists C.
            # The width heuristic assumes C can be chosen independently for each direction, 
            # which is not possible. So LP R >= Width Heuristic R.
            # However, the width heuristic S was used to select theta.
            # We should compare S_opt with the previous best_S.
            # Note: best_S was calculated using width heuristic.
            # If S_opt < best_S, it means the width heuristic overestimated? 
            # No, width heuristic R_w = max(W)/2.
            # LP R_lp is the true minimal R.
            # It is possible that for the optimal theta, the "conflict" between directions is small, 
            # so R_lp is close to R_w.
            # We should accept S_opt if it's valid.
            
            if S_opt < best_S:
                best_S = S_opt
                best_center = (cx, cy)
            else:
                # Even if S_opt is not better than the heuristic estimate for this theta,
                # we should use the valid center for the best configuration found.
                # But wait, if S_opt > best_S (heuristic), it means the heuristic was optimistic.
                # We must return a VALID packing.
                # So we must use S_opt and best_center.
                best_S = S_opt
                best_center = (cx, cy)
                
        else:
            # Fallback to width heuristic center (0,0) and S
            # This might be invalid, but better than nothing?
            # Let's use the midpoint of the bounding box as a guess for center
            # And S from width heuristic
            best_center = (0.0, 0.0)
            
    except Exception:
        best_center = (0.0, 0.0)

    # Format output
    inner_rows = []
    for cx, cy in best_config:
        inner_rows.append([float(cx), float(cy), 0.0])
        
    center = (float(best_center[0]), float(best_center[1]))
    side = float(best_S)
    angle_deg = float(math.degrees(best_theta))
    
    return inner_rows, center, side, angle_deg
# EVOLVE-BLOCK-END
