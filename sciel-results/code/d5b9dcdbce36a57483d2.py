import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        # Fallback: return a valid, though not optimal, configuration
        # Simple grid packing
        pts = []
        # 3x3 grid + 2 extra
        for q in range(3):
            for r in range(3):
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                pts.append([x, y, 0.0])
        # Add 2 more
        pts.append([math.sqrt(3) * (0 + 3 / 2.0), 1.5 * 3, 0.0])
        pts.append([math.sqrt(3) * (3 + 3 / 2.0), 1.5 * 3, 0.0])
        # Center
        cx = np.mean([p[0] for p in pts])
        cy = np.mean([p[1] for p in pts])
        # Estimate side
        max_dist = max(math.hypot(p[0]-cx, p[1]-cy) + 1.0 for p in pts)
        side = (max_dist + APOTHEM) / APOTHEM
        return pts, [float(cx), float(cy)], float(side), 0.0

    # 1. Generate diverse lattice topologies
    # Axial coordinates (q, r)
    # x = sqrt(3)*(q + r/2), y = 1.5*r
    base_configs = [
        # "Flower" + 5
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)],
        # "Triangle" 3x3 + 2
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)],
        # "Line" variations
        [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)],
        # "Compact"
        [(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)],
        # "Diagonal"
        [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)],
        # "Star"
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (1,2)],
        # "Cross"
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (1,2), (-1,2), (0,3)],
        # "Block"
        [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2), (1,3)],
        # "L-Shape"
        [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (2,2), (0,3), (1,3), (0,4), (1,4)],
        # "T-Shape"
        [(0,0), (1,0), (2,0), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3), (3,3)],
        # "Pentagon-like"
        [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,1), (1,2), (-1,2)],
        # "H-Shape"
        [(0,0), (1,0), (0,1), (1,1), (0,2), (1,2), (2,0), (2,1), (2,2), (3,1), (-1,1)],
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # 2. Precompute geometry for inner hexagons (rotation 0)
    # Vertices of unit hexagon
    angles = np.arange(6) * (math.pi / 3.0)
    unit_hex_verts = np.array([
        [math.cos(a), math.sin(a)] for a in angles
    ])
    # Normals to sides of inner hexagon (outward pointing)
    # Side normals are at angles 30, 90, 150, ...
    inner_normals = np.array([
        [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)] 
        for k in range(6)
    ])
    # Support function of inner hexagon in direction of its own normals is APOTHEM
    # But we need the vertices to check containment in the OUTER hexagon?
    # No, the LP formulation below handles containment via half-planes of the OUTER hexagon.
    # The inner hexagons are just rigid bodies.
    # Wait, the LP variables are cx, cy, S.
    # Constraints:
    # 1. Non-overlap: For each pair (i, j), and for each normal n of hex i (rotated 0),
    #    (c_j - c_i) . n >= APOTHEM.
    #    Since inner hexagons have fixed relative positions (from axial coords), c_j - c_i is fixed.
    #    So we just check if the fixed configuration is valid. If not, skip this topology.
    #    Actually, axial coords guarantee non-overlap for aligned hexagons.
    #    Distance between centers in axial (1,0) is sqrt(3).
    #    (c_j - c_i) . n_max = sqrt(3).
    #    We need sqrt(3) >= APOTHEM = sqrt(3)/2. Yes.
    #    So any valid axial configuration is non-overlapping.
    
    # 2. Containment: For each inner hex i, and for each normal m of the OUTER hexagon (rotated by beta),
    #    max_{v in inner_i} (v + c_i) . m <= c_outer . m + S * APOTHEM
    #    AND
    #    min_{v in inner_i} (v + c_i) . m >= c_outer . m - S * APOTHEM
    
    # Let's simplify.
    # The outer hexagon is defined by 6 half-planes.
    # H_k: x . n_k <= c_outer . n_k + D, where D = S * APOTHEM.
    # H_k': x . n_k >= c_outer . n_k - D.
    # n_k = (cos(beta + pi/6 + k*pi/3), sin(beta + pi/6 + k*pi/3)).
    
    # For a specific inner hexagon i with center c_i and vertices v_i (relative to c_i):
    # The condition is:
    # max_{v in v_i} (v + c_i) . n_k <= c_outer . n_k + D
    # min_{v in v_i} (v + c_i) . n_k >= c_outer . n_k - D
    
    # This is linear in c_outer and D.
    # c_outer . n_k - D <= min_i,k (min_{v in v_i} (v + c_i) . n_k)
    # c_outer . n_k + D >= max_i,k (max_{v in v_i} (v + c_i) . n_k)
    
    # Let L_k = min_i (min_{v in v_i} (v + c_i) . n_k)
    # Let U_k = max_i (max_{v in v_i} (v + c_i) . n_k)
    
    # Constraints:
    # c_outer . n_k - D <= L_k  =>  c_outer . n_k - L_k <= D
    # c_outer . n_k + D >= U_k  =>  -(c_outer . n_k) + U_k <= D
    
    # So for each k:
    # D >= c_outer . n_k - L_k
    # D >= -(c_outer . n_k) + U_k
    
    # We want to minimize S = D / APOTHEM.
    # Variables: cx, cy, D.
    # Minimize D (scaled).
    
    def solve_for_topology(pts, beta):
        # pts: (11, 2) array of inner centers
        # beta: outer rotation angle
        
        # Compute normals for outer hexagon
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3), math.sin(beta + math.pi/6 + k*math.pi/3)] 
            for k in range(6)
        ])
        
        # Precompute max/min projections of inner hexagons onto these normals
        # For each inner hex i, vertices are pts[i] + unit_hex_verts
        # We need max/min of (pts[i] + v) . n_k for all i, v, k
        
        # (11, 6, 6) -> proj[i, v, k]
        # Let's vectorize
        # pts: (11, 2)
        # unit_hex_verts: (6, 2)
        # normals: (6, 2)
        
        # Expand pts to (11, 6, 2)
        pts_exp = pts[:, np.newaxis, :] # (11, 1, 2)
        verts_exp = unit_hex_verts[np.newaxis, :, :] # (1, 6, 2)
        # Sum: (11, 6, 2)
        all_verts = pts_exp + verts_exp
        
        # Project onto normals: (11, 6, 6)
        # all_verts @ normals.T -> (11, 6, 6)
        projs = all_verts @ normals.T
        
        # L_k = min over i, v of projs[i, v, k]
        L = np.min(projs, axis=(0, 1)) # (6,)
        # U_k = max over i, v of projs[i, v, k]
        U = np.max(projs, axis=(0, 1)) # (6,)
        
        # LP Variables: x = [cx, cy, D]
        # Minimize D (index 2)
        c = np.array([0.0, 0.0, 1.0])
        
        # Constraints: A_ub @ x <= b_ub
        # For each k in 0..5:
        # 1. cx*n_k[0] + cy*n_k[1] - D <= L_k
        #    cx*n_k[0] + cy*n_k[1] - D - L_k <= 0
        # 2. -cx*n_k[0] - cy*n_k[1] + D <= U_k
        #    -cx*n_k[0] - cy*n_k[1] + D - U_k <= 0
        
        A_ub = []
        b_ub = []
        
        for k in range(6):
            nk = normals[k]
            # Constraint 1
            row1 = np.array([nk[0], nk[1], -1.0])
            b1 = L[k]
            A_ub.append(row1)
            b_ub.append(b1)
            
            # Constraint 2
            row2 = np.array([-nk[0], -nk[1], 1.0])
            b2 = U[k]
            A_ub.append(row2)
            b_ub.append(b2)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        
        # Bounds
        # D >= 0
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D_opt = res.x[2]
                cx_opt = res.x[0]
                cy_opt = res.x[1]
                S_opt = D_opt / APOTHEM
                return S_opt, cx_opt, cy_opt
            else:
                return float('inf'), 0, 0
        except:
            return float('inf'), 0, 0

    best_S = float('inf')
    best_result = None
    
    # Strategy:
    # 1. For each topology, find the best beta.
    #    We can scan beta from 0 to 60 degrees (symmetry).
    #    Or use minimize_scalar on beta.
    
    for cfg in base_configs:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(cfg)
        
        # Define objective for beta
        def obj_beta(beta):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e100
            S, cx, cy = solve_for_topology(pts, beta)
            return S
        
        # Scan a few betas to find a good starting point
        # Symmetry period is 60 deg.
        # Try 0, 15, 30, 45
        best_local_S = float('inf')
        best_local_beta = 0.0
        
        # Coarse scan
        for b in np.linspace(0, math.pi/3, 13):
            S, _, _ = solve_for_topology(pts, b)
            if S < best_local_S:
                best_local_S = S
                best_local_beta = b
                
        # Refine with minimize_scalar
        try:
            res_beta = minimize_scalar(obj_beta, bounds=(0, math.pi/3), method='bounded', 
                                       options={'xatol': 1e-6})
            if res_beta.success and res_beta.fun < best_local_S:
                best_local_S = res_beta.fun
                best_local_beta = res_beta.x
        except:
            pass
        
        # Get final cx, cy for best beta
        if best_local_S < best_S:
            S_final, cx_final, cy_final = solve_for_topology(pts, best_local_beta)
            if S_final < best_S:
                best_S = S_final
                best_result = (pts, cx_final, cy_final, best_local_beta)

    if best_result is None:
        # Fallback to first config with beta=0
        pts = axial_to_cartesian(base_configs[0])
        S, cx, cy = solve_for_topology(pts, 0.0)
        if S == float('inf'):
            return [[0.0, 0.0, 0.0]], [0.0, 0.0], 10.0, 0.0
        best_result = (pts, cx, cy, 0.0)
        best_S = S

    pts_opt, cx_opt, cy_opt, beta_opt = best_result
    
    # Add tiny epsilon to S to ensure validity
    final_S = best_S * (1.0 + 1e-9)
    
    inner = []
    for x, y in pts_opt:
        inner.append([float(x), float(y), float(0.0)])
        
    center = [float(cx_opt), float(cy_opt)]
    side = float(final_S)
    rotation = float(math.degrees(beta_opt))
    
    return inner, center, side, rotation

