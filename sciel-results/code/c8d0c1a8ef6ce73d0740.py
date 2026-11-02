import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 145.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT:
        # Fallback: Return a valid configuration immediately
        # 3x4 grid minus 1
        axial = [(q, r) for r in range(3) for q in range(4)]
        axial = axial[:11]
        pts = axial_to_cartesian(axial)
        # Simple estimate
        S = 4.0 
        center = [float(np.mean(pts[:,0])), float(np.mean(pts[:,1]))]
        inner = [[float(x), float(y), 0.0] for x,y in pts]
        return inner, center, S, 0.0

    # --- Geometry Helpers ---
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def get_normals(beta):
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def solve_lp_for_beta(pts, beta):
        normals = get_normals(beta)
        n_pts = len(pts)
        A_ub = np.zeros((12 * n_pts, 3))
        b_ub = np.zeros(12 * n_pts)
        
        for k in range(6):
            nx, ny = normals[k]
            idx1 = 2 * k * n_pts
            idx2 = idx1 + n_pts
            
            # (px - cx)*nx + (py - cy)*ny <= D
            # -nx*cx - ny*cy - D <= -px*nx - py*ny
            A_ub[idx1:idx1+n_pts, 0] = -nx
            A_ub[idx1:idx1+n_pts, 1] = -ny
            A_ub[idx1:idx1+n_pts, 2] = -1.0
            b_ub[idx1:idx1+n_pts] = -(pts[:, 0] * nx + pts[:, 1] * ny)
            
            # (px - cx)*nx + (py - cy)*ny >= -D
            # nx*cx + ny*cy - D <= px*nx + py*ny
            A_ub[idx2:idx2+n_pts, 0] = nx
            A_ub[idx2:idx2+n_pts, 1] = ny
            A_ub[idx2:idx2+n_pts, 2] = -1.0
            b_ub[idx2:idx2+n_pts] = pts[:, 0] * nx + pts[:, 1] * ny

        c_obj = np.array([0.0, 0.0, 1.0])
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy)
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    def optimize_orientation(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        
        # Coarse Grid: 0 to 60 deg, step 2 deg (30 steps)
        for i in range(30):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            beta = i * (math.pi / 6) / 30
            S, center = solve_lp_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center: best_center = center
                
        # Fine Grid: around best_beta, step 0.5 deg
        lo = max(0.0, best_beta - math.pi / 60)
        hi = min(math.pi / 6, best_beta + math.pi / 60)
        n_fine = 20
        for i in range(n_fine):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            beta = lo + i * (hi - lo) / (n_fine - 1) if n_fine > 1 else best_beta
            S, center = solve_lp_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                if center: best_center = center
                
        return best_S, best_beta, best_center

    # --- Generate Candidates ---
    candidates = []
    
    # 1. 3x4 Block (12) remove 1 corner
    for r in range(3):
        for q in range(4):
            candidates.append((q, r))
    candidates.append((0, 3)) # 12th
    # Take first 11
    candidates.append(candidates[:11])
    
    # 2. 4x3 Block (12) remove 1 corner
    c2 = []
    for r in range(4):
        for q in range(3):
            c2.append((q, r))
    c2.append((0, 4))
    candidates.append(c2[:11])
    
    # 3. 3x3 Block (9) + 2
    c3 = []
    for r in range(3):
        for q in range(3):
            c3.append((q, r))
    c3.append((3, 1))
    c3.append((3, 2))
    candidates.append(c3)
    
    # 4. Hex Cluster (19) take 11 central
    # Center (0,0), 6 neighbors, 6 next ring
    c4 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1)]
    c4 += [(2,0), (2,-1), (1,-2), (0,-2), (-1,-2), (-2,-1), (-2,0), (-2,1), (-1,2), (0,2), (1,2), (2,1)]
    candidates.append(c4[:11])
    
    # 5. 5+6 Rows
    c5 = []
    for q in range(5): c5.append((q, 0))
    for q in range(6): c5.append((q, 1))
    candidates.append(c5)
    
    # 6. 2+3+4+2
    c6 = []
    for q in range(2): c6.append((q, 0))
    for q in range(3): c6.append((q, 1))
    for q in range(4): c6.append((q, 2))
    for q in range(2): c6.append((q, 3))
    candidates.append(c6)
    
    # 7. 1+2+3+2+1+2 (Pyramid)
    c7 = []
    c7.append((0,0))
    for q in range(2): c7.append((q,1))
    for q in range(3): c7.append((q,2))
    for q in range(2): c7.append((q,3))
    c7.append((0,4))
    c7.append((1,4))
    candidates.append(c7)
    
    # 8. 4+4+3
    c8 = []
    for q in range(4): c8.append((q,0))
    for q in range(4): c8.append((q,1))
    for q in range(3): c8.append((q,2))
    candidates.append(c8)
    
    # 9. 3+4+4
    c9 = []
    for q in range(3): c9.append((q,0))
    for q in range(4): c9.append((q,1))
    for q in range(4): c9.append((q,2))
    candidates.append(c9)
    
    # 10. 2+4+5
    c10 = []
    for q in range(2): c10.append((q,0))
    for q in range(4): c10.append((q,1))
    for q in range(5): c10.append((q,2))
    candidates.append(c10)

    # --- Optimization Loop ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if time.time() - START_TIME > TIME_LIMIT:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_orientation(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Finalize ---
    if best_global_pts is None:
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        S, beta, center = optimize_orientation(pts)
        best_global_S = S
        best_global_pts = pts
        best_global_beta = beta
        best_global_center = center

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), 0.0])
        
    center_out = [float(cx), float(cy)]
    side_out = float(best_global_S)
    angle_out = float(angle_deg)
    
    return inner, center_out, side_out, angle_out
