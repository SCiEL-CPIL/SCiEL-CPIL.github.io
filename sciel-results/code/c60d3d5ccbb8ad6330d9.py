import math
import numpy as np
import time
from scipy.optimize import minimize, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
SQRT3 = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 12.0

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Candidate axial configurations ---
    # Each candidate is a list of (q, r) axial coordinates for N=11 unit hexagons.
    # We provide a diverse set of starting packings.

    def axial_to_cartesian(axial_list):
        pts = np.zeros((len(axial_list), 2))
        for i, (q, r) in enumerate(axial_list):
            pts[i, 0] = SQRT3 * (q + r / 2.0)
            pts[i, 1] = 1.5 * r
        return pts

    # Generate multiple diverse candidates
    candidates_axial = []

    # 1. Standard hexagonal cluster (radius 2 minus corners)
    hex_cluster = [(0,0)]
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if abs(q) <= 2 and abs(r) <= 2 and abs(s) <= 2:
                if (q, r) != (0, 0):
                    hex_cluster.append((q, r))
    # This gives 19 hexagons, we need 11. Take a subset.
    # Let's just use predefined good-looking packings.

    # Candidate 1: Central + 6 around + 4 more
    c1 = [(0,0), (1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1), (2,0),(0,2),(-2,2),(2,-2)]
    candidates_axial.append(c1)

    # Candidate 2: Row-based
    c2 = [(0,0),(1,0),(2,0),(3,0),
          (0,1),(1,1),(2,1),
          (0,2),(1,2),(2,2),
          (1,3)]
    candidates_axial.append(c2)

    # Candidate 3: Diagonal
    c3 = [(0,0), (1,0), (2,0),
          (0,1), (1,1), (2,1),
          (0,2), (1,2), (2,2),
          (-1,1), (3,1)]
    candidates_axial.append(c3)

    # Candidate 4: Compact 3x3 + 2
    c4 = [(0,0),(1,0),(2,0),
          (0,1),(1,1),(2,1),
          (0,2),(1,2),(2,2),
          (-1,1),(3,1)]
    candidates_axial.append(c4)

    # Candidate 5: Another row variation
    c5 = [(0,0),(1,0),
          (0,1),(1,1),(2,1),
          (0,2),(1,2),(2,2),
          (0,3),(1,3),(2,3)]
    candidates_axial.append(c5)

    # Candidate 6: Staggered
    c6 = [(0,0),(1,0),(2,0),(3,0),
          (1,1),(2,1),(3,1),
          (0,2),(1,2),(2,2),(3,2)]
    candidates_axial.append(c6)

    # Candidate 7: Cross-like
    c7 = [(0,0),
          (1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),
          (2,0),(-2,0),(0,2),(1,2)]
    candidates_axial.append(c7)

    # Candidate 8: L-shape
    c8 = [(0,0),(1,0),(2,0),(3,0),
          (0,1),
          (0,2),
          (0,3),
          (0,4),
          (1,1),(1,2),(1,3)]
    candidates_axial.append(c8)

    # Candidate 9: T-shape
    c9 = [(0,0),(1,0),(2,0),(3,0),
          (1,1),(2,1),
          (1,2),(2,2),
          (1,3),(2,3)]
    candidates_axial.append(c9)

    # Candidate 10: Dense
    c10 = [(0,0),(1,0),(2,0),
           (0,1),(1,1),(2,1),
           (0,2),(1,2),(2,2),
           (1,3),(2,3)]
    candidates_axial.append(c10)

    # Generate cartesian points for each candidate
    candidate_pts = []
    for ax in candidates_axial:
        if len(ax) == N:
            pts = axial_to_cartesian(ax)
            candidate_pts.append(pts)

    if not candidate_pts:
        # Fallback: simple grid
        ax = [(i % 3, i // 3) for i in range(9)] + [(0,3),(1,3)]
        candidate_pts.append(axial_to_cartesian(ax))

    # --- Optimization ---

    def compute_enclosing_hexagon(pts, beta, cx, cy):
        """Compute the required side length S for an enclosing hexagon.
        pts: (N, 2) array of inner hexagon centers
        beta: rotation of outer hexagon
        cx, cy: center of outer hexagon
        Returns: (S, D, max_proj)
        """
        normals = np.array([
            [math.cos(beta + math.pi/6 + k*math.pi/3),
             math.sin(beta + math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        # (6, 2)
        
        pts_c = pts - np.array([cx, cy])  # (N, 2)
        projs = pts_c @ normals.T  # (N, 6)
        
        # For each normal direction, the constraint is:
        # |proj| <= D where D is the apothem of the outer hexagon
        # So D = max over all points and normals of |proj|
        D = np.max(np.abs(projs))
        S = (D + APOTHEM) / APOTHEM
        return S, D, D

    def optimize_single(pts):
        """Optimize position, rotation, and center for a single candidate."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return None
        
        # Initial estimate: centroid, beta=0
        cx0 = float(np.mean(pts[:, 0]))
        cy0 = float(np.mean(pts[:, 1]))
        
        # Try multiple initial betas
        best_result = None
        best_S = float('inf')
        
        beta_inits = [0.0, math.pi/12, math.pi/6, math.pi/18, math.pi/9, 5*math.pi/36]
        
        for beta_init in beta_inits:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            def objective(params):
                cx, cy, beta = params
                beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                S, D, _ = compute_enclosing_hexagon(pts, beta, cx, cy)
                return S
            
            x0 = [cx0, cy0, beta_init]
            
            try:
                res = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-10,
                                      'adaptive': True})
                if res.fun < best_S:
                    best_S = res.fun
                    best_result = (res.x[0], res.x[1], res.x[2])
            except Exception:
                pass
        
        # Also try with optimized center via LP for fixed beta
        for beta_init in [0.0, math.pi/12, math.pi/6]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            # Find optimal center for this beta using LP
            normals = np.array([
                [math.cos(beta_init + math.pi/6 + k*math.pi/3),
                 math.sin(beta_init + math.pi/6 + k*math.pi/3)]
                for k in range(6)
            ])
            
            # Minimize D such that for all points i and normals k:
            # (px - cx) * nx + (py - cy) * ny <= D
            # -(px - cx) * nx - (py - cy) * ny <= D
            # => -nx*px + nx*cx - ny*py + ny*cy <= D
            # => nx*px - nx*cx + ny*py - ny*cy <= D
            
            # Variables: cx, cy, D
            # Constraints: for each (i, k):
            # nx_i * (px_i - cx) + ny_i * (py_i - cy) <= D
            # -nx_i * (px_i - cx) - ny_i * (py_i - cy) <= D
            
            A_ub = []
            b_ub = []
            for k in range(6):
                nx, ny = normals[k]
                for i in range(len(pts)):
                    px, py = pts[i]
                    # nx*px - nx*cx + ny*py - ny*cy - D <= 0
                    A_ub.append([-nx, -ny, -1.0])
                    b_ub.append(-(nx*px + ny*py))
                    # -nx*px + nx*cx - ny*py + ny*cy - D <= 0
                    A_ub.append([nx, ny, -1.0])
                    b_ub.append(nx*px + ny*py)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            c_obj = np.array([0.0, 0.0, 1.0])
            bounds = [(None, None), (None, None), (0, None)]
            
            try:
                res_lp = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res_lp.status == 0:
                    cx_lp = res_lp.x[0]
                    cy_lp = res_lp.x[1]
                    D_lp = res_lp.x[2]
                    S_lp = (D_lp + APOTHEM) / APOTHEM
                    
                    if S_lp < best_S:
                        best_S = S_lp
                        best_result = (cx_lp, cy_lp, beta_init)
            except Exception:
                pass
        
        if best_result is None:
            # Fallback
            S, D, _ = compute_enclosing_hexagon(pts, 0.0, cx0, cy0)
            best_result = (cx0, cy0, 0.0)
            best_S = S
        
        return (best_S, best_result[0], best_result[1], best_result[2])

    # Optimize each candidate
    best_global_S = float('inf')
    best_global_pts = None
    best_global_cx = 0.0
    best_global_cy = 0.0
    best_global_beta = 0.0
    
    for idx, pts in enumerate(candidate_pts):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        result = optimize_single(pts)
        if result is not None:
            S, cx, cy, beta = result
            if S < best_global_S:
                best_global_S = S
                best_global_pts = pts
                best_global_cx = cx
                best_global_cy = cy
                best_global_beta = beta
    
    # If no valid result, use first candidate with default
    if best_global_pts is None:
        best_global_pts = candidate_pts[0]
        best_global_cx = float(np.mean(best_global_pts[:, 0]))
        best_global_cy = float(np.mean(best_global_pts[:, 1]))
        best_global_beta = 0.0
        S, D, _ = compute_enclosing_hexagon(best_global_pts, 0.0, best_global_cx, best_global_cy)
        best_global_S = S

    # Final polish: try to further optimize the best solution
    if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def final_objective(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
            S, D, _ = compute_enclosing_hexagon(best_global_pts, beta, cx, cy)
            return S
        
        x0 = [best_global_cx, best_global_cy, best_global_beta]
        try:
            res = minimize(final_objective, x0, method='Nelder-Mead',
                         options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11,
                                  'adaptive': True})
            if res.fun < best_global_S:
                best_global_S = res.fun
                best_global_cx = res.x[0]
                best_global_cy = res.x[1]
                best_global_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass

    # Compute final side length with a tiny safety margin
    S_final, D_final, _ = compute_enclosing_hexagon(best_global_pts, best_global_beta, best_global_cx, best_global_cy)
    # Add tiny margin to ensure validity
    S_final = S_final + 1e-10

    angle_deg = math.degrees(best_global_beta)
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(best_global_cx), float(best_global_cy)]
    side = float(S_final)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
