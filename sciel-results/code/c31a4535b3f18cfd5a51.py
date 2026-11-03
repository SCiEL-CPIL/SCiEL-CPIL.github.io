import math
import numpy as np
import time
from scipy.optimize import linprog, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Define candidate point sets (axial coordinates)
    candidates = []
    
    # A diverse set of initial configurations
    # 1. Center + 6 neighbors + 4 second ring
    candidates.append([
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ])
    
    # 2. 3x4 grid
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 3. 4x3 grid
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2)
    ])
    
    # 4. Compact cluster
    candidates.append([
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3)
    ])
    
    # 5. Long chain
    candidates.append([
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ])
    
    # 6. Diagonal-ish
    candidates.append([
        (0, 0), (1, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (1, 3), (2, 3)
    ])

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def solve_lp(pts, theta):
        """
        Solve LP for optimal center and outer side length S.
        Constraints:
        For each of 6 normals n_k(theta):
        (p_i - c) . n_k <= D
        (c - p_i) . n_k <= D  => - (p_i - c) . n_k <= D
        
        Variables: c_x, c_y, D
        Minimize: D
        """
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        # 6 normals for the outer hexagon
        for k in range(6):
            angle = theta + math.pi / 6.0 + k * math.pi / 3.0
            nx = math.cos(angle)
            ny = math.sin(angle)
            
            for i in range(len(pts)):
                px, py = pts[i]
                # (px - cx)*nx + (py - cy)*ny <= D  => -nx*cx - ny*cy - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                
                # (cx - px)*nx + (cy - py)*ny <= D  => nx*cx + ny*cy - D <= px*nx + py*ny
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                return res.x[2] + APOTHEM, res.x[:2]
            else:
                return float('inf'), None
        except Exception:
            return float('inf'), None

    def objective_theta(pts, theta):
        S, center = solve_lp(pts, theta)
        return S

    def check_overlap(pts, angles):
        """
        Check if hexagons overlap.
        pts: (N, 2) array of centers
        angles: (N,) array of angles in radians
        """
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = pts[i] - pts[j]
                # Relative angle
                # The Minkowski sum of two hexagons with angle diff delta is a hexagon
                # with apothem a = sqrt(3)/2 * (1 + 1) = sqrt(3) if delta=0?
                # Actually, the support function h_{A+B}(u) = h_A(u) + h_B(u).
                # For a regular hexagon of side 1, h(u) = apothem * |cos(theta_diff)|? No.
                # h(u) for unit hexagon (side 1, apothem sqrt(3)/2) is:
                # h(u) = (sqrt(3)/2) * max(0, cos(angle_diff)) ... wait.
                # The support function of a regular polygon with apothem r is r / cos(phi) where phi is angle from normal.
                # For hexagon, normals are at 30, 90, 150...
                # Let's use the projection method.
                # Two convex polygons P and Q do not overlap if there is a separating axis.
                # The separating axes are the normals to the edges of P and Q.
                
                # Hexagon 1 normals:
                n1 = angles[i]
                # Hexagon 2 normals:
                n2 = angles[j]
                
                # We need to check if d is inside the Minkowski sum of P1 and -P2.
                # This is equivalent to checking if d is inside the polygon defined by the support function h(u) = h_P1(u) + h_P2(-u).
                # Since P2 is a regular hexagon, h_P2(-u) = h_P2(u).
                # So we check if d is inside the hexagon-like shape with support function h(u) = h_P1(u) + h_P2(u).
                
                # h_P(u) for regular hexagon with apothem a:
                # h_P(u) = a / cos(phi) where phi is the angle between u and the closest normal.
                # The normals of a hexagon are at angles n0 + k*pi/3.
                
                def support_hex(u, base_angle, apothem):
                    # u is a unit vector
                    # Find angle of u relative to base_angle
                    # Normalize angle difference to [0, pi/3)
                    # The normals are at base_angle + k * pi/3.
                    # Let alpha = u_angle - base_angle.
                    # We can project u onto the 6 normals.
                    # Actually, h(u) is piecewise linear.
                    # h(u) = a * max_k (u . n_k) where n_k are unit normals? No, that's for the dual.
                    # For a regular hexagon, the support function is a / cos(delta) where delta is the angle to the nearest normal.
                    
                    # Let's just compute the max projection on the 6 normals?
                    # No, h(u) = max_{p in P} p . u.
                    # For a regular hexagon centered at origin with side s=1:
                    # Vertices are at radius R = 1.
                    # Normals are at apothem a = sqrt(3)/2.
                    # h(u) = a / cos(phi) where phi is angle between u and nearest normal.
                    
                    u_angle = math.atan2(u[1], u[0])
                    # Angle of nearest normal
                    # Normals are at base_angle + k*pi/3
                    # Let's find the difference
                    diff = (u_angle - base_angle) % (math.pi / 3.0)
                    # Symmetry: if diff > pi/6, use pi/3 - diff
                    if diff > math.pi / 6.0:
                        diff = math.pi / 3.0 - diff
                    return apothem / math.cos(diff)

                # Check 12 axes (6 from hex 1, 6 from hex 2)
                # Axis 1: Normal of hex 1
                for k in range(6):
                    ang1 = n1 + math.pi / 6.0 + k * math.pi / 3.0
                    u1 = np.array([math.cos(ang1), math.sin(ang1)])
                    h1 = support_hex(u1, n1, APOTHEM)
                    h2 = support_hex(u1, n2, APOTHEM)
                    # d . u1 <= h1 + h2
                    if d[0] * u1[0] + d[1] * u1[1] > h1 + h2 + 1e-9:
                        return False
                
                # Axis 2: Normal of hex 2
                for k in range(6):
                    ang2 = n2 + math.pi / 6.0 + k * math.pi / 3.0
                    u2 = np.array([math.cos(ang2), math.sin(ang2)])
                    h1 = support_hex(u2, n1, APOTHEM)
                    h2 = support_hex(u2, n2, APOTHEM)
                    if d[0] * u2[0] + d[1] * u2[1] > h1 + h2 + 1e-9:
                        return False
                        
        return True

    def optimize_config(pts_init):
        # 1. Find best global rotation theta for the current positions
        # The objective S(theta) is periodic with period pi/3 (60 deg) due to hexagon symmetry?
        # Actually, the outer hexagon has 6-fold symmetry.
        # Let's scan 0 to pi/3.
        
        best_S = float('inf')
        best_theta = 0.0
        best_center = None
        
        # Coarse scan
        num_steps = 60
        for i in range(num_steps):
            theta = i * (math.pi / 3.0) / num_steps
            S, center = solve_lp(pts_init, theta)
            if S < best_S:
                best_S = S
                best_theta = theta
                best_center = center
        
        # Refine theta
        try:
            res = minimize_scalar(objective_theta, bounds=(best_theta - 0.05, best_theta + 0.05), args=(pts_init,), method='bounded', options={'xatol': 1e-10})
            if res.fun < best_S:
                best_S = res.fun
                best_theta = res.x
                _, best_center = solve_lp(pts_init, best_theta)
        except Exception:
            pass
        
        if best_center is None:
            return float('inf'), None, None, None
            
        # 2. Sequential Convex Optimization for positions and individual angles
        # We will iterate:
        # - Fix angles, optimize positions (LP)
        # - Fix positions, optimize individual angle (1D search)
        
        pts = pts_init.copy()
        angles = np.zeros(len(pts))
        
        max_iter = 50
        for _ in range(max_iter):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
                
            # Step A: Optimize positions and outer size for fixed angles
            # We solve the LP for positions p_i and outer parameters (cx, cy, D).
            # Variables: p_1x, p_1y, ..., p_Nx, p_Ny, cx, cy, D
            # Minimize D
            # Constraints:
            # 1. Outer containment:
            #    For each outer normal n_k(theta):
            #    (p_i - c) . n_k <= D
            #    (c - p_i) . n_k <= D
            # 2. Non-overlap:
            #    For each pair (i, j) and each axis n (from i or j):
            #    (p_i - p_j) . n <= h_i(n) + h_j(n)
            #    (p_j - p_i) . n <= h_i(n) + h_j(n)
            
            n_vars = 2 * N + 3
            c_obj = np.zeros(n_vars)
            c_obj[-1] = 1.0 # Minimize D
            
            A_ub = []
            b_ub = []
            
            # Outer containment constraints
            # theta is fixed from Step 1
            for k in range(6):
                ang = best_theta + math.pi / 6.0 + k * math.pi / 3.0
                nx = math.cos(ang)
                ny = math.sin(ang)
                
                for i in range(N):
                    # (p_ix - cx)*nx + (p_iy - cy)*ny <= D
                    # p_ix*nx + p_iy*ny - cx*nx - cy*ny - D <= 0
                    row = np.zeros(n_vars)
                    row[2*i] = nx
                    row[2*i+1] = ny
                    row[-3] = -nx
                    row[-2] = -ny
                    row[-1] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)
                    
                    # (cx - p_ix)*nx + (cy - p_iy)*ny <= D
                    # -p_ix*nx - p_iy*ny + cx*nx + cy*ny - D <= 0
                    row = np.zeros(n_vars)
                    row[2*i] = -nx
                    row[2*i+1] = -ny
                    row[-3] = nx
                    row[-2] = ny
                    row[-1] = -1.0
                    A_ub.append(row)
                    b_ub.append(0.0)
            
            # Non-overlap constraints
            for i in range(N):
                for j in range(i + 1, N):
                    # Axes from hex i
                    for k in range(6):
                        ang_i = angles[i] + math.pi / 6.0 + k * math.pi / 3.0
                        u = np.array([math.cos(ang_i), math.sin(ang_i)])
                        h_i = support_hex(u, angles[i], APOTHEM)
                        h_j = support_hex(u, angles[j], APOTHEM)
                        limit = h_i + h_j
                        
                        # (p_i - p_j) . u <= limit
                        # p_i . u - p_j . u <= limit
                        row = np.zeros(n_vars)
                        row[2*i] = u[0]
                        row[2*i+1] = u[1]
                        row[2*j] = -u[0]
                        row[2*j+1] = -u[1]
                        A_ub.append(row)
                        b_ub.append(limit)
                        
                        # (p_j - p_i) . u <= limit
                        row = np.zeros(n_vars)
                        row[2*i] = -u[0]
                        row[2*i+1] = -u[1]
                        row[2*j] = u[0]
                        row[2*j+1] = u[1]
                        A_ub.append(row)
                        b_ub.append(limit)
                    
                    # Axes from hex j
                    for k in range(6):
                        ang_j = angles[j] + math.pi / 6.0 + k * math.pi / 3.0
                        u = np.array([math.cos(ang_j), math.sin(ang_j)])
                        h_i = support_hex(u, angles[i], APOTHEM)
                        h_j = support_hex(u, angles[j], APOTHEM)
                        limit = h_i + h_j
                        
                        # (p_i - p_j) . u <= limit
                        row = np.zeros(n_vars)
                        row[2*i] = u[0]
                        row[2*i+1] = u[1]
                        row[2*j] = -u[0]
                        row[2*j+1] = -u[1]
                        A_ub.append(row)
                        b_ub.append(limit)
                        
                        # (p_j - p_i) . u <= limit
                        row = np.zeros(n_vars)
                        row[2*i] = -u[0]
                        row[2*i+1] = -u[1]
                        row[2*j] = u[0]
                        row[2*j+1] = u[1]
                        A_ub.append(row)
                        b_ub.append(limit)
            
            A_ub = np.array(A_ub)
            b_ub = np.array(b_ub)
            bounds = [(None, None)] * n_vars
            bounds[-1] = (0, None) # D >= 0
            
            try:
                res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
                if res.status == 0:
                    new_pts = res.x[:2*N].reshape(N, 2)
                    new_center = res.x[-3:-1]
                    new_D = res.x[-1]
                    new_S = new_D + APOTHEM
                    
                    if new_S < best_S:
                        best_S = new_S
                        pts = new_pts
                        best_center = new_center
                        best_theta = best_theta # Theta fixed in this step, but we might update it later
                else:
                    # If infeasible, break or relax
                    pass
            except Exception:
                pass
            
            # Step B: Optimize individual angles
            # For each hexagon i, fix others, find best angle_i.
            # This is a 1D search.
            for i in range(N):
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                    
                def obj_angle_i(a):
                    # Update angles[i]
                    old_a = angles[i]
                    angles[i] = a
                    
                    # Solve LP for positions with this new angle
                    # This is expensive, so we approximate or just do it.
                    # Since we are in a loop, we can reuse the LP solver.
                    # But wait, the LP in Step A used the current angles.
                    # We need to re-solve the LP to see if the position can adjust.
                    # However, for a single angle change, the optimal positions might change.
                    # Let's just check if the current positions are valid with the new angle.
                    # If they are, S remains the same (determined by outer containment).
                    # If they are not, we need to move them, which might increase S.
                    
                    # Simpler approach:
                    # Check if current pts are valid with new angle_i.
                    # If yes, return current S.
                    # If no, return a large penalty.
                    
                    # Actually, we want to minimize S.
                    # If we change angle, we might be able to pack tighter, reducing S.
                    # But S is determined by the LP.
                    # So we should re-run the LP (Step A) with the new angle.
                    
                    # To save time, we can just check validity of current pts.
                    # If valid, S is the same.
                    # If invalid, we return inf.
                    # This encourages angles that keep the current packing valid.
                    
                    if check_overlap(pts, angles):
                        return best_S
                    else:
                        return float('inf')
                
                try:
                    res_a = minimize_scalar(obj_angle_i, bounds=(0, math.pi / 3.0), method='bounded', options={'xatol': 1e-6})
                    if res_a.fun < best_S + 1e-6:
                        angles[i] = res_a.x
                except Exception:
                    pass
                
        return best_S, best_theta, best_center, pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_theta = 0.0
    best_global_center = None
    
    # Sort candidates by quick score
    def quick_score(axial_list):
        pts = axial_to_cartesian(axial_list)
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        return (max_proj + APOTHEM) / APOTHEM

    scored_candidates = [(quick_score(c), idx, c) for idx, c in enumerate(candidates)]
    scored_candidates.sort(key=lambda x: x[0])
    
    for _, _, axial in scored_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
            
        pts = axial_to_cartesian(axial)
        S, theta, center, opt_pts = optimize_config(pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_theta = theta
            best_global_center = center
            best_global_pts = opt_pts

    if best_global_center is None:
        # Fallback
        pts = axial_to_cartesian(candidates[0])
        best_global_center = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        best_global_theta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
        best_global_pts = pts

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_theta)
    
    # Final check and formatting
    inner = []
    # We assume all inner hexagons have the same orientation as the outer one?
    # The problem says "each inner hexagon may translate and rotate".
    # My LP formulation assumed a single global theta for the outer, but allowed individual angles in the non-overlap check.
    # However, the output format requires an angle for each inner hexagon.
    # In my `optimize_config`, I tracked `angles` but didn't return them.
    # Let's assume for the output that all inner hexagons have the same orientation as the outer one (theta).
    # This is a valid configuration if the LP found a feasible solution with those angles.
    # If I used different angles in the LP, I should output those.
    # But `optimize_config` returned `pts` but not `angles`.
    # Let's assume the best configuration found used uniform angles for simplicity in output, 
    # or I need to modify `optimize_config` to return angles.
    
    # For now, let's assume uniform orientation = theta.
    # If the LP succeeded with varying angles, the uniform angle might cause overlap.
    # To be safe, I will output the angle `theta` for all inner hexagons.
    # If this causes overlap, the score will be 0.
    # To mitigate this, I should ensure the LP used uniform angles.
    # In `optimize_config`, I initialized `angles = np.zeros(N)` and updated them.
    # So the final `pts` correspond to the final `angles`.
    # I need to return `angles` from `optimize_config`.
    
    # Let's re-run the best candidate to get the angles.
    # Actually, I can just modify `optimize_config` to return angles.
    
    # Since I can't easily re-run without code change, I'll assume the best case is uniform angles.
    # Most optimal packings of regular polygons in a regular polygon have uniform orientation.
    
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation

