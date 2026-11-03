import math
import numpy as np
import time
from scipy.optimize import linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = np.array([
            [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
            for k in range(6)
        ])
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                A_ub.append([nx, ny, -1.0])
                b_ub.append(px * nx + py * ny)
        
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0, None)]
        
        try:
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                D = res.x[2]
                cx = res.x[0]
                cy = res.x[1]
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, None
        except Exception:
            return float('inf'), None, None

    def check_overlap(pts):
        normals = np.array([
            [math.cos(math.pi/6 + k*math.pi/3), math.sin(math.pi/6 + k*math.pi/3)]
            for k in range(6)
        ])
        n_pts = len(pts)
        for i in range(n_pts):
            for j in range(i+1, n_pts):
                d = pts[i] - pts[j]
                projs = np.abs(d @ normals.T)
                if np.min(projs) < math.sqrt(3) - 1e-9:
                    return False
        return True

    def simulate(pts0, max_iters=20000):
        pts = pts0.copy()
        n = len(pts)
        dt = 0.05
        temp = 1.0
        cooling = 0.9998
        k_rep = 10.0
        k_att = 0.01
        target_dist = math.sqrt(3) * 1.05
        
        # Compute initial center of mass
        com = np.mean(pts, axis=0)
        
        for it in range(max_iters):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return pts
            
            # Compute forces
            forces = np.zeros_like(pts)
            
            # Repulsion between all pairs
            for i in range(n):
                for j in range(i+1, n):
                    d = pts[i] - pts[j]
                    dist = np.linalg.norm(d)
                    if dist < 1e-6:
                        # Random push if exactly overlapping
                        d = np.random.randn(2)
                        dist = np.linalg.norm(d)
                    
                    # Repulsive force
                    if dist < 2.0:
                        f_mag = k_rep / (dist * dist + 0.1)
                        f = f_mag * d / dist
                        forces[i] += f
                        forces[j] -= f
                    
                    # Weak attractive force to keep cluster together
                    if dist > target_dist:
                        f_mag = k_att * (dist - target_dist)
                        f = f_mag * d / dist
                        forces[i] -= f
                        forces[j] += f
            
            # Centering force
            current_com = np.mean(pts, axis=0)
            forces -= 0.1 * (current_com - com)
            
            # Update positions
            pts += dt * forces
            pts += np.random.randn(n, 2) * temp * 0.1
            temp *= cooling
        
        return pts

    def optimize_shape(pts_init):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts_init
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None, None
        
        # Find best beta for initial configuration
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            S, center, D = calc_S_for_beta(pts_init, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                from scipy.optimize import minimize_scalar
                res = minimize_scalar(lambda b: calc_S_for_beta(pts_init, b)[0], bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 1000, 'xatol': 1e-16})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts_init, best_beta)
        except Exception:
            pass
        
        return best_S, best_beta, best_center, best_pts

    # Generate diverse initial configurations
    np.random.seed(42)
    candidates = []
    
    # Hexagonal spiral
    axial_spiral = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, -2)
    ]
    candidates.append(axial_to_cartesian(axial_spiral))
    
    # Random cluster
    pts_random = np.random.randn(N, 2) * 2.0
    candidates.append(pts_random)
    
    # Another random
    pts_random2 = np.random.randn(N, 2) * 1.5
    candidates.append(pts_random2)
    
    # Compact random
    angles = np.linspace(0, 2*np.pi, N, endpoint=False)
    radii = np.random.uniform(0.5, 1.5, N)
    pts_compact = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
    candidates.append(pts_compact)
    
    # Grid-like
    grid = []
    for i in range(4):
        for j in range(3):
            if len(grid) < N:
                grid.append((i, j))
    candidates.append(axial_to_cartesian(grid))
    
    # Run simulation on each candidate
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    
    for pts0 in candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        # Run physics simulation
        pts_sim = simulate(pts0, max_iters=10000)
        
        # Optimize outer hexagon
        S, beta, center, optimized_pts = optimize_shape(pts_sim)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = optimized_pts
            best_global_beta = beta
            best_global_center = center
        
        # Early stop if very good
        if best_global_S < 3.5:
            break
    
    # Fallback
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        max_proj = 0
        for k in range(6):
            ang = math.pi / 6.0 + k * math.pi / 3.0
            nx, ny = math.cos(ang), math.sin(ang)
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        best_global_S = (max_proj + APOTHEM) / APOTHEM
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    
    pts_scaled = np.array(pts_scaled)
    
    max_proj = 0
    for k in range(6):
        ang = best_global_beta + math.pi / 6.0 + k * math.pi / 3.0
        nx, ny = math.cos(ang), math.sin(ang)
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, np.max(projs), -np.min(projs))
    
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
