import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 15.0
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

    # Generate initial candidates
    candidates = []
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])

    def check_overlap(pts, min_dist=2.0 * APOTHEM):
        """
        Checks for overlap. 
        min_dist is the minimum distance for aligned hexes.
        For rotated hexes, we use a conservative check first.
        """
        n = len(pts)
        for i in range(n):
            for j in range(i + 1, n):
                d = np.linalg.norm(pts[i] - pts[j])
                if d < min_dist - 1e-6:
                    return False
        return True

    def exact_S(pts, cx, cy, beta):
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pc = pts - np.array([cx, cy])
        projs = pc @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        best_pts = pts.copy()
        
        # Stage 1: Rigid Body Optimization (Fast)
        def rigid_obj(params):
            cx, cy, beta = params
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            return exact_S(pts, cx, cy, beta)

        # Initial center
        c_init = np.mean(pts, axis=0)
        
        # Try a few rotations
        for b0 in [0, math.pi/12, math.pi/6/2]:
            init = [c_init[0], c_init[1], b0]
            try:
                res = minimize(rigid_obj, init, method='Nelder-Mead', options={'maxiter': 1000, 'xatol': 1e-9})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            except Exception:
                pass

        if best_center is None:
            return float('inf'), 0.0, None, pts

        # Stage 2: Local Search on Individual Hexagons
        # This allows the hexagons to move relative to each other to tighten the packing.
        current_pts = pts.copy()
        current_S = best_S
        current_center = np.array(best_center)
        current_beta = best_beta
        
        # We will perturb positions and angles.
        # To keep it simple, we fix the angles to 0 initially (aligned with lattice) 
        # and only move positions. Then we might try rotating individual hexes.
        # But the interface requires returning angles. 
        # If we move hexes, their relative angles might change if we rotate them.
        # Let's assume all hexes have the same angle beta (rigid body) for the local search 
        # but allow their positions to vary. This is a "soft" rigid body.
        
        def soft_rigid_obj(params):
            # params: 2*N + 3 (x1, y1, ..., xN, yN, cx, cy, beta)
            n_pts = len(pts)
            positions = params[:2*n_pts].reshape(n_pts, 2)
            cx = params[2*n_pts]
            cy = params[2*n_pts+1]
            beta = params[2*n_pts+2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
            
            # Check overlap
            for i in range(n_pts):
                for j in range(i+1, n_pts):
                    d = np.linalg.norm(positions[i] - positions[j])
                    if d < 2.0 * APOTHEM - 1e-5:
                        return 1e10 + (2.0 * APOTHEM - d)**2
            
            return exact_S(positions, cx, cy, beta)

        # Initialize for soft rigid search
        init_params = current_pts.flatten().tolist() + [current_center[0], current_center[1], current_beta]
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(soft_rigid_obj, init_params, method='Nelder-Mead', 
                               options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
                if res.fun < 1e9 and res.fun < current_S:
                    current_S = res.fun
                    current_pts = res.x[:2*N].reshape(N, 2)
                    current_center = np.array([res.x[2*N], res.x[2*N+1]])
                    current_beta = ((res.x[2*N+2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        return current_S, current_beta, (float(current_center[0]), float(current_center[1])), current_pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, axial in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        pts = axial_to_cartesian(axial)
        S, beta, center, opt_pts = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = opt_pts if opt_pts is not None else pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
        c, D = center_and_D(best_global_pts, 0.0) # Need to define center_and_D or use mean
        best_global_center = (float(np.mean(best_global_pts, axis=0)[0]), float(np.mean(best_global_pts, axis=0)[1]))
        best_global_beta = 0.0
        best_global_S = exact_S(best_global_pts, best_global_center[0], best_global_center[1], 0.0)

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Scaling to ensure no floating point errors cause overlap
    SCALE = 1.0 + 1e-14
    pts_scaled = best_global_pts.copy()
    for i in range(len(pts_scaled)):
        pts_scaled[i, 0] = cx + (pts_scaled[i, 0] - cx) * SCALE
        pts_scaled[i, 1] = cy + (pts_scaled[i, 1] - cy) * SCALE

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
