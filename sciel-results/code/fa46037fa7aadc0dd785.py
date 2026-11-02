import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def compute_enclosing_hexagon(pts, beta):
        """
        Compute the side length of the minimal enclosing regular hexagon
        with rotation beta for the given points.
        """
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        cx_min, cx_max = float('inf'), float('-inf')
        cy_min, cy_max = float('inf'), float('-inf')
        max_proj = 0.0
        
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            p_min = np.min(projs)
            p_max = np.max(projs)
            max_proj = max(max_proj, (p_max - p_min) / 2.0)
        
        # The center is the average of min and max projections
        cx = 0.0
        cy = 0.0
        for nx, ny in normals:
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            cx += (np.min(projs) + np.max(projs)) / 2.0 / nx if abs(nx) > 1e-10 else 0
            cy += (np.min(projs) + np.max(projs)) / 2.0 / ny if abs(ny) > 1e-10 else 0
        cx /= 6
        cy /= 6
        
        # Recompute max_proj with the center
        max_proj = 0.0
        for nx, ny in normals:
            projs = (pts[:, 0] - cx) * nx + (pts[:, 1] - cy) * ny
            max_proj = max(max_proj, np.max(projs), -np.min(projs))
        
        D = max_proj
        S = (D + APOTHEM) / APOTHEM
        return S, (cx, cy), D

    def optimize_shape(pts):
        """
        Optimize the enclosing hexagon for a given set of points.
        """
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse grid search over beta
        for i in range(37):
            beta = i * (math.pi / 6) / 36
            S, center, D = compute_enclosing_hexagon(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        # Refine beta with bounded scalar optimization
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: compute_enclosing_hexagon(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = compute_enclosing_hexagon(pts, best_beta)
        except Exception:
            pass
        
        # Local optimization of center and beta
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0.0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = abs((px - cx) * nx + (py - cy) * ny)
                            if proj > max_proj:
                                max_proj = proj
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [
                    [best_center[0], best_center[1], best_beta],
                    [best_center[0], best_center[1], best_beta + 0.02],
                    [best_center[0], best_center[1], best_beta - 0.02],
                    [best_center[0] + 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] + 0.02, best_beta],
                    [best_center[0] - 0.02, best_center[1], best_beta],
                    [best_center[0], best_center[1] - 0.02, best_beta],
                ]
                
                for init in polish_starts:
                    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-13})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    def relax_layout(initial_pts, iterations=200):
        """
        Relax the layout using force-directed method to find a compact, non-overlapping cluster.
        """
        pts = initial_pts.copy()
        n = len(pts)
        d_min = math.sqrt(3) # Distance between centers of touching hexagons
        learning_rate = 0.05
        decay = 0.95
        current_lr = learning_rate
        
        for it in range(iterations):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            forces = np.zeros_like(pts)
            
            # Compute center of mass
            com = np.mean(pts, axis=0)
            
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-6:
                        # Random nudge
                        diff = np.random.normal(0, 0.01, 2)
                        dist = np.linalg.norm(diff)
                    
                    # Repulsive force if overlapping or too close
                    # We want to push them apart if dist < d_min
                    # Attractive force if dist > d_min to keep them compact
                    if dist < d_min:
                        # Strong repulsion
                        force_mag = (d_min - dist) * 10.0
                        forces[i] += force_mag * diff / dist
                        forces[j] -= force_mag * diff / dist
                    else:
                        # Weak attraction to form bonds
                        force_mag = (dist - d_min) * 0.1
                        forces[i] -= force_mag * diff / dist
                        forces[j] += force_mag * diff / dist

            # Add a small force towards center of mass to prevent drift
            for i in range(n):
                forces[i] -= 0.01 * (pts[i] - com)

            pts += current_lr * forces
            current_lr *= decay
            
        # Final cleanup: ensure no overlaps by pushing apart iteratively
        for _ in range(50):
            moved = False
            for i in range(n):
                for j in range(i + 1, n):
                    diff = pts[i] - pts[j]
                    dist = np.linalg.norm(diff)
                    if dist < d_min - 1e-5:
                        overlap = d_min - dist
                        push = overlap / 2 + 1e-6
                        if dist < 1e-6:
                            direction = np.array([1.0, 0.0])
                        else:
                            direction = diff / dist
                        pts[i] += push * direction
                        pts[j] -= push * direction
                        moved = True
            if not moved:
                break
                
        return pts

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Generate diverse initial configurations
    # 1. Random
    # 2. Grid
    # 3. Hexagonal spiral
    
    np.random.seed(42)
    
    initial_configs = []
    
    # Random
    for _ in range(5):
        pts = np.random.normal(0, 2.0, (N, 2))
        initial_configs.append(pts)
        
    # Grid-like
    for _ in range(5):
        pts = np.zeros((N, 2))
        idx = 0
        for r in range(-2, 3):
            for c in range(-2, 3):
                if idx >= N:
                    break
                x = c * math.sqrt(3) + (r % 2) * math.sqrt(3) / 2
                y = r * 1.5
                pts[idx] = [x, y]
                idx += 1
        initial_configs.append(pts)
        
    # Hexagonal cluster seed
    axial_seed = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    pts_seed = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial_seed])
    initial_configs.append(pts_seed)
    
    # Run relaxation and optimization for each
    for i, init_pts in enumerate(initial_configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        relaxed_pts = relax_layout(init_pts)
        
        # Center the relaxed points to origin for stability
        com = np.mean(relaxed_pts, axis=0)
        relaxed_pts = relaxed_pts - com
        
        S, beta, center = optimize_shape(relaxed_pts)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = relaxed_pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback to a simple valid configuration
        axial_1 = [
            (0, 0),
            (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
            (1, 1), (-1, 2), (-2, 1), (2, -1)
        ]
        best_global_pts = np.array([[math.sqrt(3) * (q + r / 2.0), 1.5 * r] for q, r in axial_1])
        best_global_center = (0.0, 0.0)
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
    
    # Use a very small safety margin to ensure no overlap
    SCALE = 1.0 + 1e-12
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
