import math
import numpy as np
import time
from scipy.optimize import minimize

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

    # Generate a large set of diverse lattice configurations for N=11
    def generate_lattice_configs():
        configs = []
        
        # Flower + various extensions
        flower = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        
        # Star extensions from flower
        for extra in [
            [(2, 0)],
            [(-2, 0)],
            [(0, 2)],
            [(0, -2)],
            [(1, 1)],
            [(-1, -1)],
            [(2, 1), (-1, 2)],
            [(-2, -1), (1, -2)],
            [(1, 2), (-2, 1)],
            [(-1, -2), (2, -1)],
            [(2, -1), (-2, 2)],
            [(1, -2), (-1, 3)],
        ]:
            if len(flower) + len(extra) == N:
                configs.append(flower + extra)
        
        # Row-based configurations: rows of varying lengths
        # Total must be 11
        row_combos = [
            [5, 3, 3], [3, 5, 3], [5, 5, 1], [1, 5, 5],
            [4, 4, 3], [3, 4, 4], [4, 3, 4],
            [6, 2, 3], [3, 2, 6], [2, 6, 3], [6, 3, 2], [3, 6, 2], [2, 3, 6],
            [4, 3, 2, 2], [2, 2, 3, 4], [3, 2, 2, 4],
            [5, 2, 2, 2], [2, 5, 2, 2], [2, 2, 5, 2],
            [3, 3, 3, 2], [2, 3, 3, 3], [3, 2, 3, 3], [3, 3, 2, 3],
            [4, 2, 2, 1, 2], [2, 2, 4, 1, 2],
            [3, 1, 3, 1, 3], [1, 3, 1, 3, 3], [3, 3, 1, 3, 1],
        ]
        
        for lengths in row_combos:
            if sum(lengths) != N:
                continue
            # Build the configuration
            pts = []
            y_offset = 0
            for i, length in enumerate(lengths):
                for j in range(length):
                    # Center each row horizontally
                    x_start = -(length - 1) / 2.0
                    # In axial coords, convert fractional to proper
                    # We need integer or half-integer offsets
                    # Use proper hex grid: row offset by 0.5 for odd rows
                    if i % 2 == 0:
                        x_pos = x_start + j
                    else:
                        x_pos = x_start + j + 0.5
                    # Convert to axial-like: we'll use the cartesian formula directly
                    # Actually, let's build in cartesian directly
                    # For row i at y = 1.5 * i_offset, x spacing = sqrt(3)
                    pass
            
            # Better approach: build in cartesian directly for row configs
            pts = []
            y_start = -((len(lengths) - 1) * 1.5) / 2.0
            for i, length in enumerate(lengths):
                y = y_start + i * 1.5
                x_start = -(length - 1) * math.sqrt(3) / 2.0
                for j in range(length):
                    x = x_start + j * math.sqrt(3)
                    pts.append((x, y))
            if len(pts) == N:
                configs.append(np.array(pts))
        
        # Diamond / hexagonal patches
        # Size 3 hex patch has 19 cells, too many
        # Let's do partial patches
        # 3x3 block minus corners
        for base in range(-2, 3):
            for height in range(3, 6):
                pts = []
                for i in range(height):
                    width = min(i + 1, 3, height - i)
                    if width <= 0:
                        continue
                    x_start = -(width - 1) * math.sqrt(3) / 2.0
                    y = (i - height / 2 + 0.5) * 1.5
                    for j in range(width):
                        x = x_start + j * math.sqrt(3)
                        pts.append((x, y))
                if len(pts) == N:
                    configs.append(np.array(pts))
        
        # Striped / linear patterns
        for width in range(1, 6):
            for height in range(1, 6):
                pts = []
                for i in range(height):
                    for j in range(width):
                        x = (j - width / 2 + 0.5) * math.sqrt(3)
                        y = (i - height / 2 + 0.5) * 1.5
                        pts.append((x, y))
                if len(pts) == N:
                    configs.append(np.array(pts))
        
        # L-shapes and other irregular shapes
        l_configs = [
            # L-shape: 5 in one direction, 5 in another, 1 overlap
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
             (0, 1), (0, 2), (0, 3), (0, 4),
             (1, 1), (2, 1)],
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
             (4, 1), (4, 2), (4, 3), (4, 4),
             (3, 1), (2, 1)],
            # T-shape
            [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
             (2, 1), (2, 2), (2, 3),
             (1, 1), (3, 1)],
            # Cross shape
            [(0, 0), (1, 0), (2, 0),
             (-1, 1), (0, 1), (1, 1), (2, 1), (3, 1),
             (0, 2), (1, 2), (2, 2)],
        ]
        for axial in l_configs:
            pts = axial_to_cartesian(axial)
            if len(pts) == N:
                configs.append(pts)
        
        # Random-ish perturbations of the flower
        np.random.seed(42)
        base_flower = axial_to_cartesian(flower)
        for _ in range(20):
            pts = base_flower.copy()
            # Add 4 random points in a reasonable range
            for _ in range(4):
                angle = np.random.uniform(0, 2 * math.pi)
                radius = np.random.uniform(1.0, 3.5)
                x = math.sqrt(3) * radius * math.cos(angle)
                y = 1.5 * radius * math.sin(angle)
                # Snap to nearest hex center
                # Nearest hex centers are at positions where:
                # x = sqrt(3)*(q + r/2), y = 1.5*r
                r = round(y / 1.5)
                q = round(x / math.sqrt(3) - r / 2.0)
                nx = math.sqrt(3) * (q + r / 2.0)
                ny = 1.5 * r
                pts = np.vstack([pts, [[nx, ny]]])
            # Check for duplicates
            unique_pts = []
            seen = set()
            for pt in pts:
                key = (round(pt[0], 6), round(pt[1], 6))
                if key not in seen:
                    seen.add(key)
                    unique_pts.append(pt)
            if len(unique_pts) == N:
                configs.append(np.array(unique_pts))
        
        return configs

    def calc_S_for_beta(pts, beta):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None
        
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        # Find the minimum enclosing hexagon centered at (cx, cy) with rotation beta
        # D = max over all normals and points of |proj - center_proj|
        # We want to minimize D by choosing center
        
        # For a given beta, the optimal center is the centroid that minimizes
        # the maximum projection deviation
        # This is a LP problem
        
        from scipy.optimize import linprog
        
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        
        for nx, ny in normals:
            for px, py in pts:
                # (px - cx)*nx + (py - cy)*ny <= D
                # -cx*nx - cy*ny - D <= -px*nx - py*ny
                A_ub.append([-nx, -ny, -1.0])
                b_ub.append(-(px * nx + py * ny))
                # (px - cx)*nx + (py - cy)*ny >= -D
                # cx*nx + cy*ny - D <= px*nx + py*ny
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

    def smooth_calc_S(pts, beta, cx, cy, temperature=0.1):
        normals = []
        for k in range(6):
            ang = beta + math.pi / 6.0 + k * math.pi / 3.0
            normals.append((math.cos(ang), math.sin(ang)))
        
        projs = np.array([
            ((px - cx) * nx + (py - cy) * ny)
            for nx, ny in normals
            for px, py in pts
        ])
        all_projs = np.concatenate([projs, -projs])
        
        max_val = np.max(all_projs)
        if max_val > 100:
            return float('inf')
        
        shifted = (all_projs - max_val) / temperature
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        S = (D + APOTHEM) / APOTHEM
        return S

    def optimize_shape(pts):
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None
        
        # Coarse search over beta
        for i in range(60):
            beta = i * (math.pi / 6) / 59
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            return float('inf'), 0.0, None
        
        # Fine search
        try:
            from scipy.optimize import minimize_scalar
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-10})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Nelder-Mead polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    normals = []
                    for k in range(6):
                        ang = beta + math.pi / 6.0 + k * math.pi / 3.0
                        normals.append((math.cos(ang), math.sin(ang)))
                    
                    max_proj = 0
                    for nx, ny in normals:
                        for px, py in pts:
                            proj = (px - cx) * nx + (py - cy) * ny
                            max_proj = max(max_proj, abs(proj))
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                res = minimize(objective, [best_center[0], best_center[1], best_beta],
                             method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_center = (res.x[0], res.x[1])
                    best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
        
        return best_S, best_beta, best_center

    configs = generate_lattice_configs()
    
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for idx, config in enumerate(configs):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        if len(config) != N:
            continue
        S, beta, center = optimize_shape(config)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = config
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        # Fallback: flower + 4 more
        axial = [(0, 0), (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
                 (2, 0), (-2, 0), (0, 2), (0, -2)]
        best_global_pts = axial_to_cartesian(axial)
        best_global_center = (float(np.mean(best_global_pts[:, 0])), 
                            float(np.mean(best_global_pts[:, 1])))
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
    
    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])
        
    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
