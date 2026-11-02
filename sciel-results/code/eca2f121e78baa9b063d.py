import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar, least_squares

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9
_TIME_DEADLINE = START_TIME + TIME_LIMIT - SAFETY_MARGIN

def _time_ok():
    return time.time() < _TIME_DEADLINE

def construct():
    if not _time_ok():
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    # Pre-defined candidate configurations
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
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    # Additional diverse configurations for better rotation packing
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(4,1),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(1,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(2,2),(4,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(0,2),(4,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(2,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(1,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(3,2)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2)])

    # Contact-graph based configurations
    # For 11 hexagons, generate configurations where some hexagons are at non-lattice positions
    # by solving distance constraints
    
    def generate_contact_graph_configurations():
        """Generate additional configurations using contact graph approach."""
        configs = []
        
        # Base: hexagonal cluster with 7 inner + 4 outer
        # Try different arrangements of the 4 outer hexagons
        base_inner = [(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1)]
        
        # Different placements of 4 outer hexagons
        outer_options = [
            [(1,1),(-1,2),(-2,1),(2,-1)],
            [(1,1),(-1,2),(2,-1),(0,2)],
            [(1,1),(-1,2),(2,-1),(0,-2)],
            [(1,1),(-1,2),(-2,1),(0,2)],
            [(1,1),(-1,2),(-2,1),(0,-2)],
            [(1,1),(2,-1),(-2,1),(0,2)],
            [(1,1),(2,-1),(-2,1),(0,-2)],
            [(1,1),(2,-1),(-1,2),(0,2)],
            [(1,1),(2,-1),(-1,2),(0,-2)],
            [(1,1),(-2,1),(-1,2),(2,-1)],
            [(1,1),(-2,1),(-1,2),(0,2)],
            [(1,1),(-2,1),(-1,2),(0,-2)],
        ]
        
        for outer in outer_options:
            config = base_inner + outer
            # Check for duplicates
            config_set = frozenset(config)
            dup = False
            for c in configs:
                if frozenset(c) == config_set:
                    dup = True
                    break
            if not dup:
                configs.append(config)
        
        # Star configurations with different arm lengths
        star_configs = [
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(1,2),(-1,2)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(2,1),(-2,-1)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(2,0),(-2,0)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(1,-1),(-1,-1),(2,1),(-2,1)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(1,-1),(-1,-1)],
            [(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(2,1),(-2,1)],
        ]
        
        for sc in star_configs:
            sc_set = frozenset(sc)
            dup = False
            for c in configs:
                if frozenset(c) == sc_set:
                    dup = True
                    break
            if not dup:
                configs.append(sc)
        
        return configs
    
    contact_configs = generate_contact_graph_configurations()
    candidates.extend(contact_configs)

    # Remove duplicates from candidates
    seen = set()
    unique_candidates = []
    for c in candidates:
        key = frozenset(c)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(c)
    candidates = unique_candidates

    def calc_S_for_beta(pts, beta):
        """LP-based evaluation of minimum enclosing hexagon side length."""
        if not _time_ok():
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        n_pts = len(pts)
        n_constraints = 12 * n_pts
        A_ub = np.zeros((n_constraints, 3))
        b_ub = np.zeros(n_constraints)

        for k in range(6):
            nx, ny = normals[k]
            projs = pts[:, 0] * nx + pts[:, 1] * ny
            start = 2 * k * n_pts
            A_ub[start:start+n_pts, 0] = -nx
            A_ub[start:start+n_pts, 1] = -ny
            A_ub[start:start+n_pts, 2] = -1.0
            b_ub[start:start+n_pts] = -projs
            A_ub[start+n_pts:start+2*n_pts, 0] = nx
            A_ub[start+n_pts:start+2*n_pts, 1] = ny
            A_ub[start+n_pts:start+2*n_pts, 2] = -1.0
            b_ub[start+n_pts:start+2*n_pts] = projs

        c_obj = np.array([0.0, 0.0, 1.0])
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

    def exact_S(pts, cx, cy, beta):
        """Exact computation of S for given center and rotation."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    def smooth_calc_S(pts, cx, cy, beta, temperature=0.05):
        """Smooth surrogate using log-sum-exp."""
        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        pts_centered = pts - np.array([cx, cy])
        projs = pts_centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = np.max(all_projs)
        if max_val > 200:
            return float('inf')
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -50, 50)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        D = temperature * log_sum
        return (D + APOTHEM) / APOTHEM

    def optimize_shape(pts):
        """Multi-stage optimization for a fixed point configuration."""
        best_S = float('inf')
        best_beta = 0.0
        best_center = None
        
        if not _time_ok():
            return float('inf'), 0.0, None
        
        # Stage 1: Coarse grid search over beta in [0, pi/6]
        n_grid = 49
        for i in range(n_grid):
            if not _time_ok():
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center
        
        if best_center is None:
            return float('inf'), 0.0, None
        
        # Stage 2: Refine beta with bounded scalar optimization
        try:
            if _time_ok():
                res = minimize_scalar(lambda b: calc_S_for_beta(pts, b)[0], 
                                     bounds=(0, math.pi / 6), method='bounded',
                                     options={'maxiter': 300, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta(pts, best_beta)
        except Exception:
            pass
        
        # Stage 3: Smooth surrogate optimization with adaptive temperature
        try:
            if _time_ok():
                temperatures = [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]
                start_betas = [best_beta]
                for offset in [0.03, -0.03, 0.06, -0.06, 0.1, -0.1, 0.15, -0.15]:
                    sb = best_beta + offset
                    sb = ((sb % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    start_betas.append(sb)
                start_betas.extend([0.0, math.pi/12, math.pi/6 * 0.25, math.pi/6 * 0.75])
                
                for temp in temperatures:
                    if not _time_ok():
                        break
                    for sb in start_betas:
                        if not _time_ok():
                            break
                        
                        def smooth_obj(params, t=temp):
                            cx, cy, beta = params
                            beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S, _, _ = smooth_calc_S(pts, beta, cx, cy, temperature=t)
                            return S
                        
                        init = [best_center[0], best_center[1], sb]
                        res = minimize(smooth_obj, init, method='Nelder-Mead',
                                     options={'maxiter': 500, 'xatol': 1e-11, 'fatol': 1e-11})
                        
                        if res.fun < 1e10:
                            cx_r, cy_r, beta_r = res.x
                            beta_r = ((beta_r % (math.pi/6)) + math.pi/6) % (math.pi/6)
                            S_exact, c_exact, _ = calc_S_for_beta(pts, beta_r)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_beta = beta_r
                                best_center = c_exact
        except Exception:
            pass
        
        # Stage 4: Final high-precision polish using exact objective
        try:
            if _time_ok():
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi/6)) + math.pi/6) % (math.pi/6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    
                    pts_centered = pts - np.array([cx, cy])
                    projs = pts_centered @ normals.T
                    max_proj = np.max(np.abs(projs))
                    
                    S = (max_proj + APOTHEM) / APOTHEM
                    return S
                
                polish_starts = [[best_center[0], best_center[1], best_beta]]
                for db in [0.01, -0.01, 0.02, -0.02, 0.03, -0.03, 0.05, -0.05]:
                    bb = ((best_beta + db) % (math.pi/6) + math.pi/6) % (math.pi/6)
                    polish_starts.append([best_center[0], best_center[1], bb])
                for dc in [0.01, -0.01, 0.02, -0.02]:
                    polish_starts.append([best_center[0] + dc, best_center[1], best_beta])
                    polish_starts.append([best_center[0], best_center[1] + dc, best_beta])
                
                for init in polish_starts:
                    if not _time_ok():
                        break
                    res = minimize(objective, init, method='Nelder-Mead',
                                 options={'maxiter': 5000, 'xatol': 1e-14, 'fatol': 1e-14})
                    if res.fun < best_S:
                        best_S = res.fun
                        best_center = (res.x[0], res.x[1])
                        best_beta = ((res.x[2] % (math.pi/6)) + math.pi/6) % (math.pi/6)
        except Exception:
            pass
            
        return best_S, best_beta, best_center

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for axial in candidates:
        if not _time_ok():
            break
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = axial_to_cartesian(candidates[0])
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
    
    pts_scaled = []
    for x, y in best_global_pts:
        pts_scaled.append((float(x), float(y)))
    
    pts_scaled = np.array(pts_scaled)
    
    # Compute final side length
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
