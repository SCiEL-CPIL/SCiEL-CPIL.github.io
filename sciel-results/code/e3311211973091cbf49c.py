import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar
from scipy.spatial import ConvexHull
from itertools import permutations

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 148.0
SAFETY_MARGIN = 4.0
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

    def hexagons_disjoint_axial(axial_list):
        """Check if hexagons at given axial coords are pairwise disjoint (centers >= 2 apart)."""
        n = len(axial_list)
        for i in range(n):
            for j in range(i + 1, n):
                q1, r1 = axial_list[i]
                q2, r2 = axial_list[j]
                dq = q1 - q2
                dr = r1 - r2
                # Hex distance: max(|dq|, |dr|, |dq+dr|)
                dist = max(abs(dq), abs(dr), abs(dq + dr))
                if dist < 2:
                    return False
        return True

    # Generate candidate configurations
    candidates = []

    # Standard hexagonal cluster (7 + 4)
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # Cross
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3x3 block + 2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4x3 minus corners
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # Staircase
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Compact 3x3+2
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    # Star
    candidates.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # Elongated
    candidates.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(5,1)])
    # Asymmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # Compact with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # Wide base
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # Tall narrow
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4)])
    # Star with asymmetric outer
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,2)])
    # Star with diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    # Dense cluster
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1),(0,-1)])
    # Zigzag
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # Star with top cluster
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(1,2),(-1,2),(0,3)])
    # Trapezoid
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # Wide top
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,3)])
    # Star with inner diagonal
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,1)])
    # 4x3 variant
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # Wide with top
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # Symmetric star
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # Very tall
    candidates.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(0,4)])
    # 3x3+row
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # Dense 4-wide
    candidates.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # Star with asymmetric bottom
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(1,-2)])
    # Tall with top row
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    # Additional
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    candidates.append([(0,0),(1,0),(2,0),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(0,-1),(3,1)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    candidates.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(1,1),(-1,-1)])
    candidates.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(1,4),(2,4)])
    candidates.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3)])
    candidates.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3)])

    # NEW: Generate additional candidates via local search on axial lattice
    # Start from the best known and try swapping/adding hexagons
    def generate_additional_candidates(base_candidates, max_extra=30):
        extra = []
        seen = set()
        for c in base_candidates:
            seen.add(tuple(sorted(c)))
        
        # Try to find compact configurations by searching small regions
        # A hexagon of radius R contains all cells with hex distance <= R from center
        # For 11 hexagons, we need a region that can hold 11 non-adjacent hexagons
        
        # Systematic search: pick 11 cells from a larger grid, check disjointness
        # Use a grid of axial coords from -3 to 3
        grid_cells = []
        for q in range(-3, 4):
            for r in range(-3, 4):
                grid_cells.append((q, r))
        
        # Try various compact subsets
        # Method: start with center, add cells greedily by distance
        import random
        random.seed(42)
        
        # Generate random valid configurations
        for _ in range(200):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            # Start with a random center
            center = (random.randint(-1, 1), random.randint(-1, 1))
            chosen = [center]
            # Add 10 more cells, ensuring distance >= 2 from all chosen
            attempts = 0
            while len(chosen) < 11 and attempts < 500:
                attempts += 1
                q = random.randint(-3, 3)
                r = random.randint(-3, 3)
                new_cell = (q, r)
                if new_cell in chosen:
                    continue
                # Check distance >= 2 from all chosen
                valid = True
                for cq, cr in chosen:
                    dq = q - cq
                    dr = r - cr
                    dist = max(abs(dq), abs(dr), abs(dq + dr))
                    if dist < 2:
                        valid = False
                        break
                if valid:
                    chosen.append(new_cell)
            
            if len(chosen) == 11:
                key = tuple(sorted(chosen))
                if key not in seen:
                    seen.add(key)
                    extra.append(chosen)
                    if len(extra) >= max_extra:
                        break
        
        return extra

    # Generate additional candidates
    extra_candidates = generate_additional_candidates(candidates, max_extra=40)
    candidates.extend(extra_candidates)

    def calc_S_for_beta_fast(pts, beta):
        """
        Fast LP for minimum enclosing hexagon with given rotation.
        Uses the fact that for fixed beta, the optimal center C minimizes
        max_k max(h_k - C.n_k, C.n_k - g_k).
        This is equivalent to minimizing t subject to:
        C.n_k - t <= h_k
        -C.n_k - t <= g_k
        where h_k = max_i (P_i . n_k), g_k = min_i (P_i . n_k).
        """
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, None

        angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        
        # Compute support functions h_k and g_k
        projs = pts @ normals.T # shape (N, 6)
        h = np.max(projs, axis=0) # shape (6,)
        g = np.min(projs, axis=0) # shape (6,)

        # Setup LP: Minimize t
        # Variables: cx, cy, t
        # Constraints:
        # cx*nx + cy*ny - t <= h_k
        # -cx*nx - cy*ny - t <= g_k  => cx*nx + cy*ny + t >= -g_k => -cx*nx - cy*ny - t <= g_k
        
        A_ub = np.zeros((12, 3))
        b_ub = np.zeros(12)
        
        for k in range(6):
            nx, ny = normals[k]
            # Row 2k: nx*cx + ny*cy - t <= h_k
            A_ub[2*k, 0] = nx
            A_ub[2*k, 1] = ny
            A_ub[2*k, 2] = -1.0
            b_ub[2*k] = h[k]
            
            # Row 2k+1: -nx*cx - ny*cy - t <= g_k
            A_ub[2*k+1, 0] = -nx
            A_ub[2*k+1, 1] = -ny
            A_ub[2*k+1, 2] = -1.0
            b_ub[2*k+1] = g[k]

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

    def optimize_orientation_only(pts):
        """1D scalar minimization over outer hexagon orientation beta using fast LP."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, None

        best_S = float('inf')
        best_beta = 0.0
        best_center = None

        # Coarse grid
        n_grid = 61
        for i in range(n_grid):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            beta = i * (math.pi / 6) / (n_grid - 1)
            S, center, D = calc_S_for_beta_fast(pts, beta)
            if S < best_S:
                best_S = S
                best_beta = beta
                best_center = center

        if best_center is None:
            return float('inf'), 0.0, None

        # Fine grid around best
        window = (math.pi / 6) / (n_grid - 1)
        lo = max(0.0, best_beta - 2 * window)
        hi = min(math.pi / 6, best_beta + 2 * window)
        
        def obj_1d(b):
            S, _, _ = calc_S_for_beta_fast(pts, b)
            return S
        
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 100, 'xatol': 1e-12})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta_fast(pts, best_beta)
        except Exception:
            pass

        # Very fine polish
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                window = (math.pi / 6) / (n_grid - 1) * 0.2
                lo = max(0.0, best_beta - window)
                hi = min(math.pi / 6, best_beta + window)
                res = minimize_scalar(obj_1d, bounds=(lo, hi), method='bounded',
                                     options={'maxiter': 200, 'xatol': 1e-14})
                if res.fun < best_S:
                    best_S = res.fun
                    best_beta = res.x
                    _, best_center, _ = calc_S_for_beta_fast(pts, best_beta)
        except Exception:
            pass

        return best_S, best_beta, best_center

    def optimize_shape(pts):
        """Optimization for fixed point configuration using fast rotating calipers."""
        return optimize_orientation_only(pts)

    # Score and sort candidates
    def quick_estimate(axial_list):
        pts = axial_to_cartesian(axial_list)
        angles = math.pi / 6.0 + np.arange(6) * math.pi / 3.0
        normals = np.column_stack([np.cos(angles), np.sin(angles)])
        cx, cy = np.mean(pts[:, 0]), np.mean(pts[:, 1])
        pts_c = pts - np.array([cx, cy])
        projs = pts_c @ normals.T
        D = np.max(np.abs(projs))
        return (D + APOTHEM) / APOTHEM

    scored_candidates = []
    for axial in candidates:
        est = quick_estimate(axial)
        scored_candidates.append((est, axial))
    scored_candidates.sort(key=lambda x: x[0])

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    # Process top candidates first
    num_to_process = min(len(scored_candidates), 100)
    
    for idx in range(num_to_process):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        est, axial = scored_candidates[idx]
        pts = axial_to_cartesian(axial)
        S, beta, center = optimize_shape(pts)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center
            
            # Early termination if we hit a very good score? 
            # Theoretical min for 11 hexes is around 3.0-3.5? 
            # If S < 3.2, it's likely optimal or near-optimal.
            if best_global_S < 3.2:
                # Do a quick check on a few more just in case
                for j in range(idx+1, min(idx+5, num_to_process)):
                     if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN: break
                     pts2 = axial_to_cartesian(scored_candidates[j][1])
                     S2, b2, c2 = optimize_shape(pts2)
                     if S2 < best_global_S:
                         best_global_S = S2
                         best_global_pts = pts2
                         best_global_beta = b2
                         best_global_center = c2
                break

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
