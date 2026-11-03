import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 5.0
EPS = 1e-9

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # --- Initial candidate configurations (axial coordinates) ---
    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    candidates = []

    # Star pattern: center + 6 around + 4 more
    axial_1 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(axial_1)

    # Compact cluster
    axial_2 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (1,1), (-1,2), (-2,1), (2,-1)]
    candidates.append(axial_2)

    # Row-based
    axial_3 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (-1,1), (3,1)]
    candidates.append(axial_3)

    axial_4 = [(0,0), (1,0), (2,0), (3,0), (0,1), (1,1), (2,1), (3,1), (0,2), (1,2), (2,2)]
    candidates.append(axial_4)

    # More diverse patterns
    axial_5 = [(0,0), (1,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3), (2,3)]
    candidates.append(axial_5)

    axial_6 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (0,3), (1,3)]
    candidates.append(axial_6)

    axial_7 = [(0,0), (1,0), (-1,0), (0,1), (0,-1), (1,1), (-1,1), (2,0), (-2,0), (0,2), (0,-2)]
    candidates.append(axial_7)

    axial_8 = [(0,0), (1,0), (2,0), (3,0), (4,0), (0,1), (1,1), (2,1), (3,1), (4,1), (5,1)]
    candidates.append(axial_8)

    axial_9 = [(0,0), (1,0), (0,1), (-1,1), (-1,0), (0,-1), (1,-1), (2,1), (-2,2), (-1,3), (1,-2)]
    candidates.append(axial_9)

    axial_10 = [(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2), (1,3), (2,3)]
    candidates.append(axial_10)

    # Convert to Cartesian
    cart_candidates = [axial_to_cartesian(ax) for ax in candidates]

    # --- Geometry helpers ---
    def get_normals(beta):
        """Return 6 unit normals of the containing hexagon for rotation beta."""
        return np.array([
            (math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
             math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0))
            for k in range(6)
        ])

    def compute_D(pts, center, normals):
        """Exact: D = max over points and normals of |(p - c) . n|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T
        return float(np.max(np.abs(projs)))

    def calc_S_for_beta(pts, beta, center=None):
        """LP over center for fixed beta. Returns (S, center, D)."""
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), None, float('inf')

        normals = get_normals(beta)
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
                D = float(res.x[2])
                cx, cy = float(res.x[0]), float(res.x[1])
                S = (D + APOTHEM) / APOTHEM
                return S, (cx, cy), D
            else:
                return float('inf'), None, float('inf')
        except Exception:
            return float('inf'), None, float('inf')

    def smooth_D(pts, center, normals, temperature):
        """Smooth surrogate for D via log-sum-exp soft-max over |projections|."""
        centered = pts - np.array(center)
        projs = centered @ normals.T
        all_projs = np.concatenate([projs.ravel(), (-projs).ravel()])
        max_val = float(np.max(all_projs))
        shifted = (all_projs - max_val) / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
        return temperature * log_sum

    def smooth_calc_S(pts, beta, center, temperature):
        normals = get_normals(beta)
        D = smooth_D(pts, center, normals, temperature)
        S = (D + APOTHEM) / APOTHEM
        return S

    def compute_overlap_penalty(pts, temperature=0.1):
        """
        Smooth penalty for overlapping hexagons.
        For two unit hexagons with centers p1, p2 and same orientation,
        they overlap if the distance between centers is less than 2*APOTHEM.
        We use a smooth max of (2*APOTHEM - dist) over all pairs.
        """
        n = len(pts)
        if n < 2:
            return 0.0
        
        # Compute all pairwise distances
        diffs = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diffs**2, axis=2))
        
        # Only consider upper triangle to avoid double counting
        iu = np.triu_indices(n, k=1)
        pair_dists = dists[iu]
        
        # Violation: 2*APOTHEM - dist (positive if overlapping)
        violations = 2 * APOTHEM - pair_dists
        
        # Smooth max via log-sum-exp
        max_viol = float(np.max(violations))
        if max_viol <= 0:
            return 0.0
        
        shifted = violations / temperature
        shifted = np.clip(shifted, -60.0, 60.0)
        log_sum = np.log(np.sum(np.exp(shifted))) + max_viol / temperature
        return temperature * log_sum

    def full_objective(params, pts_template, temperature=0.1):
        """
        Full objective: outer hexagon side length + overlap penalty.
        params: [cx, cy, beta, x0, y0, x1, y1, ..., x10, y10]
        """
        cx, cy, beta = params[0], params[1], params[2]
        beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        
        # Reconstruct points with optimized offsets
        pts = np.array(pts_template)
        for i in range(len(pts)):
            pts[i, 0] += params[3 + 2*i]
            pts[i, 1] += params[4 + 2*i]
        
        # Outer hexagon cost
        normals = get_normals(beta)
        S = smooth_calc_S(pts, beta, (cx, cy), temperature)
        
        # Overlap penalty (heavy weight)
        overlap = compute_overlap_penalty(pts, temperature * 0.5)
        
        return S + 100.0 * overlap

    def optimize_positions_and_containment(pts_template):
        """
        Optimize positions of all hexagons + outer hexagon (center, rotation).
        Uses smooth surrogate for both containment and non-overlap.
        """
        best_S = float('inf')
        best_params = None
        best_pts = None
        best_beta = 0.0
        best_center = (float(np.mean(pts_template[:, 0])), float(np.mean(pts_template[:, 1])))
        
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return float('inf'), 0.0, best_center, pts_template
        
        # Initial parameters: center at mean, beta=0, zero offsets
        cx0 = float(np.mean(pts_template[:, 0]))
        cy0 = float(np.mean(pts_template[:, 1]))
        
        # Try multiple initial betas
        initial_betas = [0.0, math.pi/12, math.pi/6, math.pi/24, -math.pi/24]
        
        for temp in [0.5, 0.2, 0.1, 0.05, 0.02, 0.01]:
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                break
            
            for init_beta in initial_betas:
                if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                    break
                
                # Build initial parameter vector
                init_params = [cx0, cy0, init_beta]
                for i in range(len(pts_template)):
                    init_params.append(0.0)
                    init_params.append(0.0)
                
                try:
                    res = minimize(full_objective, init_params, args=(pts_template, temp),
                                   method='Nelder-Mead',
                                   options={'maxiter': 5000, 'xatol': 1e-10, 'fatol': 1e-10})
                    
                    if res.fun < 1e10:
                        # Extract optimized positions
                        opt_cx, opt_cy, opt_beta = res.x[0], res.x[1], res.x[2]
                        opt_beta = ((opt_beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                        
                        opt_pts = np.array(pts_template)
                        for i in range(len(pts_template)):
                            opt_pts[i, 0] += res.x[3 + 2*i]
                            opt_pts[i, 1] += res.x[4 + 2*i]
                        
                        # Check for actual overlap (exact check)
                        n = len(opt_pts)
                        has_overlap = False
                        for i in range(n):
                            for j in range(i+1, n):
                                dist = np.linalg.norm(opt_pts[i] - opt_pts[j])
                                if dist < 2 * APOTHEM - 1e-8:
                                    has_overlap = True
                                    break
                            if has_overlap:
                                break
                        
                        if not has_overlap:
                            # Compute exact S
                            S_exact, c_exact, _ = calc_S_for_beta(opt_pts, opt_beta)
                            if S_exact < best_S:
                                best_S = S_exact
                                best_params = res.x.copy()
                                best_pts = opt_pts.copy()
                                best_beta = opt_beta
                                if c_exact is not None:
                                    best_center = c_exact
                except Exception:
                    pass
        
        # If no valid configuration found, fall back to template with LP
        if best_S == float('inf'):
            S, center, _ = calc_S_for_beta(pts_template, 0.0)
            best_S = S
            best_pts = pts_template
            best_beta = 0.0
            if center is not None:
                best_center = center
        
        return best_S, best_beta, best_center, best_pts

    # --- Main loop over candidates ---
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for i, pts_template in enumerate(cart_candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        S, beta, center, pts = optimize_positions_and_containment(pts_template)
        if S < best_global_S:
            best_global_S = S
            best_global_pts = pts
            best_global_beta = beta
            best_global_center = center

    # --- Fallback ---
    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = cart_candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])),
                              float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        normals = get_normals(best_global_beta)
        max_proj = 0.0
        for nx, ny in normals:
            projs = best_global_pts[:, 0] * nx + best_global_pts[:, 1] * ny
            max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
        best_global_S = (max_proj + APOTHEM) / APOTHEM

    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # --- Final: recompute exact S with the chosen (center, beta) ---
    normals = get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (best_global_pts[:, 0] - cx) * nx + (best_global_pts[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))

    D_final = max_proj
    S_final = (D_final + APOTHEM) / APOTHEM

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(S_final)
    rotation = float(angle_deg)

    return inner, center, side, rotation
