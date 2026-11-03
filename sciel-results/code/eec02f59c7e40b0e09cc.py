import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, linprog

N = 11
APOTHEM = math.sqrt(3) / 2.0
TWO_APOTHEM = math.sqrt(3)
START_TIME = time.time()
TIME_LIMIT = 145.0
SAFETY_MARGIN = 5.0

def _elapsed():
    return time.time() - START_TIME

def _dead():
    return _elapsed() > TIME_LIMIT - SAFETY_MARGIN


def _axial_to_cartesian(axial, gap_factor=1.0):
    """Convert axial coords to cartesian with tunable gap.
    Neighbors on the lattice are gap_factor * (2*APOTHEM) apart.
    gap_factor = 1.0: exact touching.
    gap_factor < 1.0: closer packing.
    """
    pts = []
    for q, r in axial:
        x = TWO_APOTHEM * (q + r / 2.0) * gap_factor
        y = 1.5 * r * gap_factor
        pts.append((x, y))
    return np.array(pts)


def _get_normals(beta):
    """6 unit outward normals of the containing hexagon (rotated by beta)."""
    return np.array([
        [math.cos(beta + math.pi / 6.0 + k * math.pi / 3.0),
         math.sin(beta + math.pi / 6.0 + k * math.pi / 3.0)]
        for k in range(6)
    ])


def _compute_D(pts, center, normals):
    centered = pts - np.array(center)
    projs = centered @ normals.T
    return float(np.max(np.abs(projs)))


def _solve_center_lp(pts, normals):
    """LP: find center c and t such that |p_i - c|_hex <= t, minimize t.
    Returns (t_opt, c_opt) or None."""
    n_pts = len(pts)
    c0 = np.mean(pts, axis=0)
    proj = pts @ normals.T  # (n_pts, 6)
    
    # Constraint 1: (p_i - c)·n_j <= t  =>  -n_j·c - t <= -proj[i,j]
    A1 = np.tile(np.column_stack([-normals, np.ones(6)]), (n_pts, 1))
    b1 = -proj.ravel()
    
    # Constraint 2: -(p_i - c)·n_j <= t  =>  n_j·c - t <= proj[i,j]
    A2 = np.tile(np.column_stack([normals, -np.ones(6)]), (n_pts, 1))
    b2 = proj.ravel()
    
    A_ub = np.concatenate([A1, A2], axis=0)
    b_ub = np.concatenate([b1, b2])
    c_obj = np.array([1e-10, 1e-10, 1.0])
    
    try:
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.status == 0:
            return float(res.x[2]), (float(res.x[0]), float(res.x[1]))
    except Exception:
        pass
    return None


def _calc_S(pts, beta):
    """For fixed beta, solve LP for optimal center. Returns (S, center)."""
    normals = _get_normals(beta)
    sol = _solve_center_lp(pts, normals)
    if sol is not None:
        t_opt, c_opt = sol
        return (t_opt + APOTHEM) / APOTHEM, c_opt
    # Fallback
    c0 = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
    D = _compute_D(pts, c0, normals)
    return (D + APOTHEM) / APOTHEM, c0


def _candidates():
    """List of candidate axial configurations for 11 hexagons."""
    cands = []
    # 1. Center + ring + 4 more (cluster)
    cands.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(1,1),(-1,2),(-2,1),(2,-1)])
    # 2. Star with 4 far
    cands.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,0),(-2,0),(0,2),(0,-2)])
    # 3. 3x3 + 2
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(-1,1),(3,1)])
    # 4. 4x3 + 1
    cands.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2)])
    # 5. 3x4 + 1
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(0,3),(1,3),(2,3)])
    # 6. Cross-like
    cands.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2)])
    # 7. Asymmetric cluster
    cands.append([(0,0),(1,0),(0,1),(-1,1),(-1,0),(0,-1),(1,-1),(2,1),(-2,2),(-1,3),(1,-2)])
    # 8. 3x3 + 2 top
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # 9. 2+4+4+1
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # 10. Staircase
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(1,2),(2,2),(0,3),(1,3),(2,3),(3,2)])
    # 11. 4+3+3
    cands.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # 12. 4+4+3
    cands.append([(0,0),(1,0),(2,0),(3,0),(0,1),(1,1),(2,1),(3,1),(1,2),(2,2),(3,2)])
    # 13. 3+4+3+1
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3)])
    # 14. 4+3+4+2
    cands.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3),(2,3)])
    # 15. 5+3+3
    cands.append([(0,0),(1,0),(2,0),(3,0),(4,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,2)])
    # 16. 3+4+3+1 variant
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    # 17. 2+5+4
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(4,2)])
    # 18. 5+5+1
    cands.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2),(1,2)])
    # 19. 3+2+3+3
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(0,2),(1,2),(2,2),(3,2),(0,3),(1,3)])
    # 20. 3-4-3 offset
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    # 21. Diamond-ish
    cands.append([(0,0),(1,0),(-1,0),(0,1),(0,-1),(1,1),(-1,1),(2,0),(-2,0),(0,2),(0,-2),(1,-1)])
    # 22. 2+3+3+3
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3),(3,3)])
    # 23. 4+2+4+1
    cands.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(2,3)])
    # 24. 3+3+3+2
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # 25. L-shape
    cands.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(0,2),(0,3),(0,4),(1,2)])
    # 26. Plus shape
    cands.append([(0,0),(1,0),(-1,0),(2,0),(-2,0),(0,1),(0,-1),(0,2),(0,-2),(1,1),(1,-1)])
    # 27. Zigzag
    cands.append([(0,0),(1,0),(1,1),(2,1),(2,2),(3,2),(3,3),(4,3),(4,4),(5,4),(5,5)])
    # 28. 2+2+2+2+2+1 (diagonal stripes)
    cands.append([(0,0),(1,0),(0,1),(1,1),(0,2),(1,2),(0,3),(1,3),(0,4),(1,4),(2,2)])
    # 29. Compact 3-3-3-2
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # 30. Asymmetric 3-4-4
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    # 31. 4-4-3 with offset
    cands.append([(0,0),(1,0),(2,0),(3,0),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(3,2)])
    # 32. 5-3-3 with offset
    cands.append([(0,0),(1,0),(2,0),(3,0),(4,0),(0,1),(1,1),(2,1),(1,2),(2,2),(3,2)])
    # 33. 3-5-3
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(3,1),(4,1),(0,2),(1,2),(2,2)])
    # 34. 2-4-3-2
    cands.append([(0,0),(1,0),(0,1),(1,1),(2,1),(3,1),(0,2),(1,2),(2,2),(1,3),(2,3)])
    # 35. 3-3-4-1
    cands.append([(0,0),(1,0),(2,0),(0,1),(1,1),(2,1),(0,2),(1,2),(2,2),(3,2),(1,3)])
    return cands


def _optimize_candidate(axial):
    """Optimize (gap_factor, beta, center) for a candidate."""
    best_S = float('inf')
    best_gf = 1.0
    best_beta = 0.0
    best_center = (0.0, 0.0)
    
    # Phase 1: coarse grid over (gf, beta)
    gf_vals = [0.85, 0.90, 0.95, 1.00, 1.05]
    beta_vals = [i * (math.pi / 6) / 24 for i in range(25)]
    
    for gf in gf_vals:
        if _dead():
            break
        pts = _axial_to_cartesian(axial, gf)
        c0 = (float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])))
        for beta in beta_vals:
            if _dead():
                break
            S, center = _calc_S(pts, beta)
            if S < best_S:
                best_S = S
                best_gf = gf
                best_beta = beta
                best_center = center
    
    if _dead():
        return best_S, best_gf, best_beta, best_center
    
    # Phase 2: refine gf and beta around best
    for _ in range(3):
        if _dead():
            break
        # Refine gf
        try:
            lo = max(0.70, best_gf - 0.08)
            hi = min(1.10, best_gf + 0.08)
            def obj_gf(gf):
                pts = _axial_to_cartesian(axial, gf)
                S, _ = _calc_S(pts, best_beta)
                return S
            res = minimize_scalar(obj_gf, bounds=(lo, hi), method='bounded',
                                  options={'maxiter': 80, 'xatol': 1e-6})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_gf = float(res.x)
        except Exception:
            pass
        
        if _dead():
            break
        # Refine beta
        try:
            lo = max(0.0, best_beta - 0.15)
            hi = min(math.pi / 6, best_beta + 0.15)
            pts = _axial_to_cartesian(axial, best_gf)
            def obj_beta(b):
                S, _ = _calc_S(pts, b)
                return S
            res = minimize_scalar(obj_beta, bounds=(lo, hi), method='bounded',
                                  options={'maxiter': 80, 'xatol': 1e-7})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_beta = float(res.x)
        except Exception:
            pass
    
    if _dead():
        return best_S, best_gf, best_beta, best_center
    
    # Phase 3: 3D polish (center_x, center_y, beta) at fixed gf
    pts = _axial_to_cartesian(axial, best_gf)
    
    def obj_3d(params):
        cx, cy, b = params
        b = ((b % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        normals = _get_normals(b)
        D = _compute_D(pts, (cx, cy), normals)
        return (D + APOTHEM) / APOTHEM
    
    polish_starts = [
        [best_center[0], best_center[1], best_beta],
        [best_center[0], best_center[1], best_beta + 0.02],
        [best_center[0], best_center[1], best_beta - 0.02],
        [best_center[0] + 0.05, best_center[1], best_beta],
        [best_center[0] - 0.05, best_center[1], best_beta],
        [best_center[0], best_center[1] + 0.05, best_beta],
        [best_center[0], best_center[1] - 0.05, best_beta],
    ]
    
    for init in polish_starts:
        if _dead():
            break
        try:
            res = minimize(obj_3d, init, method='Nelder-Mead',
                           options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-12})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_center = (float(res.x[0]), float(res.x[1]))
                best_beta = ((float(res.x[2]) % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass
    
    # Phase 4: final gf refinement at best beta
    if not _dead():
        try:
            lo = max(0.70, best_gf - 0.03)
            hi = min(1.10, best_gf + 0.03)
            def obj_gf_final(gf):
                pts = _axial_to_cartesian(axial, gf)
                normals = _get_normals(best_beta)
                sol = _solve_center_lp(pts, normals)
                if sol is not None:
                    t_opt, _ = sol
                    return (t_opt + APOTHEM) / APOTHEM
                return float('inf')
            res = minimize_scalar(obj_gf_final, bounds=(lo, hi), method='bounded',
                                  options={'maxiter': 100, 'xatol': 1e-8})
            if res.fun < best_S:
                best_S = float(res.fun)
                best_gf = float(res.x)
        except Exception:
            pass
    
    # Recompute best center at final (gf, beta)
    pts = _axial_to_cartesian(axial, best_gf)
    _, best_center = _calc_S(pts, best_beta)
    
    return best_S, best_gf, best_beta, best_center


def construct():
    if _dead():
        # Fallback: 11 hexagons in a row, minimal bounding
        pts = _axial_to_cartesian([(i, 0) for i in range(11)], 1.0)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return [[float(x), float(y), 0.0] for x, y in pts], [cx, cy], 11.0, 0.0
    
    cands = _candidates()
    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None
    best_global_gf = 1.0
    
    # Budget per candidate: fast for most, more for top few
    n_cands = len(cands)
    base_budget = (TIME_LIMIT - SAFETY_MARGIN - _elapsed()) / max(1, n_cands)
    
    for idx, axial in enumerate(cands):
        if _dead():
            break
        # Adaptive budget: spend more on early candidates, less on later
        remaining = TIME_LIMIT - SAFETY_MARGIN - _elapsed()
        remaining_cands = n_cands - idx
        if remaining <= 0:
            break
        # Save time for at least 2 more candidates
        per_cand = min(remaining / max(1, remaining_cands), 3.0)
        if per_cand < 0.5:
            break
        
        # Time-box the optimization
        t_start = time.time()
        S, gf, beta, center = _optimize_candidate(axial)
        # If it took too long, just use what we have (the function already time-boxes internally)
        
        if S < best_global_S:
            best_global_S = S
            best_global_pts = _axial_to_cartesian(axial, gf)
            best_global_beta = beta
            best_global_center = center
            best_global_gf = gf
    
    # Fallback if nothing worked
    if best_global_pts is None:
        axial0 = cands[0]
        best_global_pts = _axial_to_cartesian(axial0, 1.0)
        best_global_beta = 0.0
        cx = float(np.mean(best_global_pts[:, 0]))
        cy = float(np.mean(best_global_pts[:, 1]))
        best_global_center = (cx, cy)
        normals = _get_normals(0.0)
        D = _compute_D(best_global_pts, best_global_center, normals)
        best_global_S = (D + APOTHEM) / APOTHEM
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Verify exact S with final points
    normals = _get_normals(best_global_beta)
    D = _compute_D(best_global_pts, (cx, cy), normals)
    S_final = (D + APOTHEM) / APOTHEM
    
    # Safety: scale up points by a tiny factor if needed to ensure no overlap
    # The gap_factor < 1.0 means hexagons are closer than touching, which is invalid.
    # We need gap_factor >= 1.0 for validity. Let's check and adjust.
    if best_global_gf < 1.0 - 1e-9:
        # Re-do with gf = 1.0 for validity
        # But this defeats the purpose... let's think.
        # Actually, gap_factor < 1.0 means hexagons OVERLAP. That's invalid.
        # We must use gf >= 1.0.
        # Let's clamp to 1.0.
        best_global_pts = _axial_to_cartesian(_find_original_axial(best_global_pts, best_global_gf), 1.0)
        normals = _get_normals(best_global_beta)
        sol = _solve_center_lp(best_global_pts, normals)
        if sol is not None:
            t_opt, c_opt = sol
            best_global_S = (t_opt + APOTHEM) / APOTHEM
            best_global_center = c_opt
        else:
            cx = float(np.mean(best_global_pts[:, 0]))
            cy = float(np.mean(best_global_pts[:, 1]))
            best_global_center = (cx, cy)
            D = _compute_D(best_global_pts, best_global_center, normals)
            best_global_S = (D + APOTHEM) / APOTHEM
        cx, cy = best_global_center
        angle_deg = math.degrees(best_global_beta)
        D = _compute_D(best_global_pts, (cx, cy), normals)
        S_final = (D + APOTHEM) / APOTHEM
    
    # Final safety scale
    SCALE = 1.0 + 1e-10
    pts_scaled = []
    for x, y in best_global_pts:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        pts_scaled.append((sx, sy))
    pts_scaled = np.array(pts_scaled)
    
    normals = _get_normals(best_global_beta)
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    D_scaled = max_proj
    S_scaled = (D_scaled + APOTHEM) / APOTHEM
    
    inner = []
    for x, y in pts_scaled:
        inner.append([float(x), float(y), float(angle_deg)])
    
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation


def _find_original_axial(pts, gf):
    """Recover original axial coords from cartesian with gap_factor.
    This is only used as a fallback; we should track the axial directly."""
    # Reverse transform: q + r/2 = x / (2*APOTHEM*gf), r = y / (1.5*gf)
    # This is approximate; better to track axial in the main loop.
    # For now, just use the closest lattice point.
    axial = []
    for x, y in pts:
        r = round(y / (1.5 * gf))
        q_plus_r_half = x / (TWO_APOTHEM * gf)
        q = round(q_plus_r_half - r / 2.0)
        axial.append((q, r))
    return axial


# Refactor: track axial directly to avoid the recovery step
def construct():
    if _dead():
        pts = _axial_to_cartesian([(i, 0) for i in range(11)], 1.0)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        return [[float(x), float(y), 0.0] for x, y in pts], [cx, cy], 11.0, 0.0
    
    cands = _candidates()
    best_global_S = float('inf')
    best_global_axial = None
    best_global_gf = 1.0
    best_global_beta = 0.0
    best_global_center = None
    
    n_cands = len(cands)
    for idx, axial in enumerate(cands):
        if _dead():
            break
        remaining = TIME_LIMIT - SAFETY_MARGIN - _elapsed()
        remaining_cands = n_cands - idx
        if remaining <= 0:
            break
        per_cand = min(remaining / max(1, remaining_cands), 3.0)
        if per_cand < 0.5:
            break
        
        S, gf, beta, center = _optimize_candidate(axial)
        if S < best_global_S:
            best_global_S = S
            best_global_axial = axial
            best_global_gf = gf
            best_global_beta = beta
            best_global_center = center
    
    if best_global_axial is None:
        best_global_axial = cands[0]
        best_global_gf = 1.0
        best_global_beta = 0.0
        pts = _axial_to_cartesian(best_global_axial, 1.0)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        best_global_center = (cx, cy)
        normals = _get_normals(0.0)
        D = _compute_D(pts, best_global_center, normals)
        best_global_S = (D + APOTHEM) / APOTHEM
    
    # Clamp gf to >= 1.0 for validity
    gf_final = max(best_global_gf, 1.0)
    pts = _axial_to_cartesian(best_global_axial, gf_final)
    
    # Recompute optimal center at (gf_final, beta)
    normals = _get_normals(best_global_beta)
    sol = _solve_center_lp(pts, normals)
    if sol is not None:
        t_opt, c_opt = sol
        best_global_S = (t_opt + APOTHEM) / APOTHEM
        best_global_center = c_opt
    
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)
    
    # Verify
    D = _compute_D(pts, (cx, cy), normals)
    S_final = (D + APOTHEM) / APOTHEM
    
    # Tiny safety scale
    SCALE = 1.0 + 1e-10
    pts_scaled = np.array([[cx + (x - cx) * SCALE, cy + (y - cy) * SCALE] for x, y in pts])
    
    max_proj = 0.0
    for nx, ny in normals:
        projs = (pts_scaled[:, 0] - cx) * nx + (pts_scaled[:, 1] - cy) * ny
        max_proj = max(max_proj, float(np.max(projs)), float(-np.min(projs)))
    S_scaled = (max_proj + APOTHEM) / APOTHEM
    
    inner = [[float(x), float(y), float(angle_deg)] for x, y in pts_scaled]
    center = [float(cx), float(cy)]
    side = float(S_scaled)
    rotation = float(angle_deg)
    
    return inner, center, side, rotation
