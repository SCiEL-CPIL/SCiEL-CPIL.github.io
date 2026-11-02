import math
import numpy as np
import time
from scipy.optimize import minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 6.0
EPS = 1e-10

def _hex_vertices(cx, cy, side, rot):
    """Vertices of a regular hexagon centered at (cx,cy), side length, rotation (radians)."""
    vs = []
    for k in range(6):
        a = rot + k * math.pi / 3.0
        vs.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return np.array(vs)

def _hex_normals(rot):
    """Unit outward normals of a hexagon with given rotation (radians)."""
    ns = []
    for k in range(6):
        a = rot + math.pi / 6.0 + k * math.pi / 3.0
        ns.append((math.cos(a), math.sin(a)))
    return np.array(ns)

def _min_side_for_positions(pts, rot_inner, rot_outer):
    """
    Given inner hexagon centers (n,2), inner rotation, outer rotation,
    find the minimal outer side length S such that all inner hexagons fit
    inside the outer hexagon (centered at origin for simplicity; we'll re-center).
    Returns (S, center_offset) where center_offset shifts outer hex center.
    """
    n = len(pts)
    # Vertices of each unit inner hexagon
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        for k in range(6):
            a = rot_inner + k * math.pi / 3.0
            verts[i, k, 0] = pts[i, 0] + math.cos(a)
            verts[i, k, 1] = pts[i, 1] + math.sin(a)
    
    # Outer normals
    on = _hex_normals(rot_outer)  # (6,2)
    
    # For a given outer center c and apothem D (S = (D + APOTHEM)/APOTHEM),
    # constraint: for all i,k,j: |dot(verts[i,k] - c, on[j])| <= D
    # Equivalent: -D <= dot(v-c, n_j) <= D for all
    # => -D - dot(v, n_j) + dot(c, n_j) <= 0 and D - dot(v, n_j) + dot(c, n_j) >= 0
    # We want to minimize D over c. This is a linear program in (c_x, c_y, D).
    
    # Collect all vertices
    all_v = verts.reshape(-1, 2)  # (6n, 2)
    m = len(all_v)
    
    # LP: minimize D
    # Variables: c_x, c_y, D
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for j in range(6):
        nx, ny = on[j]
        # dot(v - c, n_j) <= D  =>  nx*vx + ny*vy - nx*cx - ny*cy - D <= 0
        # dot(v - c, n_j) >= -D => -nx*vx - ny*vy + nx*cx + ny*cy - D <= 0
        A_ub.append([-nx, -ny, -1.0])
        b_ub.append(-(all_v[:, 0] * nx + all_v[:, 1] * ny))
        A_ub.append([nx, ny, -1.0])
        b_ub.append(all_v[:, 0] * nx + all_v[:, 1] * ny)
    
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]
    
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = res.x
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy), D
        else:
            return float('inf'), None, None
    except Exception:
        return float('inf'), None, None

def _smooth_max_proj(all_v, on, c, temp):
    """Smooth approximation of max |dot(v - c, n_j)| over all v, j."""
    # all_v: (M, 2), on: (6,2), c: (2,)
    diff = all_v - c  # (M, 2)
    projs = diff @ on.T  # (M, 6)
    abs_projs = np.abs(projs)
    flat = abs_projs.ravel()
    m = flat.max()
    if m > 1e6:
        return 1e6
    shifted = (flat - m) / temp
    shifted = np.clip(shifted, -60, 60)
    lse = m / temp + np.log(np.sum(np.exp(shifted)))
    return temp * lse

def _optimize_config(pts, rot_inner, rot_outer, time_budget=8.0):
    """
    Given initial positions and rotations, refine positions (and optionally outer rotation)
    to minimize S. Uses smooth objective + Nelder-Mead, then exact polish.
    Returns (S, pts_opt, rot_inner_opt, rot_outer_opt, center).
    """
    t0 = time.time()
    n = len(pts)
    pts = np.array(pts, dtype=float)
    
    # Build vertex array (depends on rot_inner)
    def make_verts(p, ri):
        v = np.zeros((n, 6, 2))
        for i in range(n):
            for k in range(6):
                a = ri + k * math.pi / 3.0
                v[i, k, 0] = p[i, 0] + math.cos(a)
                v[i, k, 1] = p[i, 1] + math.sin(a)
        return v.reshape(-1, 2)
    
    def exact_S(p, ri, ro):
        v = make_verts(p, ri)
        on = _hex_normals(ro)
        # LP
        m = len(v)
        c_obj = np.array([0.0, 0.0, 1.0])
        A_ub = []
        b_ub = []
        for j in range(6):
            nx, ny = on[j]
            A_ub.append([-nx, -ny, -1.0])
            b_ub.append(-(v[:, 0] * nx + v[:, 1] * ny))
            A_ub.append([nx, ny, -1.0])
            b_ub.append(v[:, 0] * nx + v[:, 1] * ny)
        A_ub = np.array(A_ub)
        b_ub = np.array(b_ub)
        bounds = [(None, None), (None, None), (0.0, None)]
        try:
            from scipy.optimize import linprog
            res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
            if res.status == 0:
                cx, cy, D = res.x
                return (D + APOTHEM) / APOTHEM, (cx, cy)
        except Exception:
            pass
        return float('inf'), None
    
    def smooth_obj(params, temp):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            return 1e6
        p = params[:2 * n].reshape(n, 2)
        ro = params[2 * n]
        ri = params[2 * n + 1]
        v = make_verts(p, ri)
        on = _hex_normals(ro)
        # Center estimate: mean of positions
        c = p.mean(axis=0)
        D = _smooth_max_proj(v, on, c, temp)
        return (D + APOTHEM) / APOTHEM
    
    # Start with exact evaluation
    S0, c0 = exact_S(pts, rot_inner, rot_outer)
    if c0 is None:
        c0 = (0.0, 0.0)
    best_S = S0
    best_p = pts.copy()
    best_ri = rot_inner
    best_ro = rot_outer
    best_c = c0
    
    # Smooth Nelder-Mead refinement with decreasing temperature
    temps = [0.5, 0.2, 0.08, 0.03, 0.01]
    for temp in temps:
        if time.time() - t0 > time_budget:
            break
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        
        x0 = best_p.flatten().tolist() + [best_ro, best_ri]
        
        def obj(params, t=temp):
            return smooth_obj(params, t)
        
        try:
            res = minimize(obj, x0, method='Nelder-Mead',
                          options={'maxiter': 800, 'xatol': 1e-11, 'fatol': 1e-11,
                                   'adaptive': True})
            if res.fun < 1e5:
                p_new = res.x[:2 * n].reshape(n, 2)
                ro_new = res.x[2 * n]
                ri_new = res.x[2 * n + 1]
                S_new, c_new = exact_S(p_new, ri_new, ro_new)
                if c_new is not None and S_new < best_S:
                    best_S = S_new
                    best_p = p_new
                    best_ri = ri_new
                    best_ro = ro_new
                    best_c = c_new
        except Exception:
            pass
    
    # Final exact polish: Nelder-Mead on exact S (piecewise linear)
    if time.time() - t0 < time_budget and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
        def exact_obj(params):
            if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
                return 1e6
            p = params[:2 * n].reshape(n, 2)
            ro = params[2 * n]
            ri = params[2 * n + 1]
            S, _ = exact_S(p, ri, ro)
            return S
        
        x0 = best_p.flatten().tolist() + [best_ro, best_ri]
        try:
            res = minimize(exact_obj, x0, method='Nelder-Mead',
                          options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-12,
                                   'adaptive': True})
            if res.fun < best_S:
                p_new = res.x[:2 * n].reshape(n, 2)
                ro_new = res.x[2 * n]
                ri_new = res.x[2 * n + 1]
                S_new, c_new = exact_S(p_new, ri_new, ro_new)
                if c_new is not None and S_new < best_S:
                    best_S = S_new
                    best_p = p_new
                    best_ri = ri_new
                    best_ro = ro_new
                    best_c = c_new
        except Exception:
            pass
    
    return best_S, best_p, best_ri, best_ro, best_c

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0
    
    # Generate candidate configurations.
    # Strategy: hexagonal lattice of centers with various rotations.
    # The 30-degree rotation trick: place hexagons in a hex lattice but rotate
    # the inner hexagons by 30 degrees relative to the outer, so they interlock.
    
    candidates = []
    
    # Configuration A: "Flower" - 1 center + 6 around + 4 more, all rotated 30 deg
    # Centers in hex lattice (axial to cartesian, side=1 hexagons touching means center dist = 2*APOTHEM = sqrt(3))
    d = 2.0 * APOTHEM  # distance between centers of touching hexagons (same orientation)
    
    # For rotated hexagons (30 deg), the touching distance changes.
    # Two unit hexagons, one rotated 30 deg relative to other: min center distance for no overlap
    # is approximately 2 * (cos(15 deg)) ~ 1.93 (slightly less than sqrt(3) ~ 1.732? No, more)
    # Actually for same orientation: min dist = 2 * APOTHEM = sqrt(3) ≈ 1.732
    # For 30 deg rotation: the apothem direction rotates, so the constraint is different.
    # Let's just use d = sqrt(3) as a safe base and let optimization refine.
    
    def hex_lattice_centers(n_rings, d):
        """Generate hex lattice centers out to n_rings."""
        centers = [(0.0, 0.0)]
        for ring in range(1, n_rings + 1):
            # 6 * ring points
            # Start at (ring, 0) and walk
            q, r = ring, 0
            for step in range(6):
                for _ in range(ring):
                    x = d * (q + r / 2.0)
                    y = d * (math.sqrt(3) / 2.0) * r
                    centers.append((x, y))
                    # Move along edge
                    if step == 0:
                        q += 1
                    elif step == 1:
                        r += 1
                    elif step == 2:
                        q -= 1; r -= 1
                    elif step == 3:
                        q -= 1
                    elif step == 4:
                        r -= 1
                    elif step == 5:
                        q += 1; r += 1
        return centers
    
    # We need exactly 11 centers. Generate a few base patterns.
    
    # Pattern 1: Center + 6 ring + 4 from second ring (select 4)
    centers1 = [(0.0, 0.0)]
    # Ring 1: 6 points at distance d
    for k in range(6):
        a = k * math.pi / 3.0
        centers1.append((d * math.cos(a), d * math.sin(a)))
    # Ring 2: 12 points, pick 4 that form a compact shape
    # Ring 2 axial: (2,0),(1,1),(0,2),(-1,2),(-2,1),(-2,0),(-1,-1),(0,-2),(1,-2),(2,-1),(2,0)... 
    # Let's pick (2,0), (0,2), (-2,1), (-1,-1) or similar
    ring2_axial = [(2,0), (1,1), (0,2), (-1,2), (-2,1), (-2,0), (-1,-1), (0,-2), (1,-2), (2,-1)]
    # Pick 4: (2,0), (0,2), (-2,1), (0,-2) for a balanced shape
    pick4 = [(2,0), (0,2), (-2,1), (0,-2)]
    for q, r in pick4:
        x = d * (q + r / 2.0)
        y = d * (math.sqrt(3) / 2.0) * r
        centers1.append((x, y))
    candidates.append((np.array(centers1), math.pi/6, 0.0))  # inner rot 30 deg, outer 0
    
    # Pattern 2: Same but inner rot 0
    candidates.append((np.array(centers1), 0.0, 0.0))
    
    # Pattern 3: 3-4-4 arrangement (rows)
    centers3 = []
    # Row 0: 3 centers
    for i in range(3):
        centers3.append((i * d, 0.0))
    # Row 1: 4 centers, offset
    for i in range(4):
        centers3.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    # Row 2: 4 centers
    for i in range(4):
        centers3.append((i * d, d * math.sqrt(3)))
    candidates.append((np.array(centers3), math.pi/6, 0.0))
    candidates.append((np.array(centers3), 0.0, 0.0))
    
    # Pattern 4: 2-4-3-2 arrangement
    centers4 = []
    for i in range(2):
        centers4.append(((i - 0.5) * d, 0.0))
    for i in range(4):
        centers4.append(((i - 1.5) * d, d * math.sqrt(3) / 2.0))
    for i in range(3):
        centers4.append(((i - 1.0) * d, d * math.sqrt(3)))
    for i in range(2):
        centers4.append(((i - 0.5) * d, d * 3 * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers4), math.pi/6, 0.0))
    candidates.append((np.array(centers4), 0.0, 0.0))
    
    # Pattern 5: 1-6-4 with tighter spacing (allow slight overlap in init, optimization fixes)
    centers5 = [(0.0, 0.0)]
    d5 = d * 0.98
    for k in range(6):
        a = k * math.pi / 3.0
        centers5.append((d5 * math.cos(a), d5 * math.sin(a)))
    pick5 = [(2,0), (1,1), (-1,2), (-2,1)]
    for q, r in pick5:
        x = d5 * (q + r / 2.0)
        y = d5 * (math.sqrt(3) / 2.0) * r
        centers5.append((x, y))
    candidates.append((np.array(centers5), math.pi/6, 0.0))
    
    # Pattern 6: Random-ish perturbation of pattern 1
    np.random.seed(42)
    centers6 = centers1.copy()
    centers6 = np.array(centers6) + np.random.normal(0, 0.1, size=(11, 2))
    candidates.append((centers6, math.pi/6, 0.0))
    
    # Pattern 7: Star shape - center + 6 at 60 deg + 4 at 30 deg offset (rotated ring)
    centers7 = [(0.0, 0.0)]
    for k in range(6):
        a = k * math.pi / 3.0
        centers7.append((d * math.cos(a), d * math.sin(a)))
    # 4 more at intermediate angles, slightly further
    for k in range(4):
        a = math.pi/6 + k * math.pi/2.0
        centers7.append((d * 1.5 * math.cos(a), d * 1.5 * math.sin(a)))
    candidates.append((np.array(centers7), math.pi/6, 0.0))
    
    # Pattern 8: Linear-ish 5-6
    centers8 = []
    for i in range(5):
        centers8.append((i * d, 0.0))
    for i in range(6):
        centers8.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers8), math.pi/6, 0.0))
    
    # Pattern 9: 4-4-3
    centers9 = []
    for i in range(4):
        centers9.append((i * d, 0.0))
    for i in range(4):
        centers9.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    for i in range(3):
        centers9.append(((i + 0.5) * d, d * math.sqrt(3)))
    candidates.append((np.array(centers9), math.pi/6, 0.0))
    
    # Pattern 10: 3-5-3
    centers10 = []
    for i in range(3):
        centers10.append(((i - 1.0) * d, 0.0))
    for i in range(5):
        centers10.append(((i - 2.0) * d, d * math.sqrt(3) / 2.0))
    for i in range(3):
        centers10.append(((i - 1.0) * d, d * math.sqrt(3)))
    candidates.append((np.array(centers10), math.pi/6, 0.0))
    
    # Pattern 11: 2-5-4
    centers11 = []
    for i in range(2):
        centers11.append(((i - 0.5) * d, 0.0))
    for i in range(5):
        centers11.append(((i - 2.0) * d, d * math.sqrt(3) / 2.0))
    for i in range(4):
        centers11.append(((i - 1.5) * d, d * math.sqrt(3)))
    candidates.append((np.array(centers11), math.pi/6, 0.0))
    
    # Pattern 12: 4-3-4
    centers12 = []
    for i in range(4):
        centers12.append((i * d, 0.0))
    for i in range(3):
        centers12.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    for i in range(4):
        centers12.append((i * d, d * math.sqrt(3)))
    candidates.append((np.array(centers12), math.pi/6, 0.0))
    
    # Pattern 13: 5-3-3
    centers13 = []
    for i in range(5):
        centers13.append((i * d, 0.0))
    for i in range(3):
        centers13.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    for i in range(3):
        centers13.append(((i + 1.0) * d, d * math.sqrt(3)))
    candidates.append((np.array(centers13), math.pi/6, 0.0))
    
    # Pattern 14: Diamond 1-3-4-3
    centers14 = [(0.0, 0.0)]
    for i in range(3):
        centers14.append(((i - 1.0) * d, d * math.sqrt(3) / 2.0))
    for i in range(4):
        centers14.append(((i - 1.5) * d, d * math.sqrt(3)))
    for i in range(3):
        centers14.append(((i - 1.0) * d, d * 3 * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers14), math.pi/6, 0.0))
    
    # Pattern 15: 1-3-3-4 (vertical diamond)
    centers15 = [(0.0, 0.0)]
    for i in range(3):
        centers15.append(((i - 1.0) * d, d * math.sqrt(3) / 2.0))
    for i in range(3):
        centers15.append(((i - 1.0) * d, d * math.sqrt(3)))
    for i in range(4):
        centers15.append(((i - 1.5) * d, d * 3 * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers15), math.pi/6, 0.0))
    
    # Pattern 16: Compact cluster - 1 center + 6 + 4 closer
    centers16 = [(0.0, 0.0)]
    d16 = d * 0.95
    for k in range(6):
        a = k * math.pi / 3.0
        centers16.append((d16 * math.cos(a), d16 * math.sin(a)))
    pick16 = [(2,0), (1,1), (0,2), (-1,2)]
    for q, r in pick16:
        x = d16 * (q + r / 2.0)
        y = d16 * (math.sqrt(3) / 2.0) * r
        centers16.append((x, y))
    candidates.append((np.array(centers16), math.pi/6, 0.0))
    
    # Pattern 17: 6-5 (two rows)
    centers17 = []
    for i in range(6):
        centers17.append((i * d, 0.0))
    for i in range(5):
        centers17.append(((i + 0.5) * d, d * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers17), math.pi/6, 0.0))
    
    # Pattern 18: 7-4
    centers18 = []
    for i in range(7):
        centers18.append((i * d, 0.0))
    for i in range(4):
        centers18.append(((i + 1.5) * d, d * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers18), math.pi/6, 0.0))
    
    # Pattern 19: 4-7
    centers19 = []
    for i in range(4):
        centers19.append(((i + 1.5) * d, 0.0))
    for i in range(7):
        centers19.append((i * d, d * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers19), math.pi/6, 0.0))
    
    # Pattern 20: 5-6
    centers20 = []
    for i in range(5):
        centers20.append(((i + 0.5) * d, 0.0))
    for i in range(6):
        centers20.append((i * d, d * math.sqrt(3) / 2.0))
    candidates.append((np.array(centers20), math.pi/6, 0.0))
    
    # Also try with outer rotation = 30 deg (inner 0)
    for i in range(len(candidates)):
        p, ri, ro = candidates[i]
        candidates.append((p, 0.0, math.pi/6))
    
    # Deduplicate and limit
    seen = set()
    unique_candidates = []
    for p, ri, ro in candidates:
        key = (tuple(np.round(p, 6)), round(ri, 6), round(ro, 6))
        if key not in seen:
            seen.add(key)
            unique_candidates.append((p, ri, ro))
    candidates = unique_candidates
    
    # Optimize each candidate
    best_S = float('inf')
    best_p = None
    best_ri = 0.0
    best_ro = 0.0
    best_c = (0.0, 0.0)
    
    # Time budget per candidate
    n_cands = len(candidates)
    time_per_cand = (TIME_LIMIT - SAFETY_MARGIN - (time.time() - START_TIME)) / max(1, n_cands)
    time_per_cand = max(1.0, min(6.0, time_per_cand))
    
    for idx, (p, ri, ro) in enumerate(candidates):
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break
        S, p_opt, ri_opt, ro_opt, c_opt = _optimize_config(p, ri, ro, time_budget=time_per_cand)
        if S < best_S:
            best_S = S
            best_p = p_opt
            best_ri = ri_opt
            best_ro = ro_opt
            best_c = c_opt
    
    # Final verification and safety scaling
    if best_p is None or best_S == float('inf'):
        # Fallback
        best_p = np.array(centers1)
        best_ri = math.pi / 6
        best_ro = 0.0
        best_S, best_p, best_ri, best_ro, best_c = _optimize_config(best_p, best_ri, best_ro, time_budget=10.0)
    
    cx, cy = best_c
    angle_inner_deg = math.degrees(best_ri)
    angle_outer_deg = math.degrees(best_ro)
    
    # Compute final S with exact LP for safety
    S_final, c_final = _exact_S_for_output(best_p, best_ri, best_ro)
    if c_final is not None:
        cx, cy = c_final
        best_S = S_final
    
    # Apply tiny safety scale to ensure no floating-point overlap
    SCALE = 1.0 + 1e-10
    p_scaled = []
    for x, y in best_p:
        sx = cx + (x - cx) * SCALE
        sy = cy + (y - cy) * SCALE
        p_scaled.append((sx, sy))
    
    # Recompute S after scaling
    p_scaled_arr = np.array(p_scaled)
    S_final2, c_final2 = _exact_S_for_output(p_scaled_arr, best_ri, best_ro)
    if c_final2 is not None:
        cx, cy = c_final2
        best_S = S_final2
    
    inner = []
    for x, y in p_scaled:
        inner.append([float(x), float(y), float(angle_inner_deg)])
    
    center = [float(cx), float(cy)]
    side = float(best_S)
    rotation = float(angle_outer_deg)
    
    return inner, center, side, rotation

def _exact_S_for_output(pts, ri, ro):
    """Exact S computation for output."""
    n = len(pts)
    verts = np.zeros((n, 6, 2))
    for i in range(n):
        for k in range(6):
            a = ri + k * math.pi / 3.0
            verts[i, k, 0] = pts[i, 0] + math.cos(a)
            verts[i, k, 1] = pts[i, 1] + math.sin(a)
    all_v = verts.reshape(-1, 2)
    on = _hex_normals(ro)
    m = len(all_v)
    c_obj = np.array([0.0, 0.0, 1.0])
    A_ub = []
    b_ub = []
    for j in range(6):
        nx, ny = on[j]
        A_ub.append([-nx, -ny, -1.0])
        b_ub.append(-(all_v[:, 0] * nx + all_v[:, 1] * ny))
        A_ub.append([nx, ny, -1.0])
        b_ub.append(all_v[:, 0] * nx + all_v[:, 1] * ny)
    A_ub = np.array(A_ub)
    b_ub = np.array(b_ub)
    bounds = [(None, None), (None, None), (0.0, None)]
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        if res.status == 0:
            cx, cy, D = res.x
            S = (D + APOTHEM) / APOTHEM
            return S, (cx, cy)
    except Exception:
        pass
    return float('inf'), None
