"""Improved program: contact-graph based packing with multi-start orientation optimization."""
import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar

N = 11
SQRT3 = math.sqrt(3)

# EVOLVE-BLOCK-START

def hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * np.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side_fast(inner, outer_center, outer_angle_deg):
    """
    Compute minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_degrees)
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    max_proj = -np.inf
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_proj:
            max_proj = cur
    if max_proj < 0:
        max_proj = 0.0
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth (log-sum-exp) surrogate for the max projection.
    Uses centered projections to reduce bias.
    """
    if not inner:
        return 0.0
    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])
    all_projs = []
    for (ix, iy, iang) in inner:
        verts = hex_vertices(ix, iy, 1.0, math.radians(iang))
        local = (verts - np.array([ox, oy])) @ R.T
        projs = local @ normals.T
        all_projs.append(projs)
    P = np.concatenate([p.ravel() for p in all_projs])
    if P.size == 0:
        return 0.0
    # Center projections to reduce bias
    P_centered = P - P.mean()
    m = P_centered.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P_centered - m)))) / beta
    # The actual max is approximately smooth_max + mean
    actual_max = smooth_max + P.mean()
    if actual_max < 0:
        actual_max = 0.0
    return 2.0 * actual_max / SQRT3

def sat_overlap(v1, v2, eps=1e-9):
    """SAT overlap check for two convex polygons (lists of (x,y))."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]; p2 = poly[(i + 1) % len(poly)]
            e = (p2[0] - p1[0], p2[1] - p1[1])
            n = (-e[1], e[0])
            ln = math.hypot(n[0], n[1])
            if ln > 1e-12:
                axes.append((n[0] / ln, n[1] / ln))
    for ax in axes:
        min1 = min(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        max1 = max(p[0] * ax[0] + p[1] * ax[1] for p in v1)
        min2 = min(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        max2 = max(p[0] * ax[0] + p[1] * ax[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_disjoint(inner, eps=1e-6):
    """Pairwise SAT check that all inner hexagons are disjoint."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap(polys[i], polys[j], eps):
                return False
    return True

def min_center_dist(a1, a2):
    """Compute minimum center-to-center distance for two unit hexagons with given angles."""
    # The minimum distance between two unit hexagons depends on relative angle
    # For parallel hexagons (angle diff ~ 0 or 60), min dist = sqrt(3)
    # For 30-degree offset, min dist is slightly less
    da = abs((a1 - a2) % 60.0)
    da = min(da, 60.0 - da)
    # Approximate: min_dist decreases slightly with angle offset
    # At 0 deg: sqrt(3) ≈ 1.732
    # At 30 deg: ~1.70 (rough estimate)
    base = SQRT3
    # Use a conservative lower bound
    return base * math.cos(math.radians(da / 2.0)) - 0.001

def overlap_penalty(inner, base_min_dist=SQRT3 - 0.002):
    """Smooth quadratic penalty when center distances fall below orientation-aware min_dist."""
    pen = 0.0
    n = len(inner)
    for i in range(n):
        xi, yi, ai = inner[i]
        for j in range(i + 1, n):
            xj, yj, aj = inner[j]
            d = math.hypot(xi - xj, yi - yj)
            md = min_center_dist(ai, aj)
            # Use a smooth transition: penalty is zero when d >= md, grows quadratically below
            if d < md:
                pen += (md - d) ** 2
            elif d < md + 0.05:
                # Small smooth transition zone
                t = (d - md) / 0.05
                pen += (0.05 * (1 - t)) ** 2
    return pen

def build_lattice_config(axial_coords, s=1.0 + 1e-8):
    """Build inner hexagons from axial coordinates with spacing s."""
    centers = [(1.5 * s * q, SQRT3 * s * (r + q / 2.0)) for q, r in axial_coords]
    return [[x, y, 0.0] for x, y in centers]

def try_orientation_optimization(inner, outer_center, outer_angle_deg, deadline):
    """Optimize outer hexagon orientation and center for fixed inner positions."""
    def obj_smooth(p):
        return compute_required_side_smooth(inner, (p[0], p[1]), p[2], beta=80.0)
    
    def obj_exact(p):
        return compute_required_side_fast(inner, (p[0], p[1]), p[2])
    
    best_val = float('inf')
    best_p = None
    # Try multiple starting angles
    for ia in range(0, 60, 5):
        if time.time() > deadline - 15:
            break
        p0 = np.array([outer_center[0], outer_center[1], float(ia)])
        try:
            r1 = minimize(obj_smooth, p0, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 500})
            r2 = minimize(obj_exact, r1.x, method='Nelder-Mead',
                          options={'maxiter': 600, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1200})
            if r2.fun < best_val:
                best_val = r2.fun
                best_p = r2.x.copy()
        except Exception:
            pass
    if best_p is not None:
        return (float(best_p[0]), float(best_p[1])), float(best_p[2]), best_val
    else:
        return outer_center, outer_angle_deg, compute_required_side_fast(inner, outer_center, outer_angle_deg)

def rigid_composite_optimize(axial_coords, deadline):
    """
    Treat the 11 hexagons as a rigid composite (6 DOF: 2 translation + 1 rotation 
    of the whole cluster + 2 outer center + 1 outer angle).
    This exploits the fact that in a lattice packing, all hexagons share the same orientation.
    """
    # Precompute local vertex offsets for each hexagon in the cluster
    # Each hexagon i has local vertices at axial position (q_i, r_i)
    local_vertices = []
    for q, r in axial_coords:
        cx = 1.5 * q
        cy = SQRT3 * (r + q / 2.0)
        # Unit hexagon vertices at angle 0
        thetas = np.arange(6) * np.pi / 3.0
        vx = cx + np.cos(thetas)
        vy = cy + np.sin(thetas)
        local_vertices.append(np.column_stack([vx, vy]))
    
    # Stack all vertices with their hexagon index
    all_local = np.vstack(local_vertices)  # (66, 2)
    
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])
    
    def obj_rigid_smooth(params, beta=100.0):
        """params: [tx, ty, cluster_angle_deg, ocx, ocy, outer_angle_deg]"""
        tx, ty, ca_deg, ocx, ocy, oa_deg = params
        ca = math.radians(ca_deg)
        oa = math.radians(oa_deg)
        
        # Rotate cluster
        cos_c, sin_c = math.cos(ca), math.sin(ca)
        Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])
        # Translate cluster
        pts = (all_local @ Rc.T) + np.array([tx, ty])
        
        # Rotate by outer angle and subtract outer center
        cos_o, sin_o = math.cos(oa), math.sin(oa)
        Ro = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
        local_pts = (pts - np.array([ocx, ocy])) @ Ro.T
        
        # Project onto 6 normals
        projs = local_pts @ normals.T  # (66, 6)
        P = projs.ravel()
        m = P.max()
        if m < -1e6:
            smooth_max = 0.0
        else:
            smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
            if smooth_max < 0:
                smooth_max = 0.0
        return 2.0 * smooth_max / SQRT3
    
    def obj_rigid_exact(params):
        tx, ty, ca_deg, ocx, ocy, oa_deg = params
        ca = math.radians(ca_deg)
        oa = math.radians(oa_deg)
        
        cos_c, sin_c = math.cos(ca), math.sin(ca)
        Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])
        pts = (all_local @ Rc.T) + np.array([tx, ty])
        
        cos_o, sin_o = math.cos(oa), math.sin(oa)
        Ro = np.array([[cos_o, sin_o], [-sin_o, cos_o]])
        local_pts = (pts - np.array([ocx, ocy])) @ Ro.T
        
        projs = local_pts @ normals.T
        max_proj = projs.max()
        if max_proj < 0:
            max_proj = 0.0
        return 2.0 * max_proj / SQRT3
    
    # Initial guess: center the cluster at origin, outer at centroid
    cx0 = all_local[:, 0].mean()
    cy0 = all_local[:, 1].mean()
    
    best_val = float('inf')
    best_params = None
    
    # Multi-start over cluster angle and outer angle
    for ca0 in range(0, 60, 10):
        for oa0 in range(0, 60, 15):
            if time.time() > deadline - 20:
                break
            p0 = np.array([0.0, 0.0, float(ca0), cx0, cy0, float(oa0)])
            try:
                r1 = minimize(obj_rigid_smooth, p0, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 800})
                r2 = minimize(obj_rigid_exact, r1.x, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 1500})
                if r2.fun < best_val:
                    best_val = r2.fun
                    best_params = r2.x.copy()
            except Exception:
                pass
    
    if best_params is None:
        return None
    
    # Convert back to individual hexagon positions
    tx, ty, ca_deg, ocx, ocy, oa_deg = best_params
    ca = math.radians(ca_deg)
    cos_c, sin_c = math.cos(ca), math.sin(ca)
    Rc = np.array([[cos_c, sin_c], [-sin_c, cos_c]])
    
    inner = []
    for q, r in axial_coords:
        cx = 1.5 * q
        cy = SQRT3 * (r + q / 2.0)
        pt = np.array([cx, cy]) @ Rc.T + np.array([tx, ty])
        inner.append([float(pt[0]), float(pt[1]), float(ca_deg)])
    
    return inner, (float(ocx), float(ocy)), float(oa_deg), float(best_val)

def joint_optimize(init_inner, init_outer_center, init_outer_angle, deadline):
    """Joint optimization of inner positions/angles and outer hexagon with DE + adaptive SA."""
    def obj_joint_smooth(params, beta=100.0):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_smooth(il, (ocx, ocy), oa, beta=beta)
        pen = overlap_penalty(il)
        return side + 800.0 * pen
    
    def obj_joint_exact(params):
        il = []
        for i in range(N):
            il.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])
        ocx, ocy, oa = params[N * 3], params[N * 3 + 1], params[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 5000.0 * pen
    
    # Build initial params
    p0 = np.zeros(N * 3 + 3)
    for i, (x, y, a) in enumerate(init_inner):
        p0[i * 3] = x
        p0[i * 3 + 1] = y
        p0[i * 3 + 2] = a
    p0[N * 3] = init_outer_center[0]
    p0[N * 3 + 1] = init_outer_center[1]
    p0[N * 3 + 2] = init_outer_angle
    
    best_params = p0.copy()
    best_val_j = obj_joint_smooth(p0)
    
    # Phase A: Differential Evolution for global exploration
    if time.time() < deadline - 30:
        from scipy.optimize import differential_evolution
        # Define bounds for DE
        bounds = []
        for i in range(N):
            bounds.extend([(-5, 5), (-5, 5), (-180, 180)])
        bounds.extend([(-3, 3), (-3, 3), (-90, 90)])
        
        np.random.seed(789)
        try:
            de_result = differential_evolution(
                obj_joint_smooth, bounds,
                seed=789, maxiter=50, popsize=15,
                tol=1e-6, polish=True,
                options={'maxfev': 5000}
            )
            if de_result.fun < best_val_j:
                best_val_j = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass
    
    # Phase B: Adaptive simulated annealing from DE result
    current = best_params.copy()
    current_val = best_val_j
    
    T = 0.2
    T_min = 0.0005
    cool = 0.9997
    max_iter = 15000
    np.random.seed(123)
    step_pos = 0.2
    step_ang = 8.0
    step_oc = 0.12
    step_oa = 3.0
    
    accept_count = 0
    window = 400
    accept_rate = 0.5
    
    for it in range(max_iter):
        if time.time() > deadline - 20:
            break
        nb = current.copy()
        r = np.random.random()
        if r < 0.55:
            idx = np.random.randint(0, N)
            dim = np.random.randint(0, 2)
            nb[idx * 3 + dim] += np.random.normal(0, step_pos)
        elif r < 0.75:
            idx = np.random.randint(0, N)
            nb[idx * 3 + 2] += np.random.normal(0, step_ang)
        elif r < 0.87:
            dim = np.random.randint(0, 2)
            nb[N * 3 + dim] += np.random.normal(0, step_oc)
        else:
            nb[N * 3 + 2] += np.random.normal(0, step_oa)
        
        nb_val = obj_joint_smooth(nb)
        accepted = False
        if nb_val < current_val or np.random.random() < math.exp(-(nb_val - current_val) / max(T, 1e-10)):
            current = nb
            current_val = nb_val
            accepted = True
            if current_val < best_val_j:
                best_params = current.copy()
                best_val_j = current_val
        
        if (it + 1) % window == 0:
            accept_rate = accept_count / window
            accept_count = 0
            if accept_rate < 0.15:
                step_pos *= 0.8
                step_ang *= 0.8
                step_oc *= 0.8
                step_oa *= 0.8
            elif accept_rate > 0.5:
                step_pos *= 1.2
                step_ang *= 1.2
                step_oc *= 1.2
                step_oa *= 1.2
            step_pos = max(0.01, min(step_pos, 1.0))
            step_ang = max(1.0, min(step_ang, 30.0))
            step_oc = max(0.01, min(step_oc, 0.5))
            step_oa = max(0.5, min(step_oa, 15.0))
        
        if accepted:
            accept_count += 1
        T *= cool
        if T < T_min:
            break
    
    # Extract best
    best_inner_j = []
    for i in range(N):
        best_inner_j.append([float(best_params[i * 3]), float(best_params[i * 3 + 1]), float(best_params[i * 3 + 2])])
    best_oc_j = (float(best_params[N * 3]), float(best_params[N * 3 + 1]))
    best_oa_j = float(best_params[N * 3 + 2])
    
    return best_inner_j, best_oc_j, best_oa_j, best_params

def final_polish(params, deadline):
    """Two-stage high-precision polish: coarse Nelder-Mead then fine with tighter tolerance."""
    def obj_polish(p):
        il = []
        for i in range(N):
            il.append([p[i * 3], p[i * 3 + 1], p[i * 3 + 2]])
        ocx, ocy, oa = p[N * 3], p[N * 3 + 1], p[N * 3 + 2]
        side = compute_required_side_fast(il, (ocx, ocy), oa)
        pen = overlap_penalty(il)
        return side + 8000.0 * pen
    
    best_res = params.copy()
    best_val = obj_polish(params)
    
    # Stage 1: Coarse multi-start
    np.random.seed(456)
    for restart in range(6):
        if time.time() > deadline - 10:
            break
        p_try = params.copy()
        n_perturb = np.random.randint(2, 5)
        idxs = np.random.choice(len(params), size=n_perturb, replace=False)
        p_try[idxs] += np.random.normal(0, 0.01, size=n_perturb)
        try:
            res = minimize(obj_polish, p_try, method='Nelder-Mead',
                           options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res.x.copy()
        except Exception:
            pass
    
    # Stage 2: Fine polish from best
    if time.time() < deadline - 5:
        try:
            res = minimize(obj_polish, best_res, method='Nelder-Mead',
                           options={'maxiter': 6000, 'xatol': 1e-15, 'fatol': 1e-17, 'maxfev': 10000})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res.x.copy()
        except Exception:
            pass
    
    # Stage 3: Ultra-fine final pass
    if time.time() < deadline - 2:
        try:
            res = minimize(obj_polish, best_res, method='Nelder-Mead',
                           options={'maxiter': 4000, 'xatol': 1e-16, 'fatol': 1e-18, 'maxfev': 6000})
            if res.fun < best_val:
                best_val = res.fun
                best_res = res.x.copy()
        except Exception:
            pass
    
    return best_res

def construct():
    t0 = time.time()
    deadline = t0 + 142.0  # 8s safety margin
    
    # --- Define multiple candidate configurations ---
    # Config 1: Standard hexagonal flower (1+6+4)
    axial1 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (1, 1), (-2, 1), (-1, -1)]
    
    # Config 2: Row-based 4-3-4
    axial2 = [(-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    # Config 3: Compact 3-5-3
    axial3 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    # Config 4: Another variant - 5-3-3
    axial4 = [(-2, -1), (-1, -1), (0, -1), (1, -1), (2, -1),
              (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1)]
    
    # Config 5: Centered with different outer placement
    axial5 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, 0), (0, 2), (-2, 1), (-2, -1)]
    
    # Config 6: Linear-ish 2-7-2
    axial6 = [(-1, -1), (0, -1),
              (-3, 0), (-2, 0), (-1, 0), (0, 0), (1, 0), (2, 0), (3, 0),
              (-1, 1), (0, 1)]
    
    # Config 7: Diamond 1-3-5-3-1 style (compact)
    axial7 = [(0, 0),
              (-1, 0), (0, -1), (1, -1),
              (-2, 1), (-1, 1), (0, 1), (1, 0), (2, -1),
              (-1, -1), (1, -2)]
    
    # Config 8: Offset row 3-4-4
    axial8 = [(-1, -1), (0, -1), (1, -1),
              (-2, 0), (-1, 0), (0, 0), (1, 0),
              (-1, 1), (0, 1), (1, 1), (2, 1)]
    
    # Config 9: Compact cluster with one offset
    axial9 = [(0, 0),
              (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
              (2, -1), (2, 0), (1, 1), (0, 2)]
    
    configs = [axial1, axial2, axial3, axial4, axial5, axial6, axial7, axial8, axial9]
    
    best_side = float('inf')
    best_inner = None
    best_outer_center = None
    best_outer_angle = 0.0
    best_params = None
    
    # Phase 0: Rigid composite optimization (fast, 6 DOF)
    for ci, axial in enumerate(configs[:6]):
        if time.time() > deadline - 50:
            break
        result = rigid_composite_optimize(axial, deadline)
        if result is not None:
            rc_inner, rc_oc, rc_oa, rc_side = result
            if check_disjoint(rc_inner, eps=1e-5) and rc_side < best_side:
                best_side = rc_side
                best_inner = rc_inner
                best_outer_center = rc_oc
                best_outer_angle = rc_oa
                p0 = np.zeros(N * 3 + 3)
                for i, (x, y, a) in enumerate(rc_inner):
                    p0[i * 3] = x
                    p0[i * 3 + 1] = y
                    p0[i * 3 + 2] = a
                p0[N * 3] = rc_oc[0]
                p0[N * 3 + 1] = rc_oc[1]
                p0[N * 3 + 2] = rc_oa
                best_params = p0.copy()
    
    # Phase 1: For each config, optimize outer hexagon (individual angles)
    for ci, axial in enumerate(configs):
        if time.time() > deadline - 40:
            break
        s0 = 1.0 + 1e-8
        init_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in init_inner) / len(init_inner)
        cy0 = sum(c[1] for c in init_inner) / len(init_inner)
        
        oc, oa, side_val = try_orientation_optimization(init_inner, (cx0, cy0), 0.0, deadline)
        
        if side_val < best_side:
            best_side = side_val
            best_inner = init_inner
            best_outer_center = oc
            best_outer_angle = oa
            # Store params for joint optimization
            p0 = np.zeros(N * 3 + 3)
            for i, (x, y, a) in enumerate(init_inner):
                p0[i * 3] = x
                p0[i * 3 + 1] = y
                p0[i * 3 + 2] = a
            p0[N * 3] = oc[0]
            p0[N * 3 + 1] = oc[1]
            p0[N * 3 + 2] = oa
            best_params = p0.copy()
    
    if best_inner is None:
        # Fallback
        axial = configs[0]
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(axial, s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
        p0 = np.zeros(N * 3 + 3)
        for i, (x, y, a) in enumerate(best_inner):
            p0[i * 3] = x
            p0[i * 3 + 1] = y
            p0[i * 3 + 2] = a
        p0[N * 3] = cx0
        p0[N * 3 + 1] = cy0
        p0[N * 3 + 2] = 0.0
        best_params = p0.copy()
    
    # Phase 2: Joint optimization from best config
    if time.time() < deadline - 25:
        joint_inner, joint_oc, joint_oa, joint_params = joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(joint_inner, eps=1e-5):
            joint_side = compute_required_side_fast(joint_inner, joint_oc, joint_oa)
            if joint_side < best_side:
                best_inner = joint_inner
                best_outer_center = joint_oc
                best_outer_angle = joint_oa
                best_side = joint_side
                best_params = joint_params
        else:
            # Try to repair: use params but verify
            test_inner = []
            for i in range(N):
                test_inner.append([float(joint_params[i * 3]), float(joint_params[i * 3 + 1]), float(joint_params[i * 3 + 2])])
            if check_disjoint(test_inner, eps=1e-4):
                test_side = compute_required_side_fast(test_inner, (float(joint_params[N*3]), float(joint_params[N*3+1])), float(joint_params[N*3+2]))
                if test_side < best_side:
                    best_inner = test_inner
                    best_outer_center = (float(joint_params[N*3]), float(joint_params[N*3+1]))
                    best_outer_angle = float(joint_params[N*3+2])
                    best_side = test_side
    
    # Phase 2.5: Second joint optimization pass from current best (if time allows)
    if time.time() < deadline - 35 and best_params is not None:
        second_inner, second_oc, second_oa, second_params = joint_optimize(
            best_inner, best_outer_center, best_outer_angle, deadline)
        
        if check_disjoint(second_inner, eps=1e-5):
            second_side = compute_required_side_fast(second_inner, second_oc, second_oa)
            if second_side < best_side:
                best_inner = second_inner
                best_outer_center = second_oc
                best_outer_angle = second_oa
                best_side = second_side
                best_params = second_params
    
    # Phase 3: Final high-precision polish
    if time.time() < deadline - 5 and best_params is not None:
        polished = final_polish(best_params, deadline)
        polish_inner = []
        for i in range(N):
            polish_inner.append([float(polished[i * 3]), float(polished[i * 3 + 1]), float(polished[i * 3 + 2])])
        polish_oc = (float(polished[N * 3]), float(polished[N * 3 + 1]))
        polish_oa = float(polished[N * 3 + 2])
        
        if check_disjoint(polish_inner, eps=1e-5):
            polish_side = compute_required_side_fast(polish_inner, polish_oc, polish_oa)
            if polish_side < best_side:
                best_inner = polish_inner
                best_outer_center = polish_oc
                best_outer_angle = polish_oa
                best_side = polish_side
    
    # Final verification
    if not check_disjoint(best_inner, eps=1e-5):
        # Fallback to first lattice config
        s0 = 1.0 + 1e-8
        best_inner = build_lattice_config(configs[0], s0)
        cx0 = sum(c[0] for c in best_inner) / len(best_inner)
        cy0 = sum(c[1] for c in best_inner) / len(best_inner)
        best_outer_center = (cx0, cy0)
        best_outer_angle = 0.0
        best_side = compute_required_side_fast(best_inner, best_outer_center, best_outer_angle)
    
    return best_inner, best_outer_center, best_side, best_outer_angle

# EVOLVE-BLOCK-END
