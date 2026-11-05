"""Hexagon packing: 12 unit hexagons inside a regular hexagon, minimize outer side."""
import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12
TIME_LIMIT = 145.0  # safety margin below 150s

def _hex_vertices(cx, cy, angle_deg, side=1.0):
    """Vertices of a regular hexagon centered at (cx,cy) with rotation angle_deg."""
    a = math.radians(angle_deg)
    ca, sa = math.cos(a), math.sin(a)
    verts = []
    for k in range(6):
        theta = math.radians(60 * k)
        x = side * math.cos(theta)
        y = side * math.sin(theta)
        verts.append((cx + ca * x - sa * y, cy + sa * x + ca * y))
    return verts

def _project_polygon(verts, axis):
    """Project polygon vertices onto axis, return (min, max)."""
    ax, ay = axis
    vals = [vx * ax + vy * ay for vx, vy in verts]
    return min(vals), max(vals)

def _axes_from_polygon(verts):
    """Get edge normals (perpendicular to edges) for SAT."""
    axes = []
    n = len(verts)
    for i in range(n):
        x1, y1 = verts[i]
        x2, y2 = verts[(i + 1) % n]
        dx, dy = x2 - x1, y2 - y1
        length = math.hypot(dx, dy)
        if length < 1e-12:
            continue
        nx, ny = -dy / length, dx / length
        axes.append((nx, ny))
    return axes

def _center_distance(vertsA, vertsB):
    """Approximate center distance between two polygons."""
    cxA = sum(v[0] for v in vertsA) / len(vertsA)
    cyA = sum(v[1] for v in vertsA) / len(vertsA)
    cxB = sum(v[0] for v in vertsB) / len(vertsB)
    cyB = sum(v[1] for v in vertsB) / len(vertsB)
    return math.hypot(cxA - cxB, cyA - cyB)

def _separating_axis_check(vertsA, vertsB, tol=1e-9):
    """Check if two convex polygons overlap using SAT. Return True if they overlap (not disjoint)."""
    if _center_distance(vertsA, vertsB) > 4.0:
        return False
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    arrA = np.array(vertsA)
    arrB = np.array(vertsB)
    for axis in axes:
        ax, ay = axis
        projA = arrA[:, 0] * ax + arrA[:, 1] * ay
        projB = arrB[:, 0] * ax + arrB[:, 1] * ay
        if projA.max() <= projB.min() + tol or projB.max() <= projA.min() + tol:
            return False
    return True

def _min_overlap_depth(vertsA, vertsB):
    """Compute minimum overlap depth between two convex polygons (0 if disjoint)."""
    axes = _axes_from_polygon(vertsA) + _axes_from_polygon(vertsB)
    arrA = np.array(vertsA)
    arrB = np.array(vertsB)
    max_overlap = 0.0
    for axis in axes:
        ax, ay = axis
        projA = arrA[:, 0] * ax + arrA[:, 1] * ay
        projB = arrB[:, 0] * ax + arrB[:, 1] * ay
        overlap = min(projA.max(), projB.max()) - max(projA.min(), projB.min())
        if overlap > max_overlap:
            max_overlap = overlap
        if overlap < -1e-9:
            return 0.0
    return max_overlap

def _outer_hex_side(centers_angles, center, angle_deg):
    """Compute minimum side length S of outer hexagon containing all inner hexagons."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    
    # Vectorized computation of all projections
    all_verts = []
    for (px, py, pangle) in centers_angles:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0
        
    verts_arr = np.array(all_verts)
    normals_arr = np.array(normals)
    
    # Project all vertices onto all normals
    dots = verts_arr @ normals_arr.T
    max_dots = np.max(dots, axis=0)
    center_proj = np.dot(center, normals_arr.T)
    vals = (max_dots - center_proj) / apothem_unit
    
    return np.max(vals)

def _is_valid(centers_angles, tol=1e-9):
    """Check all pairs of hexagons are disjoint."""
    m = len(centers_angles)
    polys = [_hex_vertices(p[0], p[1], p[2]) for p in centers_angles]
    for i in range(m):
        for j in range(i + 1, m):
            if _separating_axis_check(polys[i], polys[j], tol):
                return False
    return True

def _smooth_collision_penalty(all_verts, beta=20.0):
    """Smooth collision penalty using soft-min of overlap depths."""
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _min_overlap_depth(all_verts[i], all_verts[j])
            if depth > 0:
                penalty += depth * depth
    return penalty

def _smooth_outer_side(cfg, center, angle_deg, beta=50.0):
    """Smooth outer side length using log-sum-exp over all vertex-projection pairs."""
    a = math.radians(angle_deg)
    apothem_unit = math.sqrt(3) / 2.0
    normals = []
    for k in range(6):
        na = a + math.radians(90 + 60 * k)
        normals.append((math.cos(na), math.sin(na)))
    cx, cy = center
    
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
        
    if not all_verts:
        return 0.0

    verts_arr = np.array(all_verts)
    normals_arr = np.array(normals)
    
    dots = verts_arr @ normals_arr.T
    center_proj = np.dot(center, normals_arr.T)
    
    vals = (dots - center_proj[None, :]) / apothem_unit
    vals = vals.flatten()
    
    max_val = np.max(vals)
    shifted = vals - max_val
    lse = max_val + (1.0 / beta) * np.log(np.sum(np.exp(beta * shifted)))
    return lse

def _optimize_outer_rotation(cfg, center, time_budget=5.0):
    """
    Treat the 12 hexagons as a fixed composite body.
    Optimize the outer hexagon rotation angle and center.
    Returns (best_center, best_angle, best_side).
    """
    t_start = time.time()
    
    all_verts = []
    for (px, py, pangle) in cfg:
        verts = _hex_vertices(px, py, pangle)
        all_verts.extend(verts)
        
    if not all_verts:
        return center, 0.0, 0.0

    verts_arr = np.array(all_verts)
    
    def side_for_params(v):
        angle_deg = v[0]
        cx, cy = v[1], v[2]
        return _outer_hex_side(cfg, (cx, cy), angle_deg)
    
    x0 = np.array([0.0, center[0], center[1]])
    
    result = minimize(
        side_for_params, x0, method='Nelder-Mead',
        options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    best_angle = result.x[0] % 60.0
    best_center = (result.x[1], result.x[2])
    best_side = result.fun
    
    for start_angle in [10.0, 20.0, 30.0, 40.0, 50.0]:
        if time.time() - t_start > time_budget:
            break
        x0_test = np.array([start_angle, center[0], center[1]])
        result_test = minimize(
            side_for_params, x0_test, method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
        )
        if result_test.fun < best_side:
            best_side = result_test.fun
            best_angle = result_test.x[0] % 60.0
            best_center = (result_test.x[1], result_test.x[2])
            
    return best_center, best_angle, best_side

def construct():
    t0 = time.time()
    random.seed(42)
    np.random.seed(42)
    
    def full_score(cfg, center, angle):
        """Return outer side if valid, else None."""
        if not _is_valid(cfg):
            return None
        return _outer_hex_side(cfg, tuple(center), angle)
    
    def make_hex_lattice():
        """3x4 hexagonal lattice with offset."""
        pts = []
        row_height = math.sqrt(3) * 1.0
        col_width = 2.0 * 1.0
        for r in range(3):
            for c in range(4):
                x = c * col_width + (col_width / 2 if r % 2 else 0.0)
                y = r * row_height
                pts.append([x - 3.0, y - row_height, 0.0])
        return pts
    
    def make_ring():
        """12 hexagons in a ring, oriented outward."""
        pts = []
        R = 2.5
        for i in range(N):
            theta = 2 * math.pi * i / N
            x = R * math.cos(theta)
            y = R * math.sin(theta)
            angle_deg = math.degrees(theta)
            pts.append([x, y, angle_deg])
        return pts
    
    def make_flower():
        """Flower pattern: 1 center + 6 inner ring + 5 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_dense_cluster():
        """Tight cluster: center + 6 around + 5 more."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = math.sqrt(3) * math.cos(theta)
            y = math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = 2 * math.sqrt(3) * math.cos(theta)
            y = 2 * math.sqrt(3) * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_hexagonal_12():
        """12 hexagons: center + 5 inner + 6 outer."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * i / 5 + math.pi/5
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2 * math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_tight_hex12():
        """Tight hexagonal arrangement: 1 center + 6 ring + 5 outer, all aligned."""
        pts = []
        pts.append([0.0, 0.0, 0.0])
        R1 = math.sqrt(3)
        for i in range(6):
            theta = 2 * math.pi * i / 6
            x = R1 * math.cos(theta)
            y = R1 * math.sin(theta)
            pts.append([x, y, 0.0])
        R2 = 2.0 * math.sqrt(3)
        for i in range(5):
            theta = 2 * math.pi * (i + 0.5) / 5
            x = R2 * math.cos(theta)
            y = R2 * math.sin(theta)
            pts.append([x, y, 0.0])
        return pts
    
    def make_compact_grid():
        """Compact 2-row arrangement with slight rotation for better packing."""
        pts = []
        for i in range(6):
            x = i * 1.86
            y = 0.0
            pts.append([x - 2.79, y, 0.0])
        for i in range(6):
            x = i * 1.86 + 0.93
            y = math.sqrt(3) * 0.95
            pts.append([x - 2.79, y, 0.0])
        return pts
    
    def smooth_objective(v, beta=50.0, coll_weight=100.0):
        """
        Smooth objective: smooth outer side + collision penalty.
        v = [x1,y1,a1, ..., x12,y12,a12, cx,cy, outer_angle]
        """
        cfg = []
        for i in range(N):
            cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
        center = (v[3*N], v[3*N+1])
        outer_angle = v[3*N+2]
        
        all_verts = []
        for p in cfg:
            verts = _hex_vertices(p[0], p[1], p[2])
            all_verts.append(verts)
        
        smooth_side = _smooth_outer_side(cfg, center, outer_angle, beta=beta)
        collision_penalty = _smooth_collision_penalty(all_verts, beta=20.0)
        
        disp_penalty = 0.0
        for i in range(N):
            disp_penalty += v[3*i]**2 + v[3*i+1]**2
        disp_penalty += v[3*N]**2 + v[3*N+1]**2
        disp_penalty *= 0.01
        
        return smooth_side + coll_weight * collision_penalty + disp_penalty

    def gradient_descent_polish(cfg_init, center_init, angle_init, time_budget, beta=50.0, coll_weight=100.0):
        """L-BFGS-B polish with smooth objective."""
        def flatten(cfg, center, angle):
            v = []
            for p in cfg:
                v.extend([p[0], p[1], p[2]])
            v.extend([center[0], center[1], angle])
            return np.array(v, dtype=float)
        
        def unflatten(v):
            cfg = []
            for i in range(N):
                cfg.append([v[3*i], v[3*i+1], v[3*i+2]])
            center = [v[3*N], v[3*N+1]]
            angle = v[3*N+2]
            return cfg, center, angle
        
        def obj(v):
            if time.time() - t0 > TIME_LIMIT - 2.0:
                return 1e6
            return smooth_objective(v, beta=beta, coll_weight=coll_weight)
        
        v = flatten(cfg_init, center_init, angle_init)
        
        result = minimize(
            obj, v, method='L-BFGS-B',
            options={'maxiter': 2000, 'maxfun': 10000, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        
        best_v = v.copy()
        best_f = smooth_objective(v, beta=beta, coll_weight=coll_weight)
        
        if result.fun < best_f:
            best_v = result.x.copy()
            best_f = result.fun
        
        return best_v, best_f
    
    # Try multiple initial configurations
    best_cfg = None
    best_score = float('inf')
    best_center = [0.0, 0.0]
    best_angle = 0.0
    
    inits = [make_hex_lattice(), make_ring(), make_flower(), make_dense_cluster(), 
             make_hexagonal_12(), make_tight_hex12(), make_compact_grid()]
    for init in inits:
        s = full_score(init, (0.0, 0.0), 0.0)
        if s is not None and s < best_score:
            best_score = s
            best_cfg = [row[:] for row in init]
            best_center = [0.0, 0.0]
            best_angle = 0.0
    
    if best_cfg is None:
        cfg = []
        for i in range(N):
            cfg.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        best_cfg = cfg
        best_score = 50.0
        best_center = [0.0, 0.0]
        best_angle = 0.0
    
    cfg = [row[:] for row in best_cfg]
    center = best_center[:]
    angle = best_angle
    cur_score = best_score
    
    # Simulated Annealing phase with Smooth Objective
    T = 0.8
    T_end = 0.003
    alpha = 0.9997
    
    sa_time_limit = TIME_LIMIT - 35.0  # Reserve 35s for polishing + composite body
    
    lattice_positions = []
    row_height = math.sqrt(3)
    col_width = 2.0
    for r in range(5):
        for c in range(6):
            x = c * col_width + (col_width / 2 if r % 2 else 0.0)
            y = r * row_height
            lattice_positions.append((x - 3.0, y - 2.0 * row_height))
    
    while time.time() - t0 < sa_time_limit:
        r = random.random()
        
        # Save old state for rollback
        old_cfg = [row[:] for row in cfg]
        old_center = center[:]
        old_angle = angle
        
        if r < 0.45:
            idx = random.randrange(N)
            step = 0.2 * (1.0 + 0.5 * T)
            cfg[idx][0] += random.gauss(0, step)
            cfg[idx][1] += random.gauss(0, step)
        elif r < 0.60:
            idx = random.randrange(N)
            cfg[idx][2] += random.gauss(0, 5.0 + 3.0 * T)
        elif r < 0.70:
            step = 0.1 * (1.0 + 0.5 * T)
            center[0] += random.gauss(0, step)
            center[1] += random.gauss(0, step)
        elif r < 0.80:
            angle += random.gauss(0, 2.0 + 2.0 * T)
        elif r < 0.90:
            idx1, idx2 = random.sample(range(N), 2)
            step = 0.15 * (1.0 + 0.5 * T)
            cfg[idx1][0] += random.gauss(0, step)
            cfg[idx1][1] += random.gauss(0, step)
            cfg[idx2][0] += random.gauss(0, step)
            cfg[idx2][1] += random.gauss(0, step)
        else:
            idx = random.randrange(N)
            best_lp = None
            best_dist = float('inf')
            for lx, ly in lattice_positions:
                d = (cfg[idx][0] - lx)**2 + (cfg[idx][1] - ly)**2
                if d < best_dist:
                    best_dist = d
                    best_lp = (lx, ly)
            if best_lp is not None:
                cfg[idx][0] = cfg[idx][0] + 0.8 * (best_lp[0] - cfg[idx][0])
                cfg[idx][1] = cfg[idx][1] + 0.8 * (best_lp[1] - cfg[idx][1])
        
        # Evaluate smooth cost
        v_flat = []
        for p in cfg:
            v_flat.extend(p)
        v_flat.extend([center[0], center[1], angle])
        
        new_cost = smooth_objective(np.array(v_flat), beta=50.0, coll_weight=100.0)
        
        # Acceptance criterion
        # We compare smooth cost to current smooth cost (approximated by cur_score + penalty)
        # To keep it simple, we accept if new_cost < cur_smooth_cost or with probability exp(-(new_cost-cur_smooth_cost)/T)
        # We need to track cur_smooth_cost.
        
        # For simplicity in SA loop, we just check if it's valid and better, or if it's a small overlap, accept with prob.
        # Better: Track the smooth cost of the current best state.
        
        # Let's re-evaluate: The current code structure uses cur_score (hard).
        # To integrate smooth SA, we should track cur_smooth_cost.
        
        # Let's refactor slightly:
        # cur_smooth_cost = smooth_objective(current_state)
        
        # However, computing smooth_objective every step is expensive.
        # Alternative: Use the hard score for best tracking, but accept moves based on a "relaxed" validity.
        # If valid: accept if better.
        # If invalid: accept if overlap is small and T is high.
        
        # Let's stick to a hybrid:
        # 1. Check validity.
        # 2. If valid, compute hard score.
        # 3. If invalid, compute collision penalty.
        
        new_score = full_score(cfg, tuple(center), angle)
        
        accepted = False
        if new_score is not None:
            if new_score < cur_score:
                accepted = True
            elif T > T_end and random.random() < math.exp(-(new_score - cur_score) / T):
                accepted = True
        else:
            # Invalid: compute collision penalty
            all_verts = [_hex_vertices(p[0], p[1], p[2]) for p in cfg]
            coll_pen = _smooth_collision_penalty(all_verts)
            # Approximate cost increase
            est_cost = cur_score + 100.0 * coll_pen
            if T > T_end and random.random() < math.exp(-(est_cost - cur_score) / T):
                accepted = True
        
        if accepted:
            if new_score is not None:
                cur_score = new_score
                if new_score < best_score:
                    best_score = new_score
                    best_cfg = [row[:] for row in cfg]
                    best_center = center[:]
                    best_angle = angle
            else:
                # Update cur_score to reflect the "cost" of this invalid state for future comparisons?
                # No, keep cur_score as the best *valid* score.
                # But for SA dynamics, we need to know the "height" of the current state.
                # This hybrid approach is tricky.
                # Let's revert to a simpler robust strategy:
                # Use smooth objective for *polish*, and hard SA for *search* but with smaller steps.
                # The inspiration's gain likely came from the *polish* being able to fix small overlaps.
                # So, let's keep hard SA, but ensure the polish stages are very strong.
                
                # Actually, the current program already does this.
                # The key might be that the *initial* best_cfg from inits is not good enough.
                # Or the SA is getting stuck.
                
                # Let's try: If invalid, reject. (Standard SA).
                # But increase the number of SA iterations by making them faster?
                # The hard check is the bottleneck.
                
                # Let's try a different approach for the SA:
                # Use a "soft" validity: accept if overlap < epsilon.
                # This allows "moving through" thin overlaps.
                
                # Let's implement a soft acceptance for invalid states in the SA loop.
                pass
        
        if not accepted:
            cfg = old_cfg
            center = old_center
            angle = old_angle
            
        T = max(T_end, T * alpha)
    
    # Composite body optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 12.0:
        try:
            cb_center, cb_angle, cb_side = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.3, 8.0)
            )
            if cb_side < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side
                    best_center = cb_center
                    best_angle = cb_angle
        except Exception:
            pass
    
    # Multi-stage gradient descent polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 15.0:
        try:
            polished_v, polished_f = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.4, 15.0), beta=50.0
            )
            cfg_polished = []
            for i in range(N):
                cfg_polished.append([polished_v[3*i], polished_v[3*i+1], polished_v[3*i+2]])
            center_polished = [polished_v[3*N], polished_v[3*N+1]]
            angle_polished = polished_v[3*N+2]
            
            polished_score = full_score(cfg_polished, tuple(center_polished), angle_polished)
            if polished_score is not None and polished_score < best_score:
                best_score = polished_score
                best_cfg = cfg_polished
                best_center = center_polished
                best_angle = angle_polished
        except Exception:
            pass
    
    # Stage 2: Fine polish
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 10.0:
        try:
            polished_v2, polished_f2 = gradient_descent_polish(
                best_cfg, best_center, best_angle, min(remaining_time * 0.5, 12.0), beta=100.0
            )
            cfg_polished2 = []
            for i in range(N):
                cfg_polished2.append([polished_v2[3*i], polished_v2[3*i+1], polished_v2[3*i+2]])
            center_polished2 = [polished_v2[3*N], polished_v2[3*N+1]]
            angle_polished2 = polished_v2[3*N+2]
            
            polished_score2 = full_score(cfg_polished2, tuple(center_polished2), angle_polished2)
            if polished_score2 is not None and polished_score2 < best_score:
                best_score = polished_score2
                best_cfg = cfg_polished2
                best_center = center_polished2
                best_angle = angle_polished2
        except Exception:
            pass
    
    # Final composite body optimization
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 8.0:
        try:
            cb_center2, cb_angle2, cb_side2 = _optimize_outer_rotation(
                best_cfg, best_center, time_budget=min(remaining_time * 0.4, 5.0)
            )
            if cb_side2 < best_score:
                if _is_valid(best_cfg):
                    best_score = cb_side2
                    best_center = cb_center2
                    best_angle = cb_angle2
        except Exception:
            pass
    
    # Final quick refinement
    remaining_time = TIME_LIMIT - (time.time() - t0)
    if remaining_time > 3.0:
        for _ in range(300):
            if time.time() - t0 > TIME_LIMIT - 1.0:
                break
            cfg_try = [row[:] for row in best_cfg]
            center_try = best_center[:]
            angle_try = best_angle
            idx = random.randrange(N)
            cfg_try[idx][0] += random.gauss(0, 0.02)
            cfg_try[idx][1] += random.gauss(0, 0.02)
            score_try = full_score(cfg_try, tuple(center_try), angle_try)
            if score_try is not None and score_try < best_score:
                best_score = score_try
                best_cfg = cfg_try
    
    inner = [[float(p[0]), float(p[1]), float(p[2])] for p in best_cfg]
    return inner, (float(best_center[0]), float(best_center[1])), float(best_score), float(best_angle)

if __name__ == "__main__":
    inner, c, s, a = construct()
    print("side:", s)
