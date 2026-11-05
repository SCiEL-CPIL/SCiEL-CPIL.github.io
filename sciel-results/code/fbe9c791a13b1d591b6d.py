import numpy as np
import time
from scipy.optimize import minimize

def construct():
    """
    Pack 11 unit regular hexagons inside a regular hexagon, minimizing the outer side length.
    Uses a flower packing (1 center, 6 ring 1, 4 ring 2) with optimized rotations.
    Combines support function optimization with overlap penalties and soft-max smoothing.
    """
    start_time = time.time()
    deadline = 145.0  # Safety margin before 150s limit
    
    # Centers of the 11 hexagons
    # 1 center at (0,0)
    # 6 at distance sqrt(3) at angles 0, 60, ... 300
    # 4 at distance 3 at angles 30, 90, 150, 210
    eps = 1e-10
    centers = np.array([[0.0, 0.0]])
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers = np.vstack([centers, [r1 * np.cos(angle), r1 * np.sin(angle)]])
    
    r2 = 3.0 + 2 * eps
    # Place 4 hexagons in the gaps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers = np.vstack([centers, [r2 * np.cos(angle), r2 * np.sin(angle)]])
        
    n_hex = len(centers)
    
    # Base vertices of unit hexagon (pointy-right orientation, angle 0)
    # Vertices at 0, 60, 120, 180, 240, 300
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    def compute_outer_side_exact(theta_out, phis):
        """
        Compute the minimum outer hexagon side length given outer rotation and inner rotations.
        Uses support function: max projection onto face normals.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        max_proj = 0.0
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            max_proj = max(max_proj, np.max(dots))
        
        # For a regular hexagon with side S, apothem = S * sqrt(3) / 2
        # S = 2 * max_proj / sqrt(3)
        return 2.0 * max_proj / np.sqrt(3)
    
    def compute_outer_side_soft(theta_out, phis, beta=50.0):
        """
        Soft-max version of support function for smoother optimization.
        Uses log-sum-exp approximation to max, scaled by 1/beta.
        """
        normals = np.array([
            [np.cos(np.deg2rad(30 + k * 60) + theta_out),
             np.sin(np.deg2rad(30 + k * 60) + theta_out)]
            for k in range(6)
        ])
        
        # Collect all vertex projections
        all_projs = []
        for i in range(n_hex):
            phi = phis[i]
            c, s = np.cos(phi), np.sin(phi)
            rot = np.array([[c, -s], [s, c]])
            verts = base_vertices @ rot.T + centers[i]
            dots = verts @ normals.T
            all_projs.append(dots)
        
        all_projs = np.concatenate(all_projs)
        
        # Soft-max via log-sum-exp with temperature beta
        m = np.max(all_projs)
        if m > 1e10:
            return m
        # Correct log-sum-exp: (1/beta) * log(sum(exp(beta * (x_i - m)))) + m
        log_sum_exp = m + np.log(np.sum(np.exp(beta * (all_projs - m)))) / beta
        
        return 2.0 * log_sum_exp / np.sqrt(3)
    
    def _hex_vertices(cx, cy, angle_rad):
        """Vertices of a unit hexagon centered at (cx,cy) rotated by angle_rad."""
        c, s = np.cos(angle_rad), np.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return base_vertices @ rot.T + np.array([cx, cy])
    
    def _point_in_hex(pt, center, angle_rad):
        """Check if point is inside a unit hexagon centered at center, rotated by angle_rad."""
        # Transform point to hexagon's local frame
        local_pt = pt - center
        c, s = np.cos(-angle_rad), np.sin(-angle_rad)
        rot_inv = np.array([[c, -s], [s, c]])
        local_pt = rot_inv @ local_pt
        
        # In local frame, hexagon has vertices at angles 0,60,...,300
        # Check using half-plane tests: for each edge, point must be on the correct side
        for k in range(6):
            a1 = np.deg2rad(60 * k)
            a2 = np.deg2rad(60 * (k + 1))
            v1 = np.array([np.cos(a1), np.sin(a1)])
            v2 = np.array([np.cos(a2), np.sin(a2)])
            edge = v2 - v1
            # Normal pointing outward
            normal = np.array([edge[1], -edge[0]])
            # The center should be on the inside
            center_to_v1 = v1
            if np.dot(normal, local_pt - v1) > 1e-12:
                return False
        return True
    
    def _poly_overlap_penalty(phis, weight=100.0):
        """
        Compute penalty for overlapping inner hexagons.
        Returns 0 if all hexagons are disjoint, positive otherwise.
        Uses vertex-in-polygon and edge distance checks.
        """
        penalty = 0.0
        verts_list = []
        for i in range(n_hex):
            verts_list.append(_hex_vertices(centers[i][0], centers[i][1], phis[i]))
        
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                v1 = verts_list[i]
                v2 = verts_list[j]
                
                # Quick rejection: if center distance > 2*sqrt(3), no overlap possible
                dc = np.linalg.norm(centers[i] - centers[j])
                if dc > 2.0 * np.sqrt(3) + 0.1:
                    continue
                
                # Check if any vertex of v1 is inside v2
                overlap = False
                for v in v1:
                    if _point_in_hex(v, centers[j], phis[j]):
                        overlap = True
                        break
                if not overlap:
                    for v in v2:
                        if _point_in_hex(v, centers[i], phis[i]):
                            overlap = True
                            break
                
                if overlap:
                    penalty += weight
                else:
                    # Check edge-edge distances
                    min_dist = float('inf')
                    for a in range(6):
                        p1 = v1[a]
                        p2 = v1[(a + 1) % 6]
                        for b in range(6):
                            p3 = v2[b]
                            p4 = v2[(b + 1) % 6]
                            # Segment-segment distance
                            d = _seg_seg_dist(p1, p2, p3, p4)
                            if d < min_dist:
                                min_dist = d
                    
                    if min_dist < 0.001:
                        penalty += weight * (0.001 - min_dist) ** 2
        
        return penalty
    
    def _seg_seg_dist(p1, p2, p3, p4):
        """Minimum distance between two line segments p1-p2 and p3-p4."""
        p1 = np.array(p1, dtype=float)
        p2 = np.array(p2, dtype=float)
        p3 = np.array(p3, dtype=float)
        p4 = np.array(p4, dtype=float)
        
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-12 and e < 1e-12:
            return np.linalg.norm(p1 - p3)
        
        if a < 1e-12:
            t = f / e
            t = max(0, min(1, t))
            d = p1 - (p3 + t * d2)
            return np.linalg.norm(d)
        
        c = np.dot(d1, r)
        if e < 1e-12:
            s = c / a
            s = max(0, min(1, s))
            d = (p1 + s * d1) - p3
            return np.linalg.norm(d)
        
        b = np.dot(d1, d2)
        denom = a * e - b * b
        s = 0
        t = 0
        
        if denom > 1e-12:
            s = (b * f - c * e) / denom
            s = max(0, min(1, s))
            t = (b * s + f) / e
            if t < 0:
                t = 0
                s = max(0, min(1, c / a))
            elif t > 1:
                t = 1
                s = max(0, min(1, (b + f) / a))
            d = (p1 + s * d1) - (p3 + t * d2)
            return np.linalg.norm(d)
        else:
            t = f / e
            t = max(0, min(1, t))
            d = (p1 + t * d2) - p3
            return np.linalg.norm(d)
    
    def objective(params, use_soft=True):
        """
        params: [theta_out, phi_0, ..., phi_10]
        """
        theta_out = params[0]
        phis = params[1:]
        
        if use_soft:
            side = compute_outer_side_soft(theta_out, phis, beta=50.0)
        else:
            side = compute_outer_side_exact(theta_out, phis)
        
        overlap_pen = _poly_overlap_penalty(phis, weight=50.0)
        return side + overlap_pen
    
    def objective_exact(params):
        """Exact objective for final evaluation."""
        theta_out = params[0]
        phis = params[1:]
        return compute_outer_side_exact(theta_out, phis)
    
    # Initial guess: 
    # Outer rotation 0.
    # Inner rotations: align faces radially.
    x0 = np.zeros(n_hex + 1)
    x0[0] = 0.0
    for i in range(n_hex):
        cx, cy = centers[i]
        if abs(cx) < 1e-9 and abs(cy) < 1e-9:
            x0[i+1] = 0.0
        else:
            alpha = np.arctan2(cy, cx)
            # Align face normal (30 deg in base) to alpha
            x0[i+1] = alpha - np.deg2rad(30)
            
    best_score = float('inf')
    best_params = None
    
    # Phase 1: Multi-start Powell optimization with soft objective
    n_starts = 10
    for i in range(n_starts):
        if time.time() > deadline - 30:
            break
        
        if i == 0:
            x_start = x0.copy()
        elif i == 1:
            # All aligned at 30 degrees
            x_start = np.zeros(n_hex + 1)
            x_start[1:] = np.deg2rad(30)
        elif i == 2:
            # Random perturbations
            x_start = x0 + np.random.uniform(-0.15, 0.15, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
        elif i == 3:
            # Try 60-degree offset
            x_start = x0.copy()
            x_start[0] = np.deg2rad(60)
            x_start[1:] = x0[1:] + np.deg2rad(30)
        elif i == 4:
            # Try 15-degree offset for outer
            x_start = x0.copy()
            x_start[0] = np.deg2rad(15)
        elif i == 5:
            # Try 45-degree offset for outer
            x_start = x0.copy()
            x_start[0] = np.deg2rad(45)
        elif i == 6:
            # All inner at 0
            x_start = np.zeros(n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
        else:
            # More random
            x_start = np.random.uniform(-0.3, 0.3, n_hex + 1)
            x_start[0] = np.random.uniform(0, np.pi/3)
            
        try:
            res = minimize(objective, x_start, args=(True,), method='Powell',
                          options={'maxiter': 800, 'disp': False, 'ftol': 1e-10, 'xtol': 1e-10})
            if res.fun < best_score:
                best_score = res.fun
                best_params = res.x.copy()
        except Exception:
            pass
    
    if best_params is None:
        # Fallback to analytical solution
        return _fallback_solution()
    
    # Phase 2: Nelder-Mead polish with soft objective
    if time.time() < deadline - 10:
        try:
            res2 = minimize(objective, best_params, args=(True,), method='Nelder-Mead',
                           options={'xatol': 1e-9, 'fatol': 1e-11, 'maxiter': 3000, 'disp': False, 'adaptive': True})
            if res2.fun < best_score:
                best_score = res2.fun
                best_params = res2.x.copy()
        except Exception:
            pass
    
    # Phase 3: L-BFGS-B with exact objective for final refinement
    if time.time() < deadline - 5:
        try:
            res3 = minimize(objective_exact, best_params, method='L-BFGS-B',
                           options={'maxiter': 200, 'ftol': 1e-12, 'gtol': 1e-8})
            if res3.fun < best_score:
                best_score = res3.fun
                best_params = res3.x.copy()
        except Exception:
            pass
    
    # Phase 4: Final high-precision Nelder-Mead with exact objective
    if time.time() < deadline - 5:
        try:
            res4 = minimize(objective_exact, best_params, method='Nelder-Mead',
                           options={'xatol': 1e-11, 'fatol': 1e-13, 'maxiter': 2000, 'disp': False})
            if res4.fun < best_score:
                best_score = res4.fun
                best_params = res4.copy()
        except Exception:
            pass
    
    # Phase 5: High-precision 1D golden-section search over outer rotation
    # Fix inner rotations, optimize only theta_out with exact objective
    if time.time() < deadline - 3:
        phis_fixed = best_params[1:]
        
        def outer_only_objective(theta):
            return compute_outer_side_exact(theta, phis_fixed)
        
        # Golden-section search over [0, pi/3]
        gr = (np.sqrt(5) - 1) / 2  # golden ratio conjugate
        a, b = 0.0, np.pi / 3
        c = b - gr * (b - a)
        d = a + gr * (b - a)
        fc = outer_only_objective(c)
        fd = outer_only_objective(d)
        
        for _ in range(80):
            if time.time() > deadline - 1:
                break
            if fc < fd:
                b = d
                d = c
                fd = fc
                c = b - gr * (b - a)
                fc = outer_only_objective(c)
            else:
                a = c
                c = d
                fc = fd
                d = a + gr * (b - a)
                fd = outer_only_objective(d)
        
        theta_opt = (a + b) / 2
        side_opt = outer_only_objective(theta_opt)
        
        if side_opt < best_score - 1e-12:
            best_score = side_opt
            best_params = np.zeros(n_hex + 1)
            best_params[0] = theta_opt
            best_params[1:] = phis_fixed
    
    # Extract solution
    theta_out = best_params[0]
    phis = best_params[1:]
    
    # Verify no overlaps and compute exact side
    overlap_pen = _poly_overlap_penalty(phis, weight=1.0)
    if overlap_pen > 0.01:
        # If there's significant overlap, fall back to a known-good solution
        return _fallback_solution()
    
    side = compute_outer_side_exact(theta_out, phis)
    
    inner = []
    for i in range(n_hex):
        inner.append([centers[i][0], centers[i][1], np.rad2deg(phis[i])])
    
    return inner, (0.0, 0.0), side + 1e-9, np.rad2deg(theta_out)

def _fallback_solution():
    """
    Analytical solution: all inner hexagons at 30 degrees, outer at 30 degrees.
    """
    eps = 1e-10
    centers = []
    centers.append((0.0, 0.0))
    
    r1 = np.sqrt(3) + eps
    for k in range(6):
        angle = np.deg2rad(k * 60)
        centers.append((r1 * np.cos(angle), r1 * np.sin(angle)))
    
    r2 = 3.0 + 2 * eps
    for k in range(4):
        angle = np.deg2rad(30 + k * 60)
        centers.append((r2 * np.cos(angle), r2 * np.sin(angle)))
    
    inner = []
    for x, y in centers:
        inner.append([x, y, 30.0])
    
    # Compute exact side for this configuration
    base_vertices = np.array([
        [np.cos(np.deg2rad(60 * k)), np.sin(np.deg2rad(60 * k))] for k in range(6)
    ])
    
    theta_out = np.deg2rad(30)
    normals = np.array([
        [np.cos(np.deg2rad(30 + k * 60) + theta_out),
         np.sin(np.deg2rad(30 + k * 60) + theta_out)]
        for k in range(6)
    ])
    
    max_proj = 0.0
    for i, (cx, cy) in enumerate(centers):
        phi = np.deg2rad(30)
        c, s = np.cos(phi), np.sin(phi)
        rot = np.array([[c, -s], [s, c]])
        verts = base_vertices @ rot.T + np.array([cx, cy])
        dots = verts @ normals.T
        max_proj = max(max_proj, np.max(dots))
    
    side = 2.0 * max_proj / np.sqrt(3)
    
    return inner, (0.0, 0.0), side + 1e-9, 30.0
