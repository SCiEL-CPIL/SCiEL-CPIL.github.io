# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_verts_fast(cx, cy, side, angle_deg):
    """Fast hexagon vertices using precomputed cos/sin."""
    a = np.radians(angle_deg)
    cos_a = np.cos(a)
    sin_a = np.sin(a)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = a + np.radians(60 * i)
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts

def _point_in_polygon(pt, verts):
    """Check if point is inside polygon using winding number / cross products."""
    n = len(verts)
    for i in range(n):
        v1 = verts[i]
        v2 = verts[(i + 1) % n]
        edge = v2 - v1
        rel = pt - v1
        cross = edge[0] * rel[1] - edge[1] * rel[0]
        if cross < 0:
            return False
    return True

def _softplus(x, beta=100.0):
    """Smooth approximation of max(0, x)."""
    # Numerical stability: if beta*x is large, exp overflows.
    # log(1 + exp(beta*x)) / beta
    # If beta*x > 50, approx beta*x / beta = x.
    # If beta*x < -50, approx 0.
    bx = beta * x
    if bx > 50:
        return x
    if bx < -50:
        return 0.0
    return math.log(1.0 + math.exp(bx)) / beta

def _smooth_min_overlap_depth(h1, h2, beta=100.0):
    """
    Smooth approximation of the minimum overlap depth between two convex polygons.
    Returns 0 if disjoint (approx).
    """
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
            
    if not axes:
        return 0.0

    # Calculate overlap depths on all axes
    overlaps = []
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        # Overlap depth on this axis
        # If max1 <= min2 or max2 <= min1, overlap is <= 0
        # overlap = min(max1, max2) - max(min1, min2)
        overlap = min(max1, max2) - max(min1, min2)
        overlaps.append(overlap)
        
    overlaps = np.array(overlaps)
    
    # Smooth min: -1/beta * log(sum(exp(-beta * overlaps)))
    # We want min(overlaps).
    # If min(overlaps) > 0, they overlap.
    # If min(overlaps) <= 0, they are disjoint.
    
    # To avoid overflow in exp(-beta * overlaps):
    # Let m = min(overlaps).
    # sum(exp(-beta * o)) = exp(-beta * m) * sum(exp(-beta * (o - m)))
    # log(sum) = -beta * m + log(sum(exp(-beta * (o - m))))
    # -1/beta * log(sum) = m - 1/beta * log(sum(exp(-beta * (o - m))))
    
    m = np.min(overlaps)
    if m <= 0:
        # They are disjoint or barely touching.
        # The smooth min will be slightly negative or close to 0.
        # We apply softplus to the result to make it non-negative penalty.
        pass
        
    # Compute the log-sum-exp part
    # o - m >= 0, so -beta * (o - m) <= 0. exp is safe.
    shifted = overlaps - m
    exp_vals = np.exp(-beta * shifted)
    sum_exp = np.sum(exp_vals)
    
    # Smooth min value
    smooth_min_val = m - (1.0 / beta) * np.log(sum_exp)
    
    # Penalty is max(0, smooth_min_val) approximated by softplus
    penalty = _softplus(smooth_min_val, beta)
    
    return penalty

def _smooth_containment_penalty(inner_verts, outer_verts, beta=100.0):
    """
    Smooth penalty for inner hexagon not being fully inside outer hexagon.
    """
    penalty = 0.0
    n_out = len(outer_verts)
    for iv in inner_verts:
        for i in range(n_out):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % n_out]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            # cross > 0 means inside (assuming CCW outer). 
            # If cross < 0, it's outside.
            # Distance (signed) = cross / edge_len.
            # Violation = max(0, -cross / edge_len).
            signed_dist = cross / edge_len
            violation = -signed_dist
            penalty += _softplus(violation, beta)
    return penalty

def _compute_all_penalties_smooth(params, outer_side, outer_center, outer_angle, beta=100.0):
    """Compute all inner hex vertices and return smooth containment + disjointness penalties."""
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts_fast(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts_fast(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _smooth_containment_penalty(iv, outer_verts, beta)
    
    # Disjointness
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _smooth_min_overlap_depth(inner_verts_list[i], inner_verts_list[j], beta)
            disj_pen += depth
            
    return cont_pen + disj_pen

def _compute_all_penalties(params, outer_side, outer_center, outer_angle):
    """Compute all inner hex vertices and return exact containment + disjointness penalties."""
    # This is the exact, nonsmooth version used for final validation/polish
    inner_verts_list = []
    for i in range(N):
        cx = params[3*i]
        cy = params[3*i+1]
        ang = params[3*i+2]
        inner_verts_list.append(_hex_verts_fast(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts_fast(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment (exact)
    cont_pen = 0.0
    for iv in inner_verts_list:
        for i in range(len(outer_verts)):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % len(outer_verts)]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                cont_pen += (-cross) / edge_len
                
    # Disjointness (exact)
    disj_pen = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            depth = _sat_overlap_depth_exact(inner_verts_list[i], inner_verts_list[j])
            if depth > 0:
                disj_pen += depth
                
    return cont_pen + disj_pen

def _sat_overlap_depth_exact(h1, h2):
    """Exact minimum penetration depth between two convex polygons using SAT. Returns 0 if disjoint."""
    min_depth = float('inf')
    axes = []
    for poly in [h1, h2]:
        n = len(poly)
        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]
            edge = p2 - p1
            norm = np.linalg.norm(edge)
            if norm < 1e-10:
                continue
            normal = np.array([-edge[1], edge[0]]) / norm
            axes.append(normal)
            
    for axis in axes:
        proj1 = np.dot(h1, axis)
        proj2 = np.dot(h2, axis)
        min1, max1 = np.min(proj1), np.max(proj1)
        min2, max2 = np.min(proj2), np.max(proj2)
        
        if max1 <= min2 or max2 <= min1:
            return 0.0
        
        overlap = min(max1, max2) - max(min1, min2)
        if overlap < min_depth:
            min_depth = overlap
            
    return min_depth

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Alias for _compute_all_penalties."""
    return _compute_all_penalties(params, outer_side, outer_center, outer_angle)

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit
    
    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions
    
    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]
    
    # Generate candidate initial configurations
    configs = []
    
    # 4x3 lattice
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # 3x4 lattice
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # 2x6 lattice
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # Ring config
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # 6x2 lattice
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())
        
    # Line
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)
    
    # Add rotated versions of 4x3 lattice
    for rot_angle in [15.0, -15.0, 30.0]:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        arr = arr - centroid
        theta = np.radians(rot_angle)
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        arr = arr @ rot_mat.T
        configs.append(arr.tolist())

    best_score = float('inf')
    best_result = None
    
    # Phase 1: Quick L-BFGS-B search with multiple outer angles and sides
    # Using smoothed objective with annealing
    outer_angles = [0.0, 30.0]
    outer_sides = [5.0, 4.5, 4.2, 4.0, 3.8, 3.6, 3.5, 3.4, 3.3, 3.2, 3.1]
    
    for outer_angle in outer_angles:
        for outer_side in outer_sides:
            if time.time() > deadline - 30.0:
                break
            
            outer_center = (0.0, 0.0)
            
            for config in configs:
                if time.time() > deadline - 25.0:
                    break
                
                init_params = []
                for (x, y) in config:
                    init_params.extend([float(x), float(y), 0.0])
                init_params = np.array(init_params)
                
                # Check exact penalty first to skip if already valid
                pen_exact = _compute_all_penalties(init_params, outer_side, outer_center, outer_angle)
                if pen_exact < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (init_params.copy(), outer_center, outer_side, outer_angle)
                    continue
                
                if time.time() > deadline - 25.0:
                    continue
                
                # Smoothed optimization with annealing
                current_params = init_params.copy()
                betas = [10.0, 50.0, 100.0, 200.0]
                
                for beta in betas:
                    if time.time() > deadline - 25.0:
                        break
                    
                    def objective_smooth(p, b=beta):
                        return _compute_all_penalties_smooth(p, outer_side, outer_center, outer_angle, b)
                    
                    try:
                        result = minimize(
                            objective_smooth,
                            current_params,
                            method='L-BFGS-B',
                            options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
                        )
                        current_params = result.x
                    except Exception:
                        pass
                
                # Final exact check and polish
                pen_final = _compute_all_penalties(current_params, outer_side, outer_center, outer_angle)
                
                if pen_final < 1e-6:
                    if outer_side < best_score:
                        best_score = outer_side
                        best_result = (current_params.copy(), outer_center, outer_side, outer_angle)
                else:
                    # Try to polish with exact penalty if close
                    if pen_final < 1.0:
                        def objective_exact(p):
                            return _compute_all_penalties(p, outer_side, outer_center, outer_angle)
                        
                        try:
                            result = minimize(
                                objective_exact,
                                current_params,
                                method='L-BFGS-B',
                                options={'maxiter': 100, 'maxfun': 3000, 'ftol': 1e-14, 'gtol': 1e-10}
                            )
                            if result.fun < 1e-6:
                                if outer_side < best_score:
                                    best_score = outer_side
                                    best_result = (result.x.copy(), outer_center, outer_side, outer_angle)
                        except Exception:
                            pass

    # Phase 2: Binary search for minimal side length with more iterations
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result
        
        # Binary search for the smallest side length that still admits a valid packing
        low = max(2.8, outer_side - 0.5)
        high = outer_side
        
        for _ in range(15):
            if time.time() > deadline - 5.0:
                break
            mid = (low + high) / 2.0
            if high - low < 1e-5:
                break
            
            # Use smoothed optimization for the search step
            current_params = opt_params.copy()
            betas = [50.0, 100.0]
            for beta in betas:
                if time.time() > deadline - 5.0:
                    break
                def objective_smooth(p, b=beta):
                    return _compute_all_penalties_smooth(p, mid, outer_center, outer_angle, b)
                try:
                    result = minimize(
                        objective_smooth,
                        current_params,
                        method='L-BFGS-B',
                        options={'maxiter': 80, 'maxfun': 2000, 'ftol': 1e-12, 'gtol': 1e-8}
                    )
                    current_params = result.x
                except Exception:
                    pass
            
            # Check with exact penalty
            pen_exact = _compute_all_penalties(current_params, mid, outer_center, outer_angle)
            if pen_exact < 1e-6:
                high = mid
                opt_params = current_params.copy()
                outer_side = mid
            else:
                low = mid
                
        # Final high-precision polish at the found side length
        if time.time() < deadline - 3.0:
            # Use exact penalty for final polish
            def objective_exact(p):
                return _compute_all_penalties(p, outer_side, outer_center, outer_angle)
            
            try:
                result = minimize(
                    objective_exact,
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < 1e-8:
                    opt_params = result.x
            except Exception:
                pass
        
        best_result = (opt_params, outer_center, outer_side, outer_angle)
    
    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    opt_params, outer_center, outer_side, outer_angle = best_result
    
    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])
    
    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts_fast(cx, cy, 1.0, ang))
    outer_verts = _hex_verts_fast(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    cont_pen = 0.0
    for iv in inner_verts_list:
        for i in range(len(outer_verts)):
            v1 = outer_verts[i]
            v2 = outer_verts[(i + 1) % len(outer_verts)]
            edge = v2 - v1
            edge_len = np.linalg.norm(edge)
            if edge_len < 1e-12:
                continue
            rel = iv - v1
            cross = edge[0] * rel[1] - edge[1] * rel[0]
            if cross < 0:
                cont_pen += (-cross) / edge_len
                
    disj_pen = 0.0
    for i in range(N):
        for j in range(i+1, N):
            d = _sat_overlap_depth_exact(inner_verts_list[i], inner_verts_list[j])
            if d > 0:
                disj_pen += d
    
    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0
    
    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
