import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 115.0  # Safety margin before 120s hard limit

def _get_hex_vertices(cx, cy, angle_deg=0.0):
    """Get vertices of a unit regular hexagon centered at (cx, cy) with rotation."""
    angle_rad = math.radians(angle_deg)
    verts = []
    for k in range(6):
        a = angle_rad + math.pi * k / 3
        verts.append((cx + math.cos(a), cy + math.sin(a)))
    return verts

def _point_in_convex_polygon(p, polygon, strict=False):
    """Check if point p is inside convex polygon."""
    px, py = p
    n = len(polygon)
    sign = 0
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        cross = (x2 - x1) * (py - y1) - (y2 - y1) * (px - x1)
        if strict:
            if cross < 0:
                return False
        else:
            if cross < -1e-12:
                return False
        if cross > 0:
            if sign < 0:
                return False
            sign = 1
        elif cross < 0:
            if sign > 0:
                return False
            sign = -1
    return True

def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    c1 = cross(p3, p4, p1)
    c2 = cross(p3, p4, p2)
    c3 = cross(p1, p2, p3)
    c4 = cross(p1, p2, p4)
    if ((c1 > 0 and c2 < 0) or (c1 < 0 and c2 > 0)) and \
       ((c3 > 0 and c4 < 0) or (c3 < 0 and c4 > 0)):
        return True
    if abs(c1) < 1e-12 and _on_segment(p3, p4, p1):
        return True
    if abs(c2) < 1e-12 and _on_segment(p3, p4, p2):
        return True
    if abs(c3) < 1e-12 and _on_segment(p1, p2, p3):
        return True
    if abs(c4) < 1e-12 and _on_segment(p1, p2, p4):
        return True
    return False

def _on_segment(a, b, p):
    return (min(a[0], b[0]) - 1e-12 <= p[0] <= max(a[0], b[0]) + 1e-12 and
            min(a[1], b[1]) - 1e-12 <= p[1] <= max(a[1], b[1]) + 1e-12)

def _hexagons_overlap(verts1, verts2):
    """Check if two convex hexagons overlap (interior intersection)."""
    if _point_in_convex_polygon(verts1[0], verts2, strict=True):
        return True
    if _point_in_convex_polygon(verts2[0], verts1, strict=True):
        return True
    for k in range(6):
        a1 = verts1[k]
        b1 = verts1[(k + 1) % 6]
        for l in range(6):
            a2 = verts2[l]
            b2 = verts2[(l + 1) % 6]
            if _segments_intersect(a1, b1, a2, b2):
                return True
    return False

def _check_disjoint(inner_positions, inner_angles, tol=1e-9):
    """Check that all inner hexagons are disjoint (no overlap)."""
    n = len(inner_positions)
    all_verts = []
    for i in range(n):
        px, py = inner_positions[i]
        ang = inner_angles[i]
        all_verts.append(_get_hex_vertices(px, py, ang))
    for i in range(n):
        for j in range(i + 1, n):
            if _hexagons_overlap(all_verts[i], all_verts[j]):
                return False
    return True

def _compute_all_vertices(positions, angles):
    """Compute all 6*N vertices for given positions and angles."""
    all_verts = []
    for (px, py), ang in zip(positions, angles):
        all_verts.extend(_get_hex_vertices(px, py, ang))
    return all_verts

def _min_outer_side_lp(vertices, outer_angle_deg):
    """
    Find the minimum side length of a regular hexagon with given angle
    that contains all given vertices, using LP.
    Returns (center_x, center_y, side_length) or None.
    
    For a regular hexagon with side s, apothem = s * sqrt(3)/2.
    The 6 face normals are at angles outer_angle + 30 + 60k.
    Constraint: for all vertices v, (v - c) . n_k <= apothem for all k.
    Equivalently: v . n_k - c . n_k <= t where t = apothem = s*sqrt(3)/2.
    So: -c . n_k - t <= -v . n_k
    LP: minimize t over (cx, cy, t)
    side = t * 2 / sqrt(3)
    """
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    A = []
    b = []
    
    for vx, vy in vertices:
        for nx, ny in normals:
            v_dot_n = vx * nx + vy * ny
            # -nx*cx - ny*cy - t <= -v_dot_n
            A.append([-nx, -ny, -1.0])
            b.append(-v_dot_n)
    
    # t >= 0
    A.append([0.0, 0.0, -1.0])
    b.append(0.0)
    
    c_obj = [0.0, 0.0, 1.0]
    
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                     bounds=[(None, None), (None, None), (0, None)],
                     method='highs')
        if res.success:
            cx, cy, t = res.x
            side = t * 2.0 / math.sqrt(3)
            return cx, cy, side
        else:
            return None
    except:
        return None

def _soft_max(values, beta=50.0):
    """Smooth approximation of max using log-sum-exp."""
    m = max(values)
    if m < -1e10:
        return m
    shifted = [v - m for v in values]
    log_sum = math.log(sum(math.exp(beta * s) for s in shifted))
    return m + math.log(sum(math.exp(beta * s) for s in shifted)) / beta

def _outer_side_smooth(positions, angles, outer_cx, outer_cy, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for the outer hexagon side length using soft-max.
    This gives a differentiable objective for optimization.
    """
    all_verts = _compute_all_vertices(positions, angles)
    normals = []
    for k in range(6):
        a = math.radians(outer_angle_deg + 30 + 60 * k)
        normals.append((math.cos(a), math.sin(a)))
    
    max_projs = []
    for nx, ny in normals:
        projs = [(vx - outer_cx) * nx + (vy - outer_cy) * ny for vx, vy in all_verts]
        max_projs.append(max(projs))
    
    t = max(max_projs)  # Use exact max for robustness, soft version for gradient
    side = t * 2.0 / math.sqrt(3)
    return side

def _find_best_outer_lp(all_verts, time_budget=5.0):
    """
    Find the best outer hexagon (angle, center, side) for given vertices
    using multi-resolution LP search.
    """
    start = time.time()
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Coarse search: 0 to 30 degrees (hexagon has 60-degree symmetry, but
    # the packing breaks it, so we search 0-60)
    for outer_angle_deg in np.arange(0, 60.1, 2.0):
        if time.time() - start > time_budget:
            break
        res = _min_outer_side_lp(all_verts, outer_angle_deg)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_center = (cx, cy)
                best_angle = outer_angle_deg
    
    # Medium resolution around best
    if best_side < float('inf') and time.time() - start < time_budget:
        for outer_angle_deg in np.arange(max(0, best_angle - 2.0),
                                         min(60, best_angle + 2.0) + 0.01, 0.2):
            if time.time() - start > time_budget:
                break
            res = _min_outer_side_lp(all_verts, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = outer_angle_deg
    
    # Fine resolution
    if best_side < float('inf') and time.time() - start < time_budget:
        for outer_angle_deg in np.arange(max(0, best_angle - 0.2),
                                         min(60, best_angle + 0.2) + 0.001, 0.01):
            if time.time() - start > time_budget:
                break
            res = _min_outer_side_lp(all_verts, outer_angle_deg)
            if res:
                cx, cy, side = res
                if side < best_side:
                    best_side = side
                    best_center = (cx, cy)
                    best_angle = outer_angle_deg
    
    return best_side, best_center, best_angle

def _nelder_mead_polish(positions, angles, max_iter=150, tol=1e-7):
    """
    Polish the inner positions using Nelder-Mead on the outer hexagon side length.
    """
    n = len(positions)
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    
    def objective(params):
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:])
        
        # Check disjointness with a small tolerance
        if not _check_disjoint(pos, ang, tol=1e-8):
            return 1e6
        
        all_verts = _compute_all_vertices(pos, ang)
        
        # Quick LP search for best outer
        best_side = float('inf')
        for outer_angle in np.arange(0, 60.1, 3.0):
            res = _min_outer_side_lp(all_verts, outer_angle)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
        return best_side
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': tol, 'fatol': tol})
        if res.success:
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:])
            return new_pos, new_ang
    except:
        pass
    
    return positions, angles

def _full_polish(positions, angles, outer_cx, outer_cy, outer_angle, max_iter=100):
    """
    Full polish: optimize inner positions, angles, AND outer hexagon parameters
    simultaneously using Nelder-Mead with the LP-based objective.
    """
    n = len(positions)
    # Parameters: x0,y0, x1,y1, ..., angle0,...,angleN-1, outer_cx, outer_cy, outer_angle
    x0 = []
    for px, py in positions:
        x0.extend([px, py])
    x0.extend(angles)
    x0.extend([outer_cx, outer_cy, outer_angle])
    
    def objective(params):
        pos = [(params[2*i], params[2*i+1]) for i in range(n)]
        ang = list(params[2*n:2*n+n])
        ocx = params[2*n+n]
        ocy = params[2*n+n+1]
        oang = params[2*n+n+2]
        
        # Check disjointness
        if not _check_disjoint(pos, ang, tol=1e-8):
            return 1e6
        
        all_verts = _compute_all_vertices(pos, ang)
        
        # Use LP to find best side for this configuration
        res = _min_outer_side_lp(all_verts, oang)
        if res is None:
            return 1e6
        return res[2]
    
    try:
        res = minimize(objective, x0, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-8})
        if res.success:
            new_pos = [(res.x[2*i], res.x[2*i+1]) for i in range(n)]
            new_ang = list(res.x[2*n:2*n+n])
            new_ocx = res.x[2*n+n]
            new_ocy = res.x[2*n+n+1]
            new_oang = res.x[2*n+n+2]
            # Verify
            if _check_disjoint(new_pos, new_ang, tol=1e-8):
                return new_pos, new_ang, new_ocx, new_ocy, new_oang
    except:
        pass
    
    return positions, angles, outer_cx, outer_cy, outer_angle

def _generate_lattice_positions(N, spacing=1.0):
    """Generate positions on a triangular lattice, taking N closest to origin."""
    sqrt3 = math.sqrt(3)
    scale = spacing * 1.000001  # Slightly larger than touching to ensure disjoint
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)
    
    lattice_positions = []
    for i in range(-4, 5):
        for j in range(-4, 5):
            x = i * u[0] + j * v[0]
            y = i * u[1] + j * v[1]
            lattice_positions.append((x, y))
    
    lattice_positions.sort(key=lambda p: p[0]*p[0] + p[1]*p[1])
    return lattice_positions[:N]

def _try_configuration(positions, angles, time_budget=10.0):
    """
    For a given configuration of inner hexagons, find the best outer hexagon
    and optionally polish. Returns (side, inner, center, outer_angle).
    """
    start = time.time()
    all_verts = _compute_all_vertices(positions, angles)
    
    # Find best outer hexagon
    best_side, best_center, best_angle = _find_best_outer_lp(all_verts, time_budget)
    
    if best_side == float('inf'):
        return float('inf'), None, (0.0, 0.0), 0.0
    
    # Polish inner positions if time allows
    if time.time() - start < time_budget * 0.4:
        polished_pos, polished_ang = _nelder_mead_polish(positions, angles, max_iter=80)
        
        if polished_pos != positions or polished_ang != angles:
            if _check_disjoint(polished_pos, polished_ang, tol=1e-8):
                all_verts_p = _compute_all_vertices(polished_pos, polished_ang)
                p_side, p_center, p_angle = _find_best_outer_lp(all_verts_p, max(1.0, time_budget * 0.3))
                
                if p_side < best_side:
                    best_side = p_side
                    best_center = p_center
                    best_angle = p_angle
                    positions = polished_pos
                    angles = polished_ang
    
    # Full polish: optimize everything together
    if time.time() - start < time_budget * 0.6:
        fp_pos, fp_ang, fp_cx, fp_cy, fp_ang = _full_polish(
            positions, angles, best_center[0], best_center[1], best_angle, max_iter=60)
        
        if _check_disjoint(fp_pos, fp_ang, tol=1e-8):
            all_verts_fp = _compute_all_vertices(fp_pos, fp_ang)
            res = _min_outer_side_lp(all_verts_fp, fp_ang)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
                    best_center = (fp_cx, fp_cy)
                    best_angle = fp_ang
                    positions = fp_pos
                    angles = fp_ang
    
    inner = [[pos[0], pos[1], ang] for pos, ang in zip(positions, angles)]
    return best_side, inner, best_center, best_angle

def construct():
    global _START_TIME, _DEADLINE
    
    if time.time() > _DEADLINE - 5.0:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    
    # Generate initial lattice positions
    lattice_positions = _generate_lattice_positions(N, spacing=1.0)
    
    # Try different angle configurations
    angle_configs = [
        [0.0] * N,
        [30.0] * N,
        [0.0, 30.0] * 5 + [0.0],
        [15.0] * N,
        [45.0] * N,
    ]
    
    # Also try slightly perturbed lattice positions
    perturbations = [0.0, 0.05, 0.1]
    
    for spacing in [1.0]:
        for pert in perturbations:
            if time.time() > _DEADLINE - 15.0:
                break
            
            positions = list(lattice_positions)
            if pert > 0:
                # Add small deterministic perturbation
                for i in range(N):
                    dx = pert * math.sin(i * 2.399)  # Golden angle
                    dy = pert * math.cos(i * 2.399)
                    positions[i] = (positions[i][0] + dx, positions[i][1] + dy)
            
            for inner_angles in angle_configs:
                if time.time() > _DEADLINE - 15.0:
                    break
                
                # Check disjointness
                if not _check_disjoint(positions, inner_angles, tol=1e-7):
                    continue
                
                side, inner, center, outer_angle = _try_configuration(
                    positions, inner_angles, time_budget=8.0)
                
                if side < best_side:
                    best_side = side
                    best_inner = inner
                    best_center = center
                    best_outer_angle = outer_angle
    
    # If we have a good result, do a final high-precision polish
    if best_inner is not None and time.time() < _DEADLINE - 20.0:
        positions = [(p[0], p[1]) for p in best_inner]
        angles = [p[2] for p in best_inner]
        
        # Final polish with tight tolerance
        fp_pos, fp_ang, fp_cx, fp_cy, fp_oang = _full_polish(
            positions, angles, best_center[0], best_center[1], best_outer_angle, max_iter=80)
        
        if _check_disjoint(fp_pos, fp_ang, tol=1e-9):
            all_verts_fp = _compute_all_vertices(fp_pos, fp_ang)
            res = _min_outer_side_lp(all_verts_fp, fp_oang)
            if res:
                side = res[2]
                if side < best_side:
                    best_side = side
                    best_center = (fp_cx, fp_cy)
                    best_outer_angle = fp_oang
                    best_inner = [[pos[0], pos[1], ang] for pos, ang in zip(fp_pos, fp_ang)]
    
    if best_inner is None:
        # Fallback
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0
    
    return best_inner, best_center, best_side, best_outer_angle
