"""Direct hexagon packing: 11 unit hexagons with optimized outer hexagon using vectorized evaluation."""
import math
import time
import numpy as np

N = 11


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon with side length `side`,
    centered at (cx, cy), rotated by angle_deg degrees."""
    verts = []
    for k in range(6):
        a = math.radians(angle_deg + 60 * k)
        verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
    return verts


def _point_in_hex(px, py, cx, cy, side, angle_deg):
    """Check if point (px,py) is inside hexagon centered at (cx,cy) with side `side` rotated by angle_deg."""
    dx = px - cx
    dy = py - cy
    a = -math.radians(angle_deg)
    lx = dx * math.cos(a) - dy * math.sin(a)
    ly = dx * math.sin(a) + dy * math.cos(a)
    apothem = side * math.sqrt(3) / 2
    for k in range(6):
        a_norm = math.radians(30 + 60 * k)
        nx = math.cos(a_norm)
        ny = math.sin(a_norm)
        if lx * nx + ly * ny > apothem + 1e-9:
            return False
    return True


def _segments_intersect(p1, p2, p3, p4):
    """Check if segment p1-p2 properly intersects segment p3-p4."""
    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    d1 = cross(p3, p4, p1)
    d2 = cross(p3, p4, p2)
    d3 = cross(p1, p2, p3)
    d4 = cross(p1, p2, p4)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and \
       ((d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)):
        return True
    return False


def _hexagons_disjoint(hex1, hex2, eps=1e-9):
    """Check if two hexagons have disjoint interiors."""
    cx1, cy1, s1, a1 = hex1
    cx2, cy2, s2, a2 = hex2
    r1 = s1
    r2 = s2
    dx = cx2 - cx1
    dy = cy2 - cy1
    if dx * dx + dy * dy > (r1 + r2) ** 2 + 1e-12:
        return True
    verts1 = _hex_vertices(cx1, cy1, s1, a1)
    for vx, vy in verts1:
        if _point_in_hex(vx, vy, cx2, cy2, s2, a2):
            return False
    verts2 = _hex_vertices(cx2, cy2, s2, a2)
    for vx, vy in verts2:
        if _point_in_hex(vx, vy, cx1, cy1, s1, a1):
            return False
    for i in range(6):
        v1a = verts1[i]
        v1b = verts1[(i + 1) % 6]
        for j in range(6):
            v2a = verts2[j]
            v2b = verts2[(j + 1) % 6]
            if _segments_intersect(v1a, v1b, v2a, v2b):
                return False
    return True


def _all_disjoint(inner_hexes, eps=1e-9):
    """Check if all pairs of hexagons are disjoint."""
    n = len(inner_hexes)
    for i in range(n):
        for j in range(i + 1, n):
            if not _hexagons_disjoint(inner_hexes[i], inner_hexes[j], eps):
                return False
    return True


def _inner_in_outer(inner_hex, outer_hex):
    """Check if inner hexagon is fully inside outer hexagon."""
    cx, cy, s, a = inner_hex
    ox, oy, oside, oangle = outer_hex
    verts = _hex_vertices(cx, cy, s, a)
    for vx, vy in verts:
        if not _point_in_hex(vx, vy, ox, oy, oside, oangle):
            return False
    return True


def _min_outer_side(hexes, ox, oy, oangle=0.0):
    """Find minimum side length of outer hexagon centered at (ox, oy) with rotation oangle
    that contains all hexagons in `hexes`."""
    max_val = 0.0
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    for cx, cy, s, ang in hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        for vx, vy in verts:
            dx = vx - ox
            dy = vy - oy
            lx = dx * cos_a - dy * sin_a
            ly = dx * sin_a + dy * cos_a
            for k in range(6):
                theta = math.radians(30 + 60 * k)
                proj = lx * math.cos(theta) + ly * math.sin(theta)
                if proj > max_val:
                    max_val = proj
    return max_val * 2.0 / math.sqrt(3)


def _compute_all_projections(verts_array, cx, cy, angle_deg):
    """Compute all 6 projections of all vertices onto the 6 normal directions.
    Returns array of shape (n_verts * 6,) with all projection values."""
    a = -math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    dx = verts_array[:, 0] - cx
    dy = verts_array[:, 1] - cy
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    # 6 normal directions
    thetas = [math.radians(30 + 60 * k) for k in range(6)]
    nxs = np.array([math.cos(t) for t in thetas])
    nys = np.array([math.sin(t) for t in thetas])
    
    # Compute all projections: (n_verts, 6)
    proj_lx = lx[:, np.newaxis] * nxs[np.newaxis, :]
    proj_ly = ly[:, np.newaxis] * nys[np.newaxis, :]
    all_projs = (proj_lx + proj_ly).flatten()
    
    return all_projs


def _min_side_from_projections(all_projs):
    """Compute minimum outer side length from all projections."""
    sqrt3 = math.sqrt(3)
    max_val = np.max(all_projs)
    return max_val * 2.0 / sqrt3


def _smooth_min_side(verts_array, cx, cy, angle_deg, beta=80.0):
    """Smooth log-sum-exp approximation of the max projection."""
    a = -math.radians(angle_deg)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    dx = verts_array[:, 0] - cx
    dy = verts_array[:, 1] - cy
    lx = dx * cos_a - dy * sin_a
    ly = dx * sin_a + dy * cos_a
    
    thetas = [math.radians(30 + 60 * k) for k in range(6)]
    nxs = np.array([math.cos(t) for t in thetas])
    nys = np.array([math.sin(t) for t in thetas])
    
    proj_lx = lx[:, np.newaxis] * nxs[np.newaxis, :]
    proj_ly = ly[:, np.newaxis] * nys[np.newaxis, :]
    all_projs = (proj_lx + proj_ly).flatten()
    
    # Log-sum-exp for stability
    max_p = np.max(all_projs)
    log_sum_exp = max_p + np.log(np.sum(np.exp(all_projs - max_p)))
    smooth_max = log_sum_exp / beta
    
    sqrt3 = math.sqrt(3)
    return smooth_max * 2.0 / sqrt3


def _optimize_outer_hex(inner_hexes, initial_cx, initial_cy, initial_angle):
    """Optimize the outer hexagon (center, angle) to minimize its side length."""
    from scipy.optimize import minimize
    
    sqrt3 = math.sqrt(3)
    
    # Precompute all vertices
    all_verts = []
    for cx, cy, s, ang in inner_hexes:
        verts = _hex_vertices(cx, cy, s, ang)
        all_verts.extend(verts)
    verts_array = np.array(all_verts)
    
    # Phase 1: Use smooth surrogate for optimization
    def objective_smooth(params):
        cx, cy, angle = params
        return _smooth_min_side(verts_array, cx, cy, angle, beta=100.0)
    
    x0 = [initial_cx, initial_cy, initial_angle]
    
    try:
        res = minimize(objective_smooth, x0, method='Nelder-Mead',
                      options={'xatol': 1e-9, 'fatol': 1e-12, 'maxiter': 5000, 'maxfev': 10000})
        if res.fun < objective_smooth(*x0):
            best_cx, best_cy, best_angle = res.x
        else:
            best_cx, best_cy, best_angle = x0
    except Exception:
        best_cx, best_cy, best_angle = x0
    
    # Phase 2: Refine with exact objective
    def objective_exact(params):
        cx, cy, angle = params
        all_projs = _compute_all_projections(verts_array, cx, cy, angle)
        return _min_side_from_projections(all_projs)
    
    best_side = objective_exact([best_cx, best_cy, best_angle])
    
    try:
        res2 = minimize(objective_exact, [best_cx, best_cy, best_angle], method='Nelder-Mead',
                       options={'xatol': 1e-11, 'fatol': 1e-14, 'maxiter': 3000, 'maxfev': 8000})
        if res2.fun < best_side:
            best_cx, best_cy, best_angle = res2.x
            best_side = res2.fun
    except Exception:
        pass
    
    # Phase 3: High-precision coordinate descent polish
    step_c = 1e-8
    step_a = 1e-7
    for _ in range(60):
        improved = False
        for dcx, dcy in [(step_c, 0), (-step_c, 0), (0, step_c), (0, -step_c),
                         (step_c, step_c), (-step_c, step_c), (step_c, -step_c), (-step_c, -step_c)]:
            new_side = objective_exact([best_cx + dcx, best_cy + dcy, best_angle])
            if new_side < best_side - 1e-14:
                best_cx += dcx
                best_cy += dcy
                best_side = new_side
                improved = True
        for da in [step_a, -step_a]:
            new_angle = best_angle + da
            new_side = objective_exact([best_cx, best_cy, new_angle])
            if new_side < best_side - 1e-14:
                best_angle = new_angle
                best_side = new_side
                improved = True
        if not improved:
            step_c *= 0.5
            step_a *= 0.5
            if step_c < 1e-13:
                break
    
    return best_cx, best_cy, best_angle, best_side


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s limit
    
    sqrt3 = math.sqrt(3)
    
    # Flat-top hexagon mapping using axial coordinates:
    # For a unit hexagon (side=1) with flat-top orientation:
    # x = 1.5 * q
    # y = sqrt(3) * (r + q/2)
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)
    
    # Center + ring 1 (6 hexagons)
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)
    
    # Ring 2 positions (distance 2 from center) - 14 hexagons
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]
    
    best_side = float('inf')
    best_positions = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    
    # Try all combinations of 4 from ring2
    import itertools
    for combo in itertools.combinations(range(len(ring2)), 4):
        if time.time() > deadline:
            break
            
        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        
        # Convert to Cartesian
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]
        
        # Compute centroid as initial guess
        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0
        
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]
        
        # Try several initial outer hexagon configurations
        initial_candidates = [
            (cx_mean, cy_mean, 0.0),
            (cx_mean, cy_mean, 30.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 30.0),
            (cx_mean, cy_mean, 15.0),
            (cx_mean, cy_mean, 45.0),
        ]
        
        for ocx, ocy, oang in initial_candidates:
            if time.time() > deadline:
                break
            # First evaluate quickly
            side_quick = _min_outer_side(inner_hexes, ocx, ocy, oang)
            
            # If this looks promising, optimize
            if side_quick < best_side * 1.01:
                opt_cx, opt_cy, opt_angle, opt_side = _optimize_outer_hex(
                    inner_hexes, ocx, ocy, oang
                )
                
                if opt_side < best_side:
                    best_side = opt_side
                    best_positions = full_xy
                    best_center = (opt_cx, opt_cy)
                    best_angle = opt_angle

    if best_positions is None:
        # Fallback: just use the 7-hexagon cluster
        best_positions = [axial_to_xy(q, r) for q, r in pos_axial]
        inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]
        cx_mean = sum(x for x, y in best_positions) / 7.0
        cy_mean = sum(y for x, y in best_positions) / 7.0
        best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
            inner_hexes, cx_mean, cy_mean, 0.0
        )
        best_center = (best_cx, best_cy)

    # Build final inner hexagons
    inner_hexes = [(x, y, 1.0, 0.0) for x, y in best_positions]

    # Verify disjointness
    if not _all_disjoint(inner_hexes):
        # If there's an issue, try scaling up positions slightly
        for scale in [1.001, 1.01, 1.05]:
            scaled_hexes = [(x * scale, y * scale, 1.0, 0.0) for x, y in best_positions]
            if _all_disjoint(scaled_hexes):
                inner_hexes = scaled_hexes
                cx_mean = sum(x for x, y, s, a in inner_hexes) / 11.0
                cy_mean = sum(y for x, y, s, a in inner_hexes) / 11.0
                best_cx, best_cy, best_angle, best_side = _optimize_outer_hex(
                    inner_hexes, cx_mean, cy_mean, 0.0
                )
                best_center = (best_cx, best_cy)
                break

    # Verify all inner hexagons are inside outer
    outer_hex = (best_center[0], best_center[1], best_side, best_angle)
    valid = True
    for h in inner_hexes:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Increase side slightly to ensure containment
        best_side *= 1.001

    # Format output: inner rows (x, y, angle_degrees)
    inner = [[h[0], h[1], h[3]] for h in inner_hexes]

    return inner, (best_center[0], best_center[1]), best_side, best_angle
