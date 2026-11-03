import math
import time
import numpy as np
from scipy.optimize import linprog, minimize

N = 11
_START_TIME = time.time()
_DEADLINE = _START_TIME + 145.0
_SQRT3 = math.sqrt(3)
_INVSQRT3 = 1.0 / _SQRT3


def _hex_vertices(cx, cy, angle_deg):
    a0 = math.radians(angle_deg)
    angles = a0 + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(angles), cy + np.sin(angles)])


def _min_enclosing_hex_side(verts, outer_angle_deg):
    """
    Find minimum side length of a regular hexagon (with vertex at outer_angle_deg)
    that contains all points in verts (numpy array of shape (M,2)).
    Returns (cx, cy, side) or None.
    """
    base_angle = math.radians(outer_angle_deg + 30)
    angles = base_angle + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])

    M = len(verts)
    v_dot_n = verts @ normals.T

    A_rows = []
    b_vals = []

    for k in range(6):
        nx, ny = normals[k]
        v_dots = v_dot_n[:, k]
        A_rows.append(np.column_stack([np.full(M, -nx), np.full(M, -ny), np.full(M, -1.0)]))
        b_vals.append(-v_dots)
        A_rows.append(np.column_stack([np.full(M, nx), np.full(M, ny), np.full(M, -1.0)]))
        b_vals.append(v_dots)

    A = np.vstack(A_rows)
    b = np.concatenate(b_vals)

    A = np.vstack([A, [[0.0, 0.0, -1.0]]])
    b = np.append(b, 0.0)

    c_obj = [0.0, 0.0, 1.0]
    try:
        res = linprog(c_obj, A_ub=A, b_ub=b,
                      bounds=[(None, None), (None, None), (0, None)],
                      method='highs')
        if res.success:
            cx, cy, R = res.x
            side = R * _INVSQRT3 * 2.0
            return cx, cy, side
    except Exception:
        pass
    return None


def _min_enclosing_hex_side_fast(verts, outer_angle_deg):
    """Same as above but returns just the side length (or large penalty)."""
    res = _min_enclosing_hex_side(verts, outer_angle_deg)
    if res is None:
        return 1e6
    return res[2]


def _hexagon_distance(p1, p2, a1, a2):
    """
    Approximate minimum distance between two unit hexagons at (p1, angle a1) and (p2, angle a2).
    Returns the gap (positive if separated, negative if overlapping).
    Uses the support function approach: for two convex polygons, the minimum
    separating distance can be found via the GJK-like approach, but for hexagons
    we can use the fact that the distance between two convex sets equals the
    minimum over directions of (support1 - support2 - |d|).
    
    For a simpler but effective approach: check all edge-to-edge and vertex-to-edge
    distances. But that's expensive. Instead, we use a conservative bound:
    the distance between centers minus the sum of circumradii gives a lower bound
    on overlap, but we need the actual gap.
    
    For SLSQP, we need a smooth-ish constraint. We'll use the minimum distance
    between the two hexagons computed by checking all vertex-edge pairs.
    """
    v1 = _hex_vertices(*p1, a1)
    v2 = _hex_vertices(*p2, a2)
    
    min_dist = float('inf')
    # For each edge of hex1, compute distance from each vertex of hex2 to that edge
    for i in range(6):
        e1_start = v1[i]
        e1_end = v1[(i + 1) % 6]
        for j in range(6):
            pt = v2[j]
            # Distance from point pt to line segment e1_start-e1_end
            d = _point_segment_dist(pt, e1_start, e1_end)
            min_dist = min(min_dist, d)
    # Also check edges of hex2 vs vertices of hex1
    for i in range(6):
        e2_start = v2[i]
        e2_end = v2[(i + 1) % 6]
        for j in range(6):
            pt = v1[j]
            d = _point_segment_dist(pt, e2_start, e2_end)
            min_dist = min(min_dist, d)
    
    return min_dist


def _point_segment_dist(p, a, b):
    """Distance from point p to line segment ab."""
    ab = b - a
    ap = p - a
    t = np.dot(ap, ab) / (np.dot(ab, ab) + 1e-15)
    t = max(0.0, min(1.0, t))
    closest = a + t * ab
    return np.linalg.norm(p - closest)


def _build_initial_positions(layout_idx):
    """Generate 11 initial positions for given layout index."""
    sqrt3 = _SQRT3
    scale = 1.0000001
    u = (sqrt3 * scale, 0.0)
    v = (sqrt3 / 2 * scale, 1.5 * scale)

    if layout_idx == 0:
        # Compact hexagonal: center + 6 ring + 4 from second ring
        target = [(0, 0)]
        ring1 = [(1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1)]
        target.extend(ring1)
        ring2 = [(2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0), (-1, -1), (0, -2), (1, -2), (2, -1)]
        ring2_sorted = sorted(ring2, key=lambda p: p[0] ** 2 + p[1] ** 2)
        target.extend(ring2_sorted[:4])
        positions = []
        for ti, tj in target:
            x = ti * u[0] + tj * v[0]
            y = ti * u[1] + tj * v[1]
            positions.append([x, y])
        return positions[:N]
    elif layout_idx == 1:
        positions = []
        y1 = -1.5 * scale
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append([x, y1])
        y2 = 0.0
        for k in range(-1, 3):
            x = k * sqrt3 * scale
            positions.append([x, y2])
        y3 = 1.5 * scale
        for k in range(-1, 3):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append([x, y3])
        return positions[:N]
    elif layout_idx == 2:
        positions = []
        y1 = -1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append([x, y1])
        y2 = 0.0
        for k in range(-1, 2):
            x = k * sqrt3 * scale + sqrt3 / 2 * scale
            positions.append([x, y2])
        y3 = 1.5 * scale
        for k in range(-2, 2):
            x = k * sqrt3 * scale
            positions.append([x, y3])
        return positions[:N]
    elif layout_idx == 3:
        # Diagonal arrangement
        positions = []
        for i in range(N):
            x = i * sqrt3 * scale - 5 * sqrt3 * scale
            y = 0.0
            positions.append([x, y])
        return positions[:N]
    else:
        # Random-ish perturbation of layout 0
        base = _build_initial_positions(0)
        rng = np.random.RandomState(42 + layout_idx)
        positions = []
        for p in base:
            px = p[0] + rng.uniform(-0.3, 0.3)
            py = p[1] + rng.uniform(-0.3, 0.3)
            positions.append([px, py])
        return positions[:N]


def _objective_and_grad(x, inner_angle):
    """
    x = [x1, y1, x2, y2, ..., x11, y11, outer_angle_rad]
    Returns (side_length, None) - we don't have easy gradients for the LP-based objective.
    """
    n_positions = N
    pos_flat = x[:2 * n_positions]
    outer_angle_deg = math.degrees(x[2 * n_positions])
    # Normalize outer angle to [0, 30)
    outer_angle_deg = outer_angle_deg % 30.0
    if outer_angle_deg < 0:
        outer_angle_deg += 30.0

    positions = np.array(pos_flat).reshape(n_positions, 2)
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
    side = _min_enclosing_hex_side_fast(all_vertices, outer_angle_deg)
    return side


def _overlap_penalty(x, inner_angle, margin=0.01):
    """
    Penalty for overlapping hexagons. Returns 0 if all are separated by at least margin.
    """
    n_positions = N
    pos_flat = x[:2 * n_positions]
    positions = np.array(pos_flat).reshape(n_positions, 2)
    
    penalty = 0.0
    for i in range(n_positions):
        for j in range(i + 1, n_positions):
            dist = _hexagon_distance(positions[i], positions[j], inner_angle, inner_angle)
            if dist < margin:
                # Quadratic penalty for violation
                violation = margin - dist
                penalty += violation * violation
    return penalty


def _construct_with_joint_optimization(initial_positions, inner_angle, time_budget):
    """
    Jointly optimize positions and outer angle using SLSQP.
    """
    n_positions = N
    x0 = np.zeros(2 * n_positions + 1)
    for i, (px, py) in enumerate(initial_positions):
        x0[2 * i] = px
        x0[2 * i + 1] = py
    x0[2 * n_positions] = math.radians(0.0)  # Start outer angle at 0
    
    def obj(x):
        side = _objective_and_grad(x, inner_angle)
        pen = _overlap_penalty(x, inner_angle)
        return side + 100.0 * pen
    
    # Bounds: positions in a reasonable range, angle in [0, 2*pi)
    bounds = []
    for i in range(n_positions):
        bounds.append((-15.0, 15.0))
        bounds.append((-15.0, 15.0))
    bounds.append((0.0, 2 * math.pi))
    
    start_time = time.time()
    try:
        res = minimize(
            obj, x0, method='SLSQP', bounds=bounds,
            options={'maxiter': 200, 'ftol': 1e-10, 'disp': False},
            timeout=time_budget
        )
        elapsed = time.time() - start_time
        if res.success or res.fun < 1e5:
            x_opt = res.x
            side = _objective_and_grad(x_opt, inner_angle)
            pen = _overlap_penalty(x_opt, inner_angle)
            if pen < 0.01:  # Acceptable overlap
                positions = np.array(x_opt[:2 * n_positions]).reshape(n_positions, 2)
                outer_angle_deg = math.degrees(x_opt[2 * n_positions]) % 30.0
                # Get the center
                all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
                res_center = _min_enclosing_hex_side(all_vertices, outer_angle_deg)
                if res_center:
                    cx, cy, side_final = res_center
                    return positions, (cx, cy), side_final, outer_angle_deg, elapsed
    except Exception:
        pass
    return None


def _ternary_search_angle(verts, lo, hi, n_iter=40, deadline_margin=10.0):
    """Ternary search for the optimal outer angle in [lo, hi]."""
    for _ in range(n_iter):
        if time.time() > _DEADLINE - deadline_margin:
            break
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = _min_enclosing_hex_side_fast(verts, m1)
        f2 = _min_enclosing_hex_side_fast(verts, m2)
        if f1 < f2:
            hi = m2
        else:
            lo = m1
    mid = (lo + hi) / 2.0
    return mid


def _refine_fixed_positions(positions, inner_angle, time_budget):
    """
    For fixed positions, do a thorough search over outer angle.
    Returns (best_angle, best_side, best_center).
    """
    all_vertices = np.vstack([_hex_vertices(px, py, inner_angle) for px, py in positions])
    
    # Coarse grid
    best_side = float('inf')
    best_angle = 0.0
    best_center = None
    
    for a in np.arange(0, 30.01, 0.5):
        if time.time() > _DEADLINE - 8.0:
            break
        res = _min_enclosing_hex_side(all_vertices, a)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = a
                best_center = (cx, cy)
    
    # Fine grid around best
    lo = max(0.0, best_angle - 1.0)
    hi = min(30.0, best_angle + 1.0)
    for a in np.arange(lo, hi + 0.01, 0.01):
        if time.time() > _DEADLINE - 8.0:
            break
        res = _min_enclosing_hex_side(all_vertices, a)
        if res:
            cx, cy, side = res
            if side < best_side:
                best_side = side
                best_angle = a
                best_center = (cx, cy)
    
    # Ternary search for sub-degree precision
    lo = max(0.0, best_angle - 0.05)
    hi = min(30.0, best_angle + 0.05)
    refined = _ternary_search_angle(all_vertices, lo, hi, n_iter=30, deadline_margin=8.0)
    res = _min_enclosing_hex_side(all_vertices, refined)
    if res:
        cx, cy, side = res
        if side < best_side:
            best_side = side
            best_angle = refined
            best_center = (cx, cy)
    
    return best_angle, best_side, best_center


def construct():
    global _START_TIME, _DEADLINE

    if time.time() > _DEADLINE - 5.0:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    best_side = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_outer_angle = 0.0
    best_inner_angle = 0.0

    num_layouts = 5
    inner_angles_to_try = [0.0, 30.0, 15.0, -15.0]

    # Phase 1: Try joint optimization with a few layouts
    for layout_idx in range(num_layouts):
        if time.time() > _DEADLINE - 30.0:
            break
        
        for inner_angle in inner_angles_to_try[:2]:  # Limit to 2 angles per layout for joint opt
            if time.time() > _DEADLINE - 25.0:
                break
            
            initial_positions = _build_initial_positions(layout_idx)
            time_budget = min(20.0, _DEADLINE - time.time() - 15.0)
            if time_budget < 2.0:
                continue
            
            result = _construct_with_joint_optimization(initial_positions, inner_angle, time_budget)
            if result:
                positions, center, side, outer_angle, elapsed = result
                if side < best_side:
                    best_side = side
                    best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                    best_center = center
                    best_outer_angle = outer_angle
                    best_inner_angle = inner_angle

    # Phase 2: Refine the best result found so far with fixed-position angle search
    if best_inner is not None and time.time() < _DEADLINE - 15.0:
        positions = [[p[0], p[1]] for p in best_inner]
        angle, side, center = _refine_fixed_positions(positions, best_inner_angle, 15.0)
        if side < best_side:
            best_side = side
            best_center = center
            best_outer_angle = angle

    # Phase 3: Also try the lattice layouts with full angle search (as fallback/complement)
    for layout_idx in range(num_layouts):
        if time.time() > _DEADLINE - 10.0:
            break
        
        for inner_angle in inner_angles_to_try:
            if time.time() > _DEADLINE - 8.0:
                break
            
            positions = _build_initial_positions(layout_idx)
            angle, side, center = _refine_fixed_positions(positions, inner_angle, 5.0)
            if side < best_side:
                best_side = side
                best_inner = [[pos[0], pos[1], inner_angle] for pos in positions]
                best_center = center
                best_outer_angle = angle
                best_inner_angle = inner_angle

    if best_inner is None:
        inner = [[4.0 * k, 0.0, 0.0] for k in range(-(N // 2), N - N // 2)]
        return inner, (0.0, 0.0), 40.0, 0.0

    return best_inner, best_center, best_side, best_outer_angle
