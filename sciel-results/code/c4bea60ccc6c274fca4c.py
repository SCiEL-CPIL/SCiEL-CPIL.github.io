"""Hexagon packing: 11 unit hexagons inside a regular hexagon.
Uses a smooth log-sum-exp surrogate for the outer side length with
Nelder-Mead optimization over all 33 position variables plus outer
center and rotation. Includes a final exact polish pass."""
import math
import time
import itertools
import random

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


def _smooth_outer_side(hexes, ox, oy, oangle, beta=50.0):
    """Smooth approximation of _min_outer_side using log-sum-exp."""
    a = -math.radians(oangle)
    cos_a = math.cos(a)
    sin_a = math.sin(a)
    
    projs = []
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
                projs.append(proj)
    
    if not projs:
        return 0.0
    
    m = max(projs)
    exp_sum = sum(math.exp(p - m) for p in projs)
    soft_max = m + math.log(exp_sum) / beta
    
    return soft_max * 2.0 / math.sqrt(3)


def _separation_violation(hexes):
    """Compute total pairwise separation violation (positive means overlap)."""
    n = len(hexes)
    total = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            cx1, cy1, s1, a1 = hexes[i]
            cx2, cy2, s2, a2 = hexes[j]
            # Minimum distance between centers for non-overlapping unit hexagons
            # in the same orientation is sqrt(3) (edge-to-edge)
            # But with rotation, the minimum center distance can be as low as sqrt(3)
            # For safety, use a slightly smaller threshold
            dx = cx2 - cx1
            dy = cy2 - cy1
            dist = math.sqrt(dx * dx + dy * dy)
            min_dist = math.sqrt(3)  # For same-orientation hexagons
            if dist < min_dist:
                total += (min_dist - dist) ** 2
    return total


def _objective(params, beta):
    """Combined objective: smooth outer side + separation penalty."""
    n = len(params) // 2 + 2  # 11*2 positions + cx + cy + angle
    # Actually: 11*2 = 22 position coords + 3 (ox, oy, oangle) = 25
    positions = []
    for i in range(11):
        x = params[2 * i]
        y = params[2 * i + 1]
        positions.append((x, y, 1.0, 0.0))
    ox = params[22]
    oy = params[23]
    oangle = params[24]
    
    outer_side = _smooth_outer_side(positions, ox, oy, oangle, beta)
    sep_viol = _separation_violation(positions)
    
    return outer_side + 10.0 * sep_viol


def _nelder_mead(objective, x0, max_iter=500, tol=1e-10, alpha=1.0, gamma=2.0, rho=0.5, sigma=0.5):
    """Nelder-Mead simplex optimization."""
    n = len(x0)
    # Initialize simplex
    simplex = [list(x0)]
    for i in range(n):
        perturbed = list(x0)
        perturbed[i] += 0.1 if perturbed[i] != 0 else 0.1
        simplex.append(perturbed)
    
    # Evaluate initial simplex
    values = [objective(s) for s in simplex]
    
    for iteration in range(max_iter):
        # Sort simplex by objective value
        order = sorted(range(n + 1), key=lambda i: values[i])
        simplex = [simplex[i] for i in order]
        values = [values[i] for i in order]
        
        # Check convergence
        if abs(values[-1] - values[0]) < tol:
            break
        
        # Centroid of all points except worst
        centroid = [sum(simplex[i][j] for i in range(n)) / n for j in range(n)]
        
        # Reflection
        worst = simplex[-1]
        xr = [centroid[j] + alpha * (centroid[j] - worst[j]) for j in range(n)]
        fr = objective(xr)
        
        if fr < values[0]:
            # Expansion
            xe = [centroid[j] + gamma * (centroid[j] - worst[j]) for j in range(n)]
            fe = objective(xe)
            if fe < fr:
                simplex[-1] = xe
                values[-1] = fe
            else:
                simplex[-1] = xr
                values[-1] = fr
        elif fr < values[-2]:
            simplex[-1] = xr
            values[-1] = fr
        else:
            # Contraction
            xc = [centroid[j] + rho * (worst[j] - centroid[j]) for j in range(n)]
            fc = objective(xc)
            if fc < values[-1]:
                simplex[-1] = xc
                values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = [simplex[0][j] + sigma * (simplex[i][j] - simplex[0][j]) for j in range(n)]
                    values[i] = objective(simplex[i])
    
    # Return best point
    best_idx = min(range(n + 1), key=lambda i: values[i])
    return simplex[best_idx], values[best_idx]


def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    sqrt3 = math.sqrt(3)

    # Axial coordinate to Cartesian for flat-top hexagons
    def axial_to_xy(q, r):
        x = 1.5 * q
        y = sqrt3 * (r + q / 2.0)
        return (x, y)

    # Generate initial hexagonal lattice configuration
    pos_axial = [(0, 0)]
    ring1 = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    pos_axial.extend(ring1)

    # Ring 2 positions
    ring2 = [(2, 0), (2, -1), (2, -2), (1, -2), (0, -2), (-1, -2),
             (-2, -2), (-2, -1), (-2, 0), (-2, 1), (-2, 2), (-1, 2),
             (0, 2), (1, 2)]

    best_side = float('inf')
    best_params = None
    best_positions = None

    # Phase 1: Coarse scan over ring2 combinations and outer orientations
    all_combos = list(itertools.combinations(range(len(ring2)), 4))
    oangles = [0.0, 30.0, 15.0, -15.0, 45.0, -45.0, 60.0, -60.0, 75.0, -75.0,
               10.0, -10.0, 20.0, -20.0, 50.0, -50.0, 80.0, -80.0, 5.0, -5.0]

    for combo in all_combos:
        if time.time() > deadline - 50.0:
            break

        selected_r2 = [ring2[i] for i in combo]
        full_axial = pos_axial + selected_r2
        full_xy = [axial_to_xy(q, r) for q, r in full_axial]

        cx_mean = sum(x for x, y in full_xy) / 11.0
        cy_mean = sum(y for x, y in full_xy) / 11.0

        inner_hexes = [(x, y, 1.0, 0.0) for x, y in full_xy]

        for oang in oangles:
            side = _min_outer_side(inner_hexes, cx_mean, cy_mean, oang)
            if side < best_side:
                best_side = side
                best_positions = full_xy
                best_params = list(full_xy[0]) + list(full_xy[1]) + \
                               list(full_xy[2]) + list(full_xy[3]) + \
                               list(full_xy[4]) + list(full_xy[5]) + \
                               list(full_xy[6]) + list(full_xy[7]) + \
                               list(full_xy[8]) + list(full_xy[9]) + \
                               list(full_xy[10]) + [cx_mean, cy_mean, oang]

    if best_params is None:
        # Fallback
        full_xy = [axial_to_xy(q, r) for q, r in pos_axial]
        best_positions = full_xy
        cx_mean = sum(x for x, y in full_xy) / 7.0
        cy_mean = sum(y for x, y in full_xy) / 7.0
        best_side = _min_outer_side([(x, y, 1.0, 0.0) for x, y in full_xy], cx_mean, cy_mean, 0)
        best_params = []
        for x, y in full_xy:
            best_params.extend([x, y])
        best_params.extend([cx_mean, cy_mean, 0.0])

    # Phase 2: Nelder-Mead optimization with smooth surrogate
    if time.time() < deadline - 20.0:
        # Start from best_params
        cur_params = list(best_params)
        cur_obj = _objective(cur_params, beta=50.0)

        # Try multiple beta values for the smooth surrogate
        for beta in [100.0, 50.0, 25.0, 10.0, 5.0]:
            if time.time() > deadline - 10.0:
                break

            obj_func = lambda p: _objective(p, beta)
            result_params, result_obj = _nelder_mead(obj_func, cur_params, max_iter=300, tol=1e-12)

            if result_obj < cur_obj:
                cur_params = result_params
                cur_obj = result_obj

            # Extract positions and check actual side
            positions = []
            for i in range(11):
                positions.append((cur_params[2 * i], cur_params[2 * i + 1], 1.0, 0.0))
            ox = cur_params[22]
            oy = cur_params[23]
            oangle = cur_params[24]

            # Check disjointness
            if _all_disjoint(positions):
                actual_side = _min_outer_side(positions, ox, oy, oangle)
                if actual_side < best_side:
                    best_side = actual_side
                    best_positions = [(p[0], p[1]) for p in positions]
                    best_params = list(cur_params)

            # Update starting point for next beta
            cur_params = list(result_params)

    # Phase 3: Final high-precision polish using exact objective
    if time.time() < deadline - 5.0 and best_positions is not None:
        # Use the best parameters found
        cur_params = list(best_params)

        # Fine-tune outer hexagon parameters only (positions fixed)
        positions = []
        for i in range(11):
            positions.append((cur_params[2 * i], cur_params[2 * i + 1], 1.0, 0.0))

        # Optimize outer center and angle with small steps
        ox = cur_params[22]
        oy = cur_params[23]
        oangle = cur_params[24]
        cur_side = _min_outer_side(positions, ox, oy, oangle)

        for step_size in [0.01, 0.001, 0.0001]:
            improved = True
            while improved and time.time() < deadline - 2.0:
                improved = False
                for dx in [-step_size, 0, step_size]:
                    for dy in [-step_size, 0, step_size]:
                        for da in [-step_size * 10, 0, step_size * 10]:
                            if dx == 0 and dy == 0 and da == 0:
                                continue
                            nox = ox + dx
                            noy = oy + dy
                            noangle = oangle + da
                            s = _min_outer_side(positions, nox, noy, noangle)
                            if s < cur_side - 1e-12:
                                ox, oy, oangle, cur_side = nox, noy, noangle, s
                                improved = True

        if cur_side < best_side:
            best_side = cur_side
            best_params[22] = ox
            best_params[23] = oy
            best_params[24] = oangle

    # Build final inner hexagons
    final_positions = []
    for i in range(11):
        x = best_params[2 * i]
        y = best_params[2 * i + 1]
        final_positions.append((x, y, 1.0, 0.0))

    # Verify disjointness
    if not _all_disjoint(final_positions):
        # Scale up slightly to ensure disjointness
        for scale in [1.001, 1.01, 1.05]:
            scaled = [(x * scale, y * scale, 1.0, 0.0) for x, y, s, a in final_positions]
            if _all_disjoint(scaled):
                final_positions = scaled
                break

    # Compute final outer hexagon
    cx_mean = sum(p[0] for p in final_positions) / 11.0
    cy_mean = sum(p[1] for p in final_positions) / 11.0
    oangle = best_params[24]

    # Use the optimized center if available, otherwise use centroid
    ox = best_params[22]
    oy = best_params[23]

    final_side = _min_outer_side(final_positions, ox, oy, oangle)

    # Verify all inner hexagons are inside outer
    outer_hex = (ox, oy, final_side, oangle)
    valid = True
    for h in final_positions:
        if not _inner_in_outer(h, outer_hex):
            valid = False
            break

    if not valid:
        # Binary search for minimum valid side length
        lo, hi = final_side, final_side * 1.1
        for _ in range(50):
            mid = (lo + hi) / 2
            test_outer = (ox, oy, mid, oangle)
            ok = all(_inner_in_outer(h, test_outer) for h in final_positions)
            if ok:
                hi = mid
            else:
                lo = mid
        final_side = hi

    # Format output
    inner = [[p[0], p[1], p[3]] for p in final_positions]

    return inner, (ox, oy), final_side, oangle
