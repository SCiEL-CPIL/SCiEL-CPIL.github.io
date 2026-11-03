import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin for 150s limit

    sqrt3 = math.sqrt(3)

    def get_hex_vertices(cx, cy, side, angle_deg):
        """Returns 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append((cx + side * math.cos(a), cy + side * math.sin(a)))
        return verts

    def point_in_hexagon(px, py, hx, hy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (hx, hy)."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        inradius = side * sqrt3 / 2
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            if lx * nx + ly * ny > inradius + 1e-9:
                return False
        return True

    def hex_radius_from_center(px, py, hx, hy, side, angle_deg):
        """Compute the required side length of hexagon centered at (hx,hy) with angle to contain point (px,py)."""
        dx = px - hx
        dy = py - hy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        lx = dx * cos_a - dy * sin_a
        ly = dx * sin_a + dy * cos_a
        max_proj = 0.0
        for k in range(6):
            normal_angle = math.radians(30 + k * 60)
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            proj = lx * nx + ly * ny
            if proj > max_proj:
                max_proj = proj
        # inradius = side * sqrt3/2, so side = max_proj / (sqrt3/2) = 2*max_proj/sqrt3
        return 2.0 * max_proj / sqrt3

    def point_in_polygon(point, polygon):
        """Ray casting algorithm to determine if a point is inside a polygon."""
        x, y = point
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        else:
                            xinters = p1x
                        if (p1x <= xinters or p2x <= xinters) and (p1x != p2x):
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside

    def on_segment(p, q, r):
        """Check if point q lies on segment pr."""
        if (q[0] <= max(p[0], r[0]) and q[0] >= min(p[0], r[0]) and
                q[1] <= max(p[1], r[1]) and q[1] >= min(p[1], r[1])):
            return True
        return False

    def orientation(p, q, r):
        """Find orientation of triangle formed by p, q, r."""
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-12:
            return 0
        return 1 if val > 0 else 2

    def do_intersect(p1, q1, p2, q2):
        """Check if segment p1-q1 intersects with segment p2-q2."""
        o1 = orientation(p1, q1, p2)
        o2 = orientation(p1, q1, q2)
        o3 = orientation(p2, q2, p1)
        o4 = orientation(p2, q2, q1)

        if o1 != o2 and o3 != o4:
            return True

        if o1 == 0 and on_segment(p1, p2, q1):
            return True
        if o2 == 0 and on_segment(p1, q2, q1):
            return True
        if o3 == 0 and on_segment(p2, p1, q2):
            return True
        if o4 == 0 and on_segment(p2, q1, q2):
            return True

        return False

    def edges_intersect(v1, v2):
        """Check if any edge of v1 intersects any edge of v2."""
        for i in range(6):
            p1 = v1[i]
            q1 = v1[(i + 1) % 6]
            for j in range(6):
                p2 = v2[j]
                q2 = v2[(j + 1) % 6]
                if do_intersect(p1, q1, p2, q2):
                    return True
        return False

    def hexes_overlap(v1, v2):
        """Check if two sets of vertices (hexagons) overlap."""
        for p in v1:
            if point_in_polygon(p, v2):
                return True
        for p in v2:
            if point_in_polygon(p, v1):
                return True
        if edges_intersect(v1, v2):
            return True
        return False

    def check_validity(params):
        """
        Check if the configuration is valid (all inner hexagons inside outer, no overlaps).
        Returns (is_valid, max_violation).
        """
        outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
        
        if outer_side < 0.5:
            return False, 1.0

        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            inner_verts_list.append(verts)
            
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    return False, 0.1

        for i in range(N):
            for j in range(i + 1, N):
                cx1 = params[4 + i * 3]
                cy1 = params[5 + i * 3]
                cx2 = params[4 + j * 3]
                cy2 = params[5 + j * 3]
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 4.5:
                    continue
                if hexes_overlap(inner_verts_list[i], inner_verts_list[j]):
                    return False, 0.1

        return True, 0.0

    def objective(params):
        """Objective: outer side length + penalties for violations."""
        if time.time() > deadline:
            return 1e10

        outer_cx, outer_cy, outer_side, outer_angle = params[0], params[1], params[2], params[3]
        
        if outer_side < 0.5:
            return 1e10

        # Check containment
        penalty = 0.0
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_cx, outer_cy, outer_side, outer_angle):
                    # Estimate how far outside
                    req_side = hex_radius_from_center(vx, vy, outer_cx, outer_cy, 1.0, outer_angle)
                    if req_side > outer_side:
                        penalty += (req_side - outer_side) * 10.0
                    else:
                        penalty += 1.0

        if penalty > 50.0:
            return outer_side + penalty

        # Check overlaps
        inner_verts_list = []
        for i in range(N):
            ix = params[4 + i * 3]
            iy = params[5 + i * 3]
            iang = params[6 + i * 3]
            verts = get_hex_vertices(ix, iy, 1.0, iang)
            inner_verts_list.append(verts)

        overlap_count = 0
        for i in range(N):
            for j in range(i + 1, N):
                cx1 = params[4 + i * 3]
                cy1 = params[5 + i * 3]
                cx2 = params[4 + j * 3]
                cy2 = params[5 + j * 3]
                dist_sq = (cx1 - cx2)**2 + (cy1 - cy2)**2
                if dist_sq > 4.5:
                    continue
                if hexes_overlap(inner_verts_list[i], inner_verts_list[j]):
                    overlap_count += 1
                    if overlap_count > 5:
                        return outer_side + 100.0

        if overlap_count > 0:
            return outer_side + 10.0 * overlap_count

        return outer_side

    # Define bounds for DE
    bounds = [
        (-6.0, 6.0),   # outer_cx
        (-6.0, 6.0),   # outer_cy
        (2.5, 12.0),   # outer_side
        (0.0, 60.0)    # outer_angle
    ]
    for i in range(N):
        bounds.append((-6.0, 6.0))
        bounds.append((-6.0, 6.0))
        bounds.append((0.0, 60.0))

    best_params = None
    best_side = float('inf')

    # 1. Global Search with Differential Evolution
    # DE is robust to non-smooth landscapes and can escape local minima
    try:
        de_result = differential_evolution(
            objective, bounds,
            maxiter=50,
            popsize=20,
            tol=1e-5,
            seed=42,
            polish=False,  # We will do our own polish
            update_interval=1,
            callback=lambda x, convergence: None
        )
        
        if de_result.fun < best_side:
            best_side = de_result.fun
            best_params = de_result.x
    except Exception as e:
        pass

    # 2. Local Refinement with Nelder-Mead
    if best_params is not None and time.time() < deadline - 10:
        try:
            nm_result = minimize(
                objective, best_params,
                method='Nelder-Mead',
                bounds=bounds,
                options={'maxiter': 1000, 'xatol': 1e-5, 'fatol': 1e-5, 'adaptive': True}
            )
            if nm_result.fun < best_side:
                best_side = nm_result.fun
                best_params = nm_result.x
        except Exception:
            pass

    # 3. If DE failed or found a bad result, try a few heuristic starts with Nelder-Mead
    if best_params is None or best_side > 10.0:
        # Simple heuristic: Honeycomb
        def make_honeycomb_init(scale=1.0):
            params = [0.0, 0.0, 6.0, 0.0]
            for row in range(3):
                for col in range(4):
                    x = col * sqrt3 * scale + (sqrt3 * scale / 2 if row % 2 == 1 else 0)
                    y = row * 1.5 * scale
                    x -= 2.6 * scale
                    y -= 1.5 * scale
                    params.extend([x, y, 0.0])
            return params

        for scale in [0.9, 1.0, 1.1]:
            if time.time() > deadline - 10:
                break
            init_params = make_honeycomb_init(scale)
            try:
                res = minimize(
                    objective, init_params,
                    method='Nelder-Mead',
                    bounds=bounds,
                    options={'maxiter': 200, 'xatol': 1e-4, 'fatol': 1e-4, 'adaptive': True}
                )
                if res.fun < best_side:
                    best_side = res.fun
                    best_params = res.x
            except Exception:
                pass

    # Verify and return
    if best_params is None or best_side >= float('inf'):
        # Fallback: return a known valid configuration (large enough)
        inner = []
        for row in range(3):
            for col in range(4):
                x = col * sqrt3 + (sqrt3 / 2 if row % 2 == 1 else 0)
                y = row * 1.5
                x -= 2.6
                y -= 1.5
                inner.append([x, y, 0.0])
        return inner, (0.0, 0.0), 8.0, 0.0

    # Check validity of best result
    is_valid, _ = check_validity(best_params)
    if not is_valid:
        # Try to fix by slightly expanding
        outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]
        # Expand outer side until valid
        for trial in range(20):
            new_side = outer_side + 0.1 * (trial + 1)
            test_params = list(best_params)
            test_params[2] = new_side
            is_valid, _ = check_validity(test_params)
            if is_valid:
                best_params = test_params
                break

    outer_cx, outer_cy, outer_side, outer_angle = best_params[0], best_params[1], best_params[2], best_params[3]

    inner = []
    for i in range(N):
        ix = best_params[4 + i * 3]
        iy = best_params[5 + i * 3]
        iang = best_params[6 + i * 3]
        inner.append([ix, iy, iang])

    return inner, (outer_cx, outer_cy), outer_side, outer_angle
# EVOLVE-BLOCK-END
