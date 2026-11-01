import math
import time
import random
import numpy as np
from scipy.optimize import minimize
from itertools import combinations, product

N = 11
SQRT3 = math.sqrt(3)

def construct():
    start_time = time.time()
    deadline = start_time + 130.0

    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Precompute base hex vertices (unit side, angle 0)
    base_verts = np.array([(math.cos(math.radians(60 * i)), math.sin(math.radians(60 * i))) for i in range(6)])

    def transform_verts_np(cx, cy, S, angle_deg):
        """Return (6,2) array of vertices for a hex with center, side, rotation."""
        a = math.radians(angle_deg)
        R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        return np.array([[cx + S * (R[0,0]*vx + R[0,1]*vy), cy + S * (R[1,0]*vx + R[1,1]*vy)] for vx, vy in base_verts])

    def transform_verts(cx, cy, S, angle_deg):
        """Return list of (x,y) tuples for a hex."""
        v = transform_verts_np(cx, cy, S, angle_deg)
        return [(v[i,0], v[i,1]) for i in range(6)]

    def segments_intersect(p1, p2, p3, p4):
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

    def point_in_polygon(p, polygon):
        x, y = p
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = polygon[i]
            xj, yj = polygon[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(verts_a, verts_b):
        for v in verts_a:
            if point_in_polygon(v, verts_b):
                return False
        for v in verts_b:
            if point_in_polygon(v, verts_a):
                return False
        for i in range(6):
            a1, a2 = verts_a[i], verts_a[(i + 1) % 6]
            b1, b2 = verts_b[i], verts_b[(i + 1) % 6]
            if segments_intersect(a1, a2, b1, b2):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        """Exact max support function → required outer side length."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        a = math.radians(oangle)
        apothem_factor = SQRT3 / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0] - ocx) * nx + (v[1] - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S:
                    max_S = S_needed
        return max_S

    def compute_outer_hex_smooth(inner, ocx, ocy, oangle, tau=0.05):
        """Smooth log-sum-exp surrogate for the max support function."""
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        a = math.radians(oangle)
        apothem_factor = SQRT3 / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60 * i)
            nx, ny = math.cos(theta), math.sin(theta)
            projs = []
            for iv in inner_verts:
                for v in iv:
                    projs.append((v[0] - ocx) * nx + (v[1] - ocy) * ny)
            # log-sum-exp of projections
            max_p = max(projs)
            if max_p <= 0:
                continue
            lse = max_p + math.log(sum(math.exp(p - max_p) for p in projs))
            S_needed = lse / apothem_factor
            if S_needed > max_S:
                max_S = S_needed
        return max_S

    def all_hexagons_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            iv = transform_verts(ix, iy, 1.0, iangle)
            inner_verts.append(iv)
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def overlap_penalty(inner, ocx, ocy, S, oangle, weight=100.0):
        """Penalty for overlaps and containment violations (for smooth optimization)."""
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = []
        for (ix, iy, iangle) in inner:
            inner_verts.append(transform_verts(ix, iy, 1.0, iangle))
        penalty = 0.0
        # Containment check
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    penalty += weight
        # Pairwise disjoint check
        for i in range(len(inner_verts)):
            for j in range(i + 1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    penalty += weight
        return penalty

    def local_search(inner, ocx, ocy, oangle, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        rng = random.Random(seed)
        cur_inner = [list(x) for x in inner]
        cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
        cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)

        if not cur_valid:
            for _ in range(50):
                idx = rng.randrange(N)
                old_x, old_y, old_a = cur_inner[idx]
                cur_inner[idx][0] = old_x + rng.gauss(0, 0.5)
                cur_inner[idx][1] = old_y + rng.gauss(0, 0.5)
                cur_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
                cur_valid = all_hexagons_fit(cur_inner, ocx, ocy, cur_S, oangle)
                if cur_valid:
                    break
            if not cur_valid:
                return

        if cur_valid and cur_S < best_S:
            best_S = cur_S
            best_inner = [list(x) for x in cur_inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True

        for it in range(3000):
            if time.time() > deadline:
                break
            old = [list(x) for x in cur_inner]
            idx = rng.randrange(N)
            cur_inner[idx][0] += rng.gauss(0, 0.1)
            cur_inner[idx][1] += rng.gauss(0, 0.1)
            if rng.random() < 0.05:
                cur_inner[idx][2] = rng.choice([0, 30, 60, 90, 120, 150])

            new_S = compute_outer_hex(cur_inner, ocx, ocy, oangle)
            new_valid = all_hexagons_fit(cur_inner, ocx, ocy, new_S, oangle)

            if new_valid and new_S < cur_S:
                cur_S = new_S
                if new_S < best_S:
                    best_S = new_S
                    best_inner = [list(x) for x in cur_inner]
                    best_center = (ocx, ocy)
                    best_angle = oangle
                    best_valid = True
            elif new_valid:
                if rng.random() < 0.1:
                    cur_S = new_S
            else:
                cur_inner = old

    def try_greedy_lattice(spacing, angle_deg, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        max_coord = 12
        points = []
        a1 = (spacing, 0)
        a2 = (spacing / 2, spacing * SQRT3 / 2)
        for i in range(-max_coord, max_coord + 1):
            for j in range(-max_coord, max_coord + 1):
                x = i * a1[0] + j * a2[0]
                y = i * a1[1] + j * a2[1]
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        rad = math.radians(angle_deg)
        ca, sa = math.cos(rad), math.sin(rad)
        rotated_points = []
        for x, y in points:
            rx = x * ca - y * sa
            ry = x * sa + y * ca
            rotated_points.append((rx, ry))
        used_indices = set()
        placed_points = [rotated_points[0]]
        used_indices.add(0)
        for _ in range(N - 1):
            if time.time() > deadline:
                return
            cx = sum(p[0] for p in placed_points) / len(placed_points)
            cy = sum(p[1] for p in placed_points) / len(placed_points)
            best_idx = -1
            best_dist = float('inf')
            search_limit = min(300, len(rotated_points))
            for idx in range(search_limit):
                if idx in used_indices:
                    continue
                px, py = rotated_points[idx]
                d = (px - cx)**2 + (py - cy)**2
                if d < best_dist:
                    best_dist = d
                    best_idx = idx
            if best_idx == -1:
                for idx in range(len(rotated_points)):
                    if idx not in used_indices:
                        best_idx = idx
                        break
            if best_idx != -1:
                placed_points.append(rotated_points[best_idx])
                used_indices.add(best_idx)
        cx = sum(p[0] for p in placed_points) / N
        cy = sum(p[1] for p in placed_points) / N
        inner = []
        for x, y in placed_points:
            inner.append([x - cx, y - cy, 0.0])
        for oangle in [0, 30, 60]:
            S = compute_outer_hex(inner, 0.0, 0.0, oangle)
            if all_hexagons_fit(inner, 0.0, 0.0, S, oangle):
                if S < best_S:
                    best_S = S
                    best_inner = [list(x) for x in inner]
                    best_center = (0.0, 0.0)
                    best_angle = oangle
                    best_valid = True
                    local_search(inner, 0.0, 0.0, oangle, seed)

    def solve_contact_graph():
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        d = SQRT3
        base_positions = [(0.0, 0.0)]
        for i in range(6):
            theta = math.radians(60 * i)
            base_positions.append((d * math.cos(theta), d * math.sin(theta)))
        second_ring_angles = [30, 90, 150, 210]
        for angle in second_ring_angles:
            theta = math.radians(angle)
            base_positions.append((2 * d * math.cos(theta), 2 * d * math.sin(theta)))
        cx = sum(p[0] for p in base_positions) / N
        cy = sum(p[1] for p in base_positions) / N
        centered_positions = [(x - cx, y - cy) for x, y in base_positions]

        best_local_S = float('inf')
        best_local_angle = 0.0
        best_local_inner = None
        best_local_outer_angle = 0.0

        for cluster_angle in range(0, 360, 3):
            if time.time() > deadline:
                break
            rad = math.radians(cluster_angle)
            ca, sa = math.cos(rad), math.sin(rad)
            rotated_positions = []
            for x, y in centered_positions:
                rx = x * ca - y * sa
                ry = x * sa + y * ca
                rotated_positions.append((rx, ry))
            for outer_angle in [0, 30, 60]:
                inner = [[x, y, 0.0] for x, y in rotated_positions]
                S = compute_outer_hex(inner, 0.0, 0.0, outer_angle)
                if all_hexagons_fit(inner, 0.0, 0.0, S, outer_angle):
                    if S < best_local_S:
                        best_local_S = S
                        best_local_angle = cluster_angle
                        best_local_inner = [list(x) for x in inner]
                        best_local_outer_angle = outer_angle

        if best_local_S < best_S:
            best_S = best_local_S
            best_inner = best_local_inner
            best_center = (0.0, 0.0)
            best_angle = best_local_outer_angle
            best_valid = True
            local_search(best_inner, 0.0, 0.0, best_local_outer_angle, 12345)

    # NEW: High-precision polish using smooth surrogate + Nelder-Mead
    def polish_configuration(inner_init, oangle_init, seed):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if time.time() > deadline - 10:
            return

        # Flatten: 11 hexagons × (x, y, angle) = 33 variables
        x0 = []
        for (ix, iy, ia) in inner_init:
            x0.extend([ix, iy, ia])

        def objective(params):
            if time.time() > deadline - 5:
                return 1e6
            inner_p = []
            for k in range(N):
                ix, iy, ia = params[3*k], params[3*k+1], params[3*k+2]
                inner_p.append([ix, iy, ia])
            S = compute_outer_hex_smooth(inner_p, 0.0, 0.0, oangle_init, tau=0.03)
            pen = overlap_penalty(inner_p, 0.0, 0.0, S, oangle_init, weight=50.0)
            return S + pen

        # Run Nelder-Mead
        try:
            result = minimize(objective, x0, method='Nelder-Mead',
                             options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'adaptive': True})
        except Exception:
            return

        if time.time() > deadline - 5:
            return

        # Evaluate the polished result exactly
        polished_inner = []
        for k in range(N):
            polished_inner.append([result.x[3*k], result.x[3*k+1], result.x[3*k+2]])

        S_exact = compute_outer_hex(polished_inner, 0.0, 0.0, oangle_init)
        if all_hexagons_fit(polished_inner, 0.0, 0.0, S_exact, oangle_init):
            if S_exact < best_S:
                best_S = S_exact
                best_inner = [list(x) for x in polished_inner]
                best_center = (0.0, 0.0)
                best_angle = oangle_init
                best_valid = True

        # Also try with slightly different outer angles
        for oa in [oangle_init - 5, oangle_init + 5]:
            S_test = compute_outer_hex(polished_inner, 0.0, 0.0, oa)
            if all_hexagons_fit(polished_inner, 0.0, 0.0, S_test, oa):
                if S_test < best_S:
                    best_S = S_test
                    best_inner = [list(x) for x in polished_inner]
                    best_center = (0.0, 0.0)
                    best_angle = oa
                    best_valid = True

    # Run lattice search
    base_spacing = SQRT3
    spacings = [base_spacing * (1 + i * 0.001) for i in range(5)]
    angles = [0, 15, 30, 45, 60]

    seed_counter = 100
    for sp in spacings:
        for ang in angles:
            if time.time() > deadline:
                break
            try_greedy_lattice(sp, ang, seed_counter)
            seed_counter += 1

    # Run contact graph optimization
    solve_contact_graph()

    # Fallback if lattice search didn't find anything good
    if not best_valid:
        initial_configs = []
        sp = SQRT3
        hex_config = [[0.0, 0.0, 0.0]]
        for i in range(6):
            theta = math.radians(60 * i)
            hex_config.append([sp * math.cos(theta), sp * math.sin(theta), 0.0])
        for i in [0, 2, 4]:
            theta = math.radians(60 * i + 30)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        for i in [1, 3, 5]:
            theta = math.radians(60 * i)
            hex_config.append([2 * sp * math.cos(theta), 2 * sp * math.sin(theta), 0.0])
        while len(hex_config) < N:
            hex_config.append([random.uniform(-3, 3), random.uniform(-3, 3), 0.0])
        initial_configs.append((hex_config[:N], (0.0, 0.0), 0.0))

        for i, (cfg, c, a) in enumerate(initial_configs):
            if time.time() > deadline:
                break
            local_search(cfg, c[0], c[1], a, i * 1000 + 42)

    # FINAL POLISH: apply smooth optimization to the best configuration found
    if best_valid and best_inner is not None:
        # Polish with the current best outer angle
        polish_configuration(best_inner, best_angle, 999)

        # Try polishing with different outer angles from the same inner config
        for oa in [best_angle - 10, best_angle + 10, 0, 30, 60]:
            if time.time() > deadline - 15:
                break
            polish_configuration(best_inner, oa, 888 + oa)

    if not best_valid or best_inner is None:
        sp = SQRT3
        fallback_inner = []
        for k in range(N):
            fallback_inner.append([(k - N // 2) * sp, 0.0, 0.0])
        S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        for _ in range(100):
            if all_hexagons_fit(fallback_inner, 0.0, 0.0, S, 0.0):
                return fallback_inner, (0.0, 0.0), S, 0.0
            for k in range(N):
                fallback_inner[k][0] *= 1.05
            S = compute_outer_hex(fallback_inner, 0.0, 0.0, 0.0)
        return fallback_inner, (0.0, 0.0), S, 0.0

    return best_inner, best_center, best_S, best_angle
