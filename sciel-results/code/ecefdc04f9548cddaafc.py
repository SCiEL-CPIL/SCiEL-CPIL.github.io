"""Rewritten program for packing 12 unit hexagons in a regular hexagon.
Uses caging formulation: minimize outer hexagon side by treating the
convex hull of all inner hexagon vertices as a single body.
Optimization: simulated annealing with smooth distance penalties.
"""
import numpy as np
import time
import math

N = 12
SQRT3 = math.sqrt(3.0)


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin before 150s kill

    def hexagon_vertices(cx, cy, side, angle_deg):
        """Return 6 vertices of a regular hexagon."""
        angle_rad = math.radians(angle_deg)
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        return np.column_stack([
            cx + side * np.cos(angles),
            cy + side * np.sin(angles)
        ])

    def point_in_hexagon(px, py, cx, cy, side, angle_deg):
        """Check if point (px, py) is inside hexagon centered at (cx, cy)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            if dist > apothem + 1e-9:
                return False
        return True

    def distance_to_hexagon_edge(px, py, cx, cy, side, angle_deg):
        """Return signed distance from point to hexagon boundary (positive inside)."""
        dx = px - cx
        dy = py - cy
        angle_rad = math.radians(angle_deg)
        cos_a = math.cos(-angle_rad)
        sin_a = math.sin(-angle_rad)
        local_x = dx * cos_a - dy * sin_a
        local_y = dx * sin_a + dy * cos_a
        apothem = side * SQRT3 / 2
        normal_angles = np.radians(np.arange(30, 360, 60))
        min_dist = float('inf')
        for na in normal_angles:
            nx = math.cos(na)
            ny = math.sin(na)
            dist = local_x * nx + local_y * ny
            min_dist = min(min_dist, apothem - dist)
        return min_dist

    def polygons_disjoint_sat(v1, v2):
        """Check if two convex polygons are disjoint using SAT."""
        axes = []
        for i in range(6):
            p1 = v1[i]
            p2 = v1[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for i in range(6):
            p1 = v2[i]
            p2 = v2[(i + 1) % 6]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm > 1e-9:
                axes.append(normal / norm)
        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            if np.max(proj1) < np.min(proj2) - 1e-9 or np.max(proj2) < np.min(proj1) - 1e-9:
                return True
        return False

    def all_inside_outer(inner_positions, outer_center, outer_side, outer_angle):
        """Check all inner hexagons are inside the outer hexagon."""
        for i in range(N):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                if not point_in_hexagon(vx, vy, outer_center[0], outer_center[1], outer_side, outer_angle):
                    return False
        return True

    def pairwise_disjoint(inner_positions):
        """Check all pairs of inner hexagons are disjoint."""
        for i in range(N):
            for j in range(i + 1, N):
                v1 = hexagon_vertices(inner_positions[i][0], inner_positions[i][1], 1.0, inner_positions[i][2])
                v2 = hexagon_vertices(inner_positions[j][0], inner_positions[j][1], 1.0, inner_positions[j][2])
                if not polygons_disjoint_sat(v1, v2):
                    return False
        return True

    def is_valid(inner, oc, os, oa):
        """Full validity check."""
        return all_inside_outer(inner, oc, os, oa) and pairwise_disjoint(inner)

    def min_outer_side_for_config(inner_positions, outer_center, outer_angle):
        """Find minimum outer side length that contains all inner hexagons."""
        lo = 1.0
        hi = 20.0
        for _ in range(50):
            mid = (lo + hi) / 2
            if all_inside_outer(inner_positions, outer_center, mid, outer_angle):
                hi = mid
            else:
                lo = mid
        return hi

    def make_params(inner, oc, os, oa):
        """Convert to flat parameter array."""
        params = []
        for pos in inner:
            params.extend([pos[0], pos[1], pos[2]])
        params.extend([oc[0], oc[1], os, oa])
        return np.array(params)

    def parse_params(params):
        """Convert flat parameter array back to structured form."""
        inner = []
        for i in range(N):
            inner.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))
        oc = (params[3 * N], params[3 * N + 1])
        os = params[3 * N + 2]
        oa = params[3 * N + 3]
        return inner, oc, os, oa

    def smooth_penalty(params):
        """
        Smooth objective: minimize outer side + penalties for violations.
        Uses distance-based penalties for both containment and separation.
        """
        n_inner = N
        inner_positions = []
        for i in range(n_inner):
            inner_positions.append((params[3 * i], params[3 * i + 1], params[3 * i + 2]))

        ox = params[3 * n_inner]
        oy = params[3 * n_inner + 1]
        os = max(params[3 * n_inner + 2], 0.5)
        oa = params[3 * n_inner + 3]

        outer_center = (ox, oy)
        outer_side = os
        outer_angle = oa

        penalty = 0.0

        # Containment penalty: each inner vertex should be inside outer
        for i in range(n_inner):
            cx, cy, angle = inner_positions[i]
            verts = hexagon_vertices(cx, cy, 1.0, angle)
            for vx, vy in verts:
                d = distance_to_hexagon_edge(vx, vy, ox, oy, outer_side, outer_angle)
                if d < 0:
                    penalty += d * d  # quadratic penalty for violation

        # Separation penalty: use SAT penetration depth approximation
        for i in range(n_inner):
            for j in range(i + 1, n_inner):
                ci = inner_positions[i]
                cj = inner_positions[j]
                v1 = hexagon_vertices(ci[0], ci[1], 1.0, ci[2])
                v2 = hexagon_vertices(cj[0], cj[1], 1.0, cj[2])

                # Quick distance check first
                cx1, cy1 = ci[0], ci[1]
                cx2, cy2 = cj[0], cj[1]
                dist_centers = math.sqrt((cx1 - cx2) ** 2 + (cy1 - cy2) ** 2)
                if dist_centers > 4.0:
                    continue  # definitely disjoint

                # SAT check for penetration
                axes = []
                for k in range(6):
                    p1 = v1[k]
                    p2 = v1[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)
                for k in range(6):
                    p1 = v2[k]
                    p2 = v2[(k + 1) % 6]
                    edge = p2 - p1
                    normal = np.array([-edge[1], edge[0]])
                    norm = np.linalg.norm(normal)
                    if norm > 1e-9:
                        axes.append(normal / norm)

                min_overlap = float('inf')
                for axis in axes:
                    proj1 = np.dot(v1, axis)
                    proj2 = np.dot(v2, axis)
                    min1, max1 = np.min(proj1), np.max(proj1)
                    min2, max2 = np.min(proj2), np.max(proj2)
                    if max1 < min2 - 1e-9 or max2 < min1 - 1e-9:
                        min_overlap = 0.0
                        break
                    overlap = min(max1, max2) - max(min1, min2)
                    if overlap < 0:
                        min_overlap = 0.0
                        break
                    min_overlap = min(min_overlap, overlap)

                if min_overlap > 0:
                    penalty += min_overlap * min_overlap

        return outer_side + penalty * 100.0

    def get_initial_configurations():
        """Generate diverse initial configurations."""
        configs = []
        sqrt3 = SQRT3

        # Arrangement A: 4x3 hexagonal grid
        for outer_angle in [0.0, 30.0, 15.0]:
            positions = []
            inner_angle = outer_angle
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 3.0
                    y = (row - 1) * 2.598
                    if row % 2 == 1:
                        x += 1.5
                    positions.append([x, y, inner_angle])
            configs.append((positions, outer_angle))

        # Arrangement B: Central + 6 ring + 5 outer
        for outer_angle in [0.0, 30.0]:
            positions = [[0.0, 0.0, outer_angle]]
            for k in range(6):
                angle = k * math.pi / 3
                x = 2.0 * math.cos(angle)
                y = 2.0 * math.sin(angle)
                positions.append([x, y, outer_angle])
            for k in range(5):
                angle = k * math.pi / 3 + math.pi / 6
                x = 3.0 * math.cos(angle)
                y = 3.0 * math.sin(angle)
                positions.append([x, y, outer_angle])
            configs.append((positions[:12], outer_angle))

        # Arrangement C: Two staggered rows of 6
        for outer_angle in [0.0, 30.0]:
            positions = []
            for i in range(6):
                positions.append([(i - 2.5) * 2.1, -1.4, outer_angle])
                positions.append([(i - 2.5) * 2.1 + 1.05, 1.4, outer_angle])
            configs.append((positions, outer_angle))

        # Arrangement D: 3x4 with tighter spacing
        for outer_angle in [0.0, 30.0]:
            positions = []
            for row in range(3):
                for col in range(4):
                    x = (col - 1.5) * 2.15
                    y = (row - 1) * 2.65
                    if row % 2 == 1:
                        x += 1.075
                    positions.append([x, y, outer_angle])
            configs.append((positions, outer_angle))

        # Arrangement E: Hexagonal close packing (7 in center + 5 around)
        for outer_angle in [0.0, 30.0]:
            positions = [[0.0, 0.0, outer_angle]]
            for k in range(6):
                angle = k * math.pi / 3
                x = 2.0 * math.cos(angle)
                y = 2.0 * math.sin(angle)
                positions.append([x, y, outer_angle])
            # 5 more in second ring
            for k in range(5):
                angle = k * math.pi / 3 + math.pi / 6
                x = 3.464 * math.cos(angle)
                y = 3.464 * math.sin(angle)
                positions.append([x, y, outer_angle])
            configs.append((positions[:12], outer_angle))

        return configs

    def simulated_annealing(params, max_steps=5000, t0=2.0, t_end=0.01):
        """Run simulated annealing on the parameter vector."""
        current = params.copy()
        current_score = smooth_penalty(current)
        best = current.copy()
        best_score = current_score

        n_params = len(current)
        rng = np.random.default_rng(42)

        for step in range(max_steps):
            if time.time() > deadline - 10:
                break

            # Cooling schedule
            alpha = step / max_steps
            temperature = t0 * (t_end / t0) ** alpha

            # Generate neighbor
            neighbor = current.copy()
            # Random perturbation
            noise_scale = 0.5 * temperature
            mask = rng.random(n_params) < 0.3  # perturb 30% of params
            neighbor[mask] += rng.normal(0, noise_scale, mask.sum())

            # Clamp outer side to reasonable range
            neighbor[3 * N + 2] = max(neighbor[3 * N + 2], 0.5)

            neighbor_score = smooth_penalty(neighbor)

            # Accept or reject
            if neighbor_score < current_score:
                current = neighbor
                current_score = neighbor_score
                if current_score < best_score:
                    best = current.copy()
                    best_score = current_score
            else:
                delta = neighbor_score - current_score
                if temperature > 1e-6:
                    accept_prob = math.exp(-delta / temperature)
                    if rng.random() < accept_prob:
                        current = neighbor
                        current_score = neighbor_score

        return best, best_score

    def refine_with_nelder_mead(params):
        """Refine using Nelder-Mead on the smooth objective."""
        from scipy.optimize import minimize
        try:
            res = minimize(smooth_penalty, params, method='Nelder-Mead',
                           options={'maxiter': 500, 'xatol': 0.005, 'fatol': 0.005, 'adaptive': True})
            return res.x, res.fun
        except Exception:
            return params, smooth_penalty(params)

    # Main optimization loop
    best_params = None
    best_score = float('inf')

    configs = get_initial_configurations()

    # Sort by initial tightness
    def config_score(ct):
        positions, outer_angle = ct
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]
        if not pairwise_disjoint(inner_tuples):
            return 100.0
        os = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
        return os

    scored_configs = []
    for idx, ct in enumerate(configs):
        if time.time() > deadline - 40:
            break
        s = config_score(ct)
        scored_configs.append((s, idx, ct))
    scored_configs.sort(key=lambda x: x[0])

    for score, idx, config_tuple in scored_configs:
        if time.time() > deadline - 20:
            break
        if score >= best_score:
            break

        positions, outer_angle = config_tuple
        inner_tuples = [(p[0], p[1], p[2]) for p in positions]

        # Ensure disjoint; if not, try perturbations
        if not pairwise_disjoint(inner_tuples):
            rng = np.random.default_rng(idx * 137)
            found = False
            for attempt in range(30):
                noise = rng.uniform(-0.4, 0.4, size=2 * N)
                perturbed = [[p[0] + noise[2 * i], p[1] + noise[2 * i + 1], p[2]]
                             for i, p in enumerate(positions)]
                inner_tuples_p = [(p[0], p[1], p[2]) for p in perturbed]
                if pairwise_disjoint(inner_tuples_p):
                    inner_tuples = inner_tuples_p
                    positions = perturbed
                    found = True
                    break
            if not found:
                continue

        # Find tight outer side
        outer_side = min_outer_side_for_config(inner_tuples, [0.0, 0.0], outer_angle)
        outer_center = [0.0, 0.0]

        params = make_params(positions, outer_center, outer_side, outer_angle)

        # Simulated annealing
        if time.time() < deadline - 15:
            sa_params, sa_score = simulated_annealing(params, max_steps=3000)
            if sa_score < best_score:
                best_score = sa_score
                best_params = sa_params.copy()

            # Refine with Nelder-Mead
            if time.time() < deadline - 10:
                nm_params, nm_score = refine_with_nelder_mead(best_params)
                if nm_score < best_score:
                    best_score = nm_score
                    best_params = nm_params.copy()

    # Fallback
    if best_params is None:
        positions_a = []
        for row in range(3):
            for col in range(4):
                x = (col - 1.5) * 3.0
                y = (row - 1) * 2.598
                if row % 2 == 1:
                    x += 1.5
                positions_a.append([x, y, 0.0])
        inner_tuples = [(p[0], p[1], p[2]) for p in positions_a]
        outer_center = [0.0, 0.0]
        outer_angle = 30.0
        outer_side = 8.0
        while not all_inside_outer(inner_tuples, outer_center, outer_side, outer_angle):
            outer_side += 0.5
        best_params = make_params(positions_a, outer_center, outer_side, outer_angle)
        best_score = outer_side

    # Final validation
    inner, oc, os, oa = parse_params(best_params)

    if not is_valid(inner, oc, os, oa):
        # Expand outer hexagon until valid
        while not all_inside_outer(inner, oc, os, oa):
            os += 0.1
            if os > 25:
                break
        # If still overlapping, try to fix
        if not pairwise_disjoint(inner):
            # Last resort: spread out
            for i in range(N):
                inner[i] = (inner[i][0] * 1.2, inner[i][1] * 1.2, inner[i][2])
            os += 1.0
            while not all_inside_outer(inner, oc, os, oa):
                os += 0.1
                if os > 30:
                    break

    return inner, oc, os, oa
