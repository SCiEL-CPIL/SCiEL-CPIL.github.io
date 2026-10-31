import math
import time
import random
import numpy as np

N = 11

def construct():
    deadline = time.time() + 145.0
    best_S = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_angle = 0.0
    best_valid = False

    # Unit hexagon vertices (flat-top)
    base_verts = [(math.cos(math.radians(60*i)), math.sin(math.radians(60*i))) for i in range(6)]

    def transform_verts(cx, cy, S, angle_deg):
        a = math.radians(angle_deg)
        ca, sa = math.cos(a), math.sin(a)
        return [(cx + S*(vx*ca - vy*sa), cy + S*(vx*sa + vy*ca)) for vx, vy in base_verts]

    def cross(o, a, b):
        return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])

    def segments_intersect(p1, p2, p3, p4):
        d1 = cross(p3, p4, p1); d2 = cross(p3, p4, p2)
        d3 = cross(p1, p2, p3); d4 = cross(p1, p2, p4)
        if ((d1>0 and d2<0) or (d1<0 and d2>0)) and ((d3>0 and d4<0) or (d3<0 and d4>0)):
            return True
        return False

    def point_in_polygon(p, poly):
        x, y = p
        inside = False
        j = len(poly) - 1
        for i in range(len(poly)):
            xi, yi = poly[i]; xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj-xi)*(y-yi)/(yj-yi) + xi):
                inside = not inside
            j = i
        return inside

    def hexagons_disjoint(va, vb):
        # Fast bounding box check
        min_xa = min(v[0] for v in va); max_xa = max(v[0] for v in va)
        min_ya = min(v[1] for v in va); max_ya = max(v[1] for v in va)
        min_xb = min(v[0] for v in vb); max_xb = max(v[0] for v in vb)
        min_yb = min(v[1] for v in vb); max_yb = max(v[1] for v in vb)
        if max_xa < min_xb or max_xb < min_xa or max_ya < min_yb or max_yb < min_ya:
            return True
        for v in va:
            if point_in_polygon(v, vb): return False
        for v in vb:
            if point_in_polygon(v, va): return False
        for i in range(6):
            if segments_intersect(va[i], va[(i+1)%6], vb[i], vb[(i+1)%6]):
                return False
        return True

    def compute_outer_hex(inner, ocx, ocy, oangle):
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        a = math.radians(oangle)
        apothem_factor = math.sqrt(3) / 2
        max_S = 0.0
        for i in range(6):
            theta = a + math.radians(30 + 60*i)
            nx, ny = math.cos(theta), math.sin(theta)
            max_proj = 0.0
            for iv in inner_verts:
                for v in iv:
                    proj = (v[0]-ocx)*nx + (v[1]-ocy)*ny
                    if proj > max_proj: max_proj = proj
            if max_proj > 0:
                S_needed = max_proj / apothem_factor
                if S_needed > max_S: max_S = S_needed
        return max_S

    def all_fit(inner, ocx, ocy, S, oangle):
        outer_verts = transform_verts(ocx, ocy, S, oangle)
        inner_verts = [transform_verts(ix, iy, 1.0, ia) for ix, iy, ia in inner]
        for iv in inner_verts:
            for v in iv:
                if not point_in_polygon(v, outer_verts):
                    return False
        for i in range(len(inner_verts)):
            for j in range(i+1, len(inner_verts)):
                if not hexagons_disjoint(inner_verts[i], inner_verts[j]):
                    return False
        return True

    def update_best(inner, ocx, ocy, S, oangle):
        nonlocal best_S, best_inner, best_center, best_angle, best_valid
        if S < best_S and all_fit(inner, ocx, ocy, S, oangle):
            best_S = S
            best_inner = [list(x) for x in inner]
            best_center = (ocx, ocy)
            best_angle = oangle
            best_valid = True
            return True
        return False

    # ---- Strategy: Discrete triangular lattice search with simulated annealing ----
    # The key insight: optimal packings of regular hexagons tend to align with
    # triangular lattice structures. We search over discrete lattice positions
    # and use simulated annealing to find the tightest cluster.

    # Generate candidate positions on a triangular lattice
    # Lattice spacing must be at least sqrt(3) for unit hexagons to be disjoint
    lattice_spacing = math.sqrt(3) * 1.001
    a1 = (lattice_spacing, 0)
    a2 = (lattice_spacing/2, lattice_spacing*math.sqrt(3)/2)

    # Generate a grid of candidate positions
    max_coord = 8
    candidates = []
    for i in range(-max_coord, max_coord+1):
        for j in range(-max_coord, max_coord+1):
            x = i*a1[0] + j*a2[0]
            y = i*a1[1] + j*a2[1]
            candidates.append((x, y))

    # Sort by distance from origin to prefer compact clusters
    candidates.sort(key=lambda p: p[0]**2 + p[1]**2)

    # Take the most central candidates
    n_candidates = min(200, len(candidates))
    candidate_positions = candidates[:n_candidates]

    # Simulated annealing to find the tightest arrangement
    # State: assignment of N hexagons to N positions from candidate_positions
    # Objective: outer hexagon side length (with validity penalty)

    rng = random.Random(42)

    def evaluate_assignment(positions):
        """Evaluate an assignment of positions to hexagons."""
        if time.time() > deadline:
            return 1e10

        inner = []
        for pos in positions:
            inner.append([pos[0], pos[1], 0.0])

        # Center at origin
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]

        S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)

        if not all_fit(inner_c, 0.0, 0.0, S, 0.0):
            return S + 50.0

        return S

    # Initialize with the N most central candidates
    current_positions = candidate_positions[:N]
    current_score = evaluate_assignment(current_positions)

    if current_score < 1e9:
        inner = [[pos[0], pos[1], 0.0] for pos in current_positions]
        cx = sum(p[0] for p in inner) / N
        cy = sum(p[1] for p in inner) / N
        inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
        S = compute_outer_hex(inner_c, 0.0, 0.0, S, 0.0)
        if all_fit(inner_c, 0.0, 0.0, S, 0.0):
            best_S = S
            best_inner = [list(x) for x in inner_c]
            best_center = (0.0, 0.0)
            best_angle = 0.0
            best_valid = True

    # Simulated annealing
    T_start = 1.0
    T_end = 0.01
    n_iterations = 5000

    for it in range(n_iterations):
        if time.time() > deadline - 5.0:
            break

        T = T_start * (T_end / T_start) ** (it / n_iterations)

        # Pick a random hexagon to move
        idx = rng.randrange(N)
        old_pos = current_positions[idx]

        # Move to a random candidate position
        new_pos = candidate_positions[rng.randrange(n_candidates)]
        current_positions[idx] = new_pos

        new_score = evaluate_assignment(current_positions)

        # Accept if better, or with probability exp(-(new-old)/T)
        if new_score < current_score or rng.random() < math.exp(-(new_score - current_score) / max(T, 1e-10)):
            current_score = new_score
            if new_score < best_S:
                inner = [[pos[0], pos[1], 0.0] for pos in current_positions]
                cx = sum(p[0] for p in inner) / N
                cy = sum(p[1] for p in inner) / N
                inner_c = [[p[0]-cx, p[1]-cy, p[2]] for p in inner]
                S = compute_outer_hex(inner_c, 0.0, 0.0, 0.0)
                if all_fit(inner_c, 0.0, 0.0, S, 0.0):
                    best_S = S
                    best_inner = [list(x) for x in inner_c]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
                    best_valid = True
        else:
            current_positions[idx] = old_pos

    # ---- Local refinement: try small perturbations around the best found ----
    if best_valid and best_inner is not None:
        for refine_iter in range(200):
            if time.time() > deadline - 2.0:
                break

            cur = [list(x) for x in best_inner]
            improved = False

            for _ in range(100):
                if time.time() > deadline - 2.0:
                    break

                idx = rng.randrange(N)
                old = list(cur[idx])
                cur[idx][0] += rng.gauss(0, 0.05)
                cur[idx][1] += rng.gauss(0, 0.05)

                S_new = compute_outer_hex(cur, 0.0, 0.0, 0.0)
                if S_new < best_S and all_fit(cur, 0.0, 0.0, S_new, 0.0):
                    best_S = S_new
                    best_inner = [list(x) for x in cur]
                    best_center = (0.0, 0.0)
                    best_angle = 0.0
                    best_valid = True
                    improved = True
                else:
                    cur[idx] = old

            if not improved:
                break

    # ---- Fallback: simple lattice blob ----
    if not best_valid or best_inner is None:
        sp = math.sqrt(3) * 1.001
        a1 = (sp, 0)
        a2 = (sp/2, sp*math.sqrt(3)/2)
        points = []
        for i in range(-8, 9):
            for j in range(-8, 9):
                x = i*a1[0] + j*a2[0]
                y = i*a1[1] + j*a2[1]
                points.append((x, y))
        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = points[:N]
        cx = sum(p[0] for p in selected)/N
        cy = sum(p[1] for p in selected)/N
        inner = [[p[0]-cx, p[1]-cy, 0.0] for p in selected]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        for _ in range(200):
            inner = [[p[0]*1.02, p[1]*1.02, p[2]] for p in inner]
            S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
            if all_fit(inner, 0.0, 0.0, S, 0.0):
                return inner, (0.0, 0.0), S, 0.0

    if best_valid and best_inner is not None:
        return best_inner, best_center, best_S, best_angle

    # Absolute fallback
    sp = math.sqrt(3)
    inner = [[(k - N//2)*sp, 0.0, 0.0] for k in range(N)]
    S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    for _ in range(200):
        if all_fit(inner, 0.0, 0.0, S, 0.0):
            return inner, (0.0, 0.0), S, 0.0
        inner = [[p[0]*1.05, p[1]*1.05, p[2]] for p in inner]
        S = compute_outer_hex(inner, 0.0, 0.0, 0.0)
    return inner, (0.0, 0.0), S, 0.0
