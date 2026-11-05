"""Unit hexagons in regular hexagon - rigid-cluster + outer-rotation grid search."""
import time
import math
import numpy as np
import itertools

N = 11

# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s kill

    # Base vertices of unit hexagon (pointy-right orientation, rotation 0).
    base_vertices = np.array([
        [math.cos(math.radians(60 * k)), math.sin(math.radians(60 * k))] for k in range(6)
    ])

    # --- Geometry / validity helpers ---
    def hex_vertices(cx, cy, side, angle_deg):
        a = math.radians(angle_deg)
        return [(cx + side * math.cos(a + k * math.pi / 3.0),
                 cy + side * math.sin(a + k * math.pi / 3.0)) for k in range(6)]

    def point_in_convex(p, poly):
        n = len(poly)
        for i in range(n):
            x1, y1 = poly[i]
            x2, y2 = poly[(i + 1) % n]
            if (x2 - x1) * (p[1] - y1) - (y2 - y1) * (p[0] - x1) < -1e-12:
                return False
        return True

    def polygon_area_signed(verts):
        area = 0.0
        for i in range(len(verts)):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % len(verts)]
            area += x1 * y2 - x2 * y1
        return area / 2.0

    def line_intersect(p1, p2, p3, p4):
        x1, y1 = p1; x2, y2 = p2; x3, y3 = p3; x4, y4 = p4
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denom) < 1e-15:
            return p1
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        return (x1 + t * (x2 - x1), y1 + t * (y2 - y1))

    def polygon_overlap_area(v1, v2):
        def clip_polygon(subject, clip):
            output = list(subject)
            for i in range(len(clip)):
                if not output:
                    break
                inp = output
                output = []
                S = inp[-1]
                for E in inp:
                    if point_in_convex(E, clip):
                        if not point_in_convex(S, clip):
                            output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                        output.append(E)
                    elif point_in_convex(S, clip):
                        output.append(line_intersect(S, E, clip[i], clip[(i + 1) % len(clip)]))
                    S = E
            return output
        clipped = clip_polygon(v1, v2)
        if len(clipped) < 3:
            return 0.0
        return abs(polygon_area_signed(clipped))

    def is_valid(inner_params, outer_center, outer_side, outer_angle):
        cx, cy = outer_center
        outer_verts = hex_vertices(cx, cy, outer_side, outer_angle)
        # containment
        for (ix, iy, iang) in inner_params:
            if not all(point_in_convex(v, outer_verts) for v in hex_vertices(ix, iy, 1.0, iang)):
                return False
        # non-overlap
        verts_list = [hex_vertices(p[0], p[1], 1.0, p[2]) for p in inner_params]
        for i in range(N):
            for j in range(i + 1, N):
                if polygon_overlap_area(verts_list[i], verts_list[j]) > 1e-8:
                    return False
        return True

    # --- Rigid-cluster support function (all inner hexagons rotation 0) ---
    def compute_side_rigid(centers, theta_out_deg):
        theta_out = math.radians(theta_out_deg)
        normals = np.array([
            [math.cos(math.radians(30 + k * 60) + theta_out),
             math.sin(math.radians(30 + k * 60) + theta_out)] for k in range(6)
        ])
        max_proj = 0.0
        for c in centers:
            verts = base_vertices + c
            dots = verts @ normals.T
            max_proj = max(max_proj, float(np.max(np.abs(dots))))
        return 2.0 * max_proj / math.sqrt(3)

    # --- Candidate cluster generators (all inner hexagons rotation 0) ---
    def make_cluster(r1, r2, ring1_angles, ring2_angles, eps=1e-9):
        centers = [[0.0, 0.0]]
        for a in ring1_angles:
            rad = math.radians(a)
            centers.append([r1 * math.cos(rad), r1 * math.sin(rad)])
        for a in ring2_angles:
            rad = math.radians(a)
            centers.append([r2 * math.cos(rad), r2 * math.sin(rad)])
        return np.array(centers)

    # Enumerate cluster topologies.
    # Ring1: 6 hexagons at the 6 flat-side directions (30,90,...,330) or vertex (0,60,...).
    # Ring2: choose 4 of the 6 gap directions.
    ring1_sets = [[30 + 60 * k for k in range(6)], [60 * k for k in range(6)]]
    gap_sets = [[60 * k for k in range(6)], [30 + 60 * k for k in range(6)]]

    candidates = []
    for ring1 in ring1_sets:
        # ring2 gaps are the directions opposite to ring1 (i.e. the other set)
        ring2_full = gap_sets[0] if ring1 == ring1_sets[0] else gap_sets[1]
        for combo in itertools.combinations(ring2_full, 4):
            for r1 in (math.sqrt(3), 1.7320508075688772 + 1e-7):
                for r2 in (3.0, 2.9, 3.1, 3.2):
                    candidates.append(make_cluster(r1, r2, ring1, list(combo)))

    # Also keep the current program's analytic arrangement as a candidate (rotation 30 inner).
    # We normalize to rotation 0 by rotating the whole cluster by -30.
    def rotate_cluster(centers, ang_deg):
        rad = math.radians(ang_deg)
        c, s = math.cos(rad), math.sin(rad)
        out = []
        for (x, y) in centers:
            out.append([c * x - s * y, s * x + c * y])
        return np.array(out)

    seed_centers = [[0.0, 0.0]]
    for k in range(6):
        a = math.radians(k * 60)
        seed_centers.append([(math.sqrt(3) + 1e-9) * math.cos(a), (math.sqrt(3) + 1e-9) * math.sin(a)])
    for k in range(4):
        a = math.radians(30 + k * 60)
        seed_centers.append([(3.0 + 2e-9) * math.cos(a), (3.0 + 2e-9) * math.sin(a)])
    candidates.append(rotate_cluster(seed_centers, -30.0))

    # --- Grid search over outer rotation for each candidate ---
    best_score = float('inf')
    best_centers = None
    best_theta = 0.0

    # Precompute a coarse grid once.
    coarse = np.linspace(0.0, 60.0, 1200)

    for centers in candidates:
        if time.time() > deadline - 10:
            break
        # Coarse pass
        local_best = float('inf')
        local_theta = 0.0
        for t in coarse:
            if time.time() > deadline - 10:
                break
            s = compute_side_rigid(centers, t)
            if s < local_best:
                local_best = s
                local_theta = t
        if local_best >= best_score:
            continue
        # Local refinement around best
        step = 60.0 / 1200.0
        lo = max(0.0, local_theta - 8 * step)
        hi = min(60.0, local_theta + 8 * step)
        fine = np.linspace(lo, hi, 2000)
        for t in fine:
            s = compute_side_rigid(centers, t)
            if s < local_best:
                local_best = s
                local_theta = t
        # Second finer pass
        if local_best < best_score:
            step2 = (hi - lo) / 2000.0
            lo2 = max(0.0, local_theta - 6 * step2)
            hi2 = min(60.0, local_theta + 6 * step2)
            finer = np.linspace(lo2, hi2, 2000)
            for t in finer:
                s = compute_side_rigid(centers, t)
                if s < local_best:
                    local_best = s
                    local_theta = t
        if local_best < best_score:
            best_score = local_best
            best_centers = centers
            best_theta = local_theta

    # Fallback: guarantee a valid configuration even if search found nothing.
    if best_centers is None or not np.isfinite(best_score):
        centers = make_cluster(math.sqrt(3) + 1e-9, 3.0 + 2e-9,
                               [30 + 60 * k for k in range(6)], [0, 60, 120, 180])
        best_centers = centers
        best_theta = 0.0
        best_score = compute_side_rigid(centers, 0.0)

    # Build inner list (all inner hexagons rotation 0).
    inner = [[float(centers[i][0]), float(centers[i][1]), 0.0] for i in range(len(best_centers))]

    outer_side = best_score
    outer_angle = float(best_theta)

    # Final validity check; if invalid, fall back to the analytic seed (known valid).
    if is_valid(inner, (0.0, 0.0), outer_side, outer_angle):
        return inner, (0.0, 0.0), outer_side, outer_angle

    # Fallback to the analytic solution (rotation 30 inner, known to be valid).
    eps = 1e-9
    seed_inner = [(0.0, 0.0, 30.0)]
    for k in range(6):
        a = math.radians(k * 60)
        seed_inner.append(((math.sqrt(3) + eps) * math.cos(a),
                           (math.sqrt(3) + eps) * math.sin(a), 30.0))
    for k in range(4):
        a = math.radians(30 + k * 60)
        seed_inner.append(((3.0 + 2 * eps) * math.cos(a),
                           (3.0 + 2 * eps) * math.sin(a), 30.0))
    max_x = (2 + eps) * math.sqrt(3)
    max_y = 4 + 2 * eps
    side = max(2 * max_x / math.sqrt(3), max_y) + 1e-9
    inner_out = [[float(x), float(y), float(a)] for (x, y, a) in seed_inner]
    return inner_out, (0.0, 0.0), side, 30.0
# EVOLVE-BLOCK-END
