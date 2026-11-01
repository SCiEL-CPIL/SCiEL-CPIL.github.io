"""Packing unit hexagons inside a regular hexagon - DE + polish approach."""
import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = SQRT3 / 2.0
DEADLINE = 140.0  # safety margin before 150s kill


def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon."""
    verts = []
    for i in range(6):
        theta = angle_rad + i * math.pi / 3.0
        vx = cx + side * math.cos(theta)
        vy = cy + side * math.sin(theta)
        verts.append((vx, vy))
    return verts


def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    return 2.0 * max_projection / SQRT3


def compute_required_side_smooth(inner, outer_center, outer_angle_deg, temperature=0.01):
    """
    Smooth approximation using log-sum-exp over all support function values.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    # Collect all 12 support values (6 normals × 2 directions)
    support_values = []
    for nx, ny in normals:
        max_proj = -float('inf')
        min_proj = float('inf')
        for (ix, iy, iangle) in inner:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a
                proj = lx * nx + ly * ny
                if proj > max_proj:
                    max_proj = proj
                if proj < min_proj:
                    min_proj = proj
        support_values.append(max_proj)
        support_values.append(-min_proj)

    if not support_values:
        return 0.0

    sv = np.array(support_values, dtype=np.float64)
    amax = np.max(sv)
    if not np.isfinite(amax):
        return float('inf')
    smooth_max = amax + temperature * math.log(np.sum(np.exp((sv - amax) / temperature)))
    return 2.0 * smooth_max / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))
    for i in range(len(v2)):
        p1 = v2[i]
        p2 = v2[(i + 1) % len(v2)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True


def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    for i in range(n):
        for j in range(i + 1, n):
            vi = get_hex_vertices(inner[i][0], inner[i][1], 1.0, math.radians(inner[i][2]))
            vj = get_hex_vertices(inner[j][0], inner[j][1], 1.0, math.radians(inner[j][2]))
            if sat_overlap_check(vi, vj, eps):
                return False
    return True


def check_inside_outer(inner, outer_center, outer_side, outer_angle_deg, eps=1e-6):
    """Check all inner hexagons are inside the outer hexagon."""
    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    apothem = outer_side * SQRT3 / 2.0

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for i in range(6):
                theta = math.radians(30 + i * 60)
                nx = math.cos(theta)
                ny = math.sin(theta)
                if lx * nx + ly * ny > apothem + eps:
                    return False
    return True


def objective(params, temperature=0.01):
    """
    Joint objective: outer side + overlap penalty.
    params: [x1, y1, a1, ..., xN, yN, aN, cx, cy, cangle]
    """
    inner_list = []
    for i in range(N):
        inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])

    cx = params[N * 3]
    cy = params[N * 3 + 1]
    angle = params[N * 3 + 2]

    side = compute_required_side_smooth(inner_list, (cx, cy), angle, temperature)

    # Overlap penalty using center distance (fast proxy)
    penalty = 0.0
    min_dist = SQRT3 - 0.005
    for i in range(N):
        for j in range(i + 1, N):
            xi, yi, _ = inner_list[i]
            xj, yj, _ = inner_list[j]
            dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            if dist < min_dist:
                penalty += (min_dist - dist) ** 2

    return side + 50.0 * penalty


def objective_exact(params, temperature=0.005):
    """Objective with exact SAT check for overlaps."""
    inner_list = []
    for i in range(N):
        inner_list.append([params[i * 3], params[i * 3 + 1], params[i * 3 + 2]])

    cx = params[N * 3]
    cy = params[N * 3 + 1]
    angle = params[N * 3 + 2]

    side = compute_required_side(inner_list, (cx, cy), angle)

    if not check_all_disjoint(inner_list):
        return side + 10.0

    return side


def axial_to_cartesian(axial_list, scale=1.0 + 1e-5):
    """Convert axial coordinates to cartesian."""
    pts = []
    for q, r in axial_list:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        pts.append((x, y))
    return pts


def generate_initial_configs():
    """Generate diverse initial configurations."""
    configs = []

    # Classic hexagonal cluster (7 + 4 outer ring)
    hex7_4 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2),
    ]
    configs.append(axial_to_cartesian(hex7_4))

    # Alternative ring
    alt_ring = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (1, 2),
    ]
    configs.append(axial_to_cartesian(alt_ring))

    # Rectangular 3x3 + 2
    rect = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1),
    ]
    configs.append(axial_to_cartesian(rect))

    # Row-based
    rows = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ]
    configs.append(axial_to_cartesian(rows))

    # Star-like
    star = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 1), (1, 2), (-1, 2), (0, 3),
    ]
    configs.append(axial_to_cartesian(star))

    # Compact 3x3 + corners
    compact = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, -1), (2, 3),
    ]
    configs.append(axial_to_cartesian(compact))

    # Dense cluster
    dense = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (2, 3),
    ]
    configs.append(axial_to_cartesian(dense))

    # Diamond
    diamond = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1),
        (0, -1), (1, -1), (-1, -1),
        (2, 0), (0, 2),
    ]
    configs.append(axial_to_cartesian(diamond))

    # Elongated
    elong = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ]
    configs.append(axial_to_cartesian(elong))

    # T-shape
    tshape = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (2, 1),
        (1, 1), (3, 1),
        (1, 2), (2, 2), (3, 2),
    ]
    configs.append(axial_to_cartesian(tshape))

    # L-shape
    lshape = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (0, 3), (0, 4),
    ]
    configs.append(axial_to_cartesian(lshape))

    # Plus shape
    plus = [
        (0, 0),
        (1, 0), (-1, 0),
        (0, 1), (0, -1),
        (2, 0), (-2, 0),
        (0, 2), (0, -2),
        (1, 1), (-1, -1),
        (1, -1), (-1, 1),
    ]
    configs.append(axial_to_cartesian(plus))

    return configs


def polish_solution(params, time_limit=5.0):
    """Polish a solution using Nelder-Mead on exact objective."""
    start = time.time()
    try:
        res = minimize(
            lambda p: objective_exact(p),
            params,
            method='Nelder-Mead',
            options={'maxiter': 3000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000}
        )
        if time.time() - start > time_limit:
            return params
        return res.x
    except Exception:
        return params


def polish_outer_only(inner_list, outer_center, outer_angle, time_limit=3.0):
    """Polish only the outer hexagon parameters."""
    start = time.time()
    try:
        res = minimize(
            lambda p: compute_required_side(inner_list, (p[0], p[1]), p[2]),
            [outer_center[0], outer_center[1], outer_angle],
            method='Nelder-Mead',
            options={'maxiter': 5000, 'xatol': 1e-13, 'fatol': 1e-15, 'maxfev': 10000}
        )
        if time.time() - start > time_limit:
            return outer_center, outer_angle
        return (res.x[0], res.x[1]), res.x[2]
    except Exception:
        return outer_center, outer_angle


def construct():
    start_time = time.time()

    best_side = float('inf')
    best_inner = None
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    initial_configs = generate_initial_configs()

    # Phase 1: Differential Evolution for global search
    # Use a subset of the best initial configs as starting points
    np.random.seed(42)

    # Define bounds for DE
    # Inner hexagons: positions in [-5, 5], angles in [-180, 180]
    # Outer: center in [-5, 5], angle in [-180, 180]
    bounds = []
    for i in range(N):
        bounds.extend([(-5, 5), (-5, 5), (-180, 180)])
    bounds.extend([(-5, 5), (-5, 5), (-180, 180)])

    # Run DE with a reasonable number of iterations
    de_time_budget = 60.0  # seconds for DE
    de_start = time.time()

    try:
        de_result = differential_evolution(
            lambda p: objective(p, temperature=0.02),
            bounds=bounds,
            seed=42,
            maxiter=300,
            popsize=20,
            tol=1e-8,
            mutation=(0.5, 1.0),
            recombination=0.7,
            polish=False,
            updating='deferred',
            callback=lambda xk, convergence: (
                True if time.time() - de_start > de_time_budget else False
            )
        )

        if de_result.success or de_result.fun < best_side:
            params = de_result.x
            inner_list = [[params[i * 3], params[i * 3 + 1], params[i * 3 + 2]] for i in range(N)]
            oc = (params[N * 3], params[N * 3 + 1])
            oa = params[N * 3 + 2]
            side = compute_required_side(inner_list, oc, oa)
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                best_side = side
                best_inner = inner_list
                best_outer_center = oc
                best_outer_angle = oa

    except Exception as e:
        print(f"DE failed: {e}")

    # Phase 2: Multi-start local optimization from diverse initial configs
    remaining_time = DEADLINE - (time.time() - start_time)
    configs_per_phase = max(3, int(remaining_time / 15))

    for idx, centers in enumerate(initial_configs[:configs_per_phase]):
        if time.time() - start_time > DEADLINE - 20:
            break

        initial_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Try multiple outer angles
        for init_angle in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > DEADLINE - 20:
                break

            init_params = []
            for x, y, a in initial_inner:
                init_params.extend([x, y, a])
            init_params.extend([cx_avg, cy_avg, float(init_angle)])

            # Nelder-Mead polish
            try:
                res = minimize(
                    lambda p: objective_exact(p),
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-11, 'fatol': 1e-13, 'maxfev': 3000}
                )

                if res.fun < best_side:
                    inner_list = [[res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]] for i in range(N)]
                    oc = (res.x[N * 3], res.x[N * 3 + 1])
                    oa = res.x[N * 3 + 2]
                    side = compute_required_side(inner_list, oc, oa)
                    if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                        best_side = side
                        best_inner = inner_list
                        best_outer_center = oc
                        best_outer_angle = oa
            except Exception:
                pass

    # Phase 3: Final polish on the best solution
    if best_inner is not None and time.time() - start_time < DEADLINE - 10:
        # Polish outer only
        best_outer_center, best_outer_angle = polish_outer_only(
            best_inner, best_outer_center, best_outer_angle, time_limit=5.0
        )
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

        # Full polish
        if time.time() - start_time < DEADLINE - 5:
            params = []
            for x, y, a in best_inner:
                params.extend([x, y, a])
            params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

            polished = polish_solution(params, time_limit=8.0)

            inner_list = [[polished[i * 3], polished[i * 3 + 1], polished[i * 3 + 2]] for i in range(N)]
            oc = (polished[N * 3], polished[N * 3 + 1])
            oa = polished[N * 3 + 2]
            side = compute_required_side(inner_list, oc, oa)

            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_side:
                    best_side = side
                    best_inner = inner_list
                    best_outer_center = oc
                    best_outer_angle = oa

    # Fallback
    if best_inner is None:
        centers = axial_to_cartesian(initial_configs[0], scale=1.02)
        best_inner = [[x, y, 0.0] for x, y in centers]
        best_outer_center = (0.0, 0.0)
        best_outer_angle = 0.0
        best_side = compute_required_side(best_inner, best_outer_center, best_outer_angle)

    # Final verification
    if not check_all_disjoint(best_inner) or not check_inside_outer(best_inner, best_outer_center, best_side, best_outer_angle):
        for scale_fix in [1.01, 1.02, 1.03, 1.05]:
            centers = axial_to_cartesian(initial_configs[0], scale=scale_fix)
            test_inner = [[x, y, 0.0] for x, y in centers]
            test_center = (0.0, 0.0)
            test_angle = 0.0
            test_side = compute_required_side(test_inner, test_center, test_angle)
            if check_all_disjoint(test_inner) and check_inside_outer(test_inner, test_center, test_side, test_angle):
                best_inner = test_inner
                best_outer_center = test_center
                best_outer_angle = test_angle
                best_side = test_side
                break

    return best_inner, best_outer_center, best_side, best_outer_angle
