"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 135.0  # safety margin before 150s kill


def logsumexp(a, b=1.0):
    """Stable log-sum-exp for array a with temperature b."""
    a = np.asarray(a, dtype=np.float64)
    amax = np.max(a)
    if not np.isfinite(amax):
        return amax
    return amax + math.log(np.sum(np.exp((a - amax) / b))) * b


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
    # Normals for a hexagon with angle 0 (vertices at 0, 60, ...):
    # Outward normals are at 30, 90, 150, 210, 270, 330 degrees.
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))

    max_projection = 0.0

    # Rotate outer frame to match outer_angle
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))

    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            # Translate to outer center
            dx = v[0] - ox
            dy = v[1] - oy
            # Rotate points by -outer_angle to align with standard hex axes
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a

            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > max_projection:
                    max_projection = proj

    # S * sqrt(3) / 2 >= max_projection
    # S >= 2 * max_projection / sqrt(3)
    return 2.0 * max_projection / SQRT3


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
    axes = []

    # Get axes from edges of v1
    for i in range(len(v1)):
        p1 = v1[i]
        p2 = v1[(i + 1) % len(v1)]
        edge = (p2[0] - p1[0], p2[1] - p1[1])
        normal = (-edge[1], edge[0])
        norm = math.sqrt(normal[0] ** 2 + normal[1] ** 2)
        if norm > 1e-9:
            axes.append((normal[0] / norm, normal[1] / norm))

    # Get axes from edges of v2
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

        # Check for gap
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


def construct():
    start_time = time.time()

    # 1. Define 11 hexagons in a dense 3-4-3 cluster (hexagonal arrangement)
    # Row-based: 3 hexagons in top row, 4 in middle, 3 in bottom
    # This is the most compact packing of 11 unit hexagons
    epsilon = 1e-6
    scale = 1.0 + epsilon

    # 3-4-3 cluster: positions in axial coordinates
    # Top row (3): (-1, 1), (0, 1), (1, 1)
    # Middle row (4): (-1, 0), (0, 0), (1, 0), (2, 0)
    # Bottom row (3): (-1, -1), (0, -1), (1, -1)
    # But this gives 10. Need 11. Let's use a different arrangement.
    
    # Actually, for 11 hexagons, a good arrangement is:
    # Center + 6 around it + 4 more
    # Or: a hexagonal number pattern
    # Let's use: center at origin, then rings
    
    # Better: use the standard hexagonal lattice positions for 11 hexagons
    # that form a roughly hexagonal shape
    axial_coords = [
        (0, 0),      # center
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),  # ring 1
        (2, 0), (1, 1), (0, 2), (-1, 2), (-2, 1), (-2, 0),  # partial ring 2
    ]
    
    # That's 12, take first 11
    axial_coords = axial_coords[:11]

    # Convert axial to Cartesian
    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    # Initial layout: All hexagons aligned (angle 0)
    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 2. Optimize Outer Hexagon using smooth log-sum-exp surrogate
    # This makes the objective smooth and amenable to gradient-based optimization
    
    def compute_required_side_smooth(inner_list, outer_center, outer_angle_deg, temperature=0.01):
        """Smooth version of compute_required_side using log-sum-exp."""
        if not inner_list:
            return 0.0

        ox, oy = outer_center
        normals = []
        for i in range(6):
            theta = math.radians(30 + i * 60)
            normals.append((math.cos(theta), math.sin(theta)))

        cos_a = math.cos(math.radians(outer_angle_deg))
        sin_a = math.sin(math.radians(outer_angle_deg))

        all_projections = []
        for (ix, iy, iangle) in inner_list:
            verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
            for v in verts:
                dx = v[0] - ox
                dy = v[1] - oy
                lx = dx * cos_a + dy * sin_a
                ly = -dx * sin_a + dy * cos_a

                for nx, ny in normals:
                    proj = lx * nx + ly * ny
                    all_projections.append(proj)

        # Use log-sum-exp to smoothly approximate the max
        if len(all_projections) == 0:
            return 0.0
        
        proj_array = np.array(all_projections, dtype=np.float64)
        # Smooth max using log-sum-exp
        smooth_max = logsumexp(proj_array, temperature)
        
        # S * sqrt(3) / 2 >= max_projection
        # S >= 2 * max_projection / sqrt(3)
        return 2.0 * smooth_max / SQRT3

    # Initial guess: centroid and 0
    cx_avg = sum(c[0] for c in centers) / len(centers)
    cy_avg = sum(c[1] for c in centers) / len(centers)

    # Try multiple diverse initial angles to avoid local minima
    best_result = None
    best_val = float('inf')

    init_angles = [0, 10, 20, 30, 45, 60, 75, 90, 120, 150, 180, 210, 240, 270, 300, 330]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 25:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]

        try:
            # Use BFGS with the smooth objective for better convergence
            res = minimize(
                lambda p: compute_required_side_smooth(initial_inner, (p[0], p[1]), p[2], temperature=0.01),
                init_params,
                method='BFGS',
                options={'maxiter': 200, 'gtol': 1e-8}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass

    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)
    else:
        # Fallback
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(initial_inner, best_outer_center, best_outer_angle)

    # 3. High-precision polish pass: fine-tune outer parameters with Nelder-Mead
    # using the exact (nonsmooth) objective
    if time.time() - start_time < DEADLINE - 15:
        try:
            polish_res = minimize(
                lambda p: compute_required_side(initial_inner, (p[0], p[1]), p[2]),
                [best_outer_center[0], best_outer_center[1], best_outer_angle],
                method='Nelder-Mead',
                options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000}
            )
            if polish_res.fun < best_side:
                best_outer_center = (polish_res.x[0], polish_res.x[1])
                best_outer_angle = polish_res.x[2]
                best_side = polish_res.fun
        except Exception:
            pass

    # 4. Joint optimization with smooth surrogate for better convergence
    def objective_joint_smooth(params, temperature=0.01):
        inner_list = []
        for i in range(N):
            x = params[i * 3]
            y = params[i * 3 + 1]
            a = params[i * 3 + 2]
            inner_list.append([x, y, a])

        cx = params[N * 3]
        cy = params[N * 3 + 1]
        angle = params[N * 3 + 2]

        side = compute_required_side_smooth(inner_list, (cx, cy), angle, temperature)

        # Penalty for overlaps: use center distance as proxy
        penalty = 0.0
        min_dist = SQRT3 - 0.02  # Safe margin
        for i in range(N):
            for j in range(i + 1, N):
                xi, yi, _ = inner_list[i]
                xj, yj, _ = inner_list[j]
                dist = math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        return side + 10.0 * penalty

    # Initial joint params
    init_params = []
    for x, y, a in initial_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    # Try joint optimization with a few restarts
    best_joint_side = best_side
    best_joint_inner = [list(x) for x in initial_inner]
    best_joint_center = best_outer_center
    best_joint_angle = best_outer_angle

    for restart in range(5):
        if time.time() - start_time > DEADLINE - 15:
            break

        current_params = list(init_params)
        if restart > 0:
            # Add small random perturbation
            np.random.seed(42 + restart)
            for i in range(len(current_params)):
                if i < N * 3:
                    current_params[i] += np.random.uniform(-0.05, 0.05)

        try:
            # Use L-BFGS-B for better handling of higher-dimensional problems
            res = minimize(
                lambda p: objective_joint_smooth(p, temperature=0.01),
                current_params,
                method='L-BFGS-B',
                options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
            )

            # Extract result
            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i * 3], res.x[i * 3 + 1], res.x[i * 3 + 2]])
            oc = (res.x[N * 3], res.x[N * 3 + 1])
            oa = res.x[N * 3 + 2]

            side = compute_required_side(inner_list, oc, oa)

            # Check validity
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 5. Final polish on the best joint solution
    if best_joint_side < best_side and time.time() - start_time < DEADLINE - 5:
        try:
            # Polish the outer parameters for the best joint solution
            polish_params = [best_joint_center[0], best_joint_center[1], best_joint_angle]
            polish_res = minimize(
                lambda p: compute_required_side(best_joint_inner, (p[0], p[1]), p[2]),
                polish_params,
                method='Nelder-Mead',
                options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000}
            )
            if polish_res.fun < best_joint_side:
                best_joint_center = (polish_res.x[0], polish_res.x[1])
                best_joint_angle = polish_res.x[2]
                best_joint_side = polish_res.fun
        except Exception:
            pass

    # 6. If joint optimization didn't find valid, use outer-optimized result
    if best_joint_side >= best_side:
        best_joint_inner = [list(x) for x in initial_inner]
        best_joint_side = best_side
        best_joint_center = best_outer_center
        best_joint_angle = best_outer_angle

    # 7. Final verification and fallback
    final_inner = best_joint_inner
    final_center = best_joint_center
    final_side = best_joint_side
    final_angle = best_joint_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        # Fallback: use original layout with safe spacing
        scale_fallback = 1.05
        fallback_centers = []
        for q, r in axial_coords:
            x = 1.5 * scale_fallback * q
            y = SQRT3 * scale_fallback * (r + q / 2.0)
            fallback_centers.append((x, y))

        fallback_inner = [[x, y, 0.0] for x, y in fallback_centers]
        final_center = (0.0, 0.0)
        final_angle = 0.0
        final_side = compute_required_side(fallback_inner, final_center, final_angle)
        final_inner = fallback_inner

    return final_inner, final_center, final_side, final_angle
