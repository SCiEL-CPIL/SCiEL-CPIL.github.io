"""Packing unit hexagons inside a regular hexagon."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
DEADLINE = 140.0  # safety margin before 150s kill


def hex_vertex_positions(cx, cy, angle_rad):
    """Return 6 vertex positions as a (6,2) numpy array for a unit hexagon."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + np.cos(thetas), cy + np.sin(thetas)])


def min_dist_between_hexagons(v1, v2):
    """
    Compute minimum distance between two convex hexagons (6x2 arrays).
    Returns 0 if they overlap or touch, positive distance otherwise.
    Uses SAT: if any separating axis found, return the gap; else return 0.
    """
    axes = []
    for v in (v1, v2):
        for i in range(6):
            p1 = v[i]
            p2 = v[(i + 1) % 6]
            edge = p2 - p1
            n = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(n)
            if norm > 1e-12:
                axes.append(n / norm)
    if not axes:
        return 0.0

    max_gap = 0.0
    for axis in axes:
        proj1 = v1 @ axis
        proj2 = v2 @ axis
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        gap = min(max1 - min2, max2 - min1)
        if gap < -1e-9:
            return 0.0
        max_gap = max(max_gap, gap)
    return max_gap


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


def sat_overlap_check(v1, v2, eps=1e-9):
    """
    Check if two convex polygons (defined by their vertices) overlap using SAT.
    v1, v2: lists of (x, y) tuples.
    Returns True if they overlap (including touching).
    """
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


def compute_cluster_extent(inner):
    """
    Compute the extent of the inner hexagon cluster.
    Returns the maximum distance from centroid to any vertex.
    """
    if not inner:
        return 0.0

    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)

    if not all_verts:
        return 0.0

    cx = sum(v[0] for v in all_verts) / len(all_verts)
    cy = sum(v[1] for v in all_verts) / len(all_verts)

    max_dist = 0.0
    for v in all_verts:
        d = math.sqrt((v[0] - cx) ** 2 + (v[1] - cy) ** 2)
        if d > max_dist:
            max_dist = d

    return max_dist


def force_directed_compact(initial_centers, initial_angles, time_budget=25.0):
    """
    Force-directed granular packing: treat hexagons as particles with
    repulsive forces and a soft confining potential.
    
    Returns list of [x, y, angle_deg] for N hexagons.
    """
    n = len(initial_centers)
    positions = np.array([[c[0], c[1]] for c in initial_centers], dtype=float)
    angles = np.array(initial_angles, dtype=float)

    def get_verts(i):
        return hex_vertex_positions(positions[i, 0], positions[i, 1], angles[i])

    repulse_strength = 50.0
    repulse_radius = 3.5
    centering_strength = 0.1
    dt = 0.005
    damping = 0.92
    max_steps = int(time_budget / dt)

    def compute_extent():
        all_pts = []
        for i in range(n):
            all_pts.append(get_verts(i))
        all_pts = np.vstack(all_pts)
        cx = all_pts[:, 0].mean()
        cy = all_pts[:, 1].mean()
        return np.max(np.sqrt((all_pts[:, 0] - cx)**2 + (all_pts[:, 1] - cy)**2))

    best_extent = compute_extent()
    best_positions = positions.copy()
    best_angles = angles.copy()

    start_sim = time.time()
    step = 0
    while step < max_steps and (time.time() - start_sim) < time_budget:
        forces = np.zeros_like(positions)
        torques = np.zeros(n)

        for i in range(n):
            vi = get_verts(i)
            for j in range(i + 1, n):
                vj = get_verts(j)
                d = positions[j] - positions[i]
                dist = np.linalg.norm(d)
                if dist > repulse_radius:
                    continue
                gap = min_dist_between_hexagons(vi, vj)
                if gap < 0.01:
                    force_mag = repulse_strength * max(0, 0.01 - gap) ** 2 / max(dist, 0.1)
                    if dist > 1e-6:
                        f = force_mag * d / dist
                    else:
                        f = np.array([force_mag * 0.01, 0.0])
                    forces[i] -= f
                    forces[j] += f
                    if gap < 0.005:
                        torques[j] += 0.02
                        torques[i] -= 0.02

        centroid = positions.mean(axis=0)
        for i in range(n):
            r = positions[i] - centroid
            forces[i] -= centering_strength * r

        vel = forces * dt
        vel *= damping
        positions += vel
        angles += torques * dt

        if step % 50 == 0:
            ext = compute_extent()
            if ext < best_extent:
                best_extent = ext
                best_positions = positions.copy()
                best_angles = angles.copy()

        step += 1

    result = []
    for i in range(n):
        result.append([best_positions[i, 0], best_positions[i, 1], math.degrees(best_angles[i])])
    return result, best_extent


def smooth_max(values, beta=100.0):
    """Smooth maximum using log-sum-exp."""
    values = np.array(values)
    return np.log(np.sum(np.exp(beta * values))) / beta


def objective_compact(params):
    """
    Objective for compacting the cluster: minimize the maximum projection
    onto 6 hexagon axes (smoothed), plus penalty for overlaps.
    params: [x1,y1,a1, ..., xN,yN,aN]
    """
    inner = [[params[i*3], params[i*3+1], params[i*3+2]] for i in range(N)]
    
    # Compute all vertices
    all_verts = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        all_verts.extend(verts)
    
    if not all_verts:
        return 0.0
    
    all_verts = np.array(all_verts)
    cx = all_verts[:, 0].mean()
    cy = all_verts[:, 1].mean()
    
    # Max distance from centroid (smooth surrogate for extent)
    dists = np.sqrt((all_verts[:, 0] - cx)**2 + (all_verts[:, 1] - cy)**2)
    extent = smooth_max(dists, beta=50.0)
    
    # Penalty for overlaps using center distance as proxy
    penalty = 0.0
    min_dist = SQRT3 - 0.005
    for i in range(N):
        for j in range(i+1, N):
            d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < min_dist:
                penalty += (min_dist - d)**2 * 200.0
    
    return extent + penalty


def objective_joint(params):
    """
    Joint objective: minimize outer side length with smooth surrogate,
    plus penalties for overlaps and containment violations.
    params: [x1,y1,a1, ..., xN,yN,aN, cx, cy, angle]
    """
    inner = []
    for i in range(N):
        inner.append([params[i*3], params[i*3+1], params[i*3+2]])
    
    cx = params[N*3]
    cy = params[N*3+1]
    angle = params[N*3+2]
    
    # Compute required side (smoothed)
    ox, oy = cx, cy
    cos_a = math.cos(math.radians(angle))
    sin_a = math.sin(math.radians(angle))
    
    normals = []
    for i in range(6):
        theta = math.radians(30 + i * 60)
        normals.append((math.cos(theta), math.sin(theta)))
    
    projections = []
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                projections.append(proj)
    
    projections = np.array(projections)
    max_proj = smooth_max(projections, beta=80.0)
    side = 2.0 * max_proj / SQRT3
    
    # Penalty for overlaps
    penalty = 0.0
    min_dist = SQRT3 - 0.005
    for i in range(N):
        for j in range(i+1, N):
            d = math.sqrt((inner[i][0]-inner[j][0])**2 + (inner[i][1]-inner[j][1])**2)
            if d < min_dist:
                penalty += (min_dist - d)**2 * 100.0
    
    # Penalty for containment violation
    apothem = side * SQRT3 / 2.0
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        for v in verts:
            dx = v[0] - ox
            dy = v[1] - oy
            lx = dx * cos_a + dy * sin_a
            ly = -dx * sin_a + dy * cos_a
            for nx, ny in normals:
                proj = lx * nx + ly * ny
                if proj > apothem:
                    penalty += (proj - apothem)**2 * 50.0
    
    return side + penalty


def construct():
    start_time = time.time()

    # 1. Define 11 hexagons on a hexagonal lattice
    axial_coords = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (-1, 2), (-2, 1)
    ]

    if len(axial_coords) > N:
        axial_coords = axial_coords[:N]

    # 2. Convert axial to Cartesian
    epsilon = 1e-4
    scale = 1.0 + epsilon

    centers = []
    for q, r in axial_coords:
        x = 1.5 * scale * q
        y = SQRT3 * scale * (r + q / 2.0)
        centers.append((x, y))

    initial_inner = [[x, y, 0.0] for x, y in centers]

    best_side = float('inf')
    best_inner = [list(x) for x in initial_inner]
    best_outer_center = (0.0, 0.0)
    best_outer_angle = 0.0

    # 3. Force-directed compaction
    initial_centers_list = [(x, y) for x, y, a in initial_inner]
    initial_angles_rad = [math.radians(a) for x, y, a in initial_inner]

    best_compact_extent = compute_cluster_extent(initial_inner)
    best_compact_inner = [list(x) for x in initial_inner]

    num_fd_restarts = 4
    for restart in range(num_fd_restarts):
        if time.time() - start_time > DEADLINE - 70:
            break

        if restart == 0:
            fc = initial_centers_list
            fa = initial_angles_rad
        else:
            np.random.seed(123 + restart * 17)
            fc = []
            fa = []
            for i in range(N):
                cx, cy = initial_centers_list[i]
                ca = initial_angles_rad[i]
                pc = np.random.uniform(-0.05, 0.05, size=2)
                pa = np.random.uniform(-15, 15) * math.pi / 180
                fc.append((cx + pc[0], cy + pc[1]))
                fa.append(ca + pa)

        remaining = DEADLINE - 70 - (time.time() - start_time)
        time_per_restart = max(3.0, min(12.0, remaining / (num_fd_restarts - restart)))

        try:
            fd_result, fd_extent = force_directed_compact(fc, fa, time_budget=time_per_restart)
            if check_all_disjoint(fd_result):
                if fd_extent < best_compact_extent:
                    best_compact_extent = fd_extent
                    best_compact_inner = fd_result
        except Exception:
            pass

    # 4. Gradient-based compacting using smooth objective
    if time.time() - start_time < DEADLINE - 50:
        nm_params = []
        for x, y, a in best_compact_inner:
            nm_params.extend([x, y, a])

        try:
            res = minimize(objective_compact, nm_params, method='L-BFGS-B',
                          options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            il = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(N)]
            if check_all_disjoint(il):
                ext = compute_cluster_extent(il)
                if ext < best_compact_extent:
                    best_compact_extent = ext
                    best_compact_inner = il
        except Exception:
            pass

    # Also try Nelder-Mead polish
    if time.time() - start_time < DEADLINE - 45:
        nm_params = []
        for x, y, a in best_compact_inner:
            nm_params.extend([x, y, a])

        try:
            res = minimize(objective_compact, nm_params, method='Nelder-Mead',
                          options={'maxiter': 300, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 600})
            il = [[res.x[i*3], res.x[i*3+1], res.x[i*3+2]] for i in range(N)]
            if check_all_disjoint(il):
                ext = compute_cluster_extent(il)
                if ext < best_compact_extent:
                    best_compact_extent = ext
                    best_compact_inner = il
        except Exception:
            pass

    compact_inner = best_compact_inner

    # 5. Optimize Outer Hexagon
    def objective_outer(params, inner_list):
        cx, cy, angle_deg = params
        side = compute_required_side(inner_list, (cx, cy), angle_deg)
        return side

    compact_centers = [(c[0], c[1]) for c in compact_inner]
    cx_avg = sum(c[0] for c in compact_centers) / len(compact_centers)
    cy_avg = sum(c[1] for c in compact_centers) / len(compact_centers)

    best_result = None
    best_val = float('inf')

    init_angles = [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165, 180]

    for init_angle in init_angles:
        if time.time() - start_time > DEADLINE - 35:
            break

        init_params = [cx_avg, cy_avg, float(init_angle)]

        try:
            res = minimize(
                lambda p: objective_outer(p, compact_inner),
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 350}
            )
            if res.fun < best_val:
                best_val = res.fun
                best_result = res.x
        except Exception:
            pass

    if best_result is not None:
        best_outer_center = (best_result[0], best_result[1])
        best_outer_angle = best_result[2]
        best_side = best_val
    else:
        best_outer_center = (cx_avg, cy_avg)
        best_outer_angle = 0.0
        best_side = compute_required_side(compact_inner, best_outer_center, best_outer_angle)

    # 6. Joint optimization with smooth objective
    init_params = []
    for x, y, a in compact_inner:
        init_params.extend([x, y, a])
    init_params.extend([best_outer_center[0], best_outer_center[1], best_outer_angle])

    best_joint_side = best_side
    best_joint_inner = [list(x) for x in compact_inner]
    best_joint_center = best_outer_center
    best_joint_angle = best_outer_angle

    # Try L-BFGS-B for joint optimization
    if time.time() - start_time < DEADLINE - 20:
        try:
            res = minimize(objective_joint, init_params, method='L-BFGS-B',
                          options={'maxiter': 800, 'ftol': 1e-12, 'gtol': 1e-10})
            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            oc = (res.x[N*3], res.x[N*3+1])
            oa = res.x[N*3+2]
            side = compute_required_side(inner_list, oc, oa)
            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # Try Nelder-Mead with restarts
    num_restarts = 6
    for restart in range(num_restarts):
        if time.time() - start_time > DEADLINE - 10:
            break

        current_params = list(init_params)
        if restart > 0:
            np.random.seed(42 + restart * 7)
            perturb_scale = 0.02 + restart * 0.006
            for i in range(len(current_params)):
                if i < N * 3:
                    current_params[i] += np.random.uniform(-perturb_scale, perturb_scale)
                elif i == N * 3 + 2:
                    current_params[i] += np.random.uniform(-5, 5)

        try:
            res = minimize(
                objective_joint,
                current_params,
                method='Nelder-Mead',
                options={'maxiter': 400, 'xatol': 1e-7, 'fatol': 1e-9, 'maxfev': 1000}
            )

            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            oc = (res.x[N*3], res.x[N*3+1])
            oa = res.x[N*3+2]

            side = compute_required_side(inner_list, oc, oa)

            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 7. Final high-precision polish
    if time.time() - start_time < DEADLINE - 3:
        polish_params = []
        for x, y, a in best_joint_inner:
            polish_params.extend([x, y, a])
        polish_params.extend([best_joint_center[0], best_joint_center[1], best_joint_angle])

        try:
            res = minimize(
                objective_joint,
                polish_params,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'ftol': 1e-14, 'gtol': 1e-12}
            )

            inner_list = []
            for i in range(N):
                inner_list.append([res.x[i*3], res.x[i*3+1], res.x[i*3+2]])
            oc = (res.x[N*3], res.x[N*3+1])
            oa = res.x[N*3+2]

            side = compute_required_side(inner_list, oc, oa)

            if check_all_disjoint(inner_list) and check_inside_outer(inner_list, oc, side, oa):
                if side < best_joint_side:
                    best_joint_side = side
                    best_joint_inner = inner_list
                    best_joint_center = oc
                    best_joint_angle = oa
        except Exception:
            pass

    # 8. Final verification and fallback
    final_inner = best_joint_inner
    final_center = best_joint_center
    final_side = best_joint_side
    final_angle = best_joint_angle

    if not check_all_disjoint(final_inner) or not check_inside_outer(final_inner, final_center, final_side, final_angle):
        scale_fallback = 1.02
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
