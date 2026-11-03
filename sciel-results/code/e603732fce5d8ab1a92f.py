import math
import time
import numpy as np

N = 12
DEADLINE = 142.0
SQRT3 = math.sqrt(3)

_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i + 1) % 6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i + 1) % 6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx * _nx + _ny * _ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

_UX = _UNIT_HEX_VERTS[:, 0]
_UY = _UNIT_HEX_VERTS[:, 1]
_ENX = _EDGE_NORMALS[:, 0]
_ENY = _EDGE_NORMALS[:, 1]


def _hex_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1, a2 = h1[2], h2[2]
    c1, s1 = math.cos(a1), math.sin(a1)
    c2, s2 = math.cos(a2), math.sin(a2)
    v1x = h1[0] + _UX * c1 - _UY * s1
    v1y = h1[1] + _UX * s1 + _UY * c1
    v2x = h2[0] + _UX * c2 - _UY * s2
    v2y = h2[1] + _UX * s2 + _UY * c2
    en1x = _ENX * c1 - _ENY * s1
    en1y = _ENX * s1 + _ENY * c1
    en2x = _ENX * c2 - _ENY * s2
    en2y = _ENX * s2 + _ENY * c2
    for i in range(6):
        nx, ny = en1x[i], en1y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    for i in range(6):
        nx, ny = en2x[i], en2y[i]
        p1 = v1x * nx + v1y * ny
        p2 = v2x * nx + v2y * ny
        if p1.max() < p2.min() or p2.max() < p1.min():
            return True
    return False


def _is_valid(positions, angles_rad):
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_rad):
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + positions[:, 0:1]
    verts_y = ry + positions[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    def eval_phi(phi_deg):
        phi = np.radians(phi_deg)
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / SQRT3) * max_projs
        return float(np.max(side_vals))

    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx)

    # Fine tune with golden section
    lo = max(0.0, best_phi - 1.0)
    hi = min(60.0, best_phi + 1.0)
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(60):
        if f1 < f2:
            hi = x2
            x2 = x1
            f2 = f1
            x1 = hi - gr * (hi - lo)
            f1 = eval_phi(x1)
        else:
            lo = x1
            x1 = x2
            f1 = f2
            x2 = lo + gr * (hi - lo)
            f2 = eval_phi(x2)
    best_phi = (lo + hi) / 2.0
    best_side = min(best_side, eval_phi(best_phi))

    return best_side, best_phi, float(center[0]), float(center[1])


def _md_simulate(pos0, ang0, time_limit):
    t_start = time.time()
    pos = pos0.astype(float).copy()
    ang = ang0.astype(float).copy()
    vel = np.zeros((N, 2))
    ang_vel = np.zeros(N)

    # Track best valid configuration
    if _is_valid(pos, ang):
        best_side, _, _, _ = _compute_min_outer_side(pos, ang)
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_side = best_side
    else:
        best_pos = pos.copy()
        best_ang = ang.copy()
        best_side = float('inf')

    dt = 0.005
    damping = 0.995
    k_repel = 50.0
    k_attract = 0.5
    k_ang = 0.05  # small angular spring to align with 0

    max_steps = 200000

    for step in range(max_steps):
        if time.time() - t_start > time_limit:
            break

        # Compute forces
        f = np.zeros((N, 2))
        t_ang = np.zeros(N)

        # Attraction to origin (soft)
        f -= k_attract * pos

        # Angular spring to 0
        t_ang -= k_ang * ang

        # Pairwise repulsion via SAT
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 4.5:
                    continue

                a1, a2 = ang[i], ang[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)

                v1x = pos[i, 0] + _UX * c1 - _UY * s1
                v1y = pos[i, 1] + _UX * s1 + _UY * c1
                v2x = pos[j, 0] + _UX * c2 - _UY * s2
                v2y = pos[j, 1] + _UX * s2 + _UY * c2

                en1x = _ENX * c1 - _ENY * s1
                en1y = _ENX * s1 + _ENY * c1
                en2x = _ENX * c2 - _ENY * s2
                en2y = _ENX * s2 + _ENY * c2

                min_overlap = 0.0
                best_nx, best_ny = dx / (math.sqrt(dist_sq) + 1e-12), dy / (math.sqrt(dist_sq) + 1e-12)
                # Check if overlapping and find minimum penetration
                overlap = False
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # Overlap if intervals intersect
                    lo1, hi1 = p1.min(), p1.max()
                    lo2, hi2 = p2.min(), p2.max()
                    if lo1 < hi2 and lo2 < hi1:
                        # Compute penetration depth along this normal
                        # The gap is negative if overlapping
                        gap = min(hi1 - lo2, hi2 - lo1)
                        if gap < 0:
                            overlap = True
                            pen = -gap
                            if pen > min_overlap:
                                min_overlap = pen
                                best_nx, best_ny = nx, ny
                    else:
                        # Separated along this axis, so no overlap
                        overlap = False
                        break

                if overlap:
                    # Apply repulsive force proportional to overlap
                    force_mag = k_repel * min_overlap
                    f[i, 0] += force_mag * best_nx
                    f[i, 1] += force_mag * best_ny
                    f[j, 0] -= force_mag * best_nx
                    f[j, 1] -= force_mag * best_ny
                    # Small angular torque to separate
                    t_ang[i] += 0.1 * min_overlap * (1.0 if best_nx > 0 else -1.0)
                    t_ang[j] -= 0.1 * min_overlap * (1.0 if best_nx > 0 else -1.0)

        # Velocity Verlet integration
        # Update positions
        pos += vel * dt + 0.5 * f * dt * dt
        ang += ang_vel * dt + 0.5 * t_ang * dt * dt

        # Compute new forces (reuse same function logic, but for simplicity, we recompute)
        # For efficiency, we'll approximate by using the same f for now (semi-implicit Euler style)
        # Actually, let's do proper Verlet: recompute forces at new positions
        f_new = np.zeros((N, 2))
        t_new = np.zeros(N)
        f_new -= k_attract * pos
        t_new -= k_ang * ang

        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 4.5:
                    continue

                a1, a2 = ang[i], ang[j]
                c1, s1 = math.cos(a1), math.sin(a1)
                c2, s2 = math.cos(a2), math.sin(a2)

                v1x = pos[i, 0] + _UX * c1 - _UY * s1
                v1y = pos[i, 1] + _UX * s1 + _UY * c1
                v2x = pos[j, 0] + _UX * c2 - _UY * s2
                v2y = pos[j, 1] + _UX * s2 + _UY * c2

                en1x = _ENX * c1 - _ENY * s1
                en1y = _ENX * s1 + _ENY * c1
                en2x = _ENX * c2 - _ENY * s2
                en2y = _ENX * s2 + _ENY * c2

                min_overlap = 0.0
                best_nx, best_ny = dx / (math.sqrt(dist_sq) + 1e-12), dy / (math.sqrt(dist_sq) + 1e-12)
                overlap = False
                for k in range(6):
                    nx, ny = en1x[k], en1y[k]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    lo1, hi1 = p1.min(), p1.max()
                    lo2, hi2 = p2.min(), p2.max()
                    if lo1 < hi2 and lo2 < hi1:
                        gap = min(hi1 - lo2, hi2 - lo1)
                        if gap < 0:
                            overlap = True
                            pen = -gap
                            if pen > min_overlap:
                                min_overlap = pen
                                best_nx, best_ny = nx, ny
                    else:
                        overlap = False
                        break

                if overlap:
                    force_mag = k_repel * min_overlap
                    f_new[i, 0] += force_mag * best_nx
                    f_new[i, 1] += force_mag * best_ny
                    f_new[j, 0] -= force_mag * best_nx
                    f_new[j, 1] -= force_mag * best_ny
                    t_new[i] += 0.1 * min_overlap * (1.0 if best_nx > 0 else -1.0)
                    t_new[j] -= 0.1 * min_overlap * (1.0 if best_nx > 0 else -1.0)

        # Update velocities
        vel = (vel + 0.5 * (f + f_new) * dt) * damping
        ang_vel = (ang_vel + 0.5 * (t_ang + t_new) * dt) * damping

        # Check validity and update best
        if _is_valid(pos, ang):
            # Quick estimate: compute outer side occasionally to save time
            if step % 10 == 0:
                side, _, _, _ = _compute_min_outer_side(pos, ang)
                if side < best_side:
                    best_side = side
                    best_pos = pos.copy()
                    best_ang = ang.copy()

    return best_side, best_pos, best_ang


def _coordinate_descent_polish(pos, ang_rad, time_limit, initial_step=0.03):
    current_side, _, _, _ = _compute_min_outer_side(pos, ang_rad)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang_rad.copy()

    t_start = time.time()
    step = initial_step
    min_step = 0.0005

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        for idx in range(N):
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()

                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step * math.pi / 180.0  # radians

                    valid = True
                    h_moved = (new_pos[idx, 0], new_pos[idx, 1], new_ang[idx])
                    for j in range(N):
                        if j == idx:
                            continue
                        h_other = (new_pos[j, 0], new_pos[j, 1], new_ang[j])
                        if not _hex_disjoint_fast(h_moved, h_other):
                            valid = False
                            break

                    if not valid:
                        continue

                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)

                    if new_side < best_side - 1e-12:
                        best_pos = new_pos
                        best_ang = new_ang
                        best_side = new_side
                        improved = True
                        break

                if time.time() - t_start > time_limit:
                    break
            if time.time() - t_start > time_limit:
                break
            if improved:
                break
        else:
            step *= 0.7
            continue
        break

    return best_side, best_pos, best_ang


def construct():
    t0 = time.time()

    def make_honeycomb():
        a1 = np.array([SQRT3, 0.0])
        a2 = np.array([SQRT3 / 2.0, 1.5])
        lattice_coords = [
            (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
            (1, -1), (-1, 1), (2, -1), (-2, 1),
            (1, 1), (-1, -1), (2, 0),
        ]
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for idx, (i, j) in enumerate(lattice_coords):
            pos[idx] = i * a1 + j * a2
            ang[idx] = 0.0
        return pos, ang

    def make_random_cluster():
        # Dense random cluster near origin
        pos = np.random.uniform(-2.5, 2.5, (N, 2))
        ang = np.random.uniform(-0.3, 0.3, N)
        return pos, ang

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    # Try multiple initializations with MD
    inits = [
        make_honeycomb(),
        make_random_cluster(),
        make_random_cluster(),
        make_random_cluster(),
    ]

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 20.0:
            break

        # Run MD simulation
        md_time = min(25.0, (DEADLINE - 20.0 - (time.time() - t0)) / max(1, len(inits) - init_idx))
        md_side, md_pos, md_ang = _md_simulate(pos0, ang0, md_time)

        if md_side < best_side:
            best_side = md_side
            best_positions = md_pos.copy()
            best_angles = md_ang.copy()

    # Final polish with coordinate descent
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0 and best_positions is not None:
        cd_time = min(remaining_time * 0.4, 15.0)
        cd_side, cd_pos, cd_ang = _coordinate_descent_polish(
            best_positions, best_angles, cd_time, initial_step=0.02)

        if cd_side < best_side:
            best_side = cd_side
            best_positions = cd_pos
            best_angles = cd_ang

    # Final verification and compute outer side
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = make_honeycomb()
        best_positions = pos0.copy()
        best_angles = ang0.copy()

    best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    # Convert angles to degrees
    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]),
                      float(np.degrees(best_angles[i]))])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
