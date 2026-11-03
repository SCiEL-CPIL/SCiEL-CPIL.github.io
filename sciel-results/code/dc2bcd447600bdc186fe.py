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


def _is_valid(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
    for i in range(N):
        for j in range(i + 1, N):
            h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
            h2 = (positions[j, 0], positions[j, 1], angles_rad[j])
            if not _hex_disjoint_fast(h1, h2):
                return False
    return True


def _compute_min_outer_side(positions, angles_deg):
    angles_rad = np.radians(angles_deg)
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

    # Coarse scan
    phi_vals = np.radians(np.arange(0, 60, 0.5))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / SQRT3) * max_projs
    best_side = float(np.max(side_vals).min())
    best_idx = int(np.argmin(np.max(side_vals, axis=1)))
    best_phi = float(best_idx * 0.5)

    # Fine tune with golden-section
    lo = max(0.0, best_phi - 0.5)
    hi = min(60.0, best_phi + 0.5)
    gr = (math.sqrt(5) - 1) / 2
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1 = eval_phi(x1)
    f2 = eval_phi(x2)
    for _ in range(50):
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


def _make_honeycomb():
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


def _make_ring():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        theta = i * np.pi / 3
        pos[i, 0] = 2.5 * SQRT3 * np.cos(theta)
        pos[i, 1] = 2.5 * SQRT3 * np.sin(theta)
    for i in range(6):
        theta = (i + 0.5) * np.pi / 3
        pos[6 + i, 0] = 1.2 * SQRT3 * np.cos(theta)
        pos[6 + i, 1] = 1.2 * SQRT3 * np.sin(theta)
    return pos, ang


def _make_two_rows():
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for i in range(6):
        pos[i, 0] = i * SQRT3
        pos[i, 1] = 0.0
    for i in range(6):
        pos[6 + i, 0] = i * SQRT3 + SQRT3 / 2
        pos[6 + i, 1] = 1.5
    return pos, ang


def _make_compact_hex_cluster():
    # 7 hexagons in a flower + 5 around
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])
    coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1), (1, -1), (-1, 1),
        (2, -1), (-2, 1), (2, 0), (-2, 0), (0, 2),
    ]
    pos = np.zeros((N, 2))
    ang = np.zeros(N)
    for idx, (i, j) in enumerate(coords):
        pos[idx] = i * a1 + j * a2
        ang[idx] = 0.0
    return pos, ang


def _make_random():
    pos = np.random.uniform(-3, 3, (N, 2))
    ang = np.random.uniform(-30, 30, N)
    return pos, ang


def _try_fix_validity(pos, ang, max_iters=200):
    """Attempt to make a configuration valid by small random perturbations."""
    for _ in range(max_iters):
        if _is_valid(pos, ang):
            return True
        # Perturb a random hexagon
        idx = np.random.randint(N)
        pos[idx, 0] += np.random.uniform(-0.15, 0.15)
        pos[idx, 1] += np.random.uniform(-0.15, 0.15)
        ang[idx] += np.random.uniform(-5, 5)
    return _is_valid(pos, ang)


def _local_search(pos, ang, time_limit, rng):
    """Best-improvement local search with adaptive step size."""
    current_side, _, _, _ = _compute_min_outer_side(pos, ang)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()

    t_start = time.time()
    step = 0.08
    min_step = 0.0001
    no_improve_count = 0
    max_no_improve = 8

    while time.time() - t_start < time_limit and step > min_step:
        improved = False
        # Try each hexagon in random order
        order = rng.permutation(N)
        for idx in order:
            if time.time() - t_start > time_limit:
                break
            best_dim_move = None
            best_dim_side = best_side
            for dim in range(3):
                for direction in [1, -1]:
                    new_pos = best_pos.copy()
                    new_ang = best_ang.copy()
                    if dim == 0:
                        new_pos[idx, 0] += direction * step
                    elif dim == 1:
                        new_pos[idx, 1] += direction * step
                    else:
                        new_ang[idx] += direction * step

                    if not _is_valid(new_pos, new_ang):
                        continue

                    new_side, _, _, _ = _compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_dim_side - 1e-12:
                        best_dim_side = new_side
                        best_dim_move = (dim, direction, new_pos, new_ang)

            if best_dim_move is not None:
                dim, direction, new_pos, new_ang = best_dim_move
                best_pos = new_pos
                best_ang = new_ang
                best_side = best_dim_side
                improved = True
                no_improve_count = 0

        if not improved:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                step *= 0.6
                no_improve_count = 0
            else:
                step *= 0.85

    return best_side, best_pos, best_ang


def construct():
    t0 = time.time()
    rng = np.random.RandomState(42)

    inits = [
        _make_honeycomb(),
        _make_ring(),
        _make_two_rows(),
        _make_compact_hex_cluster(),
        _make_random(),
        _make_random(),
        _make_random(),
    ]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for init_idx, (pos0, ang0) in enumerate(inits):
        if time.time() - t0 > DEADLINE - 15.0:
            break

        # Ensure validity
        if not _is_valid(pos0, ang0):
            _try_fix_validity(pos0, ang0, max_iters=300)
            if not _is_valid(pos0, ang0):
                continue

        s0, phi0, cx0, cy0 = _compute_min_outer_side(pos0, ang0)
        if s0 < best_side:
            best_side = s0
            best_positions = pos0.copy()
            best_angles = ang0.copy()
            best_phi = phi0
            best_cx = cx0
            best_cy = cy0

        # Local search with time budget
        remaining = DEADLINE - 12.0 - (time.time() - t0)
        if remaining < 3.0:
            break
        # Allocate time: more for earlier inits
        time_budget = min(remaining * 0.6, 12.0)
        if init_idx > 3:
            time_budget = min(remaining * 0.4, 6.0)

        ls_side, ls_pos, ls_ang = _local_search(pos0, ang0, time_budget, rng)
        if ls_side < best_side:
            best_side = ls_side
            best_positions = ls_pos
            best_angles = ls_ang
            best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

        # If we found a good solution, do a second round of finer local search
        if time.time() - t0 < DEADLINE - 20.0 and best_side < 10.5:
            remaining2 = DEADLINE - 18.0 - (time.time() - t0)
            if remaining2 > 5.0:
                ls2_side, ls2_pos, ls2_ang = _local_search(
                    best_positions, best_angles, min(remaining2 * 0.5, 10.0), rng)
                if ls2_side < best_side:
                    best_side = ls2_side
                    best_positions = ls2_pos
                    best_angles = ls2_ang
                    best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(
                        best_positions, best_angles)

    # Final verification
    if best_positions is None or not _is_valid(best_positions, best_angles):
        pos0, ang0 = _make_honeycomb()
        _try_fix_validity(pos0, ang0, max_iters=500)
        best_positions = pos0.copy()
        best_angles = ang0.copy()
        best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
