import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar, differential_evolution

N = 12
DEADLINE = 145.0

_SQRT3 = math.sqrt(3)
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


def _hexagons_disjoint_fast(h1, h2):
    dx = h1[0] - h2[0]
    dy = h1[1] - h2[1]
    dist_sq = dx * dx + dy * dy
    if dist_sq > 4.5:
        return True
    a1 = h1[2]
    a2 = h2[2]
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
            if not _hexagons_disjoint_fast(h1, h2):
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
        side_vals = (2.0 / _SQRT3) * max_projs
        return np.max(side_vals)

    phi_vals = np.radians(np.arange(0, 60, 1))
    normal_angles = phi_vals[:, None] + np.pi / 6 + np.arange(6) * np.pi / 3
    nx = np.cos(normal_angles)
    ny = np.sin(normal_angles)
    proj = offsets[:, 0:1, None] * nx[None, :, :] + offsets[:, 1:2, None] * ny[None, :, :]
    abs_proj = np.abs(proj)
    max_projs = np.max(abs_proj, axis=0)
    side_vals = (2.0 / _SQRT3) * max_projs
    best_side = np.max(side_vals).min()
    best_idx = np.argmin(np.max(side_vals, axis=1))
    best_phi = float(best_idx)

    try:
        lo = max(0.0, best_phi - 1.0)
        hi = min(60.0, best_phi + 1.0)
        res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                              options={'xatol': 1e-10, 'maxiter': 100})
        if res.success and res.fun < best_side:
            best_side = float(res.fun)
            best_phi = float(res.x)
    except Exception:
        pass

    return float(best_side), float(best_phi), float(center[0]), float(center[1])


def _overlap_penalty(params, beta=10.0):
    """Compute smooth overlap penalty using log-sum-exp over gap violations."""
    pos = params[:2 * N].reshape(N, 2)
    ang = params[2 * N:]
    angles_rad = np.radians(ang)
    
    total_penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist_sq = dx * dx + dy * dy
            if dist_sq > 4.5:
                continue

            a1 = angles_rad[i]
            a2 = angles_rad[j]
            c1, s1 = np.cos(a1), np.sin(a1)
            c2, s2 = np.cos(a2), np.sin(a2)

            v1x = pos[i, 0] + _UX * c1 - _UY * s1
            v1y = pos[i, 1] + _UX * s1 + _UY * c1
            v2x = pos[j, 0] + _UX * c2 - _UY * s2
            v2y = pos[j, 1] + _UX * s2 + _UY * c2

            en1x = _ENX * c1 - _ENY * s1
            en1y = _ENX * s1 + _ENY * c1
            en2x = _ENX * c2 - _ENY * s2
            en2y = _ENX * s2 + _ENY * c2

            gaps = []
            for k in range(6):
                nx, ny = en1x[k], en1y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap1 = p2.min() - p1.max()
                gap2 = p1.min() - p2.max()
                gaps.append(min(0.0, gap1, gap2))

            for k in range(6):
                nx, ny = en2x[k], en2y[k]
                p1 = v1x * nx + v1y * ny
                p2 = v2x * nx + v2y * ny
                gap1 = p2.min() - p1.max()
                gap2 = p1.min() - p2.max()
                gaps.append(min(0.0, gap1, gap2))

            gaps = np.array(gaps)
            # We want to penalize if any gap is negative (overlap)
            # Use log-sum-exp of exp(-gap/beta) to get a smooth max of -gap
            # If all gaps >= 0, -gap <= 0, exp(-gap/beta) <= 1, logsumexp is small
            # If some gap < 0, -gap > 0, exp(-gap/beta) > 1, logsumexp is larger
            
            # Actually, simpler: penalty = softplus(max(-gaps))
            # But max is not differentiable. Use logsumexp.
            # logsumexp(-gaps / beta) * beta
            # If gaps are all large positive, -gaps are large negative, exp is ~0, logsumexp is ~log(12)
            # We want penalty to be 0 when disjoint.
            # Let's shift: we care about negative gaps.
            # penalty = logsumexp(-gaps/beta + 1) - log(12)  ? No.
            
            # Let's use: penalty = logsumexp(-gaps / beta) * beta - log(12) * beta
            # This is 0 when all gaps are 0 (touching).
            # If gap is negative (overlap), -gap is positive, penalty increases.
            # If gap is positive (separated), -gap is negative, penalty is negative.
            # We want penalty >= 0.
            # So: penalty = max(0, logsumexp(-gaps/beta)*beta - log(12)*beta)
            # Or just use softplus on the max gap violation.
            
            # Let's stick to a robust smooth approximation of max(0, -min_gap)
            min_gap = np.min(gaps)
            # softplus(-min_gap)
            # If min_gap >= 0, -min_gap <= 0, softplus is small
            # If min_gap < 0, -min_gap > 0, softplus is ~ -min_gap
            
            # Use a scaled softplus to make it steeper
            penalty = np.log1p(np.exp(beta * (-min_gap))) / beta
            total_penalty += penalty

    return total_penalty


def _objective(params, weight=100.0):
    """
    Objective function for optimization.
    Returns outer_side + weight * overlap_penalty
    """
    pos = params[:2 * N].reshape(N, 2)
    ang = params[2 * N:]
    
    # Compute outer side (exact, but non-smooth)
    # For optimization, we can use a smoothed version or just the exact one
    # L-BFGS-B can handle some non-smoothness if we use enough iterations
    # But it's better to use a smooth approximation for the outer side too
    
    # For now, let's use the exact outer side calculation
    # It might be slow, but let's see
    outer_side, _, _, _ = _compute_min_outer_side(pos, ang)
    
    penalty = _overlap_penalty(params)
    
    return outer_side + weight * penalty


def _smooth_objective(params, weight=100.0):
    """
    Smooth objective function using log-sum-exp for outer side as well.
    """
    pos = params[:2 * N].reshape(N, 2)
    ang = params[2 * N:]
    angles_rad = np.radians(ang)
    
    # Compute vertices
    c = np.cos(angles_rad)
    s = np.sin(angles_rad)
    rx = _UX[None, :] * c[:, None] - _UY[None, :] * s[:, None]
    ry = _UX[None, :] * s[:, None] + _UY[None, :] * c[:, None]
    verts_x = rx + pos[:, 0:1]
    verts_y = ry + pos[:, 1:2]
    verts_flat = np.stack((verts_x, verts_y), axis=-1).reshape(-1, 2)
    center = np.mean(verts_flat, axis=0)
    offsets = verts_flat - center

    # Smooth outer side: max over 6 directions of (2/sqrt3)*max|proj|
    # Use a fixed set of candidate phi values (e.g., 12 directions) and smooth max
    phi_candidates = np.radians(np.arange(0, 60, 5))  # 12 directions
    side_components = []
    for phi in phi_candidates:
        normal_angles = phi + np.pi / 6 + np.arange(6) * np.pi / 3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, 0:1] * nx[None, :] + offsets[:, 1:2] * ny[None, :]
        abs_proj = np.abs(proj)
        # Smooth max over 72 vertices
        m = abs_proj.max(axis=0)
        side_vals = (2.0 / _SQRT3) * m
        # Smooth max over 6 sides
        beta_side = 10.0
        # logsumexp
        m_arr = side_vals.max()
        smax = m_arr + np.log(np.sum(np.exp((side_vals - m_arr) / beta_side))) * beta_side
        side_components.append(smax)

    side_components = np.array(side_components)
    # Smooth max over phi candidates
    beta_phi = 5.0
    m_comp = side_components.max()
    outer_side = m_comp + np.log(np.sum(np.exp((side_components - m_comp) / beta_phi))) * beta_phi

    penalty = _overlap_penalty(params)
    
    return outer_side + weight * penalty


def _polish(pos, ang, time_limit, initial_step=0.05):
    """Coordinate descent polish."""
    current_side, _, _, _ = _compute_min_outer_side(pos, ang)
    best_side = current_side
    best_pos = pos.copy()
    best_ang = ang.copy()

    t_start = time.time()
    step = initial_step
    min_step = 0.001

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
                        new_ang[idx] += direction * step

                    angles_rad_new = np.radians(new_ang)
                    valid = True
                    h_moved = (new_pos[idx, 0], new_pos[idx, 1], angles_rad_new[idx])
                    for j in range(N):
                        if j == idx:
                            continue
                        h_other = (new_pos[j, 0], new_pos[j, 1], angles_rad_new[j])
                        if not _hexagons_disjoint_fast(h_moved, h_other):
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

    # Define bounds for differential evolution
    # Positions: -5 to 5, Angles: -30 to 30
    bounds = [(-5, 5), (-5, 5)] * N + [(-30, 30)] * N

    def de_objective(params):
        # Use the smooth objective for DE
        return _smooth_objective(params, weight=50.0)

    # Run differential evolution
    best_result = None
    best_side = float('inf')

    # DE is slow, so we limit iterations
    try:
        res = differential_evolution(
            de_objective,
            bounds,
            seed=42,
            maxiter=50,
            tol=1e-6,
            popsize=15,
            update_interval=10,
            polish=False,
            workers=1
        )
        if time.time() - t0 > DEADLINE - 20:
            raise TimeoutError("DE took too long")
        
        if res.fun < best_side:
            best_result = res.x
            best_side = res.fun
    except Exception:
        pass

    # If DE didn't find a good solution or timed out, try L-BFGS-B from multiple starts
    if best_result is None or time.time() - t0 > DEADLINE - 30:
        # Try L-BFGS-B from a few good starting points
        starts = []
        
        # Honeycomb
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
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
        starts.append(np.concatenate([pos.flatten(), ang]))

        # Random
        for _ in range(3):
            pos = np.random.uniform(-3, 3, (N, 2))
            ang = np.random.uniform(-30, 30, N)
            starts.append(np.concatenate([pos.flatten(), ang]))

        for start in starts:
            if time.time() - t0 > DEADLINE - 15:
                break
            
            try:
                res = minimize(
                    _smooth_objective,
                    start,
                    method='L-BFGS-B',
                    options={'maxiter': 2000, 'maxfun': 3000, 'ftol': 1e-12, 'gtol': 1e-8}
                )
                if res.fun < best_side:
                    best_side = res.fun
                    best_result = res.x
            except Exception:
                pass

    if best_result is None:
        # Fallback: return a valid honeycomb
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
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
        best_result = np.concatenate([pos.flatten(), ang])

    # Extract positions and angles
    best_pos = best_result[:2 * N].reshape(N, 2)
    best_ang = best_result[2 * N:]

    # Check validity
    if not _is_valid(best_pos, best_ang):
        # If invalid, try to fix by pushing apart or just use fallback
        # For now, use fallback
        a1 = np.array([_SQRT3, 0.0])
        a2 = np.array([_SQRT3 / 2.0, 1.5])
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
        best_pos = pos
        best_ang = ang

    # Polish
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        polish_time = min(remaining_time * 0.6, 30.0)
        _, polished_pos, polished_ang = _polish(best_pos, best_ang, polish_time)
        if _is_valid(polished_pos, polished_ang):
            best_pos = polished_pos
            best_ang = polished_ang

    # Compute final outer side
    best_side, best_phi, best_cx, best_cy = _compute_min_outer_side(best_pos, best_ang)

    inner = []
    for i in range(N):
        inner.append([float(best_pos[i, 0]), float(best_pos[i, 1]), float(best_ang[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
