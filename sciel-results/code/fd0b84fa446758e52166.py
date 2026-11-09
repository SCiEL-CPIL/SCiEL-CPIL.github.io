"""Packing 11 unit regular hexagons inside a regular hexagon.

Smooth SDF-based optimization with log-sum-exp for differentiable max/min.
Uses proper hexagon SDF via half-plane distances with smooth aggregation.
"""
import numpy as np
import time
import random
import math
from scipy.optimize import minimize

N = 11
DEADLINE = 145.0
START_TIME = time.time()


def _remaining_time():
    return DEADLINE - (time.time() - START_TIME)


def _hex_halfplane_dists(lx, ly, inradius):
    """Compute the 6 half-plane signed distances for a point in local hex frame.
    Returns array of 6 values: d_k = lx*nx_k + ly*ny_k - inradius.
    Inside the hexagon: all d_k <= 0. Outside: at least one d_k > 0.
    The max of these gives a measure of how far outside the hexagon the point is.
    """
    dists = np.empty(6)
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        dists[k] = lx * nx + ly * ny - inradius
    return dists


def _smooth_max(vals, temperature=0.05):
    """Log-sum-exp based smooth maximum."""
    v = np.asarray(vals, dtype=np.float64)
    vmax = np.max(v)
    if vmax < -1e10:
        return vmax
    exps = np.exp((v - vmax) / temperature)
    return vmax + temperature * math.log(np.sum(exps))


def _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2, a2, temperature=0.05):
    """
    Smooth differentiable overlap penalty between two unit hexagons.
    Uses the max of half-plane violations from each hexagon's perspective.
    If hex1 has a vertex outside hex2, or vice versa, there's overlap.
    """
    inradius = math.sqrt(3) / 2.0
    
    # Transform hex1 vertices into hex2's local frame
    a1_rad = math.radians(a1)
    a2_rad = math.radians(a2)
    
    # Rotation from world to hex2 local frame
    ca2 = math.cos(-a2_rad)
    sa2 = math.sin(-a2_rad)
    
    # Hex1 vertices in world frame
    h1_verts = np.empty((6, 2))
    for k in range(6):
        a = a1_rad + k * math.pi / 3
        h1_verts[k, 0] = cx1 + math.cos(a)
        h1_verts[k, 1] = cy1 + math.sin(a)
    
    # Transform to hex2 local frame
    dx = h1_verts[:, 0] - cx2
    dy = h1_verts[:, 1] - cy2
    lx = dx * ca2 - dy * sa2
    ly = dx * sa2 + dy * ca2
    
    # Half-plane distances for each hex1 vertex in hex2 frame
    # If any vertex of hex1 is outside hex2 (max of half-plane dists > 0), overlap
    # We want to penalize when max_k (lx*nx_k + ly*ny_k - inradius) > 0
    h1_in_h2 = np.empty(6)
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        h1_in_h2[k] = lx * nx + ly * ny - inradius
    
    # The max over vertices of the max over half-planes
    # For each vertex, the "outside-ness" is the max of its 6 half-plane dists
    vertex_outside = np.max(h1_in_h2, axis=1)  # shape (6,)
    penalty1 = _smooth_max(vertex_outside, temperature)
    
    # Similarly, hex2 vertices in hex1 frame
    ca1 = math.cos(-a1_rad)
    sa1 = math.sin(-a1_rad)
    
    h2_verts = np.empty((6, 2))
    for k in range(6):
        a = a2_rad + k * math.pi / 3
        h2_verts[k, 0] = cx2 + math.cos(a)
        h2_verts[k, 1] = cy2 + math.sin(a)
    
    dx2 = h2_verts[:, 0] - cx1
    dy2 = h2_verts[:, 1] - cy1
    lx2 = dx2 * ca1 - dy2 * sa1
    ly2 = dx2 * sa1 + dy2 * ca1
    
    h2_in_h1 = np.empty(6)
    for k in range(6):
        a = math.radians(30 + 60 * k)
        nx = math.cos(a)
        ny = math.sin(a)
        h2_in_h1[k] = lx2 * nx + ly2 * ny - inradius
    
    vertex_outside2 = np.max(h2_in_h1, axis=1)
    penalty2 = _smooth_max(vertex_outside2, temperature)
    
    return penalty1 + penalty2


def _hex_overlap_penalty_grad(cx1, cy1, a1, cx2, cy2, a2, temperature=0.05):
    """
    Gradient of smooth overlap penalty w.r.t. (cx1, cy1, a1, cx2, cy2, a2).
    Uses finite differences for angle, analytic for position.
    """
    # Use central finite differences for all parameters
    eps = 1e-4
    grads = np.zeros(6)
    
    f0 = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2, a2, temperature)
    
    # d/dc1
    f_p = _hex_overlap_penalty_smooth(cx1 + eps, cy1, a1, cx2, cy2, a2, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1 - eps, cy1, a1, cx2, cy2, a2, temperature)
    grads[0] = (f_p - f_m) / (2 * eps)
    
    # d/cy1
    f_p = _hex_overlap_penalty_smooth(cx1, cy1 + eps, a1, cx2, cy2, a2, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1, cy1 - eps, a1, cx2, cy2, a2, temperature)
    grads[1] = (f_p - f_m) / (2 * eps)
    
    # d/da1
    f_p = _hex_overlap_penalty_smooth(cx1, cy1, a1 + eps, cx2, cy2, a2, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1, cy1, a1 - eps, cx2, cy2, a2, temperature)
    grads[2] = (f_p - f_m) / (2 * eps)
    
    # d/cx2
    f_p = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2 + eps, cy2, a2, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2 - eps, cy2, a2, temperature)
    grads[3] = (f_p - f_m) / (2 * eps)
    
    # d/cy2
    f_p = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2 + eps, a2, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2 - eps, a2, temperature)
    grads[4] = (f_p - f_m) / (2 * eps)
    
    # d/da2
    f_p = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2, a2 + eps, temperature)
    f_m = _hex_overlap_penalty_smooth(cx1, cy1, a1, cx2, cy2, a2 - eps, temperature)
    grads[5] = (f_p - f_m) / (2 * eps)
    
    return f0, grads


def _min_enclosing_side(inner_positions, outer_center, outer_angle):
    """Find minimum side length of outer hexagon to contain all inner hexagons.
    The outer hexagon has center (cx,cy), rotation outer_angle, and side S.
    A point (px,py) is inside if for all 6 edge normals n_k:
      dot((px-cx, py-cy) rotated to outer frame, n_k) <= S*sqrt(3)/2
    So S must be >= max over all vertices of (max_k dot / (sqrt(3)/2))
    """
    cx, cy = outer_center
    oa_rad = math.radians(outer_angle)
    ca = math.cos(-oa_rad)
    sa = math.sin(-oa_rad)
    
    best_side = 0.0
    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)

            dx = vx - cx
            dy = vy - cy
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            for m in range(6):
                am = math.radians(30 + m * 60)
                nx = math.cos(am)
                ny = math.sin(am)
                proj = lx * nx + ly * ny
                # proj <= S * sqrt(3)/2  =>  S >= proj / (sqrt(3)/2)
                if proj > 0:
                    required = proj * 2.0 / math.sqrt(3)
                    if required > best_side:
                        best_side = required

    return best_side


def _hexagons_disjoint(cx1, cy1, a1, cx2, cy2, a2, tol=1e-7):
    """Check if two unit hexagons are disjoint using SAT."""
    def get_verts(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            verts.append([cx + math.cos(a), cy + math.sin(a)])
        return verts

    v1 = get_verts(cx1, cy1, a1)
    v2 = get_verts(cx2, cy2, a2)

    all_norms = []
    for verts in [v1, v2]:
        for i in range(6):
            j = (i + 1) % 6
            ex = verts[j][0] - verts[i][0]
            ey = verts[j][1] - verts[i][1]
            length = math.sqrt(ex * ex + ey * ey)
            if length < 1e-12:
                all_norms.append((1.0, 0.0))
            else:
                all_norms.append((-ey / length, ex / length))

    for nx, ny in all_norms:
        proj1 = [vx * nx + vy * ny for vx, vy in v1]
        proj2 = [vx * nx + vy * ny for vx, vy in v2]
        if max(proj1) < min(proj2) - tol or max(proj2) < min(proj1) - tol:
            return True

    return False


def _check_validity(inner_positions, outer_center, outer_side, outer_angle, tol=1e-5):
    """Check if configuration is valid."""
    cx, cy = outer_center
    inradius = outer_side * math.sqrt(3) / 2.0
    oa_rad = math.radians(outer_angle)

    for (ix, iy, iangle) in inner_positions:
        angle_rad = math.radians(iangle)
        for k in range(6):
            a = angle_rad + k * math.pi / 3
            vx = ix + math.cos(a)
            vy = iy + math.sin(a)

            dx = vx - cx
            dy = vy - cy
            ca = math.cos(-oa_rad)
            sa = math.sin(-oa_rad)
            lx = dx * ca - dy * sa
            ly = dx * sa + dy * ca

            for m in range(6):
                am = math.radians(30 + m * 60)
                nx = math.cos(am)
                ny = math.sin(am)
                if lx * nx + ly * ny > inradius + tol:
                    return False

    for i in range(N):
        for j in range(i + 1, N):
            if not _hexagons_disjoint(inner_positions[i][0], inner_positions[i][1],
                                      inner_positions[i][2],
                                      inner_positions[j][0], inner_positions[j][1],
                                      inner_positions[j][2], tol=tol):
                return False

    return True


def _sdf_optimize(positions, outer_angle, max_iters=500):
    """
    Optimize positions using SDF-based gradient descent with proper overlap detection.
    """
    state = np.array([p for pos in positions for p in pos], dtype=np.float64)
    lr = 0.02
    best_state = state.copy()
    best_side = _min_enclosing_side([tuple(p) for p in state.reshape(N, 3)], (0.0, 0.0), outer_angle)
    temp = 0.05

    for it in range(max_iters):
        if _remaining_time() < 5.0:
            break

        pos = state.reshape(N, 3)
        grad = np.zeros_like(state)

        # Repulsion between hexagons using smooth SDF penalty
        for i in range(N):
            for j in range(i + 1, N):
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 16.0:
                    continue
                _, g = _hex_overlap_penalty_grad(
                    pos[i, 0], pos[i, 1], pos[i, 2],
                    pos[j, 0], pos[j, 1], pos[j, 2], temp
                )
                # g = [dc1, dcy1, da1, dc2, dcy2, da2]
                grad[3 * i] += g[0]
                grad[3 * i + 1] += g[1]
                grad[3 * i + 2] += g[2]
                grad[3 * j] += g[3]
                grad[3 * j + 1] += g[4]
                grad[3 * j + 2] += g[5]

        # Attraction to center (to keep them compact)
        for i in range(N):
            grad[3 * i] += 0.005 * pos[i, 0]
            grad[3 * i + 1] += 0.005 * pos[i, 1]

        # Gradient step with clipping
        state -= lr * grad
        # Keep angles in reasonable range
        state[2::3] = state[2::3] % 360

        # Evaluate
        cur_pos = [tuple(p) for p in state.reshape(N, 3)]
        cur_side = _min_enclosing_side(cur_pos, (0.0, 0.0), outer_angle)

        if cur_side < best_side:
            best_side = cur_side
            best_state = state.copy()

        # Adaptive learning rate
        if it % 100 == 0:
            lr *= 0.85

    return [tuple(p) for p in best_state.reshape(N, 3)], best_side


def _scipy_optimize(positions, outer_angle, time_budget):
    """Use scipy L-BFGS-B with smooth SDF-based overlap and containment penalties."""
    init_side = _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    x0 = np.concatenate([np.array([p for pos in positions for p in pos]), [init_side]])

    oa_rad = math.radians(outer_angle)
    # Precompute outer hexagon edge normals in world frame
    outer_normals = np.empty((6, 2))
    for k in range(6):
        a_local = math.radians(30 + 60 * k)
        a_world = a_local + oa_rad
        outer_normals[k, 0] = math.cos(a_world)
        outer_normals[k, 1] = math.sin(a_world)
    
    # Precompute inner hexagon edge normals (local frame, before rotation)
    inner_normals_local = np.empty((6, 2))
    for k in range(6):
        a = math.radians(30 + 60 * k)
        inner_normals_local[k, 0] = math.cos(a)
        inner_normals_local[k, 1] = math.sin(a)
    
    # Precompute inner hexagon vertices in local frame (unit hexagon)
    inner_verts_local = np.empty((6, 2))
    for k in range(6):
        a = k * math.pi / 3
        inner_verts_local[k, 0] = math.cos(a)
        inner_verts_local[k, 1] = math.sin(a)

    temp = 0.05  # smooth max temperature

    def objective(x):
        pos = x[:3 * N].reshape(N, 3)
        S = x[3 * N]
        
        # Overlap penalty using smooth SDF
        overlap_penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                # Quick reject: if centers are far apart, skip
                dx = pos[j, 0] - pos[i, 0]
                dy = pos[j, 1] - pos[i, 1]
                dist_sq = dx * dx + dy * dy
                if dist_sq > 16.0:  # > 4 apart, can't overlap
                    continue
                op = _hex_overlap_penalty_smooth(
                    pos[i, 0], pos[i, 1], pos[i, 2],
                    pos[j, 0], pos[j, 1], pos[j, 2], temp
                )
                overlap_penalty += max(0.0, op) ** 2

        # Containment penalty using smooth max of half-plane violations
        cont_penalty = 0.0
        limit = S * math.sqrt(3) / 2.0
        ca = math.cos(-oa_rad)
        sa = math.sin(-oa_rad)
        
        for i in range(N):
            angle_rad_i = math.radians(pos[i, 2])
            ci = math.cos(angle_rad_i)
            si = math.sin(angle_rad_i)
            
            for k in range(6):
                # Vertex k of inner hex i in world frame
                lvx = inner_verts_local[k, 0]
                lvy = inner_verts_local[k, 1]
                wx = pos[i, 0] + lvx * ci - lvy * si
                wy = pos[i, 1] + lvx * si + lvy * ci
                
                # Transform to outer hex frame
                dx = wx
                dy = wy
                lx = dx * ca - dy * sa
                ly = dx * sa + dy * ca
                
                # Half-plane violations
                for m in range(6):
                    proj = lx * outer_normals[m, 0] + ly * outer_normals[m, 1]
                    violation = proj - limit
                    if violation > 0:
                        cont_penalty += violation ** 2

        return S + 2.0 * overlap_penalty + 10.0 * cont_penalty

    def jac(x):
        """Finite difference Jacobian for L-BFGS-B."""
        n = len(x)
        g = np.zeros(n)
        eps = 1e-5
        f0 = objective(x)
        for i in range(n):
            if _remaining_time() < 5.0:
                break
            x[i] += eps
            f_p = objective(x)
            x[i] -= 2 * eps
            f_m = objective(x)
            x[i] += eps
            g[i] = (f_p - f_m) / (2 * eps)
        return g

    result = minimize(
        objective,
        x0,
        jac=jac,
        method='L-BFGS-B',
        options={'maxiter': 200, 'maxfun': 5000, 'ftol': 1e-14, 'gtol': 1e-8}
    )

    best_x = result.x
    best_pos = best_x[:3 * N].reshape(N, 3)
    best_pos[:, 2] = best_pos[:, 2] % 360

    true_S = _min_enclosing_side([tuple(p) for p in best_pos], (0.0, 0.0), outer_angle)
    return [tuple(p) for p in best_pos], true_S


def _initial_lattice():
    """Hexagonal lattice: center + 6 neighbors + 4 more in second ring."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # First ring: 6 hexagons at distance 2*sqrt(3)
    dist1 = 2.0 * math.sqrt(3)
    for k in range(6):
        a = math.radians(60 * k)
        positions.append([dist1 * math.cos(a), dist1 * math.sin(a), 0.0])
    # Second ring: 4 more at distance 2*sqrt(3)*sqrt(3) = 6
    dist2 = 6.0
    for k in range(4):
        a = math.radians(30 + 60 * k)
        positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
    return positions


def _initial_compact():
    """Compact arrangement: center + tight ring + partial second ring."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # Tight first ring
    dist1 = 2.0 * math.sqrt(3) - 0.1  # slightly tighter
    for k in range(6):
        a = math.radians(60 * k)
        positions.append([dist1 * math.cos(a), dist1 * math.sin(a), 0.0])
    # Second ring: 4 hexagons
    dist2 = dist1 * math.sqrt(3)
    for k in range(4):
        a = math.radians(30 + 60 * k)
        positions.append([dist2 * math.cos(a), dist2 * math.sin(a), 0.0])
    return positions


def _initial_circular():
    """Center + ring of 10 at various radii."""
    positions = []
    positions.append([0.0, 0.0, 0.0])
    # 6 in inner ring
    r1 = 3.4
    for i in range(6):
        a = 2 * math.pi * i / 6
        positions.append([r1 * math.cos(a), r1 * math.sin(a), 0.0])
    # 4 in outer ring
    r2 = 5.5
    for i in range(4):
        a = 2 * math.pi * (i + 0.5) / 4
        positions.append([r2 * math.cos(a), r2 * math.sin(a), 0.0])
    return positions


def _initial_random():
    positions = []
    positions.append([0.0, 0.0, 0.0])
    for i in range(1, N):
        r = random.uniform(2.0, 5.5)
        a = random.uniform(0, 2 * math.pi)
        angle = random.uniform(0, 60)
        positions.append([r * math.cos(a), r * math.sin(a), angle])
    return positions


def _polish(positions, outer_angle, time_budget=10.0):
    """High-precision polish pass using scipy with tighter tolerances."""
    if _remaining_time() < time_budget:
        return positions, _min_enclosing_side(positions, (0.0, 0.0), outer_angle)
    
    try:
        polished, side = _scipy_optimize(positions, outer_angle, time_budget)
        return polished, side
    except Exception:
        return positions, _min_enclosing_side(positions, (0.0, 0.0), outer_angle)


def _try_arrangement(positions, outer_angle):
    if _remaining_time() < 10.0:
        return None, float('inf')

    best_pos = None
    best_side = float('inf')

    # Try SDF gradient descent optimization
    try:
        sdf_pos, sdf_side = _sdf_optimize(positions, outer_angle)
        if sdf_side < best_side:
            best_side = sdf_side
            best_pos = sdf_pos
    except Exception:
        pass

    # Try scipy optimization from the best SDF result or initial
    start_pos = best_pos if best_pos is not None else positions
    try:
        sci_pos, sci_side = _scipy_optimize(start_pos, outer_angle, 8.0)
        if sci_side < best_side:
            best_side = sci_side
            best_pos = sci_pos
    except Exception:
        pass

    if best_pos is None:
        return None, float('inf')

    # Polish pass
    if _remaining_time() > 15.0:
        try:
            pol_pos, pol_side = _polish(best_pos, outer_angle, 10.0)
            if pol_side < best_side:
                best_side = pol_side
                best_pos = pol_pos
        except Exception:
            pass

    side = _min_enclosing_side(best_pos, (0.0, 0.0), outer_angle)

    if _check_validity(best_pos, (0.0, 0.0), side, outer_angle, tol=1e-4):
        return best_pos, side

    return None, float('inf')


def construct():
    random.seed(42)
    np.random.seed(42)

    best_result = None
    best_score = float('inf')

    outer_angles = [0.0, 30.0]

    initial_configs = [
        _initial_lattice(),
        _initial_compact(),
        _initial_circular(),
        _initial_random(),
        _initial_random(),
    ]

    for oa in outer_angles:
        if _remaining_time() < 25.0:
            break

        for init_pos in initial_configs:
            if _remaining_time() < 15.0:
                break

            positions = [list(p) for p in init_pos]
            result, side = _try_arrangement(positions, oa)

            if result is not None and side < best_score:
                best_score = side
                best_result = (result, (0.0, 0.0), side, oa)

    if best_result is not None:
        return best_result

    # Fallback
    positions = _initial_lattice()
    side = _min_enclosing_side(positions, (0.0, 0.0), 0.0)
    return positions, (0.0, 0.0), side, 0.0
