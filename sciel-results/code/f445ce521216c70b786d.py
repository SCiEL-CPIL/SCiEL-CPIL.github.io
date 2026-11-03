import math
import time
import random
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 12
DEADLINE = 143.0  # Safety margin before 150s wall-clock kill
SQRT3 = math.sqrt(3)

# Precompute unit hexagon vertices and edge normals (angle 0)
_HEX_ANGLES = np.radians(np.arange(0, 360, 60))
_UNIT_HEX_VERTS = np.column_stack((np.cos(_HEX_ANGLES), np.sin(_HEX_ANGLES)))
_EDGE_NORMALS = np.zeros((6, 2))
for _i in range(6):
    _ex = _UNIT_HEX_VERTS[(_i+1)%6, 0] - _UNIT_HEX_VERTS[_i, 0]
    _ey = _UNIT_HEX_VERTS[(_i+1)%6, 1] - _UNIT_HEX_VERTS[_i, 1]
    _nx, _ny = -_ey, _ex
    _norm = math.sqrt(_nx*_nx + _ny*_ny)
    _EDGE_NORMALS[_i] = [_nx / _norm, _ny / _norm]

def construct():
    t0 = time.time()
    sqrt3 = SQRT3
    _ux = _UNIT_HEX_VERTS[:, 0]
    _uy = _UNIT_HEX_VERTS[:, 1]
    _enx = _EDGE_NORMALS[:, 0]
    _eny = _EDGE_NORMALS[:, 1]

    # --- Core geometry helpers ---
    def get_all_vertices(positions, angles_rad):
        c = np.cos(angles_rad)
        s = np.sin(angles_rad)
        rx = _ux[None, :] * c[:, None] - _uy[None, :] * s[:, None]
        ry = _ux[None, :] * s[:, None] + _uy[None, :] * c[:, None]
        verts_x = rx + positions[:, 0:1]
        verts_y = ry + positions[:, 1:2]
        return np.stack((verts_x, verts_y), axis=-1)

    def hexagons_disjoint_fast(h1, h2):
        dx = h1[0] - h2[0]
        dy = h1[1] - h2[1]
        dist_sq = dx*dx + dy*dy
        if dist_sq > 4.5:
            return True
        a1 = h1[2]
        a2 = h2[2]
        c1, s1 = math.cos(a1), math.sin(a1)
        c2, s2 = math.cos(a2), math.sin(a2)
        v1x = h1[0] + _ux * c1 - _uy * s1
        v1y = h1[1] + _ux * s1 + _uy * c1
        v2x = h2[0] + _ux * c2 - _uy * s2
        v2y = h2[1] + _ux * s2 + _uy * c2
        en1x = _enx * c1 - _eny * s1
        en1y = _enx * s1 + _eny * c1
        en2x = _enx * c2 - _eny * s2
        en2y = _enx * s2 + _eny * c2
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

    def compute_min_outer_side(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)

        def eval_phi(phi_deg):
            phi = np.radians(phi_deg)
            na = phi + np.pi/6 + np.arange(6) * np.pi/3
            nxx = np.cos(na)
            nyy = np.sin(na)
            pr = offsets[:, 0:1] * nxx[None, :] + offsets[:, 1:2] * nyy[None, :]
            return (2.0 / sqrt3) * np.max(np.abs(pr), axis=0).max()

        try:
            lo = max(0.0, best_phi - 1.0)
            hi = min(60.0, best_phi + 1.0)
            res = minimize_scalar(eval_phi, bounds=(lo, hi), method='bounded',
                                  options={'xatol': 1e-8})
            if res.success and res.fun < best_side:
                best_side = res.fun
                best_phi = res.x
        except Exception:
            pass

        return best_side, best_phi, center[0], center[1]

    def fast_side_length(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        all_verts = get_all_vertices(positions, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        phi_vals = np.radians(np.arange(0, 60, 1))
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)
        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)
        max_projs = np.max(abs_proj, axis=0)
        side_vals = (2.0 / sqrt3) * max_projs
        side_per_phi = np.max(side_vals, axis=1)

        best_idx = np.argmin(side_per_phi)
        best_side = side_per_phi[best_idx]
        best_phi = float(best_idx)
        return best_side, best_phi, center[0], center[1]

    def is_valid_configuration(positions, angles_deg):
        angles_rad = np.radians(angles_deg)
        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i][0], positions[i][1], angles_rad[i])
                h2 = (positions[j][0], positions[j][1], angles_rad[j])
                if not hexagons_disjoint_fast(h1, h2):
                    return False
        return True

    # --- Cluster morphing representation ---
    # Base lattice: honeycomb positions for 12 hexagons
    a1 = np.array([sqrt3, 0.0])
    a2 = np.array([sqrt3/2.0, 1.5])
    lattice_coords = [
        (0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
        (1, -1), (-1, 1), (2, -1), (-2, 1),
        (1, 1), (-1, -1), (2, 0),
    ]
    base_positions = np.zeros((N, 2))
    for idx, (i, j) in enumerate(lattice_coords):
        base_positions[idx] = i * a1 + j * a2

    # Deformation field: each hexagon gets a radial displacement parameter
    # params = [r0, r1, ..., r11, theta0, theta1, ..., theta11, scale, rotation]
    # Total: 12 + 12 + 1 + 1 = 26 parameters
    # r_i: radial perturbation magnitude (positive = outward, negative = inward)
    # theta_i: angular perturbation (degrees)
    # scale: global scaling factor
    # rotation: global rotation of the entire cluster (degrees)

    def decode_params(params):
        """Convert deformation parameters to (positions, angles_deg)."""
        r = params[:N]
        theta = params[N:2*N]
        scale = params[2*N]
        rotation = params[2*N + 1]

        # Apply global scaling and rotation to base positions
        rot_rad = np.radians(rotation)
        cr, sr = math.cos(rot_rad), math.sin(rot_rad)
        rot_matrix = np.array([[cr, -sr], [sr, cr]])

        pos = np.zeros((N, 2))
        ang = np.zeros(N)

        for i in range(N):
            # Base position scaled and rotated
            bp = base_positions[i] * scale
            bp = rot_matrix @ bp

            # Radial perturbation: displace along direction from cluster center
            center = np.zeros(2)
            # Direction from center to this hexagon
            dist = np.linalg.norm(bp)
            if dist > 1e-10:
                direction = bp / dist
            else:
                # Use the theta parameter for center hexagon
                direction = np.array([math.cos(np.radians(theta[i])), math.sin(np.radians(theta[i]))])

            # Add radial perturbation
            pos[i] = bp + r[i] * direction

            # Angle: base angle 0 + individual angular perturbation
            ang[i] = theta[i] * 0.1  # Small angular perturbations

        return pos, ang

    def smooth_collision_potential(positions, angles_deg, beta=30.0):
        """
        Differentiable collision potential using soft max over separating axes.
        Returns (penalty, gradient_wrt_positions, gradient_wrt_angles).
        """
        angles_rad = np.radians(angles_deg)
        penalty = 0.0
        grad_pos = np.zeros_like(positions)
        grad_ang = np.zeros_like(angles_deg)

        for i in range(N):
            for j in range(i+1, N):
                h1 = (positions[i, 0], positions[i, 1], angles_rad[i])
                h2 = (positions[j, 0], positions[j, 1], angles_rad[j])

                # Quick distance check
                dx = h1[0] - h2[0]
                dy = h1[1] - h2[1]
                dist_sq = dx*dx + dy*dy
                if dist_sq > 6.0:
                    continue

                c1, s1 = math.cos(h1[2]), math.sin(h1[2])
                c2, s2 = math.cos(h2[2]), math.sin(h2[2])

                v1x = h1[0] + _ux * c1 - _uy * s1
                v1y = h1[1] + _ux * s1 + _uy * c1
                v2x = h2[0] + _ux * c2 - _uy * s2
                v2y = h2[1] + _ux * s2 + _uy * c2

                en1x = _enx * c1 - _eny * s1
                en1y = _enx * s1 + _eny * c1
                en2x = _enx * c2 - _eny * s2
                en2y = _enx * s2 + _eny * c2

                # Collect all separating axis violations
                violations = []
                for i_ax in range(6):
                    nx, ny = en1x[i_ax], en1y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    # Violation: p1.max() >= p2.min() means overlap
                    viol = p1.max() - p2.min()
                    if viol > 0:
                        violations.append((viol, 'pos1', i_ax, nx, ny, c1, s1))
                for i_ax in range(6):
                    nx, ny = en2x[i_ax], en2y[i_ax]
                    p1 = v1x * nx + v1y * ny
                    p2 = v2x * nx + v2y * ny
                    viol = p2.max() - p1.min()
                    if viol > 0:
                        violations.append((viol, 'pos2', i_ax, nx, ny, c2, s2))

                if not violations:
                    continue

                # Soft max of violations
                viol_vals = np.array([v[0] for v in violations])
                max_viol = viol_vals.max()
                if max_viol < 1e-10:
                    continue

                # Soft max gradient
                exp_terms = np.exp(beta * (viol_vals - max_viol))
                weights = exp_terms / exp_terms.sum()

                for w, (viol, src, ax_idx, nx, ny, c, s) in zip(weights, violations):
                    if w < 1e-8:
                        continue
                    penalty += w * viol

                    # Gradient contribution
                    if src == 'pos1':
                        # p1.max() - p2.min(): gradient wrt pos1 is +nx, +ny
                        grad_pos[i, 0] += w * nx
                        grad_pos[i, 1] += w * ny
                        # Gradient wrt angle1: d(p1.max())/d(a1)
                        # p1 = v1x*nx + v1y*ny
                        # dv1x/da = -_ux*s - _uy*c (from rotation)
                        # Actually, for the max projection, the gradient flows through
                        # the vertex that achieves the max. Approximate with average.
                        dp1_da = (-_ux * s * nx + _ux * c * ny) * 0.1  # rough
                        grad_ang[i] += w * np.mean(dp1_da) * 0.01
                    else:
                        # p2.max() - p1.min(): gradient wrt pos2 is -nx, -ny
                        grad_pos[j, 0] -= w * nx
                        grad_pos[j, 1] -= w * ny
                        dp2_da = (-_ux * s * nx + _ux * c * ny) * 0.1
                        grad_ang[j] -= w * np.mean(dp2_da) * 0.01

        return penalty, grad_pos, grad_ang

    def smooth_objective(params, beta=30.0):
        """
        Smooth objective: outer hexagon side length + collision penalty.
        Uses log-sum-exp for smooth max over projection directions.
        """
        pos, ang = decode_params(params)
        angles_rad = np.radians(ang)
        all_verts = get_all_vertices(pos, angles_rad)
        verts_flat = all_verts.reshape(-1, 2)
        center = np.mean(verts_flat, axis=0)
        offsets = verts_flat - center

        # Scan over outer hexagon orientations
        phi_vals = np.radians(np.arange(0, 60, 2))  # 30 angles
        normal_angles = phi_vals[:, None] + np.pi/6 + np.arange(6) * np.pi/3
        nx = np.cos(normal_angles)
        ny = np.sin(normal_angles)

        proj = offsets[:, None, 0:1] * nx[None, :, :] + offsets[:, None, 1:2] * ny[None, :, :]
        abs_proj = np.abs(proj)

        # Soft max over vertices for each normal direction
        max_abs = np.max(abs_proj, axis=0, keepdims=True)
        exp_term = np.exp(beta * (abs_proj - max_abs))
        soft_max = max_abs + (1.0 / beta) * np.log(np.sum(exp_term, axis=0))

        # Soft max over the 6 normal directions for each phi
        max_soft = np.max(soft_max, axis=1, keepdims=True)
        exp_term2 = np.exp(beta * (soft_max - max_soft))
        soft_max_phi = max_soft + (1.0 / beta) * np.log(np.sum(exp_term2, axis=1))

        # Soft min over phi values
        neg_x = -soft_max_phi
        max_neg_x = np.max(neg_x)
        exp_neg = np.exp(beta * (neg_x - max_neg_x))
        soft_min_val = -(max_neg_x + (1.0/beta) * np.log(np.sum(exp_neg)))

        side = (2.0 / sqrt3) * soft_min_val

        # Collision penalty
        coll_pen, _, _ = smooth_collision_potential(pos, ang, beta)
        penalty_weight = 100.0
        total = side + penalty_weight * coll_pen

        return total

    def compute_gradient(params, beta=30.0, eps=1e-5):
        """Finite difference gradient for the smooth objective."""
        grad = np.zeros_like(params)
        f0 = smooth_objective(params, beta)
        for k in range(len(params)):
            params_p = params.copy()
            params_p[k] += eps
            f_p = smooth_objective(params_p, beta)
            params_p[k] -= 2*eps
            f_m = smooth_objective(params_p, beta)
            params_p[k] += eps
            grad[k] = (f_p - f_m) / (2*eps)
        return grad

    # --- Initial configurations ---
    def make_honeycomb():
        pos = base_positions.copy()
        ang = np.zeros(N)
        return pos, ang

    def make_ring_config():
        pos = np.zeros((N, 2))
        ang = np.zeros(N)
        for i in range(6):
            theta = i * np.pi / 3
            pos[i] = [2 * sqrt3 * math.cos(theta), 2 * sqrt3 * math.sin(theta)]
        for i in range(6):
            theta = (i + 0.5) * np.pi / 3
            pos[i + 6] = [4 * sqrt3 * math.cos(theta), 4 * sqrt3 * math.sin(theta)]
        return pos, ang

    def make_tight_cluster():
        pos = base_positions + np.random.randn(N, 2) * 0.3
        ang = np.random.randn(N) * 5.0
        return pos, ang

    init_configs = [make_honeycomb(), make_ring_config(), make_tight_cluster()]

    best_side = float('inf')
    best_positions = None
    best_angles = None
    best_phi = 0.0
    best_cx = 0.0
    best_cy = 0.0

    for pos0, ang0 in init_configs:
        if is_valid_configuration(pos0, ang0):
            s, phi, cx, cy = compute_min_outer_side(pos0, ang0)
            if s < best_side:
                best_side = s
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_phi = phi
                best_cx = cx
                best_cy = cy

    if best_positions is None:
        best_positions, best_angles = make_honeycomb()
        best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # --- Phase 1: Gradient-based cluster morphing ---
    # Start from honeycomb, encode as deformation params
    # Base params: zero radial, zero theta, scale=1, rotation=0
    init_params = np.zeros(2*N + 2)
    init_params[2*N] = 1.0  # scale

    # Try multiple betas for the smooth surrogate
    for beta in [15.0, 30.0, 60.0]:
        if time.time() - t0 > DEADLINE - 15:
            break

        try:
            res = minimize(smooth_objective, init_params, args=(beta,),
                           method='L-BFGS-B',
                           jac=lambda p, b=beta: compute_gradient(p, b),
                           options={'maxiter': 3000, 'ftol': 1e-12, 'gtol': 1e-10})

            if res.fun < best_side + 1.0 and res.fun < 200:
                new_pos, new_ang = decode_params(res.x)
                if is_valid_configuration(new_pos, new_ang):
                    new_side, new_phi, new_cx, new_cy = compute_min_outer_side(new_pos, new_ang)
                    if new_side < best_side:
                        best_side = new_side
                        best_positions = new_pos
                        best_angles = new_ang
                        best_phi = new_phi
                        best_cx = new_cx
                        best_cy = new_cy
                        init_params = res.x.copy()
        except Exception:
            pass

    # --- Phase 2: Local search with coordinate descent on deformation params ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 5.0:
        # Decode current best back to param space approximately
        # This is an approximation - we search in (pos, ang) space directly
        current_pos = best_positions.copy()
        current_ang = best_angles.copy()
        current_side, _, _, _ = fast_side_length(current_pos, current_ang)

        step = 0.05
        min_step = 0.001
        t_start = time.time()
        time_limit = min(remaining_time * 0.5, 30.0)

        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                if time.time() - t_start > time_limit:
                    break
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = current_pos.copy()
                        new_ang = current_ang.copy()

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
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break

                        if not valid:
                            continue

                        new_side, _, _, _ = fast_side_length(new_pos, new_ang)

                        if new_side < current_side - 1e-12:
                            current_pos = new_pos
                            current_ang = new_ang
                            current_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            if not improved:
                step *= 0.6

        if current_side < best_side:
            best_side = current_side
            best_positions = current_pos
            best_angles = current_ang
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    # --- Phase 3: Final high-precision polish ---
    remaining_time = DEADLINE - (time.time() - t0)
    if remaining_time > 3.0:
        step = 0.01
        min_step = 0.0001
        t_start = time.time()
        time_limit = min(remaining_time * 0.7, 20.0)

        while time.time() - t_start < time_limit and step > min_step:
            improved = False
            for idx in range(N):
                if time.time() - t_start > time_limit:
                    break
                for dim in range(3):
                    for direction in [1, -1]:
                        new_pos = best_positions.copy()
                        new_ang = best_angles.copy()

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
                            if not hexagons_disjoint_fast(h_moved, h_other):
                                valid = False
                                break

                        if not valid:
                            continue

                        new_side, _, _, _ = compute_min_outer_side(new_pos, new_ang)

                        if new_side < best_side - 1e-12:
                            best_positions = new_pos
                            best_angles = new_ang
                            best_side = new_side
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    continue
            if not improved:
                step *= 0.5

    # Final verification
    if not is_valid_configuration(best_positions, best_angles):
        for pos0, ang0 in init_configs:
            if is_valid_configuration(pos0, ang0):
                best_positions = pos0.copy()
                best_angles = ang0.copy()
                best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)
                break
        else:
            best_positions, best_angles = make_honeycomb()
            best_side, best_phi, best_cx, best_cy = compute_min_outer_side(best_positions, best_angles)

    inner = []
    for i in range(N):
        inner.append([float(best_positions[i, 0]), float(best_positions[i, 1]), float(best_angles[i])])

    center = (float(best_cx), float(best_cy))
    side = float(best_side)
    angle = float(best_phi)

    return inner, center, side, angle
