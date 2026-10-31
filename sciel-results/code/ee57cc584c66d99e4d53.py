"""Circle packing in unit square - 26 circles maximizing sum of radii.

Structural approach: parameterize the packing by a small set of global parameters
(corner/edge radii, center ring, outer ring) and optimize that low-dimensional
vector with L-BFGS-B.  The 26 circle centers and radii are then derived from
the parameters, giving a much smaller and better-conditioned search space than
optimizing all 78 variables directly.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 26

# EVOLVE-BLOCK-START


def _is_valid(pos, radii, tol=1e-9):
    """Check validity: circles inside [0,1]^2 with disjoint interiors."""
    for i in range(len(radii)):
        if pos[i, 0] - radii[i] < -tol or pos[i, 0] + radii[i] > 1.0 + tol:
            return False
        if pos[i, 1] - radii[i] < -tol or pos[i, 1] + radii[i] > 1.0 + tol:
            return False
    for i in range(len(radii)):
        for j in range(i + 1, len(radii)):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < radii[i] + radii[j] - tol:
                return False
    return True


def _build_from_params(p):
    """
    Build 26 circle positions and radii from a 10-element parameter vector.

    Parameters:
      p[0]  corner_r   – radius of the 4 corner circles
      p[1]  edge_r     – radius of the 4 edge-midpoint circles
      p[2]  center_r   – radius of the central circle
      p[3]  cx         – x of central circle
      p[4]  cy         – y of central circle
      p[5]  ring1_off  – radial offset of the 8 inner-ring circles
      p[6]  ring1_r    – radius of the 8 inner-ring circles
      p[7]  ring2_off  – radial offset of the 6 outer-ring circles
      p[8]  ring2_r    – radius of the 6 outer-ring circles
      p[9]  ring1_ang  – angular offset for inner ring (rotation)

    Returns (pos, radii) where pos is (26,2) and radii is (26,).
    """
    corner_r = p[0]
    edge_r = p[1]
    center_r = p[2]
    cx = p[3]
    cy = p[4]
    ring1_off = p[5]
    ring1_r = p[6]
    ring2_off = p[7]
    ring2_r = p[8]
    ring1_ang = p[9]

    pos = np.zeros((N, 2))
    radii = np.zeros(N)

    # 4 corner circles
    corner_positions = np.array([
        [corner_r, corner_r],
        [1.0 - corner_r, corner_r],
        [corner_r, 1.0 - corner_r],
        [1.0 - corner_r, 1.0 - corner_r],
    ])
    for i in range(4):
        pos[i] = corner_positions[i]
        radii[i] = corner_r

    # 4 edge-midpoint circles
    edge_positions = np.array([
        [0.5, edge_r],
        [0.5, 1.0 - edge_r],
        [edge_r, 0.5],
        [1.0 - edge_r, 0.5],
    ])
    for i in range(4):
        pos[4 + i] = edge_positions[i]
        radii[4 + i] = edge_r

    # 1 central circle
    pos[8] = [cx, cy]
    radii[8] = center_r

    # 8 inner-ring circles
    for i in range(8):
        angle = ring1_ang + 2.0 * np.pi * i / 8.0
        x = cx + ring1_off * np.cos(angle)
        y = cy + ring1_off * np.sin(angle)
        pos[9 + i] = [x, y]
        radii[9 + i] = ring1_r

    # 6 outer-ring circles
    for i in range(6):
        angle = 2.0 * np.pi * i / 6.0
        x = cx + ring2_off * np.cos(angle)
        y = cy + ring2_off * np.sin(angle)
        pos[17 + i] = [x, y]
        radii[17 + i] = ring2_r

    return pos, radii


def _objective_struct(p, deadline):
    """
    Objective: minimize negative sum of radii plus penalty for constraint violations.
    Uses a smooth penalty: sum of (max(0, violation))^2.
    """
    if time.time() > deadline:
        return 1e10

    pos, radii = _build_from_params(p)

    # Sum of radii (to maximize)
    total_r = np.sum(radii)

    # Boundary violations
    boundary_viol = 0.0
    for i in range(N):
        boundary_viol += max(0.0, radii[i] - pos[i, 0]) ** 2
        boundary_viol += max(0.0, pos[i, 0] + radii[i] - 1.0) ** 2
        boundary_viol += max(0.0, radii[i] - pos[i, 1]) ** 2
        boundary_viol += max(0.0, pos[i, 1] + radii[i] - 1.0) ** 2

    # Overlap violations
    overlap_viol = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist) ** 2

    penalty = 100.0 * (boundary_viol + overlap_viol)
    return -total_r + penalty


def _post_process(pos, radii, deadline):
    """Iteratively shrink circles to ensure validity."""
    pos = pos.copy()
    radii = radii.copy()
    n = N

    for _ in range(200):
        if time.time() > deadline:
            break
        valid = True
        # Boundary fix
        for i in range(n):
            max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
            if radii[i] > max_r:
                radii[i] = max(1e-6, max_r * 0.999)
                valid = False
        # Overlap fix
        for i in range(n):
            for j in range(i + 1, n):
                dx = pos[i, 0] - pos[j, 0]
                dy = pos[i, 1] - pos[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = radii[i] + radii[j]
                if dist < min_dist:
                    if radii[i] <= radii[j]:
                        ratio = (min_dist - dist) / (2.0 * radii[i] + 1e-12)
                        radii[i] *= max(0.5, 1.0 - ratio * 0.999)
                    else:
                        ratio = (min_dist - dist) / (2.0 * radii[j] + 1e-12)
                        radii[j] *= max(0.5, 1.0 - ratio * 0.999)
                    valid = False
        if valid:
            break

    for i in range(n):
        max_r = min(pos[i, 0], 1.0 - pos[i, 0], pos[i, 1], 1.0 - pos[i, 1])
        radii[i] = max(1e-6, min(radii[i], max_r))
    return pos, radii


def _direct_objective(x, deadline):
    """
    Objective for direct 78-dim optimization: x = [x_1,...,x_26, y_1,...,y_26, r_1,...,r_26].
    Minimize -sum(r_i) + penalty for violations.
    """
    if time.time() > deadline:
        return 1e10
    n = N
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[:, 0] = x[:n]
    pos[:, 1] = x[n:2*n]
    radii = x[2*n:3*n]

    total_r = np.sum(radii)

    # Boundary violations (smooth)
    boundary_viol = 0.0
    for i in range(n):
        boundary_viol += max(0.0, radii[i] - pos[i, 0]) ** 2
        boundary_viol += max(0.0, pos[i, 0] + radii[i] - 1.0) ** 2
        boundary_viol += max(0.0, radii[i] - pos[i, 1]) ** 2
        boundary_viol += max(0.0, pos[i, 1] + radii[i] - 1.0) ** 2

    # Overlap violations
    overlap_viol = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[i, 0] - pos[j, 0]
            dy = pos[i, 1] - pos[j, 1]
            dist = np.sqrt(dx * dx + dy * dy)
            min_dist = radii[i] + radii[j]
            if dist < min_dist:
                overlap_viol += (min_dist - dist) ** 2

    penalty = 50.0 * (boundary_viol + overlap_viol)
    return -total_r + penalty


def _slsqp_polish(pos, radii, deadline):
    """SLSQP with hard constraints for final polish."""
    pos = pos.copy()
    radii = radii.copy()
    x0 = np.concatenate([pos[:, 0], pos[:, 1], radii])
    n = N

    bounds = []
    for i in range(n):
        bounds.append((0.001, 0.999))  # x
        bounds.append((0.001, 0.999))  # y
    for i in range(n):
        bounds.append((1e-4, 0.35))    # r

    cons = []
    for i in range(n):
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[i] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[i] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: x[n + i] - x[2*n + i]})
        cons.append({'type': 'ineq', 'fun': lambda x, i=i: 1.0 - x[n + i] - x[2*n + i]})
    for i in range(n):
        for j in range(i + 1, n):
            cons.append({'type': 'ineq',
                         'fun': lambda x, i=i, j=j: np.sqrt(
                             (x[i] - x[j])**2 + (x[n+i] - x[n+j])**2
                         ) - x[2*n + i] - x[2*n + j]})

    try:
        result = minimize(lambda x: -np.sum(x[2*n:]), x0, method='SLSQP',
                          bounds=bounds, constraints=cons,
                          options={'maxiter': 500, 'ftol': 1e-14})
        pos_opt = np.column_stack([result.x[:n], result.x[n:2*n]])
        radii_opt = result.x[2*n:]
        pos_pp, radii_pp = _post_process(pos_opt, radii_opt, deadline)
        if _is_valid(pos_pp, radii_pp):
            return pos_pp, radii_pp
    except Exception:
        pass
    return pos, radii


def _make_grid(n):
    """5x5 grid + 1 extra at center."""
    pos = np.zeros((n, 2))
    side = 5
    for i in range(side):
        for j in range(side):
            idx = i * side + j
            if idx < n:
                pos[idx] = [(i + 0.5) / side, (j + 0.5) / side]
    if n > 25:
        pos[25] = [0.5, 0.5]
    # Equal radii based on grid spacing
    r = 0.1
    radii = np.full(n, r)
    return pos, radii


def _make_hexagonal(n):
    """Hexagonal packing initial layout."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    r = 0.1
    rows = 5
    count = 0
    for row in range(rows):
        y = 0.1 + row * r * np.sqrt(3)
        if row % 2 == 0:
            xs = np.arange(5) * 2 * r + r
        else:
            xs = np.arange(4) * 2 * r + 2 * r
        for x in xs:
            if count < n:
                pos[count] = [x, y]
                radii[count] = r
                count += 1
    # Fill remaining
    while count < n:
        pos[count] = [0.5, 0.5 + 0.05 * (count - 24)]
        radii[count] = r
        count += 1
    return pos, radii


def _make_center_ring(n):
    """Center + rings layout."""
    pos = np.zeros((n, 2))
    radii = np.zeros(n)
    pos[0] = [0.5, 0.5]
    radii[0] = 0.12
    count = 1
    # Inner ring: 8
    d1 = 0.25
    for i in range(8):
        if count < n:
            angle = 2 * np.pi * i / 8
            pos[count] = [0.5 + d1 * np.cos(angle), 0.5 + d1 * np.sin(angle)]
            radii[count] = 0.09
            count += 1
    # Outer ring: remaining
    d2 = 0.42
    remaining = n - count
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1)
        pos[count] = [0.5 + d2 * np.cos(angle), 0.5 + d2 * np.sin(angle)]
        radii[count] = 0.07
        count += 1
    return pos, radii


def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # safety margin under 150s

    best_pos = None
    best_radii = None
    best_score = -np.inf

    def update_best(pos, radii):
        nonlocal best_pos, best_radii, best_score
        if time.time() > deadline:
            return
        pos_pp, radii_pp = _post_process(pos, radii, deadline)
        if _is_valid(pos_pp, radii_pp):
            score = np.sum(radii_pp)
            if score > best_score:
                best_score = score
                best_pos = pos_pp.copy()
                best_radii = radii_pp.copy()

    # ---------- Track A: Structural parameter optimization ----------
    struct_bounds = [
        (0.04, 0.28),   # corner_r
        (0.04, 0.25),   # edge_r
        (0.04, 0.22),   # center_r
        (0.30, 0.70),   # cx
        (0.30, 0.70),   # cy
        (0.15, 0.40),   # ring1_off
        (0.03, 0.16),   # ring1_r
        (0.30, 0.50),   # ring2_off
        (0.02, 0.12),   # ring2_r
        (0.0, 0.5),     # ring1_ang
    ]

    struct_seeds = [
        [0.12, 0.10, 0.14, 0.5, 0.5, 0.28, 0.09, 0.42, 0.06, 0.0],
        [0.13, 0.11, 0.13, 0.5, 0.5, 0.30, 0.08, 0.44, 0.05, 0.2],
        [0.11, 0.09, 0.15, 0.5, 0.5, 0.26, 0.10, 0.40, 0.07, 0.1],
        [0.14, 0.12, 0.12, 0.5, 0.5, 0.32, 0.07, 0.46, 0.04, 0.3],
    ]

    for s0 in struct_seeds:
        if time.time() > deadline - 30:
            break
        try:
            res = minimize(_objective_struct, np.array(s0), args=(deadline,),
                           method='L-BFGS-B', bounds=struct_bounds,
                           options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10})
            pos_s, radii_s = _build_from_params(res.x)
            update_best(pos_s, radii_s)
        except Exception:
            pass

    # ---------- Track B: Direct optimization from diverse seeds ----------
    seeds = [
        _make_grid(N),
        _make_hexagonal(N),
        _make_center_ring(N),
    ]
    # Also add struct-derived seeds
    for s0 in struct_seeds[:2]:
        pos_s, radii_s = _build_from_params(s0)
        seeds.append((pos_s, radii_s))

    for pos0, radii0 in seeds:
        if time.time() > deadline - 20:
            break
        pos_pp, radii_pp = _post_process(pos0, radii0, deadline)
        if not _is_valid(pos_pp, radii_pp):
            continue
        x0 = np.concatenate([pos_pp[:, 0], pos_pp[:, 1], radii_pp])
        bounds = []
        for i in range(N):
            bounds.append((0.005, 0.995))
            bounds.append((0.005, 0.995))
        for i in range(N):
            bounds.append((0.005, 0.30))
        try:
            result = minimize(_direct_objective, x0, args=(deadline,),
                              method='L-BFGS-B', bounds=bounds,
                              options={'maxiter': 300, 'ftol': 1e-12})
            pos_opt = np.column_stack([result.x[:N], result.x[N:2*N]])
            radii_opt = result.x[2*N:]
            update_best(pos_opt, radii_opt)
        except Exception:
            pass

    # ---------- Final polish ----------
    if best_pos is not None:
        if time.time() < deadline - 10:
            try:
                pos_p, radii_p = _slsqp_polish(best_pos, best_radii, deadline)
                if _is_valid(pos_p, radii_p):
                    s = np.sum(radii_p)
                    if s > best_score:
                        best_score = s
                        best_pos = pos_p.copy()
                        best_radii = radii_p.copy()
            except Exception:
                pass

        # One more L-BFGS-B pass on the polished result
        if time.time() < deadline - 5:
            try:
                x0 = np.concatenate([best_pos[:, 0], best_pos[:, 1], best_radii])
                bounds = []
                for i in range(N):
                    bounds.append((0.005, 0.995))
                    bounds.append((0.005, 0.995))
                for i in range(N):
                    bounds.append((0.005, 0.30))
                result = minimize(_direct_objective, x0, args=(deadline,),
                                  method='L-BFGS-B', bounds=bounds,
                                  options={'maxiter': 200, 'ftol': 1e-13})
                pos_opt = np.column_stack([result.x[:N], result.x[N:2*N]])
                radii_opt = result.x[2*N:]
                update_best(pos_opt, radii_opt)
            except Exception:
                pass

    # Fallback
    if best_pos is None:
        pos, radii = _make_grid(N)
        best_pos, best_radii = _post_process(pos, radii, deadline)

    return np.column_stack([best_pos, best_radii])


# EVOLVE-BLOCK-END
