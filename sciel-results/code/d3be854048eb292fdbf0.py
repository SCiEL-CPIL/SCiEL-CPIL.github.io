"""Direct construction of 12 unit hexagons inside a regular hexagon, minimizing outer side."""
N = 12

# EVOLVE-BLOCK-START
import numpy as np
from scipy.optimize import minimize
import time
import math

def _hex_verts(cx, cy, side, angle_deg):
    """Vertices of a regular hexagon centered at (cx, cy) with given side and rotation."""
    a = np.radians(angle_deg)
    return np.array([
        [cx + side * np.cos(a + np.radians(60 * i)),
         cy + side * np.sin(a + np.radians(60 * i))]
        for i in range(6)
    ])

def _hex_verts_batch(params, side=1.0):
    """Batch compute vertices for multiple hexagons. params shape (N,3) = (x, y, angle_deg)."""
    n = len(params)
    angles = np.radians(params[:, 2])
    base_angles = np.radians(60 * np.arange(6))
    all_angles = angles[:, np.newaxis] + base_angles[np.newaxis, :]
    cos_a = np.cos(all_angles)
    sin_a = np.sin(all_angles)
    verts = np.stack([
        params[:, 0:1] + side * cos_a,
        params[:, 1:2] + side * sin_a
    ], axis=-1)
    return verts  # (n, 6, 2)

def _containment_penalty_fast(inner_verts, outer_verts):
    """
    Fast containment penalty. inner_verts: (n, 6, 2), outer_verts: (6, 2).
    Returns sum of penetration depths for all inner hex vertices against outer hex edges.
    """
    n_outer = len(outer_verts)
    penalty = 0.0
    for i in range(n_outer):
        v1 = outer_verts[i]
        v2 = outer_verts[(i + 1) % n_outer]
        edge = v2 - v1
        edge_len = np.linalg.norm(edge)
        if edge_len < 1e-12:
            continue
        normal = np.array([-edge[1], edge[0]]) / edge_len
        d = np.sum((inner_verts - v1) * normal, axis=-1)
        neg_d = np.maximum(-d, 0.0)
        penalty += np.sum(neg_d)
    return penalty

def _pairwise_sat_depth_all(inner_verts_list):
    """
    Compute sum of SAT overlap depths for all pairs of inner hexagons.
    inner_verts_list: list of N arrays, each (6, 2).
    Returns total disjointness penalty.
    """
    total = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            h1 = inner_verts_list[i]
            h2 = inner_verts_list[j]
            # Collect 12 edge normals (6 from each)
            axes = []
            for poly in [h1, h2]:
                for k in range(6):
                    p1 = poly[k]
                    p2 = poly[(k + 1) % 6]
                    edge = p2 - p1
                    norm = np.linalg.norm(edge)
                    if norm < 1e-10:
                        continue
                    normal = np.array([-edge[1], edge[0]]) / norm
                    axes.append(normal)
            
            min_depth = float('inf')
            for axis in axes:
                proj1 = h1 @ axis
                proj2 = h2 @ axis
                min1, max1 = proj1.min(), proj1.max()
                min2, max2 = proj2.min(), proj2.max()
                if max1 <= min2 or max2 <= min1:
                    min_depth = 0.0
                    break
                overlap = min(max1, max2) - max(min1, min2)
                if overlap < min_depth:
                    min_depth = overlap
            
            if min_depth > 0 and min_depth < float('inf'):
                total += min_depth
    return total

def _total_penalty(params, outer_side, outer_center, outer_angle):
    """Total penalty: containment + pairwise disjointness."""
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = params[3*i], params[3*i+1], params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)
    
    # Containment
    cont_pen = 0.0
    for iv in inner_verts_list:
        cont_pen += _containment_penalty_fast(np.array([iv]), outer_verts)
    
    # Disjointness
    disj_pen = _pairwise_sat_depth_all(inner_verts_list)
    
    return cont_pen + disj_pen

def _check_valid(params, outer_side, outer_center, outer_angle, tol=1e-6):
    """Check if configuration is valid (no overlaps, all contained)."""
    pen = _total_penalty(params, outer_side, outer_center, outer_angle)
    return pen < tol

def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # safety margin before 150s limit

    def build_hexagonal_lattice(nrows, ncols):
        """Build a hexagonal lattice of positions (offset rows)."""
        positions = []
        for r in range(nrows):
            for c in range(ncols):
                x = c * 2.0 + (1.0 if r % 2 == 1 else 0.0)
                y = r * np.sqrt(3)
                positions.append((x, y))
        return positions

    def build_ring_config():
        """6 around center + 5 more in second ring."""
        positions = [(0.0, 0.0)]
        for i in range(6):
            ang = np.radians(60 * i)
            positions.append((2.0 * np.cos(ang), 2.0 * np.sin(ang)))
        for i in range(6):
            ang = np.radians(60 * i + 30)
            positions.append((3.464 * np.cos(ang), 3.464 * np.sin(ang)))
        return positions[:12]

    def build_2332_config():
        """2-3-3-2 row layout for compact packing."""
        positions = []
        rows = [2, 3, 3, 2]
        y_spacing = np.sqrt(3) * 0.95
        for r, count in enumerate(rows):
            y = (r - 1.5) * y_spacing
            for c in range(count):
                x = (c - (count - 1) / 2.0) * 2.0
                if r % 2 == 1:
                    x += 1.0
                positions.append((x, y))
        return positions

    def make_params_from_positions(positions, angles=None):
        """Create parameter vector from positions and optional angles."""
        if angles is None:
            angles = [0.0] * len(positions)
        params = []
        for i, (x, y) in enumerate(positions):
            params.extend([float(x), float(y), float(angles[i])])
        return np.array(params)

    # Generate candidate initial configurations
    configs = []

    # 3 rows x 4 cols hex grid
    pos_3x4 = build_hexagonal_lattice(3, 4)
    if len(pos_3x4) == N:
        arr = np.array(pos_3x4)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 4 rows x 3 cols hex grid
    pos_4x3 = build_hexagonal_lattice(4, 3)
    if len(pos_4x3) == N:
        arr = np.array(pos_4x3)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 2 rows x 6 cols
    pos_2x6 = build_hexagonal_lattice(2, 6)
    if len(pos_2x6) == N:
        arr = np.array(pos_2x6)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # Ring configuration
    ring_pos = build_ring_config()
    if len(ring_pos) == N:
        arr = np.array(ring_pos)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 6 rows x 2 cols
    pos_6x2 = build_hexagonal_lattice(6, 2)
    if len(pos_6x2) == N:
        arr = np.array(pos_6x2)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # 12 in a line
    line_pos = [(float(i - 5.5), 0.0) for i in range(12)]
    configs.append(line_pos)

    # 2-3-3-2 compact layout
    pos_2332 = build_2332_config()
    if len(pos_2332) == N:
        arr = np.array(pos_2332)
        centroid = arr.mean(axis=0)
        configs.append((arr - centroid).tolist())

    # Rotated 3x4 lattice (30 degree rotation of positions)
    if len(configs) > 0:
        base = np.array(configs[0])
        rot_angle = np.radians(30)
        rot_mat = np.array([[np.cos(rot_angle), -np.sin(rot_angle)],
                            [np.sin(rot_angle), np.cos(rot_angle)]])
        rotated = base @ rot_mat.T
        centroid = rotated.mean(axis=0)
        configs.append((rotated - centroid).tolist())

    # Additional configurations with different angles
    for base_config in configs[:4]:
        for ang_offset in [30.0, 60.0]:
            angled = []
            for (x, y) in base_config:
                angled.append((x, y, ang_offset))
            configs.append(angled)

    best_score = float('inf')
    best_result = None

    outer_angles = [0.0, 15.0, 30.0, 45.0]

    for outer_angle in outer_angles:
        if time.time() > deadline - 20.0:
            break

        outer_center = (0.0, 0.0)

        # Binary search on outer side
        lo, hi = 3.0, 6.0

        # First, find a feasible upper bound
        for test_side in [5.0, 4.5, 4.0, 3.8, 3.6, 3.4, 3.2, 3.0]:
            if time.time() > deadline - 25.0:
                break

            found = False
            for config in configs:
                if time.time() > deadline - 22.0:
                    break

                if len(config[0]) == 2:
                    init_params = make_params_from_positions(config)
                else:
                    init_params = np.array([item for tri in config for item in tri])

                # Quick check
                pen = _total_penalty(init_params, test_side, outer_center, outer_angle)

                if pen < 1e-6:
                    found = True
                    if test_side < best_score:
                        best_score = test_side
                        best_result = (init_params.copy(), outer_center, test_side, outer_angle)
                    break

                # Try local optimization
                if time.time() > deadline - 18.0:
                    continue

                def objective(p):
                    return _total_penalty(p, test_side, outer_center, outer_angle)

                try:
                    result = minimize(
                        objective,
                        init_params,
                        method='L-BFGS-B',
                        options={'maxiter': 200, 'maxfun': 6000, 'ftol': 1e-14, 'gtol': 1e-10}
                    )
                    opt_params = result.x
                    opt_val = result.fun
                except Exception:
                    continue

                if opt_val < 1e-6:
                    found = True
                    if test_side < best_score:
                        best_score = test_side
                        best_result = (opt_params, outer_center, test_side, outer_angle)
                    break

            if found:
                hi = test_side + 0.1
            else:
                lo = test_side - 0.1
                if lo <= 3.0:
                    break

        # Now binary search between lo and hi for tighter fit
        if time.time() < deadline - 30.0 and hi > lo:
            for _ in range(15):  # 15 bisection steps for tighter convergence
                if time.time() > deadline - 25.0:
                    break

                mid = (lo + hi) / 2.0
                found = False

                for config in configs[:6]:
                    if time.time() > deadline - 22.0:
                        break

                    if len(config[0]) == 2:
                        init_params = make_params_from_positions(config)
                    else:
                        init_params = np.array([item for tri in config for item in tri])

                    pen = _total_penalty(init_params, mid, outer_center, outer_angle)

                    if pen < 1e-6:
                        found = True
                        if mid < best_score:
                            best_score = mid
                            best_result = (init_params.copy(), outer_center, mid, outer_angle)
                        break

                    if time.time() > deadline - 20.0:
                        continue

                    def objective(p):
                        return _total_penalty(p, mid, outer_center, outer_angle)

                    try:
                        result = minimize(
                            objective,
                            init_params,
                            method='L-BFGS-B',
                            options={'maxiter': 150, 'maxfun': 4000, 'ftol': 1e-14, 'gtol': 1e-10}
                        )
                        opt_params = result.x
                        opt_val = result.fun
                    except Exception:
                        continue

                    if opt_val < 1e-6:
                        found = True
                        if mid < best_score:
                            best_score = mid
                            best_result = (opt_params, outer_center, mid, outer_angle)
                        break

                if found:
                    hi = mid
                else:
                    lo = mid

    # Final refinement: try to squeeze out more with high-precision polish
    if best_result is not None and time.time() < deadline - 5.0:
        opt_params, outer_center, outer_side, outer_angle = best_result

        # Try progressively smaller sides with tight tolerances
        for delta in [0.05, 0.03, 0.02, 0.01, 0.005, 0.002, 0.001]:
            if time.time() > deadline - 3.0:
                break
            new_side = outer_side - delta
            if new_side < 3.0:
                break

            def final_objective(p):
                return _total_penalty(p, new_side, outer_center, outer_angle)

            try:
                result = minimize(
                    final_objective,
                    opt_params,
                    method='L-BFGS-B',
                    options={'maxiter': 300, 'maxfun': 10000, 'ftol': 1e-15, 'gtol': 1e-13}
                )
                if result.fun < 1e-6:
                    outer_side = new_side
                    opt_params = result.x
            except Exception:
                pass

        best_result = (opt_params, outer_center, outer_side, outer_angle)

    # Fallback
    if best_result is None:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

    opt_params, outer_center, outer_side, outer_angle = best_result

    inner = []
    for i in range(N):
        x = float(opt_params[3*i])
        y = float(opt_params[3*i+1])
        ang = float(opt_params[3*i+2])
        inner.append([x, y, ang])

    # Final validation
    inner_verts_list = []
    for i in range(N):
        cx, cy, ang = opt_params[3*i], opt_params[3*i+1], opt_params[3*i+2]
        inner_verts_list.append(_hex_verts(cx, cy, 1.0, ang))
    outer_verts = _hex_verts(outer_center[0], outer_center[1], outer_side, outer_angle)

    cont_pen = sum(_containment_penalty_fast(np.array([iv]), outer_verts) for iv in inner_verts_list)
    disj_pen = _pairwise_sat_depth_all(inner_verts_list)

    if cont_pen > 1e-4 or disj_pen > 1e-4:
        inner = []
        pos = build_hexagonal_lattice(3, 4)
        arr = np.array(pos)
        centroid = arr.mean(axis=0)
        for (x, y) in (arr - centroid):
            inner.append([float(x), float(y), 0.0])
        return inner, (0.0, 0.0), 5.5, 0.0

    return inner, (float(outer_center[0]), float(outer_center[1])), float(outer_side), float(outer_angle)

# EVOLVE-BLOCK-END
