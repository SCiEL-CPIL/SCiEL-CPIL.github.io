import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    safety_margin = 5.0

    d = math.sqrt(3)

    # Unit hexagon vertices (flat-top, centered at origin)
    # Vertices at angles 0, 60, 120, 180, 240, 300 degrees
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def sat_overlap(v1, v2):
        """Return maximum overlap depth (positive if overlapping, negative if separated)."""
        max_overlap = 0.0
        for v in (v1, v2):
            for i in range(6):
                p1 = v[i]
                p2 = v[(i + 1) % 6]
                edge = p2 - p1
                normal = np.array([-edge[1], edge[0]])
                norm = np.linalg.norm(normal)
                if norm < 1e-10:
                    continue
                normal = normal / norm
                proj1 = np.dot(v1, normal)
                proj2 = np.dot(v2, normal)
                overlap = min(proj1.max(), proj2.max()) - max(proj1.min(), proj2.min())
                max_overlap = max(max_overlap, overlap)
        return max_overlap

    def compute_S(params, outer_alpha=0.0):
        """Compute the enclosing regular hexagon side length for a set of hexagons."""
        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]

        # Collect all vertices
        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)

        # Center the points for better numerical behavior
        centroid = np.mean(all_pts, axis=0)

        # Side normals at angles outer_alpha + k*pi/3 for k=0..5
        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)

        # Projection of each point onto each normal
        dots = np.sum((all_pts - centroid)[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)

        # For each normal direction, find the extent (max - min)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)

        # Side length S: for a regular hexagon with side S, the distance between
        # opposite sides is S * sqrt(3). So S = max_extent / sqrt(3)
        return max_extent / math.sqrt(3)

    def objective(params):
        if time.time() > deadline - safety_margin:
            return 1e10

        n_hex = len(params) // 3
        xs = params[0::3]
        ys = params[1::3]
        thetas = params[2::3]

        # SAT overlap penalty
        verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                overlap = sat_overlap(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * overlap * 1000.0

        S = compute_S(params)
        return S + penalty

    # Generate diverse initial configurations
    init_configs = []

    # Configuration 1: Hexagonal lattice - center + ring 1 + 4 from ring 2
    base_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        base_centers.append((x, y))

    ring2_centers = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                spacing = 2 * d
                x = spacing * (q + r / 2.0)
                y = spacing * (d / 2.0) * r
                ring2_centers.append((x, y))

    # Try different symmetric subsets of ring 2
    for offset in range(3):
        ring2_indices = [offset, offset + 3, offset + 6, offset + 9]
        final_centers = base_centers + [ring2_centers[i] for i in ring2_indices]
        params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = 0.0
        init_configs.append(params)

    # Configuration 2: Staggered arrangement with rotations
    for rot in [0.0, math.pi / 6, math.pi / 3]:
        final_centers = []
        # Center hexagon
        final_centers.append((0.0, 0.0))
        # Ring 1 with slight rotation
        for i in range(6):
            angle = i * math.pi / 3 + rot
            x = 2 * d * 0.95 * math.cos(angle)
            y = 2 * d * 0.95 * math.sin(angle)
            final_centers.append((x, y))
        # Ring 2 partial
        for i in range(4):
            angle = i * math.pi / 2 + rot
            x = 3.5 * d * 0.8 * math.cos(angle)
            y = 3.5 * d * 0.8 * math.sin(angle)
            final_centers.append((x, y))

        params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers[:N]):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = rot
        init_configs.append(params)

    # Configuration 3: Compact cluster with mixed rotations
    np.random.seed(42)
    for _ in range(3):
        params = np.zeros(N * 3)
        # Place hexagons in a compact region
        positions = []
        positions.append((0.0, 0.0))
        for i in range(6):
            angle = i * math.pi / 3
            r = 2 * d * (0.9 + 0.1 * np.random.randn())
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y))
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4
            r = 3.2 * d * (0.85 + 0.1 * np.random.randn())
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y))

        for i, (cx, cy) in enumerate(positions):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = np.random.uniform(-math.pi / 6, math.pi / 6)
        init_configs.append(params)

    # Optimize each initial configuration
    best_S = float('inf')
    best_params = None
    best_outer_alpha = 0.0

    for init_params in init_configs:
        if time.time() > deadline - safety_margin:
            break

        try:
            # Use L-BFGS-B with bounds for better convergence
            bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N
            res = minimize(objective, init_params, method='L-BFGS-B',
                           bounds=bounds,
                           options={'maxiter': 200, 'ftol': 1e-8, 'gtol': 1e-6})

            if res.fun < best_S:
                best_S = res.fun
                best_params = res.x.copy()

        except Exception:
            continue

        # Also try Nelder-Mead from the same start
        if time.time() > deadline - safety_margin:
            break
        try:
            res2 = minimize(objective, init_params, method='Nelder-Mead',
                            options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})

            if res2.fun < best_S:
                best_S = res2.fun
                best_params = res2.x.copy()

        except Exception:
            continue

    # If we have a valid result, optimize the outer hexagon rotation
    if best_params is not None:
        if time.time() < deadline - safety_margin:
            def outer_obj(alpha):
                return compute_S(best_params, alpha)

            try:
                res_outer = minimize_scalar(outer_obj, bounds=(0, math.pi / 3), method='bounded',
                                            options={'xatol': 1e-10})
                if res_outer.success:
                    best_outer_alpha = res_outer.x
                    best_S = res_outer.fun
            except Exception:
                best_outer_alpha = 0.0

        # Final refinement with multiple outer rotations
        if time.time() < deadline - safety_margin:
            best_final_S = float('inf')
            best_final_params = best_params.copy()
            best_final_alpha = best_outer_alpha

            for alpha_test in np.linspace(0, math.pi / 3, 7):
                def obj_refine(params, alpha=alpha_test):
                    if time.time() > deadline - safety_margin:
                        return 1e10
                    n_hex = len(params) // 3
                    xs = params[0::3]
                    ys = params[1::3]
                    thetas = params[2::3]
                    verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]
                    penalty = 0.0
                    for i in range(n_hex):
                        for j in range(i + 1, n_hex):
                            overlap = sat_overlap(verts_list[i], verts_list[j])
                            if overlap > 0:
                                penalty += overlap * overlap * 1000.0
                    S = compute_S(params, alpha)
                    return S + penalty

                try:
                    bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N
                    res_refine = minimize(obj_refine, best_params, method='L-BFGS-B',
                                         bounds=bounds,
                                         options={'maxiter': 100, 'ftol': 1e-9})
                    if res_refine.fun < best_final_S:
                        best_final_S = res_refine.fun
                        best_final_params = res_refine.x.copy()
                        best_final_alpha = alpha_test
                except Exception:
                    continue

            best_S = best_final_S
            best_params = best_final_params
            best_outer_alpha = best_final_alpha
    else:
        # Fallback: use the first initial config
        best_params = init_configs[0]
        best_S = compute_S(best_params)
        best_outer_alpha = 0.0

    # Construct the inner hexagons from best_params
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        angle_deg = math.degrees(best_params[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
