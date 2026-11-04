import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize_scalar, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 145.0  # safety margin before 150s kill
    safety_margin = 5.0

    d = math.sqrt(3)

    # Unit hexagon vertices (pointy-top, centered at origin)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def sat_overlap_depth(v1, v2):
        """Return maximum overlap depth (positive if overlapping, 0 if separated)."""
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
                if overlap > max_overlap:
                    max_overlap = overlap
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

        # Side normals at angles outer_alpha + k*pi/3 for k=0..5
        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)

        # Projection of each point onto each normal
        dots = np.sum(all_pts[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)

        # For each normal direction, find the extent (max - min)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)

        # Side length S: distance between opposite sides = S * sqrt(3)
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
                overlap = sat_overlap_depth(verts_list[i], verts_list[j])
                if overlap > 0:
                    penalty += overlap * overlap * 1000.0

        S = compute_S(params)
        return S + penalty

    # Define bounds for differential evolution
    # Centers can range from about -15 to 15
    # Angles from 0 to pi/3 (hexagon has 6-fold symmetry, so 60 degrees is enough)
    x_bounds = [(-15, 15)] * N
    y_bounds = [(-15, 15)] * N
    theta_bounds = [(0, math.pi / 3)] * N
    bounds = x_bounds + y_bounds + theta_bounds

    # Initial guess: hexagonal lattice arrangement
    # Center + ring 1 (6 hexagons) + ring 2 (4 hexagons for symmetry)
    base_centers = [(0.0, 0.0)]
    for i in range(6):
        angle = i * math.pi / 3
        x = 2 * d * math.cos(angle)
        y = 2 * d * math.sin(angle)
        base_centers.append((x, y))

    # Ring 2: pick 4 symmetric positions
    ring2 = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            s = -q - r
            if max(abs(q), abs(r), abs(s)) == 2:
                spacing = 2 * d
                x = spacing * (q + r / 2.0)
                y = spacing * (d / 2.0) * r
                ring2.append((x, y))

    # Pick 4 ring-2 positions for symmetry (every 3rd one)
    ring2_selected = [ring2[i] for i in range(0, 12, 3)]

    initial_centers = base_centers + ring2_selected
    initial_params = np.zeros(N * 3)
    for i, (cx, cy) in enumerate(initial_centers):
        initial_params[3 * i] = cx
        initial_params[3 * i + 1] = cy
        initial_params[3 * i + 2] = 0.0

    best_S = float('inf')
    best_params = initial_params.copy()

    # Run differential evolution
    if time.time() < deadline - safety_margin:
        try:
            de_result = differential_evolution(
                objective,
                bounds=bounds,
                seed=42,
                maxiter=30,
                tol=1e-6,
                popsize=15,
                mutation=(0.5, 1.0),
                recombination=0.7,
                polish=False,
                init='latinhypercube'
            )
            if de_result.fun < best_S:
                best_S = de_result.fun
                best_params = de_result.x.copy()
        except Exception:
            pass

    # Refine with Nelder-Mead
    if time.time() < deadline - safety_margin:
        try:
            nm_result = minimize(
                objective,
                best_params,
                method='Nelder-Mead',
                options={'maxiter': 200, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True}
            )
            if nm_result.fun < best_S:
                best_S = nm_result.fun
                best_params = nm_result.x.copy()
        except Exception:
            pass

    # Optimize outer hexagon rotation
    best_outer_alpha = 0.0
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

    # Construct the inner hexagons from best_params
    inner = []
    for i in range(N):
        x = best_params[3 * i]
        y = best_params[3 * i + 1]
        angle_deg = math.degrees(best_params[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S, math.degrees(best_outer_alpha)
