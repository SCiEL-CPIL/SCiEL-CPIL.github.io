import math
import time
import numpy as np
from scipy.optimize import differential_evolution, minimize

N = 11

def construct():
    start_time = time.time()
    deadline = 150.0
    safety_margin = 15.0

    # Geometry constants
    d = math.sqrt(3)
    spacing = 2 * d  # Distance between centers of touching unit hexagons (side 1)

    # Relative vertices of a unit hexagon (pointy-top, centered at origin)
    # Angles: 0, 60, 120, 180, 240, 300
    verts_rel = np.array([
        [math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)
    ])

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

    def compute_S_from_points(points, theta):
        """
        Compute minimal side length S of a regular hexagon with rotation theta
        that encloses the given points.
        """
        k_angles = theta + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)
        dots = np.dot(points, normals.T)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        return max_extent / math.sqrt(3)

    def objective(params, theta):
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
        S = compute_S_from_points(np.vstack(verts_list), theta)
        return S + penalty

    # Generate initial configurations: honeycomb patterns
    def generate_honeycomb_config(center_idx, ring1_count=6, ring2_count=4):
        """Generate a honeycomb-like configuration."""
        centers = [(0.0, 0.0)]
        # Ring 1: 6 hexagons
        for i in range(6):
            angle = i * math.pi / 3
            x = spacing * math.cos(angle)
            y = spacing * math.sin(angle)
            centers.append((x, y))
        # Ring 2: select 4 from 12 positions
        ring2_positions = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = spacing * (q + r / 2.0)
                    y = spacing * (d / 2.0) * r
                    ring2_positions.append((x, y))
        
        # Select 4 ring 2 positions (spread out)
        selected = []
        for i in range(0, 12, 3):
            selected.append(ring2_positions[i])
        
        centers.extend(selected)
        return centers

    # Try multiple initial configurations
    best_S_final = float('inf')
    best_params_final = None
    best_theta_final = 0.0

    # Configuration 1: Standard honeycomb
    config1 = generate_honeycomb_config(0)
    
    # Configuration 2: Different ring 2 selection
    def generate_alt_config():
        centers = [(0.0, 0.0)]
        for i in range(6):
            angle = i * math.pi / 3
            x = spacing * math.cos(angle)
            y = spacing * math.sin(angle)
            centers.append((x, y))
        # Select different 4 ring 2 positions
        ring2_positions = []
        for q in range(-2, 3):
            for r in range(-2, 3):
                s = -q - r
                if max(abs(q), abs(r), abs(s)) == 2:
                    x = spacing * (q + r / 2.0)
                    y = spacing * (d / 2.0) * r
                    ring2_positions.append((x, y))
        # Select positions 1, 4, 7, 10 (different spread)
        for i in [1, 4, 7, 10]:
            centers.append(ring2_positions[i])
        return centers

    config2 = generate_alt_config()

    # Configuration 3: More compact arrangement
    def generate_compact_config():
        centers = [(0.0, 0.0)]
        # 6 around center, but slightly compressed
        for i in range(6):
            angle = i * math.pi / 3
            x = spacing * 0.95 * math.cos(angle)
            y = spacing * 0.95 * math.sin(angle)
            centers.append((x, y))
        # 4 more at 120-degree intervals at larger radius
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4
            x = spacing * 1.8 * math.cos(angle)
            y = spacing * 1.8 * math.sin(angle)
            centers.append((x, y))
        return centers

    config3 = generate_compact_config()

    configs = [config1, config2, config3]

    for cfg_idx, config in enumerate(configs):
        if time.time() > deadline - safety_margin:
            break

        # Initialize parameters
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(config):
            init_params[3 * i] = cx
            init_params[3 * i + 1] = cy
            init_params[3 * i + 2] = 0.0

        # Define bounds for Differential Evolution
        bounds = []
        for i in range(N):
            bounds.append((-5.0, 5.0))  # x
            bounds.append((-5.0, 5.0))  # y
            bounds.append((0.0, math.pi / 3))  # angle
        bounds.append((0.0, math.pi / 3))  # outer theta

        def de_objective(x):
            p = x[:33]
            t = x[33]
            return objective(p, t)

        try:
            # Use Differential Evolution for global search
            de_res = differential_evolution(de_objective, bounds, maxiter=80, popsize=20, tol=1e-6,
                                            seed=42 + cfg_idx, polish=False, update_interval=15,
                                            init='sobol')

            if de_res.fun < best_S_final:
                p_res = de_res.x[:33]
                t_res = de_res.x[33]

                # Check validity
                n_hex = len(p_res) // 3
                xs = p_res[0::3]
                ys = p_res[1::3]
                thetas = p_res[2::3]
                verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]

                max_overlap = 0.0
                for i in range(n_hex):
                    for j in range(i + 1, n_hex):
                        ov = sat_overlap(verts_list[i], verts_list[j])
                        if ov > max_overlap:
                            max_overlap = ov

                if max_overlap < 1e-6:
                    all_pts = np.vstack(verts_list)
                    S_actual = compute_S_from_points(all_pts, t_res)

                    if S_actual < best_S_final:
                        best_S_final = S_actual
                        best_params_final = p_res.copy()
                        best_theta_final = t_res

        except Exception:
            continue

    # If DE didn't find a valid solution, fall back to Nelder-Mead
    if best_params_final is None:
        config = config1
        init_params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(config):
            init_params[3 * i] = cx
            init_params[3 * i + 1] = cy
            init_params[3 * i + 2] = 0.0

        def nm_objective(x):
            p = x[:33]
            t = x[33]
            return objective(p, t)

        try:
            nm_res = minimize(nm_objective, np.append(init_params, 0.0), method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-8, 'fatol': 1e-8})
            
            if nm_res.fun < best_S_final:
                p_res = nm_res.x[:33]
                t_res = nm_res.x[33]

                # Check validity
                n_hex = len(p_res) // 3
                xs = p_res[0::3]
                ys = p_res[1::3]
                thetas = p_res[2::3]
                verts_list = [get_hex_verts(xs[i], ys[i], thetas[i]) for i in range(n_hex)]

                max_overlap = 0.0
                for i in range(n_hex):
                    for j in range(i + 1, n_hex):
                        ov = sat_overlap(verts_list[i], verts_list[j])
                        if ov > max_overlap:
                            max_overlap = ov

                if max_overlap < 1e-6:
                    all_pts = np.vstack(verts_list)
                    S_actual = compute_S_from_points(all_pts, t_res)

                    if S_actual < best_S_final:
                        best_S_final = S_actual
                        best_params_final = p_res.copy()
                        best_theta_final = t_res

        except Exception:
            pass

    # Final fallback: use the first configuration as-is
    if best_params_final is None:
        config = config1
        best_params_final = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(config):
            best_params_final[3 * i] = cx
            best_params_final[3 * i + 1] = cy
            best_params_final[3 * i + 2] = 0.0
        best_theta_final = 0.0
        best_S_final = compute_S_from_points(np.vstack([get_hex_verts(cx, cy, 0.0) for cx, cy in config]), best_theta_final)

    inner = []
    for i in range(N):
        x = best_params_final[3 * i]
        y = best_params_final[3 * i + 1]
        angle_deg = math.degrees(best_params_final[3 * i + 2])
        inner.append([x, y, angle_deg])

    return inner, (0.0, 0.0), best_S_final, math.degrees(best_theta_final)
