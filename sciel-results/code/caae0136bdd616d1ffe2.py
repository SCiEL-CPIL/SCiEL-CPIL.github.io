import numpy as np
from scipy.optimize import differential_evolution, minimize, basinhopping
import time

def construct():
    start_time = time.time()
    deadline = start_time + 145.0  # Safety margin before 150s kill

    # --- Geometry Helpers ---

    def hex_vertices(cx, cy, side, angle_rad):
        """Returns 6x2 array of vertices for a regular hexagon."""
        angles = angle_rad + np.radians(np.arange(0, 360, 60))
        x = cx + side * np.cos(angles)
        y = cy + side * np.sin(angles)
        return np.column_stack([x, y])

    def get_normals(verts):
        """Returns 6x2 array of outward normals for a CCW polygon."""
        edges = np.roll(verts, -1, axis=0) - verts
        normals = np.column_stack([edges[:, 1], -edges[:, 0]])
        norms = np.linalg.norm(normals, axis=1)
        normals = normals / (norms[:, np.newaxis] + 1e-15)
        return normals

    def sat_overlap(v1, v2):
        """
        Returns min overlap depth (negative if disjoint) using SAT.
        """
        normals1 = get_normals(v1)
        normals2 = get_normals(v2)
        normals = np.vstack([normals1, normals2])

        p1 = v1 @ normals.T
        p2 = v2 @ normals.T

        min1 = p1.min(axis=0)
        max1 = p1.max(axis=0)
        min2 = p2.min(axis=0)
        max2 = p2.max(axis=0)

        overlaps = np.minimum(max1, max2) - np.maximum(min1, min2)
        min_overlap = np.min(overlaps)

        return min_overlap

    def compute_min_side_fixed_orientation(hexes, center=(0.0, 0.0)):
        """
        Calculate the minimum side length of a regular hexagon (centered at given center, fixed orientation)
        that contains all given hexagons.
        """
        normal_angles = np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        max_proj = 0.0
        for h in hexes:
            proj = (h - center) @ normals.T
            max_proj = max(max_proj, np.max(proj))

        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def generate_hexes_from_base(x1, y1, a1, x2, y2, a2):
        """
        Generate 12 hexagons using 6-fold rotational symmetry from two base hexagons.
        Returns list of 12 (6x2) vertex arrays.
        """
        base1 = hex_vertices(x1, y1, 1.0, a1)
        base2 = hex_vertices(x2, y2, 1.0, a2)

        hexes = []
        for k in range(6):
            angle = np.radians(60 * k)
            c = np.cos(angle)
            s_ = np.sin(angle)
            rot = np.array([[c, -s_], [s_, c]])

            h1 = base1 @ rot.T
            h2 = base2 @ rot.T

            hexes.append(h1)
            hexes.append(h2)

        return hexes

    def compute_smooth_objective(params, beta=50.0):
        """
        Smooth objective for global search using soft-max penalty for overlaps.
        params: [x1, y1, a1, x2, y2, a2]
        """
        if time.time() > deadline:
            return 1e10

        x1, y1, a1, x2, y2, a2 = params

        hexes = generate_hexes_from_base(x1, y1, a1, x2, y2, a2)

        s_min = compute_min_side_fixed_orientation(hexes)

        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                # Quick bounding box check
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)

                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue

                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    # Soft-max smooth penalty
                    penalty += np.log(1.0 + np.exp(beta * overlap)) / beta

        return s_min + penalty

    def compute_objective(params):
        """
        Objective: min_side + penalty for overlaps.
        params: [x1, y1, a1, x2, y2, a2]
        """
        if time.time() > deadline:
            return 1e10

        x1, y1, a1, x2, y2, a2 = params

        hexes = generate_hexes_from_base(x1, y1, a1, x2, y2, a2)

        s_min = compute_min_side_fixed_orientation(hexes)

        penalty = 0.0
        for i in range(12):
            for j in range(i + 1, 12):
                min1 = hexes[i].min(axis=0)
                max1 = hexes[i].max(axis=0)
                min2 = hexes[j].min(axis=0)
                max2 = hexes[j].max(axis=0)

                if min1[0] > max2[0] or max1[0] < min2[0] or min1[1] > max2[1] or max1[1] < min2[1]:
                    continue

                overlap = sat_overlap(hexes[i], hexes[j])
                if overlap > 0:
                    penalty += overlap * 10.0

        return s_min + penalty

    def compute_min_side_rotated(hexes, angle_deg):
        """
        Calculate min side length of outer hexagon with given rotation angle.
        """
        angle_rad = np.radians(angle_deg)
        normal_angles = angle_rad + np.radians([30, 90, 150, 210, 270, 330])
        normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

        max_proj = 0.0
        for h in hexes:
            proj = h @ normals.T
            max_proj = max(max_proj, np.max(proj))

        s = max_proj * 2.0 / np.sqrt(3.0)
        return s

    def optimize_outer_rotation(hexes):
        """
        Optimize the outer hexagon rotation angle to minimize side length.
        Uses golden section search over [0, 60) degrees.
        """
        best_angle = 0.0
        best_s = np.inf

        # Golden section search
        gr = (np.sqrt(5) - 1) / 2
        a, b = 0.0, 60.0

        x1 = b - gr * (b - a)
        x2 = a + gr * (b - a)
        f1 = compute_min_side_rotated(hexes, x1)
        f2 = compute_min_side_rotated(hexes, x2)

        for _ in range(50):
            if time.time() > deadline:
                break

            if f1 < f2:
                b = x2
                x2 = x1
                f2 = f1
                x1 = b - gr * (b - a)
                f1 = compute_min_side_rotated(hexes, x1)
            else:
                a = x1
                x1 = x2
                f1 = f2
                x2 = a + gr * (b - a)
                f2 = compute_min_side_rotated(hexes, x2)

        # Check endpoints and midpoint
        candidates = [0.0, 30.0, (a + b) / 2]
        for ang in candidates:
            s = compute_min_side_rotated(hexes, ang)
            if s < best_s:
                best_s = s
                best_angle = ang

        return best_s, best_angle

    # --- Optimization ---

    # Bounds for the 2 base hexagons
    bounds = [
        (-4.0, 4.0),  # x1
        (-4.0, 4.0),  # y1
        (0.0, np.pi / 3),  # a1
        (-4.0, 4.0),  # x2
        (-4.0, 4.0),  # y2
        (0.0, np.pi / 3),  # a2
    ]

    best_s = np.inf
    best_params = None

    # Phase 1: Differential Evolution with smooth objective
    try:
        res_de = differential_evolution(
            lambda p: compute_smooth_objective(p, beta=30.0),
            bounds,
            maxiter=200,
            popsize=20,
            tol=1e-8,
            seed=42,
            polish=True
        )
        if res_de.fun < best_s:
            best_s = res_de.fun
            best_params = res_de.x.copy()
    except Exception:
        pass

    # Phase 2: Basin Hopping
    if best_params is not None and time.time() < deadline - 30:
        try:
            def local_minimize(x):
                res = minimize(
                    compute_objective,
                    x,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 100, 'ftol': 1e-14}
                )
                return res.fun, res.x, {}

            res_bh = basinhopping(
                local_minimize,
                best_params,
                niter=20,
                T=0.5,
                stepsize=0.3,
                seed=42
            )
            if res_bh.fun < best_s:
                best_s = res_bh.fun
                best_params = res_bh.x.copy()
        except Exception:
            pass

    # Phase 3: Local Refinement with high precision
    if best_params is not None and time.time() < deadline - 15:
        try:
            res_local = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 1000, 'ftol': 1e-15, 'gtol': 1e-15}
            )
            if res_local.fun < best_s:
                best_s = res_local.fun
                best_params = res_local.x.copy()
        except Exception:
            pass

    # Phase 4: Random Restarts with better sampling
    if time.time() < deadline - 20:
        n_restarts = 12
        np.random.seed(123)
        for r in range(n_restarts):
            if time.time() > deadline - 15:
                break

            # Sample from a structured range
            r1 = np.random.uniform(1.0, 3.5)
            th1 = np.random.uniform(0, 2 * np.pi)
            x1 = r1 * np.cos(th1)
            y1 = r1 * np.sin(th1)
            a1 = np.random.uniform(0, np.pi / 3)

            r2 = np.random.uniform(1.5, 4.5)
            th2 = np.random.uniform(0, 2 * np.pi)
            x2 = r2 * np.cos(th2)
            y2 = r2 * np.sin(th2)
            a2 = np.random.uniform(0, np.pi / 3)

            p = np.array([x1, y1, a1, x2, y2, a2])

            try:
                res_local = minimize(
                    compute_objective,
                    p,
                    method='L-BFGS-B',
                    bounds=bounds,
                    options={'maxiter': 200, 'ftol': 1e-14}
                )
                if res_local.fun < best_s:
                    best_s = res_local.fun
                    best_params = res_local.x.copy()
            except Exception:
                pass

    # Phase 5: Final high-precision polish
    if best_params is not None and time.time() < deadline - 5:
        try:
            res_polish = minimize(
                compute_objective,
                best_params,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 2000, 'ftol': 1e-16, 'gtol': 1e-16, 'maxls': 200}
            )
            if res_polish.fun < best_s:
                best_s = res_polish.fun
                best_params = res_polish.x.copy()
        except Exception:
            pass

    # Fallback
    if best_params is None:
        best_params = np.array([1.5, 0.0, 0.0, 2.5, 0.0, 0.0])

    x1, y1, a1, x2, y2, a2 = best_params

    # Generate the 12 inner hexagons
    inner = []
    hexes_list = []
    for k in range(6):
        angle = np.radians(60 * k)
        c = np.cos(angle)
        s_ = np.sin(angle)

        # Rotate base1 position and orientation
        x1_rot = x1 * c - y1 * s_
        y1_rot = x1 * s_ + y1 * c
        a1_rot = a1 + angle

        # Rotate base2 position and orientation
        x2_rot = x2 * c - y2 * s_
        y2_rot = x2 * s_ + y2 * c
        a2_rot = a2 + angle

        inner.append([float(x1_rot), float(y1_rot), float(np.degrees(a1_rot))])
        inner.append([float(x2_rot), float(y2_rot), float(np.degrees(a2_rot))])

        hexes_list.append(hex_vertices(x1_rot, y1_rot, 1.0, a1_rot))
        hexes_list.append(hex_vertices(x2_rot, y2_rot, 1.0, a2_rot))

    # Optimize outer hexagon rotation
    s_final, outer_angle_deg = optimize_outer_rotation(hexes_list)

    outer_center = (0.0, 0.0)
    outer_side = float(s_final)

    return inner, outer_center, outer_side, float(outer_angle_deg)
