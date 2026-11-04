import math
import time
import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.spatial import ConvexHull

N = 11

def construct():
    start_time = time.time()
    deadline = 140.0
    safety_margin = 10.0

    d = math.sqrt(3)

    # Unit hexagon vertices (flat-top, centered at origin)
    verts_rel = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def get_hex_verts(cx, cy, angle):
        c = math.cos(angle)
        s = math.sin(angle)
        rx = verts_rel[:, 0] * c - verts_rel[:, 1] * s
        ry = verts_rel[:, 0] * s + verts_rel[:, 1] * c
        return np.column_stack([rx + cx, ry + cy])

    def point_in_polygon(pt, poly):
        """Check if point is inside polygon using ray casting."""
        x, y = pt
        n = len(poly)
        inside = False
        j = n - 1
        for i in range(n):
            xi, yi = poly[i]
            xj, yj = poly[j]
            if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
                inside = not inside
            j = i
        return inside

    def seg_seg_dist(p1, p2, p3, p4):
        """Compute minimum distance between two line segments."""
        # Use vector approach
        d1 = p2 - p1
        d2 = p4 - p3
        r = p1 - p3
        a = np.dot(d1, d1)
        e = np.dot(d2, d2)
        f = np.dot(d2, r)
        
        if a < 1e-10 and e < 1e-10:
            return np.linalg.norm(p1 - p3)
        elif a < 1e-10:
            t2 = f / e
            t2 = np.clip(t2, 0, 1)
            closest1 = p1
            closest2 = p3 + t2 * d2
            return np.linalg.norm(closest1 - closest2)
        elif e < 1e-10:
            t1 = np.dot(r, d1) / a
            t1 = np.clip(t1, 0, 1)
            closest1 = p1 + t1 * d1
            closest2 = p3
            return np.linalg.norm(closest1 - closest2)
        else:
            c = np.dot(d1, r)
            b = np.dot(d1, d2)
            denom = a * e - b * b
            if denom < 1e-10:
                t1 = 0.0
                t2 = f / e
                t2 = np.clip(t2, 0, 1)
                closest1 = p1
                closest2 = p3 + t2 * d2
                return np.linalg.norm(closest1 - closest2)
            else:
                t1 = (b * f - c * e) / denom
                t2 = (a * f - c * b) / denom
                if t1 < 0:
                    t1 = 0
                    t2 = f / e
                    t2 = np.clip(t2, 0, 1)
                elif t1 > 1:
                    t1 = 1
                    t2 = (f + b) / e
                    t2 = np.clip(t2, 0, 1)
                elif t2 < 0:
                    t2 = 0
                    t1 = -c / a
                    t1 = np.clip(t1, 0, 1)
                elif t2 > 1:
                    t2 = 1
                    t1 = (f + b) / a
                    t1 = np.clip(t1, 0, 1)
                closest1 = p1 + t1 * d1
                closest2 = p3 + t2 * d2
                return np.linalg.norm(closest1 - closest2)

    def poly_min_dist(v1, v2):
        """Compute minimum distance between two convex polygons (negative if overlapping)."""
        min_dist = float('inf')
        
        # Check all edge pairs
        for i in range(6):
            for j in range(6):
                dist = seg_seg_dist(v1[i], v1[(i+1)%6], v2[j], v2[(j+1)%6])
                min_dist = min(min_dist, dist)
        
        # Check if any vertex of v1 is inside v2
        for pt in v1:
            if point_in_polygon(pt, v2):
                return -1.0  # Overlapping
        
        # Check if any vertex of v2 is inside v1
        for pt in v2:
            if point_in_polygon(pt, v1):
                return -1.0  # Overlapping
        
        return min_dist

    def smooth_overlap_penalty(verts_list):
        """Smooth differentiable penalty based on minimum distance between hexagons."""
        n_hex = len(verts_list)
        penalty = 0.0
        for i in range(n_hex):
            for j in range(i + 1, n_hex):
                dist = poly_min_dist(verts_list[i], verts_list[j])
                if dist < 0:
                    # Overlapping: use quadratic penalty
                    penalty += (-dist) ** 2
                elif dist < 0.1:
                    # Close but not overlapping: small penalty to encourage separation
                    penalty += (0.1 - dist) ** 2 * 0.1
        return penalty

    def sat_overlap(v1, v2):
        """Return maximum overlap depth (positive if overlapping, 0 if touching or separated)."""
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

        all_pts = []
        for i in range(n_hex):
            verts = get_hex_verts(xs[i], ys[i], thetas[i])
            all_pts.append(verts)
        all_pts = np.vstack(all_pts)

        centroid = np.mean(all_pts, axis=0)

        k_angles = outer_alpha + np.arange(6) * math.pi / 3
        normals = np.stack([np.cos(k_angles), np.sin(k_angles)], axis=-1)

        dots = np.sum((all_pts - centroid)[np.newaxis, :, :] * normals[:, np.newaxis, :], axis=-1)
        extents = np.max(dots, axis=1) - np.min(dots, axis=1)
        max_extent = np.max(extents)
        return max_extent / math.sqrt(3)

    def total_overlap(params):
        """Total overlap across all pairs (for penalty)."""
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
                    penalty += overlap
        return penalty

    def objective(params, outer_alpha=0.0):
        if time.time() > deadline - safety_margin:
            return 1e10
        S = compute_S(params, outer_alpha)
        ov = total_overlap(params)
        return S + ov * ov * 500.0

    # Generate diverse initial configurations for DE
    def make_lattice_config(offset=0, rot=0.0):
        """Create a hex-lattice-based configuration."""
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

        ring2_indices = [offset, offset + 3, offset + 6, offset + 9]
        final_centers = base_centers + [ring2_centers[i] for i in ring2_indices]

        params = np.zeros(N * 3)
        for i, (cx, cy) in enumerate(final_centers[:N]):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = rot
        return params

    def make_random_config(seed):
        """Create a random compact configuration."""
        np.random.seed(seed)
        params = np.zeros(N * 3)
        # Place in a compact region
        positions = []
        positions.append((0.0, 0.0))
        for i in range(6):
            angle = i * math.pi / 3 + np.random.uniform(-0.3, 0.3)
            r = 2 * d * np.random.uniform(0.85, 1.1)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y))
        for i in range(4):
            angle = i * math.pi / 2 + math.pi / 4 + np.random.uniform(-0.3, 0.3)
            r = 3.2 * d * np.random.uniform(0.8, 1.0)
            x = r * math.cos(angle)
            y = r * math.sin(angle)
            positions.append((x, y))

        for i, (cx, cy) in enumerate(positions[:N]):
            params[3 * i] = cx
            params[3 * i + 1] = cy
            params[3 * i + 2] = np.random.uniform(-math.pi / 6, math.pi / 6)
        return params

    # DE bounds
    bounds = [(-10, 10), (-10, 10), (-math.pi, math.pi)] * N

    # Population for DE
    pop_size = 30
    np.random.seed(42)

    # Build initial population: mix of lattice configs and random
    population = []
    for offset in range(3):
        for rot in [0.0, math.pi / 6]:
            config = make_lattice_config(offset, rot)
            # Add small noise
            config += np.random.normal(0, 0.1, N * 3)
            population.append(config)

    # Fill rest with random configs
    while len(population) < pop_size:
        population.append(make_random_config(len(population)))

    population = np.array(population)

    # Differential Evolution
    best_S = float('inf')
    best_params = None
    best_outer_alpha = 0.0

    # Evaluate initial population
    for i in range(pop_size):
        S_val = compute_S(population[i])
        ov = total_overlap(population[i])
        fitness = S_val + ov * ov * 500.0
        if fitness < best_S:
            best_S = fitness
            best_params = population[i].copy()

    # DE iterations
    F = 0.8  # scale factor
    CR = 0.9  # crossover rate
    max_de_iter = 50

    for gen in range(max_de_iter):
        if time.time() > deadline - safety_margin:
            break

        gen_best = best_S
        gen_best_params = best_params.copy()

        for i in range(pop_size):
            if time.time() > deadline - safety_margin:
                break

            # Select 3 distinct random individuals
            a, b, c = np.random.choice(pop_size, 3, replace=False)

            # Mutation
            mutant = population[a] + F * (population[b] - population[c])

            # Crossover
            trial = np.zeros_like(population[i])
            jrand = np.random.randint(0, N * 3)
            for j in range(N * 3):
                if np.random.uniform() < CR or j == jrand:
                    trial[j] = mutant[j]
                else:
                    trial[j] = population[i][j]

            # Clamp to bounds
            for j in range(N * 3):
                trial[j] = np.clip(trial[j], bounds[j][0], bounds[j][1])

            # Evaluate
            S_trial = compute_S(trial)
            ov_trial = total_overlap(trial)
            fitness_trial = S_trial + ov_trial * ov_trial * 500.0

            # Selection
            S_i = compute_S(population[i])
            ov_i = total_overlap(population[i])
            fitness_i = S_i + ov_i * ov_i * 500.0

            if fitness_trial < fitness_i:
                population[i] = trial
                if fitness_trial < gen_best:
                    gen_best = fitness_trial
                    gen_best_params = trial.copy()

        # Update global best
        if gen_best < best_S:
            best_S = gen_best
            best_params = gen_best_params.copy()

    # Local refinement from DE result
    if best_params is not None and time.time() < deadline - safety_margin:
        # Optimize outer hexagon rotation
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

        # Local refinement with L-BFGS-B
        if time.time() < deadline - safety_margin:
            def obj_refine(params):
                if time.time() > deadline - safety_margin:
                    return 1e10
                S = compute_S(params, best_outer_alpha)
                ov = total_overlap(params)
                return S + ov * ov * 500.0

            try:
                res_refine = minimize(obj_refine, best_params, method='L-BFGS-B',
                                      bounds=bounds,
                                      options={'maxiter': 200, 'ftol': 1e-9, 'gtol': 1e-7})
                if res_refine.fun < best_S:
                    best_S = res_refine.fun
                    best_params = res_refine.x.copy()
            except Exception:
                pass

        # Additional refinement with Nelder-Mead
        if time.time() < deadline - safety_margin:
            def obj_nm(params):
                if time.time() > deadline - safety_margin:
                    return 1e10
                S = compute_S(params, best_outer_alpha)
                ov = total_overlap(params)
                return S + ov * ov * 500.0

            try:
                res_nm = minimize(obj_nm, best_params, method='Nelder-Mead',
                                  options={'maxiter': 300, 'xatol': 1e-7, 'fatol': 1e-7, 'adaptive': True})
                if res_nm.fun < best_S:
                    best_S = res_nm.fun
                    best_params = res_nm.x.copy()
            except Exception:
                pass

        # Try multiple outer rotations for final refinement
        if time.time() < deadline - safety_margin:
            best_final_S = best_S
            best_final_params = best_params.copy()
            best_final_alpha = best_outer_alpha

            for alpha_test in np.linspace(0, math.pi / 3, 7):
                if time.time() > deadline - safety_margin:
                    break

                def obj_alpha(params, alpha=alpha_test):
                    if time.time() > deadline - safety_margin:
                        return 1e10
                    S = compute_S(params, alpha)
                    ov = total_overlap(params)
                    return S + ov * ov * 500.0

                try:
                    res_a = minimize(obj_alpha, best_final_params, method='L-BFGS-B',
                                     bounds=bounds,
                                     options={'maxiter': 100, 'ftol': 1e-9})
                    if res_a.fun < best_final_S:
                        best_final_S = res_a.fun
                        best_final_params = res_a.x.copy()
                        best_final_alpha = alpha_test
                except Exception:
                    continue

            best_S = best_final_S
            best_params = best_final_params
            best_outer_alpha = best_final_alpha
    else:
        # Fallback
        best_params = make_lattice_config(0, 0.0)
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
