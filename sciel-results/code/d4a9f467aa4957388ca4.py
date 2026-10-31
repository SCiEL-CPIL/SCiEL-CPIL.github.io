<<<<<<< SEARCH
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple candidate configurations
    candidates = []

    # Candidate 1: Compact 3-4-3 hexagonal cluster
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_1, "cluster343"))

    # Candidate 2: Cross 1-6-4 (opposite pairs in ring 2)
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_2, "cross164"))

    # Candidate 3: Compact block 3x3 + 2
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]
    candidates.append((axial_3, "block3x3"))

    # Candidate 4: 4-4-3
    axial_4 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_4, "443"))

    # Candidate 5: 5-6 two rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append((axial_5, "56rows"))

    # Candidate 6: Diamond-like
    axial_6 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_6, "diamond"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        epsilon = 1e-5
        scale = 1.0 + epsilon

        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer container with fixed inner layout
        def objective_outer_smooth(params):
            cx, cy, angle_deg = params
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=50.0)

        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)

        best_result = None
        best_val = float('inf')

        init_angles = list(range(0, 360, 30))

        for init_angle in init_angles:
            if time.time() > deadline - 25:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 800, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1500})

                res2 = minimize(objective_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 1000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2500})

                if res2.fun < best_val:
                    best_val = res2.fun
                    best_result = res2.x
            except Exception:
                pass

        if best_result is not None:
            outer_center = (best_result[0], best_result[1])
            outer_angle = best_result[2]
            side_val = best_val
        else:
            outer_center = (cx_avg, cy_avg)
            outer_angle = 0.0
            side_val = compute_required_side(initial_inner, outer_center, outer_angle)

        # Phase 2: Joint optimization with simulated annealing + force-directed penalty
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 * 1.15:
                    adjacency.add((i, j))

        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            side = compute_required_side(inner_list, (ocx, ocy), oangle)
            penalty = compute_overlap_penalty(inner_list)
            fd_penalty = compute_force_directed_penalty(params, adjacency, k_rep=50.0, k_spring=2.0)

            return side + 100.0 * penalty + 0.1 * fd_penalty

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val

        T = 0.8
        T_min = 0.003
        cooling = 0.9999

        max_iter = 8000
        for i in range(max_iter):
            if time.time() > deadline - 20:
                break

            neighbor = current_params.copy()

            if np.random.random() < 0.75:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, 0.15)
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 2.0)
                else:
                    delta = np.random.normal(0, 0.1)
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val

            T *= cooling
            if T < T_min:
                break

        best_inner_joint = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_joint.append([x, y, a])

        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]

        # Phase 3: Final high-precision polish on outer container
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner_joint, (cx, cy), angle_deg)

        polish_starts = [
            [best_ocx, best_ocy, best_oangle],
            [best_ocx, best_ocy, best_oangle + 3],
            [best_ocx, best_ocy, best_oangle - 3],
            [best_ocx, best_ocy, best_oangle + 6],
            [best_ocx, best_ocy, best_oangle - 6],
            [best_ocx + 0.05, best_ocy, best_oangle],
            [best_ocx - 0.05, best_ocy, best_oangle],
            [best_ocx, best_ocy + 0.05, best_oangle],
            [best_ocx, best_ocy - 0.05, best_oangle],
        ]

        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 5:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness
        if check_all_disjoint(final_inner, eps=1e-5):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fall back to initial layout if overlapping
            fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
            if fallback_side < best_global_side:
                best_global_side = fallback_side
                best_global_inner = initial_inner
                best_global_center = (cx_avg, cy_avg)
                best_global_angle = 0.0

    if best_global_inner is None:
        # Fallback: use first candidate
        axial_coords = candidates[0][0][:N]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle
=======
def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Best known 11-hexagon packing: center + ring of 6 + 4 in ring 2
    # Using axial coordinates (q, r) for hex grid
    # Ring 2 positions chosen to minimize bounding hexagon
    axial_best = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]

    # Alternative: more symmetric ring 2 placement
    axial_sym = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]

    # Compact 3-4-3 arrangement
    axial_343 = [
        (0, 0), (1, 0), (-1, 0),
        (0, 1), (1, 1), (-1, 1), (0, -1), (1, -1), (-1, -1),
        (0, 2), (0, -2)
    ]

    candidates = [
        (axial_best, "best_known"),
        (axial_sym, "symmetric"),
        (axial_343, "343"),
    ]

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 35:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]

        # Convert axial to Cartesian with slight gap for non-overlap
        gap = 1e-4
        scale = 1.0 + gap

        centers = []
        for q, r in axial_coords:
            x = 1.5 * scale * q
            y = SQRT3 * scale * (r + q / 2.0)
            centers.append((x, y))

        initial_inner = [[x, y, 0.0] for x, y in centers]

        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)

        # Phase 1: Optimize outer container with fixed inner layout
        # Try multiple initial angles
        best_outer_val = float('inf')
        best_outer_params = [cx_avg, cy_avg, 0.0]

        for init_angle in [0, 30, 60, 90, 120, 150]:
            if time.time() > deadline - 30:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            def obj_outer(params):
                return compute_required_side(initial_inner, (params[0], params[1]), params[2])

            try:
                res = minimize(obj_outer, init_params, method='Nelder-Mead',
                              options={'maxiter': 500, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 1000})
                if res.fun < best_outer_val:
                    best_outer_val = res.fun
                    best_outer_params = res.x
            except Exception:
                pass

        outer_center = (best_outer_params[0], best_outer_params[1])
        outer_angle = best_outer_params[2]

        # Phase 2: Joint optimization - move inner hexagons to tighten packing
        # Use a simple local search with overlap penalty
        def objective_joint(params):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            side = compute_required_side(inner_list, (ocx, ocy), oangle)
            penalty = compute_overlap_penalty(inner_list)
            return side + 1000.0 * penalty

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint(current_params)
        best_params = current_params.copy()
        best_val_joint = current_val

        # Simulated annealing
        T = 0.5
        T_min = 0.001
        cooling = 0.99995

        # Estimate time per iteration
        t0 = time.time()
        for i in range(10):
            neighbor = current_params.copy()
            dim_idx = np.random.randint(0, N*3+3)
            if dim_idx >= N*3+2:
                delta = np.random.normal(0, 1.0)
            else:
                delta = np.random.normal(0, 0.1)
            neighbor[dim_idx] += delta
            _ = objective_joint(neighbor)
        t1 = time.time()
        avg_time_per_iter = (t1 - t0) / 10

        # Determine max iterations based on remaining time
        remaining_time = deadline - 10 - time.time()
        max_iter = max(100, int(remaining_time / max(avg_time_per_iter, 1e-6)))
        max_iter = min(max_iter, 20000)

        for i in range(max_iter):
            if time.time() > deadline - 10:
                break

            neighbor = current_params.copy()

            # Random perturbation
            if np.random.random() < 0.8:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, 0.12)
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 1.5)
                else:
                    delta = np.random.normal(0, 0.08)
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / T):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val

            T *= cooling
            if T < T_min:
                break

        best_inner_joint = []
        for i in range(N):
            x = best_params[i*3]
            y = best_params[i*3+1]
            a = best_params[i*3+2]
            best_inner_joint.append([x, y, a])

        best_ocx = best_params[N*3]
        best_ocy = best_params[N*3+1]
        best_oangle = best_params[N*3+2]

        # Phase 3: Final polish on outer container
        def objective_outer_final(params):
            return compute_required_side(best_inner_joint, (params[0], params[1]), params[2])

        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        polish_starts = [
            [best_ocx, best_ocy, best_oangle],
            [best_ocx, best_ocy, best_oangle + 5],
            [best_ocx, best_ocy, best_oangle - 5],
            [best_ocx + 0.03, best_ocy, best_oangle],
            [best_ocx - 0.03, best_ocy, best_oangle],
            [best_ocx, best_ocy + 0.03, best_oangle],
            [best_ocx, best_ocy - 0.03, best_oangle],
        ]

        for ps in polish_starts:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(objective_outer_final, ps, method='Nelder-Mead',
                              options={'maxiter': 1500, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 3000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness
        if check_all_disjoint(final_inner, eps=1e-5):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Fall back to initial layout if overlapping
            fallback_side = compute_required_side(initial_inner, (cx_avg, cy_avg), 0.0)
            if fallback_side < best_global_side:
                best_global_side = fallback_side
                best_global_inner = initial_inner
                best_global_center = (cx_avg, cy_avg)
                best_global_angle = 0.0

    if best_global_inner is None:
        # Fallback: use first candidate
        axial_coords = candidates[0][0][:N]
        centers = []
        for q, r in axial_coords:
            x = 1.5 * q
            y = SQRT3 * (r + q / 2.0)
            centers.append((x, y))
        best_global_inner = [[x, y, 0.0] for x, y in centers]
        cx_avg = sum(c[0] for c in centers) / len(centers)
        cy_avg = sum(c[1] for c in centers) / len(centers)
        best_global_center = (cx_avg, cy_avg)
        best_global_angle = 0.0
        best_global_side = compute_required_side(best_global_inner, best_global_center, best_global_angle)

    return best_global_inner, best_global_center, best_global_side, best_global_angle
>>>>>>> REPLACE
