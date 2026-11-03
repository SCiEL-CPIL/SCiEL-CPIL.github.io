<<<<<<< SEARCH
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 140.0  # Safety margin before 150s limit

    # Helper: get vertices of a regular hexagon with side length 1
    def hex_vertices(cx, cy, angle_deg):
        angle_rad = math.radians(angle_deg)
        verts = []
        for i in range(6):
            a = angle_rad + i * math.pi / 3
            x = cx + math.cos(a)
            y = cy + math.sin(a)
            verts.append((x, y))
        return np.array(verts)

    # Helper: check if two hexagons overlap using SAT (Separating Axis Theorem)
    def hexagons_overlap(v1, v2):
        """Check if two convex hexagons overlap using SAT."""
        axes = []
        for i in range(6):
            edge = v1[i] - v1[(i+1) % 6]
            normal = np.array([-edge[1], edge[0]])
            axes.append(normal)
            edge2 = v2[i] - v2[(i+1) % 6]
            normal2 = np.array([-edge2[1], edge2[0]])
            axes.append(normal2)

        for axis in axes:
            proj1 = np.dot(v1, axis)
            proj2 = np.dot(v2, axis)
            min1, max1 = proj1.min(), proj1.max()
            min2, max2 = proj2.min(), proj2.max()
            if max1 < min2 or max2 < min1:
                return False
        return True

    def check_all_overlaps(positions, angles):
        """Check if any pair of hexagons overlaps."""
        verts = [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]
        for i in range(N):
            for j in range(i+1, N):
                if hexagons_overlap(verts[i], verts[j]):
                    return True
        return False

    # Helper: compute minimum outer hexagon side length needed to contain all inner hexagons
    def compute_min_side(positions, angles, outer_angle_deg):
        """
        positions: list of (x, y) for each inner hexagon
        angles: list of rotation angles for each inner hexagon
        outer_angle_deg: rotation of outer hexagon
        Returns the minimum side length of outer hexagon to contain all inner hexagons.
        """
        all_verts = []
        for i in range(N):
            cx, cy = positions[i]
            a = angles[i]
            verts = hex_vertices(cx, cy, a)
            all_verts.append(verts)

        if not all_verts:
            return 0.0

        all_verts = np.vstack(all_verts)

        theta = math.radians(outer_angle_deg)
        normals = []
        for k in range(6):
            a = theta + k * math.pi / 3
            normals.append(np.array([math.cos(a), math.sin(a)]))

        max_proj = 0
        for normal in normals:
            proj = np.dot(all_verts, normal)
            max_proj = max(max_proj, proj.max())

        side = 2 * max_proj / math.sqrt(3)
        return side

    def get_config_score(positions, angles, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(positions, angles):
            return float('inf')
        return compute_min_side(positions, angles, outer_angle)

    # === NEW: Graph-based honeycomb layout approach ===
    # Define honeycomb adjacency for 12 hexagons in a compact arrangement
    # We use a hexagonal lattice coordinate system (axial coordinates)
    # and map them to Cartesian coordinates

    def axial_to_cartesian(q, r):
        """Convert axial coordinates to Cartesian for a hex grid with side length 1."""
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    def generate_honeycomb_layouts():
        """Generate multiple honeycomb-like layouts for 12 hexagons."""
        layouts = []

        # Layout 1: 3x4 grid (rows of 4, 3, 4, 3 or similar)
        # Use axial coordinates for a compact hexagonal shape
        # Center hexagon + 6 neighbors + 5 more = 12
        # Hexagonal numbers: 1, 7, 19... so we need a partial hexagon

        # Option A: Compact hexagonal shape (center + ring1 + part of ring2)
        # Ring 0: (0,0)
        # Ring 1: 6 hexagons around center
        # Ring 2: 12 hexagons, take 5 of them to make 12 total
        layout_a = [(0, 0)]  # center
        # Ring 1 directions in axial coords
        ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
        layout_a.extend(ring1_dirs)  # 7 total
        # Ring 2: extend from ring 1
        ring2_candidates = []
        for (q, r) in ring1_dirs:
            for dq, dr in ring1_dirs:
                nq, nr = q + dq, r + dr
                if (nq, nr) not in layout_a and (nq, nr) not in ring2_candidates:
                    ring2_candidates.append((nq, nr))
        # Take 5 from ring 2 to make 12 total
        layout_a.extend(ring2_candidates[:5])
        layouts.append(layout_a)

        # Option B: 3 rows of 4 (staggered)
        layout_b = []
        for row in range(3):
            offset = (row % 2) * 0.5  # stagger
            for col in range(4):
                q = col + offset
                r = row
                layout_b.append((q, r))
        layouts.append(layout_b)

        # Option C: 4 rows of 3 (staggered)
        layout_c = []
        for row in range(4):
            offset = (row % 2) * 0.5
            for col in range(3):
                q = col + offset
                r = row
                layout_c.append((q, r))
        layouts.append(layout_c)

        # Option D: 2 rows of 6
        layout_d = []
        for row in range(2):
            offset = (row % 2) * 0.5
            for col in range(6):
                q = col + offset
                r = row
                layout_d.append((q, r))
        layouts.append(layout_d)

        # Option E: 6 rows of 2
        layout_e = []
        for row in range(6):
            offset = (row % 2) * 0.5
            for col in range(2):
                q = col + offset
                r = row
                layout_e.append((q, r))
        layouts.append(layout_e)

        # Option F: More compact - try to minimize bounding hexagon
        # Use a "flower" pattern: center + 6 + 5 arranged symmetrically
        layout_f = [(0, 0)]
        layout_f.extend(ring1_dirs)
        # Choose 5 from ring 2 that are most symmetric
        # Pick every other one from ring 2 for symmetry
        ring2_selected = []
        for i, (q, r) in enumerate(ring2_candidates):
            if i % 2 == 0:
                ring2_selected.append((q, r))
            if len(ring2_selected) == 5:
                break
        layout_f.extend(ring2_selected)
        layouts.append(layout_f)

        return layouts

    def layout_to_positions(layout, rotation_deg=0.0):
        """Convert a honeycomb layout (axial coords) to Cartesian positions, centered."""
        positions = []
        for q, r in layout:
            x, y = axial_to_cartesian(q, r)
            positions.append((x, y))

        # Center the layout
        cx = sum(p[0] for p in positions) / len(positions)
        cy = sum(p[1] for p in positions) / len(positions)
        positions = [(x - cx, y - cy) for x, y in positions]

        # Apply rotation if needed
        if rotation_deg != 0:
            theta = math.radians(rotation_deg)
            cos_t, sin_t = math.cos(theta), math.sin(theta)
            rotated = []
            for x, y in positions:
                nx = x * cos_t - y * sin_t
                ny = x * sin_t + y * cos_t
                rotated.append((nx, ny))
            positions = rotated

        return positions

    def optimize_layout(positions, angles, outer_angle, max_iter=2000):
        """
        Optimize a layout using gradient-free search with overlap constraints.
        Uses a penalty-based approach with local search.
        """
        pos = np.array(positions, dtype=float)
        n = len(pos)

        def objective(x):
            if time.time() > deadline - 5:
                return 1e10
            pos = x.reshape(n, 2)
            verts_list = [hex_vertices(pos[i][0], pos[i][1], angles[i]) for i in range(n)]

            # Check overlaps with penalty
            overlap_penalty = 0.0
            for i in range(n):
                for j in range(i+1, n):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        # Compute distance-based penalty
                        d = pos[j] - pos[i]
                        dist = np.linalg.norm(d)
                        overlap_penalty += max(0, 2.0 - dist) ** 2

            if overlap_penalty > 1e-6:
                return 1000.0 + overlap_penalty

            return compute_min_side(list(map(tuple, pos)), angles, outer_angle)

        try:
            result = minimize(objective, pos.flatten(), method='Nelder-Mead',
                             options={'maxiter': max_iter, 'xatol': 1e-5, 'fatol': 1e-5})
            final_pos = result.x.reshape(n, 2)

            # Verify validity
            verts_list = [hex_vertices(final_pos[i][0], final_pos[i][1], angles[i]) for i in range(n)]
            valid = True
            for i in range(n):
                for j in range(i+1, n):
                    if hexagons_overlap(verts_list[i], verts_list[j]):
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                score = compute_min_side(list(map(tuple, final_pos)), angles, outer_angle)
                return list(map(tuple, final_pos)), angles, outer_angle, score
            else:
                return None
        except Exception:
            return None

    def local_search(positions, angles, outer_angle, max_iters=5000):
        """
        Local search: try small perturbations to each hexagon position.
        """
        pos = np.array(positions, dtype=float)
        n = len(pos)
        best_score = get_config_score(list(map(tuple, pos)), angles, outer_angle)
        best_pos = pos.copy()

        if best_score == float('inf'):
            return list(map(tuple, pos)), angles, outer_angle, best_score

        step_size = 0.1
        for it in range(max_iters):
            if time.time() > deadline - 5:
                break

            improved = False
            for i in range(n):
                # Try small random perturbations
                for _ in range(10):
                    dx = np.random.uniform(-step_size, step_size)
                    dy = np.random.uniform(-step_size, step_size)
                    new_pos = pos.copy()
                    new_pos[i][0] += dx
                    new_pos[i][1] += dy

                    new_score = get_config_score(list(map(tuple, new_pos)), angles, outer_angle)
                    if new_score < best_score:
                        best_score = new_score
                        best_pos = new_pos.copy()
                        pos = new_pos.copy()
                        improved = True
                        break

                if improved:
                    break

            if not improved:
                step_size *= 0.8
                if step_size < 1e-6:
                    break

        return list(map(tuple, best_pos)), angles, outer_angle, best_score

    # Main search: try multiple layouts and optimize each
    best_score = float('inf')
    best_positions = None
    best_angles = None
    best_outer_angle = 0.0

    layouts = generate_honeycomb_layouts()

    for layout in layouts:
        if time.time() > deadline - 20:
            break

        # Try multiple rotations and outer angles
        for rot_deg in [0, 15, 30, 45, 60, 75, 90]:
            if time.time() > deadline - 10:
                break

            for outer_angle in [0, 30, 60, 90, 120, 150]:
                if time.time() > deadline - 5:
                    break

                positions = layout_to_positions(layout, rotation_deg=rot_deg)
                angles = [0.0] * N

                score = get_config_score(positions, angles, outer_angle)

                if score < best_score:
                    best_score = score
                    best_positions = positions
                    best_angles = angles
                    best_outer_angle = outer_angle

                # If valid, try to optimize
                if score < float('inf') and time.time() < deadline - 15:
                    # Try Nelder-Mead optimization
                    opt_result = optimize_layout(positions, angles, outer_angle, max_iter=500)
                    if opt_result is not None:
                        opt_pos, opt_ang, opt_oa, opt_score = opt_result
                        if opt_score < best_score:
                            best_score = opt_score
                            best_positions = opt_pos
                            best_angles = opt_ang
                            best_outer_angle = opt_oa

                    # Try local search if time permits
                    if time.time() < deadline - 10:
                        ls_result = local_search(best_positions, best_angles, best_outer_angle, max_iters=2000)
                        ls_pos, ls_ang, ls_oa, ls_score = ls_result
                        if ls_score < best_score:
                            best_score = ls_score
                            best_positions = ls_pos
                            best_angles = ls_ang
                            best_outer_angle = ls_oa

    # Fallback: If no valid configuration was found
    if best_positions is None or best_score == float('inf'):
        # Use a simple grid as fallback
        positions = []
        angles = []
        for i in range(N):
            r = i // 4
            c = i % 4
            x = c * 2.2 + (r % 2) * 1.1
            y = r * 1.9
            positions.append((x, y))
            angles.append(0.0)

        cx = sum(p[0] for p in positions) / N
        cy = sum(p[1] for p in positions) / N
        positions = [(x - cx, y - cy) for x, y in positions]

        best_positions = positions
        best_angles = angles
        best_outer_angle = 0.0
        best_score = compute_min_side(positions, angles, 0.0)

    # Final cleanup: center the configuration
    if best_positions:
        cx = sum(p[0] for p in best_positions) / N
        cy = sum(p[1] for p in best_positions) / N
        final_positions = [(x - cx, y - cy) for x, y in best_positions]
    else:
        final_positions = [(0, 0)] * N
        best_angles = [0] * N
        best_score = 10.0

    inner = [[final_positions[i][0], final_positions[i][1], best_angles[i]] for i in range(N)]

    return inner, (0.0, 0.0), best_score, best_outer_angle
# EVOLVE-BLOCK-END
=======
# EVOLVE-BLOCK-START
def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin before 150s limit

    # Precomputed unit hexagon vertices (angle 0)
    unit_hex_verts = np.array([[math.cos(i * math.pi / 3), math.sin(i * math.pi / 3)] for i in range(6)])

    def hex_vertices(cx, cy, angle_deg):
        if angle_deg == 0.0:
            return unit_hex_verts + np.array([cx, cy])
        angle_rad = math.radians(angle_deg)
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        rot = np.array([[c, -s], [s, c]])
        return unit_hex_verts @ rot.T + np.array([cx, cy])

    def fast_overlap(v1, v2):
        """Fast SAT check for two hexagons."""
        # 12 axes (6 from each hex)
        for i in range(6):
            # Axis from hex1 edge
            edge = v1[i] - v1[(i+1) % 6]
            axis = np.array([-edge[1], edge[0]])
            proj1 = v1 @ axis
            proj2 = v2 @ axis
            if proj1.max() < proj2.min() or proj2.max() < proj1.min():
                return False
            # Axis from hex2 edge
            edge2 = v2[i] - v2[(i+1) % 6]
            axis2 = np.array([-edge2[1], edge2[0]])
            proj1b = v1 @ axis2
            proj2b = v2 @ axis2
            if proj1b.max() < proj2b.min() or proj2b.max() < proj1b.min():
                return False
        return True

    def check_all_overlaps(verts_list):
        """Check if any pair of hexagons overlaps."""
        for i in range(N):
            for j in range(i+1, N):
                if fast_overlap(verts_list[i], verts_list[j]):
                    return True
        return False

    def compute_min_side(verts_list, outer_angle_deg):
        """Compute minimum outer hexagon side length to contain all inner hexagons."""
        all_verts = np.vstack(verts_list)
        theta = math.radians(outer_angle_deg)
        max_proj = 0.0
        for k in range(6):
            a = theta + k * math.pi / 3
            normal = np.array([math.cos(a), math.sin(a)])
            proj = all_verts @ normal
            max_proj = max(max_proj, proj.max())
        return 2.0 * max_proj / math.sqrt(3)

    def get_score(verts_list, outer_angle):
        """Return side length if valid, else infinity."""
        if check_all_overlaps(verts_list):
            return float('inf')
        return compute_min_side(verts_list, outer_angle)

    def positions_to_verts(positions, angles):
        return [hex_vertices(p[0], p[1], a) for p, a in zip(positions, angles)]

    # Best honeycomb layout: center + 6 ring1 + 5 symmetric ring2
    ring1_dirs = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1)]
    
    def axial_to_cart(q, r):
        x = math.sqrt(3) * (q + r / 2.0)
        y = 1.5 * r
        return x, y

    # Generate the compact flower layout
    layout = [(0, 0)] + ring1_dirs
    # Ring 2 candidates
    ring2_set = set()
    for (q, r) in ring1_dirs:
        for dq, dr in ring1_dirs:
            nq, nr = q + dq, r + dr
            if (nq, nr) not in layout and (nq, nr) not in ring2_set:
                ring2_set.add((nq, nr))
    ring2_list = sorted(ring2_set)
    # Pick 5 symmetric ones (every other)
    for i in range(0, len(ring2_list), 2):
        layout.append(ring2_list[i])
        if len(layout) == 12:
            break

    # Convert to centered Cartesian
    positions = []
    for q, r in layout:
        x, y = axial_to_cart(q, r)
        positions.append((x, y))
    cx = sum(p[0] for p in positions) / N
    cy = sum(p[1] for p in positions) / N
    positions = [(x - cx, y - cy) for x, y in positions]

    # Try different outer angles to find best initial configuration
    best_score = float('inf')
    best_positions = positions
    best_angles = [0.0] * N
    best_outer_angle = 0.0

    for outer_angle in [0, 30, 60, 90, 120, 150]:
        verts = positions_to_verts(positions, best_angles)
        score = get_score(verts, outer_angle)
        if score < best_score:
            best_score = score
            best_outer_angle = outer_angle

    # Local search optimization
    pos = np.array(best_positions, dtype=float)
    n = N
    angles = best_angles
    outer_angle = best_outer_angle
    current_score = best_score

    step_size = 0.15
    no_improve_count = 0
    max_no_improve = 200

    while time.time() < deadline - 10:
        improved = False
        for i in range(n):
            if time.time() > deadline - 10:
                break
            for attempt in range(8):
                dx = np.random.uniform(-step_size, step_size)
                dy = np.random.uniform(-step_size, step_size)
                new_pos = pos.copy()
                new_pos[i][0] += dx
                new_pos[i][1] += dy

                new_verts = positions_to_verts(list(map(tuple, new_pos)), angles)
                new_score = get_score(new_verts, outer_angle)

                if new_score < current_score:
                    current_score = new_score
                    pos = new_pos.copy()
                    improved = True
                    break

        if improved:
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count >= max_no_improve:
                break
            step_size *= 0.85
            if step_size < 0.001:
                break

    # Try optimizing outer angle after position optimization
    final_verts = positions_to_verts(list(map(tuple, pos)), angles)
    for outer_angle_try in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
        s = get_score(final_verts, outer_angle_try)
        if s < current_score:
            current_score = s
            outer_angle = outer_angle_try

    # Final centering
    cx = sum(p[0] for p in pos) / N
    cy = sum(p[1] for p in pos) / N
    final_positions = [(x - cx, y - cy) for x, y in pos]

    inner = [[final_positions[i][0], final_positions[i][1], angles[i]] for i in range(N)]
    return inner, (0.0, 0.0), current_score, outer_angle
# EVOLVE-BLOCK-END
>>>>>>> REPLACE
