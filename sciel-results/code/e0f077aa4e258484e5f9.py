"""Packing 12 unit hexagons in a regular hexagon; minimize outer side length.
Combined approach: physics-based repulsion + SAT overlap detection + outer-side minimization.
"""
import numpy as np
from scipy.optimize import minimize
import time

N = 12
DEADLINE = 145.0  # safety margin before 150s kill


def hexagon_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx, cy)."""
    angle_rad = np.radians(angle_deg)
    angles = angle_rad + np.arange(6) * np.pi / 3
    return np.column_stack([cx + side * np.cos(angles), cy + side * np.sin(angles)])


def _sat_overlap_depth(v1, v2):
    """
    Compute the minimum translation distance to separate two convex polygons v1, v2.
    Returns 0 if no overlap, positive value if overlapping.
    """
    def get_outward_normals(verts):
        normals = []
        n = len(verts)
        center = np.mean(verts, axis=0)
        for i in range(n):
            p1 = verts[i]
            p2 = verts[(i + 1) % n]
            edge = p2 - p1
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-15:
                continue
            normal = normal / norm
            mid = (p1 + p2) / 2
            if np.dot(normal, center - mid) < 0:
                normal = -normal
            normals.append(normal)
        return np.array(normals)

    n1 = get_outward_normals(v1)
    n2 = get_outward_normals(v2)
    all_normals = np.vstack([n1, n2])

    min_overlap = float('inf')
    for normal in all_normals:
        proj1 = v1 @ normal
        proj2 = v2 @ normal
        min1, max1 = proj1.min(), proj1.max()
        min2, max2 = proj2.min(), proj2.max()
        if max1 <= min2 or max2 <= min1:
            return 0.0
        overlap = min(max1, max2) - max(min1, min2)
        min_overlap = min(min_overlap, overlap)

    if min_overlap == float('inf'):
        return 0.0
    return min_overlap


def min_outer_side_for_points(points, outer_center, outer_angle_deg):
    """
    Compute minimum side length s of a regular hexagon centered at outer_center
    with rotation outer_angle_deg that contains all given points.
    """
    cx, cy = outer_center
    angle_rad = np.radians(outer_angle_deg)
    normal_angles = angle_rad + np.pi / 6 + np.arange(6) * np.pi / 3
    normals = np.column_stack([np.cos(normal_angles), np.sin(normal_angles)])

    rel_points = points - np.array([cx, cy])
    dots = rel_points @ normals.T
    max_dot = np.max(dots, axis=0)
    r_needed = np.max(max_dot)
    if r_needed < 0:
        r_needed = 0.0
    s = 2.0 / np.sqrt(3.0) * r_needed
    return s


def compute_min_side(inner, outer_center, outer_angle_deg):
    """Compute minimum outer side length for given inner configuration."""
    all_points = []
    for x, y, ang in inner:
        v = hexagon_vertices(x, y, 1.0, ang)
        all_points.append(v)
    all_points = np.vstack(all_points)
    return min_outer_side_for_points(all_points, outer_center, outer_angle_deg)


def check_disjoint(inner, tol=1e-6):
    """Check if all inner hexagons are disjoint (no overlapping interiors)."""
    for i in range(N):
        for j in range(i + 1, N):
            v1 = hexagon_vertices(inner[i][0], inner[i][1], 1.0, inner[i][2])
            v2 = hexagon_vertices(inner[j][0], inner[j][1], 1.0, inner[j][2])
            depth = _sat_overlap_depth(v1, v2)
            if depth > tol:
                return False
    return True


def physics_objective(params):
    """
    Physics-based objective: energy = repulsion between overlapping hexagons
    + attraction toward center (to keep them compact) + small spread term.
    params: [x1, y1, a1, x2, y2, a2, ..., x12, y12, a12]
    """
    pos = params[:2 * N].reshape(N, 2)
    angles = params[2 * N: 3 * N]

    energy = 0.0

    # Repulsion: for each pair, if overlapping, add squared overlap depth
    for i in range(N):
        v1 = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
        for j in range(i + 1, N):
            v2 = hexagon_vertices(pos[j, 0], pos[j, 1], 1.0, angles[j])
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                energy += depth * depth * 100.0

    # Containment: soft constraint to keep hexagons within a radius
    # Use the current best side length as a guide
    center = np.array([0.0, 0.0])
    for i in range(N):
        d = np.linalg.norm(pos[i] - center)
        # Soft boundary at ~4.0 (roughly half the expected outer side)
        if d > 4.0:
            energy += (d - 4.0) ** 2 * 5.0

    # Small regularization to avoid degenerate configurations
    energy += 0.001 * np.sum(pos ** 2)

    return energy


def physics_repulsion_force(pos, angles):
    """
    Compute repulsion forces on each hexagon based on SAT overlap.
    Returns array of shape (N, 2) with force vectors.
    """
    forces = np.zeros((N, 2))
    for i in range(N):
        v1 = hexagon_vertices(pos[i, 0], pos[i, 1], 1.0, angles[i])
        for j in range(i + 1, N):
            v2 = hexagon_vertices(pos[j, 0], pos[j, 1], 1.0, angles[j])
            depth = _sat_overlap_depth(v1, v2)
            if depth > 0:
                # Direction from j to i
                direction = pos[i] - pos[j]
                dist = np.linalg.norm(direction)
                if dist < 1e-10:
                    direction = np.array([1.0, 0.0])
                    dist = 1.0
                direction /= dist
                # Force magnitude proportional to depth
                force_mag = depth * 50.0
                forces[i] += force_mag * direction
                forces[j] -= force_mag * direction
    return forces


def construct():
    start_time = time.time()

    best_score = float('inf')
    best_inner = None
    best_center = (0.0, 0.0)
    best_side = 40.0
    best_angle = 0.0

    def try_config(inner, center, angle):
        nonlocal best_score, best_inner, best_center, best_side, best_angle
        s = compute_min_side(inner, center, angle)
        if s < best_score:
            best_score = s
            best_inner = [list(x) for x in inner]
            best_center = center
            best_side = s
            best_angle = angle

    sqrt3 = np.sqrt(3)

    # --- Structured initial configurations ---

    # Config 1: 3x4 hexagonal grid
    positions = []
    for i in range(3):
        for j in range(4):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions.append((x, y))
    cx = np.mean([p[0] for p in positions])
    cy = np.mean([p[1] for p in positions])
    inner_grid = [[p[0] - cx, p[1] - cy, 0.0] for p in positions]
    try_config(inner_grid, (0.0, 0.0), 0.0)

    # Config 2: 4x3 hexagonal grid
    positions2 = []
    for i in range(4):
        for j in range(3):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions2.append((x, y))
    cx2 = np.mean([p[0] for p in positions2])
    cy2 = np.mean([p[1] for p in positions2])
    inner_grid2 = [[p[0] - cx2, p[1] - cy2, 0.0] for p in positions2]
    try_config(inner_grid2, (0.0, 0.0), 0.0)

    # Config 3: Center + 2 rings
    positions3 = [(0, 0)]
    for k in range(6):
        ang = k * np.pi / 3
        positions3.append((2 * sqrt3 * np.cos(ang), 2 * sqrt3 * np.sin(ang)))
    for k in range(5):
        ang = k * 2 * np.pi / 5 + np.pi / 5
        positions3.append((4 * sqrt3 * np.cos(ang), 4 * sqrt3 * np.sin(ang)))
    cx3 = np.mean([p[0] for p in positions3])
    cy3 = np.mean([p[1] for p in positions3])
    inner_ring = [[p[0] - cx3, p[1] - cy3, 0.0] for p in positions3]
    try_config(inner_ring, (0.0, 0.0), 0.0)

    # Config 4: 2x6 rectangular-ish
    positions4 = []
    for i in range(2):
        for j in range(6):
            x = j * 2 * sqrt3 + (i % 2) * sqrt3
            y = i * 3
            positions4.append((x, y))
    cx4 = np.mean([p[0] for p in positions4])
    cy4 = np.mean([p[1] for p in positions4])
    inner_rect = [[p[0] - cx4, p[1] - cy4, 0.0] for p in positions4]
    try_config(inner_rect, (0.0, 0.0), 0.0)

    # Try different outer rotations
    for cfg in [inner_grid, inner_grid2, inner_ring, inner_rect]:
        if time.time() - start_time > DEADLINE:
            break
        for outer_ang in [0, 30, 60, 90, 120, 150]:
            if time.time() - start_time > DEADLINE:
                break
            try_config(cfg, (0.0, 0.0), float(outer_ang))

    # --- Physics-based relaxation from best config ---
    if best_inner is not None:
        if time.time() - start_time < DEADLINE - 15:
            init_pos = np.array([[x, y] for x, y, a in best_inner])
            init_angles = np.array([a for x, y, a in best_inner])

            # Multiple restarts with perturbation
            n_restarts = 10
            for r in range(n_restarts):
                if time.time() - start_time > DEADLINE - 15:
                    break

                if r == 0:
                    x0_pos = init_pos.copy()
                    x0_ang = init_angles.copy()
                else:
                    scale = 0.4 * (r + 1)
                    x0_pos = init_pos + scale * np.random.randn(N, 2)
                    x0_ang = init_angles + 0.3 * np.random.randn(N)

                # Physics-based iterative relaxation
                pos = x0_pos.copy()
                angles = x0_ang.copy()
                lr = 0.05
                n_iter = 200

                for it in range(n_iter):
                    if time.time() - start_time > DEADLINE - 15:
                        break

                    forces = physics_repulsion_force(pos, angles)

                    # Add centering force
                    center = np.array([0.0, 0.0])
                    for i in range(N):
                        d = pos[i] - center
                        dist = np.linalg.norm(d)
                        if dist > 4.0:
                            forces[i] -= (d / dist) * (dist - 4.0) * 5.0

                    # Update positions
                    pos += lr * forces
                    lr *= 0.995  # decay learning rate

                # After physics relaxation, do a local optimization with L-BFGS-B
                if time.time() - start_time < DEADLINE - 10:
                    params_init = np.concatenate([pos.flatten(), angles])

                    def objective_full(params):
                        p = params[:2 * N].reshape(N, 2)
                        a = params[2 * N: 3 * N]
                        energy = 0.0
                        for i in range(N):
                            v1 = hexagon_vertices(p[i, 0], p[i, 1], 1.0, a[i])
                            for j in range(i + 1, N):
                                v2 = hexagon_vertices(p[j, 0], p[j, 1], 1.0, a[j])
                                depth = _sat_overlap_depth(v1, v2)
                                if depth > 0:
                                    energy += depth * depth * 100.0
                        # Containment
                        for i in range(N):
                            d = np.linalg.norm(p[i])
                            if d > 4.0:
                                energy += (d - 4.0) ** 2 * 5.0
                        return energy

                    try:
                        result = minimize(
                            objective_full,
                            params_init,
                            method='L-BFGS-B',
                            options={'maxiter': 300, 'ftol': 1e-10, 'gtol': 1e-8}
                        )
                        pos_opt = result.x[:2 * N].reshape(N, 2)
                        ang_opt = result.x[2 * N: 3 * N]
                        inner_opt = [[float(pos_opt[i, 0]), float(pos_opt[i, 1]), float(ang_opt[i])] for i in range(N)]

                        # Verify disjointness
                        if check_disjoint(inner_opt, tol=1e-5):
                            s_opt = compute_min_side(inner_opt, (0.0, 0.0), 0.0)
                            if s_opt < best_score:
                                best_score = s_opt
                                best_inner = inner_opt
                                best_center = (0.0, 0.0)
                                best_side = s_opt
                                best_angle = 0.0
                    except Exception:
                        pass

    # --- Final refinement: try outer rotations on best inner config ---
    if best_inner is not None:
        if time.time() - start_time < DEADLINE - 5:
            for outer_ang in [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150]:
                if time.time() - start_time > DEADLINE - 5:
                    break
                try_config(best_inner, (0.0, 0.0), float(outer_ang))

    # Add safety margin
    best_side += 0.05

    return best_inner, best_center, best_side, best_angle
