import math
import time
import numpy as np
from scipy.optimize import minimize

N = 12

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    sqrt3 = math.sqrt(3)

    # Precompute unit hexagon vertices (vertex at 0 deg, i.e., pointy-right)
    unit_verts_0 = np.array([[math.cos(math.radians(a)), math.sin(math.radians(a))] for a in range(0, 360, 60)])

    def get_hex_verts(cx, cy, side, angle_rad):
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        rot = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        verts = unit_verts_0 @ rot.T * side
        verts += np.array([cx, cy])
        return verts

    def calc_min_outer_side(positions, angles, outer_angle_rad=0.0):
        """
        Calculate the minimum side length of a regular hexagon centered at (0,0)
        with rotation outer_angle_rad that contains all inner hexagons.
        """
        normals = []
        for k in range(6):
            ang = outer_angle_rad + math.pi/6 + k * math.pi/3
            normals.append([math.cos(ang), math.sin(ang)])
        normals = np.array(normals)

        max_proj = 0.0
        for i in range(N):
            cx, cy = positions[i]
            ang = angles[i]
            verts = get_hex_verts(cx, cy, 1.0, ang)
            projs = verts @ normals.T
            max_proj_i = np.max(projs)
            if max_proj_i > max_proj:
                max_proj = max_proj_i

        return max_proj * 2.0 / math.sqrt(3)

    def find_best_outer_angle(positions, angles, n_samples=72):
        """
        Find the optimal rotation of the outer hexagon that minimizes its side length.
        The outer hexagon has 6-fold symmetry, so we only need to search [0, pi/3).
        """
        best_side = float('inf')
        best_angle = 0.0
        # Search in [0, pi/3) with fine resolution
        test_angles = np.linspace(0, math.pi/3, n_samples, endpoint=False)
        for test_angle in test_angles:
            side = calc_min_outer_side(positions, angles, test_angle)
            if side < best_side:
                best_side = side
                best_angle = test_angle
        return best_side, best_angle

    def sat_gap(poly1, poly2):
        """
        Returns the minimum separation distance between two convex polygons.
        If they overlap, returns a negative value.
        If separated, returns positive distance.
        """
        min_gap = float('inf')

        for i in range(6):
            edge = poly1[(i+1)%6] - poly1[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap

        for i in range(6):
            edge = poly2[(i+1)%6] - poly2[i]
            normal = np.array([-edge[1], edge[0]])
            norm = np.linalg.norm(normal)
            if norm < 1e-9:
                continue
            axis = normal / norm
            proj1 = poly1 @ axis
            proj2 = poly2 @ axis
            gap = max(np.min(proj1) - np.max(proj2), np.min(proj2) - np.max(proj1))
            if gap < 0:
                min_gap = min(min_gap, gap)
            else:
                return gap

        return min_gap

    def is_valid(positions, angles, tol=1e-5):
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        for i in range(N):
            for j in range(i+1, N):
                if sat_gap(polys[i], polys[j]) < -tol:
                    return False
        return True

    def objective_with_penalty(params, penalty_weight=50.0):
        """
        Objective: outer hexagon side length (with optimal rotation) + penalty for overlaps.
        params: [x0, y0, a0, x1, y1, a1, ..., x11, y11, a11]
        """
        if time.time() > deadline - 5:
            return 1e6

        positions = params[:2*N].reshape(N, 2)
        angles = params[2*N:]

        # Compute penalty for overlaps
        polys = [get_hex_verts(positions[i][0], positions[i][1], 1.0, angles[i]) for i in range(N)]
        penalty = 0.0
        for i in range(N):
            for j in range(i+1, N):
                gap = sat_gap(polys[i], polys[j])
                if gap < 0:
                    penalty += penalty_weight * gap * gap

        # Find best outer rotation
        side, _ = find_best_outer_angle(positions, angles, n_samples=36)
        return side + penalty

    # --- PHASE 1: Generate diverse valid configurations on hexagonal lattices ---

    best_config = None
    best_score = float('inf')
    best_outer_angle = 0.0

    def try_config(positions, angles):
        nonlocal best_config, best_score, best_outer_angle
        if not is_valid(positions, angles):
            return
        side, outer_ang = find_best_outer_angle(positions, angles, n_samples=72)
        if side < best_score:
            best_score = side
            best_config = (positions.copy(), angles.copy())
            best_outer_angle = outer_ang

    # Generate configurations from hexagonal lattice with various scales and rotations
    base_v1 = np.array([1.5, sqrt3/2])
    base_v2 = np.array([0.0, sqrt3])

    lattice_configs = []

    for scale in [0.95, 0.98, 1.0, 1.02, 1.05]:
        v1 = base_v1 * scale
        v2 = base_v2 * scale

        # Generate lattice points
        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)

        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = np.array(points[:N])
        centroid = np.mean(selected, axis=0)
        selected -= centroid

        # Try different global rotations of the entire cluster
        for rot_deg in range(0, 60, 15):
            rot_rad = math.radians(rot_deg)
            rot_mat = np.array([[math.cos(rot_rad), -math.sin(rot_rad)],
                                 [math.sin(rot_rad), math.cos(rot_rad)]])
            rotated = selected @ rot_mat.T
            angles = np.zeros(N)  # All hexagons aligned with the lattice
            lattice_configs.append((rotated, angles))

    # Also try configurations where individual hexagons have different orientations
    for scale in [0.98, 1.0, 1.02]:
        v1 = base_v1 * scale
        v2 = base_v2 * scale

        points = []
        for i in range(-4, 5):
            for j in range(-4, 5):
                p = i * v1 + j * v2
                points.append(p)

        points.sort(key=lambda p: p[0]**2 + p[1]**2)
        selected = np.array(points[:N])
        centroid = np.mean(selected, axis=0)
        selected -= centroid

        # Try with 30-degree rotated hexagons
        angles_30 = np.full(N, math.radians(30))
        lattice_configs.append((selected, angles_30))

        # Try with mixed orientations
        angles_mixed = np.zeros(N)
        angles_mixed[::2] = 0.0
        angles_mixed[1::2] = math.radians(30)
        lattice_configs.append((selected, angles_mixed))

    # Evaluate all lattice configurations
    for positions, angles in lattice_configs:
        if time.time() > deadline - 60:
            break
        try_config(positions, angles)

    # --- PHASE 2: Continuous refinement from best lattice config ---

    if best_config is not None:
        init_pos, init_ang = best_config

        # Refine with Nelder-Mead
        x0 = np.concatenate([init_pos.flatten(), init_ang])

        # Multiple refinement passes with different settings
        for nm_iter in [1000, 3000]:
            if time.time() > deadline - 30:
                break
            try:
                res = minimize(
                    objective_with_penalty,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': nm_iter, 'xatol': 1e-7, 'fatol': 1e-9, 'adaptive': True}
                )
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid(new_positions, new_angles):
                    side, outer_ang = find_best_outer_angle(new_positions, new_angles, n_samples=144)
                    if side < best_score:
                        best_score = side
                        best_config = (new_positions.copy(), new_angles.copy())
                        best_outer_angle = outer_ang
                        x0 = res.x.copy()
            except Exception:
                pass

        # Powell refinement
        if time.time() < deadline - 20:
            try:
                res = minimize(
                    objective_with_penalty,
                    x0,
                    method='Powell',
                    options={'maxiter': 2000, 'xatol': 1e-7, 'fatol': 1e-9}
                )
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid(new_positions, new_angles):
                    side, outer_ang = find_best_outer_angle(new_positions, new_angles, n_samples=144)
                    if side < best_score:
                        best_score = side
                        best_config = (new_positions.copy(), new_angles.copy())
                        best_outer_angle = outer_ang
            except Exception:
                pass

    # --- PHASE 3: If we don't have a good result, try random starts ---

    if best_config is None or best_score > 5.0:
        np.random.seed(42)
        random_seed = 42
        import random
        random.seed(random_seed)

        for attempt in range(20):
            if time.time() > deadline - 30:
                break

            # Random configuration
            positions = np.random.randn(N, 2) * 2.0
            positions -= np.mean(positions, axis=0)
            angles = np.random.uniform(0, 2*math.pi, N)

            x0 = np.concatenate([positions.flatten(), angles])

            try:
                res = minimize(
                    objective_with_penalty,
                    x0,
                    method='Nelder-Mead',
                    options={'maxiter': 1000, 'xatol': 1e-6, 'fatol': 1e-8, 'adaptive': True}
                )
                new_positions = res.x[:2*N].reshape(N, 2)
                new_angles = res.x[2*N:]
                if is_valid(new_positions, new_angles):
                    side, outer_ang = find_best_outer_angle(new_positions, new_angles, n_samples=72)
                    if side < best_score:
                        best_score = side
                        best_config = (new_positions.copy(), new_angles.copy())
                        best_outer_angle = outer_ang
            except Exception:
                pass

    # --- Final: Return the best configuration ---

    if best_config is None:
        # Fallback: return a simple valid configuration
        positions = np.zeros((N, 2))
        angles = np.zeros(N)
        # Place in a line (valid but not optimal)
        for i in range(N):
            positions[i, 0] = i * 2.0
        positions -= np.mean(positions, axis=0)
        side, outer_ang = find_best_outer_angle(positions, angles, n_samples=72)
        best_score = side
        best_outer_angle = outer_ang
        final_positions = positions
        final_angles = angles
    else:
        final_positions, final_angles = best_config

    # Final validation
    if not is_valid(final_positions, final_angles, tol=1e-4):
        # Try to fix by slightly expanding
        for i in range(N):
            for j in range(i+1, N):
                polys_i = get_hex_verts(final_positions[i][0], final_positions[i][1], 1.0, final_angles[i])
                polys_j = get_hex_verts(final_positions[j][0], final_positions[j][1], 1.0, final_angles[j])
                gap = sat_gap(polys_i, polys_j)
                if gap < 0:
                    # Push apart
                    dx = final_positions[j][0] - final_positions[i][0]
                    dy = final_positions[j][1] - final_positions[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-9:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    push = (-gap + 1e-4) / 2.0
                    final_positions[i][0] -= push * dx / dist
                    final_positions[i][1] -= push * dy / dist
                    final_positions[j][0] += push * dx / dist
                    final_positions[j][1] += push * dy / dist

    # Recompute final score
    final_side, final_outer_angle = find_best_outer_angle(final_positions, final_angles, n_samples=144)

    inner = []
    for i in range(N):
        angle_deg = math.degrees(final_angles[i]) % 360
        inner.append([final_positions[i][0], final_positions[i][1], angle_deg])

    return inner, (0.0, 0.0), final_side, math.degrees(final_outer_angle)

if __name__ == "__main__":
    construct()
