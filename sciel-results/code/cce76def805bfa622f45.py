"""Packing 12 unit hexagons in a regular hexagon - combinatorial grid + continuous refinement."""
import time
import math
import numpy as np
from itertools import combinations
from scipy.optimize import minimize

N = 12
SQRT3 = math.sqrt(3.0)
APOTHEM = SQRT3 / 2.0  # apothem of unit hexagon

# Precompute unit hexagon vertices (flat-top, angle 0)
UNIT_VERTS = np.array([
    [math.cos(math.radians(i * 60)), math.sin(math.radians(i * 60))]
    for i in range(6)
])

# Edge normals for flat-top hexagon (perpendicular to each edge, pointing outward)
# Edge midpoints are at angles 30, 90, 150, 210, 270, 330 degrees
EDGE_NORMALS = np.array([
    [math.cos(math.radians(30 + i * 60)), math.sin(math.radians(30 + i * 60))]
    for i in range(6)
])


def construct():
    start_time = time.time()
    deadline = start_time + 135.0  # Safety margin

    def get_vertices(cx, cy, angle_rad, side=1.0):
        """Get vertices of a regular hexagon centered at (cx,cy) with rotation angle_rad."""
        c, s = math.cos(angle_rad), math.sin(angle_rad)
        R = np.array([[c, -s], [s, c]])
        return np.array([cx, cy]) + side * (UNIT_VERTS @ R.T)

    def sat_overlap(v1, v2, n1, n2, eps=1e-9):
        """Check if two convex hexagons overlap using SAT. Returns True if overlapping."""
        for k in range(6):
            for axis in (n1[k], n2[k]):
                proj1 = v1 @ axis
                proj2 = v2 @ axis
                if proj1.max() < proj2.min() + eps or proj2.max() < proj1.min() + eps:
                    return False
        return True

    def check_all_overlap(centers, angles_rad, eps=1e-9):
        """Check if any pair of hexagons overlaps."""
        n = len(centers)
        all_verts = []
        all_normals = []
        for i in range(n):
            all_verts.append(get_vertices(centers[i][0], centers[i][1], angles_rad[i]))
            c, s = math.cos(angles_rad[i]), math.sin(angles_rad[i])
            R = np.array([[c, -s], [s, c]])
            all_normals.append(EDGE_NORMALS @ R.T)

        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                if dx * dx + dy * dy > 4.5:
                    continue
                if sat_overlap(all_verts[i], all_verts[j], all_normals[i], all_normals[j], eps):
                    return True
        return False

    def calc_L(centers, angles_rad):
        """Calculate minimum outer hexagon side length to enclose all inner hexagons."""
        max_proj = 0.0
        for i in range(N):
            verts = get_vertices(centers[i][0], centers[i][1], angles_rad[i])
            projs = verts @ EDGE_NORMALS.T
            local_max = projs.max()
            if local_max > max_proj:
                max_proj = local_max
        # Apothem of outer = max_proj, side length L = 2 * apothem / sqrt(3)
        return 2.0 * max_proj / SQRT3

    # Build candidate grid of centers
    # Use a hexagonal lattice with spacing slightly less than 2*apothem to allow tight packing
    # But we'll use exact lattice positions and check validity
    # Grid range: enough to cover a hexagon of side ~3.5
    grid_points = []
    # Hex grid: for a flat-top hexagonal lattice, neighbors are at distance 2*apothem = sqrt(3)
    # Lattice vectors: a1 = (sqrt(3), 0), a2 = (sqrt(3)/2, 3/2)
    a1 = np.array([SQRT3, 0.0])
    a2 = np.array([SQRT3 / 2.0, 1.5])

    max_coord = 4
    for q in range(-max_coord, max_coord + 1):
        for r in range(-max_coord, max_coord + 1):
            pt = q * a1 + r * a2
            # Only keep points within a reasonable radius
            if np.linalg.norm(pt) <= 4.5:
                grid_points.append(pt)

    grid_points = np.array(grid_points)
    n_grid = len(grid_points)

    # For each pair, precompute minimum distance for non-overlap at angle 0
    # Two unit hexagons at angle 0: minimum center distance to not overlap
    # They can touch side-to-side at distance sqrt(3) or vertex-to-vertex at distance 2
    # The minimum separation depends on relative orientation
    # For now, use a conservative check: if dist < sqrt(3) - eps, they definitely overlap
    # But we'll do exact SAT check

    def is_valid_subset(indices, angles_rad):
        """Check if a subset of grid points with given angles is non-overlapping."""
        centers = [grid_points[i] for i in indices]
        centers = [(c[0], c[1]) for c in centers]
        return not check_all_overlap(centers, angles_rad, eps=1e-7)

    # Strategy 1: Try all 12-subsets from a smaller grid with angle=0
    # First, try the natural hexagonal cluster: center + 6 neighbors + 5 more
    # The 19-point hex cluster (radius 2 in hex grid)
    hex_cluster = []
    for q in range(-2, 3):
        for r in range(-2, 3):
            if abs(q) <= 2 and abs(r) <= 2 and abs(q + r) <= 2:
                pt = q * a1 + r * a2
                hex_cluster.append(pt)
    hex_cluster = np.array(hex_cluster)
    n_cluster = len(hex_cluster)

    best_L = float('inf')
    best_centers = None
    best_angles = None

    # Phase 1: Combinatorial search on the 19-point hex cluster with angle=0
    # Try all C(19,12) = 50388 subsets
    if time.time() < deadline - 60:
        for combo in combinations(range(n_cluster), N):
            if time.time() > deadline - 60:
                break
            centers = [(hex_cluster[i][0], hex_cluster[i][1]) for i in combo]
            angles = [0.0] * N
            if not check_all_overlap(centers, angles, eps=1e-7):
                L = calc_L(centers, angles)
                if L < best_L:
                    best_L = L
                    best_centers = centers
                    best_angles = angles

    # Phase 2: If we have a valid solution, refine with continuous optimization
    if best_centers is not None and time.time() < deadline - 30:
        # Optimize positions and angles
        def objective(params):
            centers_opt = [(params[2 * i], params[2 * i + 1]) for i in range(N)]
            angles_opt = [params[2 * N + i] for i in range(N)]
            if check_all_overlap(centers_opt, angles_opt, eps=1e-9):
                return 100.0
            return calc_L(centers_opt, angles_opt)

        init_params = []
        for i in range(N):
            init_params.extend([best_centers[i][0], best_centers[i][1]])
            init_params.append(best_angles[i])

        try:
            res = minimize(
                objective,
                init_params,
                method='Nelder-Mead',
                options={'maxiter': 3000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
            )
            if res.fun < best_L and res.fun < 100:
                new_centers = [(res.x[2 * i], res.x[2 * i + 1]) for i in range(N)]
                new_angles = [res.x[2 * N + i] for i in range(N)]
                if not check_all_overlap(new_centers, new_angles, eps=1e-7):
                    best_L = res.fun
                    best_centers = new_centers
                    best_angles = new_angles
        except Exception:
            pass

    # Phase 3: Multi-start with random perturbations of the best solution
    if best_centers is not None and time.time() < deadline - 15:
        for trial in range(5):
            if time.time() > deadline - 15:
                break
            init_params = []
            for i in range(N):
                cx = best_centers[i][0] + (np.random.rand() - 0.5) * 0.3
                cy = best_centers[i][1] + (np.random.rand() - 0.5) * 0.3
                ang = best_angles[i] + (np.random.rand() - 0.5) * 0.2
                init_params.extend([cx, cy, ang])

            try:
                res = minimize(
                    objective,
                    init_params,
                    method='Nelder-Mead',
                    options={'maxiter': 2000, 'xatol': 1e-6, 'fatol': 1e-6, 'adaptive': True}
                )
                if res.fun < best_L and res.fun < 100:
                    new_centers = [(res.x[2 * i], res.x[2 * i + 1]) for i in range(N)]
                    new_angles = [res.x[2 * N + i] for i in range(N)]
                    if not check_all_overlap(new_centers, new_angles, eps=1e-7):
                        best_L = res.fun
                        best_centers = new_centers
                        best_angles = new_angles
            except Exception:
                pass

    # Fallback if no valid solution found
    if best_centers is None:
        # Spread hexagons in a circle
        best_centers = []
        best_angles = [0.0] * N
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 3.0
            best_centers.append((r * math.cos(theta), r * math.sin(theta)))
        best_L = calc_L(best_centers, best_angles)

    # Final verification
    if check_all_overlap(best_centers, best_angles, eps=1e-7):
        # Return a safe fallback
        inner = []
        for i in range(N):
            theta = 2 * math.pi * i / N
            r = 5.0
            inner.append([r * math.cos(theta), r * math.sin(theta), 0.0])
        L = calc_L([(5.0 * math.cos(2 * math.pi * i / N), 5.0 * math.sin(2 * math.pi * i / N)) for i in range(N)], [0.0] * N)
        return inner, (0.0, 0.0), L, 0.0

    final_L = calc_L(best_centers, best_angles)
    inner = []
    for i in range(N):
        inner.append([best_centers[i][0], best_centers[i][1], math.degrees(best_angles[i])])

    return inner, (0.0, 0.0), final_L, 0.0
