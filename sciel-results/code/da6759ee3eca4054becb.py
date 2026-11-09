import numpy as np
import time
import math
import random

N = 11
DEADLINE = 145.0  # Safety margin before 150s kill


def _hex_vertices(cx, cy, side, angle_deg):
    """Return 6 vertices of a regular hexagon centered at (cx,cy)."""
    angle = np.radians(angle_deg)
    verts = np.empty((6, 2))
    for i in range(6):
        theta = angle + np.pi * i / 3.0
        verts[i, 0] = cx + side * np.cos(theta)
        verts[i, 1] = cy + side * np.sin(theta)
    return verts


def _hex_edge_normals(angle_deg):
    """Return 6 outward normal vectors of a regular hexagon."""
    angle = np.radians(angle_deg)
    normals = np.empty((6, 2))
    for i in range(6):
        mid_angle = angle + np.pi / 6.0 + np.pi * i / 3.0
        normals[i, 0] = np.cos(mid_angle)
        normals[i, 1] = np.sin(mid_angle)
    return normals


def _min_enclosing_side_lp(centers, inner_angles, outer_angle_deg):
    """
    Compute minimum side length of a regular hexagon (fixed orientation)
    enclosing all inner hexagons using Linear Programming.
    Returns (side, center).
    """
    sqrt3 = np.sqrt(3)
    normals = _hex_edge_normals(outer_angle_deg)

    # Collect all vertices
    all_points = np.vstack([_hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
                            for i in range(len(centers))])

    # A_i = max projection in direction n_i
    projections = all_points @ normals.T
    A = projections.max(axis=0)

    # Variables: x = [cx, cy, r]
    # Minimize r
    c_obj = np.array([0.0, 0.0, 1.0])

    # Constraints: n_i . c - r <= -A_i
    A_mat = np.zeros((6, 3))
    b_vec = np.zeros(6)

    for i in range(6):
        A_mat[i, 0] = normals[i, 0]
        A_mat[i, 1] = normals[i, 1]
        A_mat[i, 2] = -1.0
        b_vec[i] = -A[i]

    # Use scipy linprog if available, otherwise fallback to manual
    try:
        from scipy.optimize import linprog
        res = linprog(c_obj, A_ub=A_mat, b_ub=b_vec, method='highs')
        if res.status != 0:
            return float('inf'), np.array([0.0, 0.0])
        cx, cy, r = res.x
        side = 2 * r / sqrt3
        return side, np.array([cx, cy])
    except ImportError:
        # Fallback: simple centering + max projection
        cx = all_points[:, 0].mean()
        cy = all_points[:, 1].mean()
        max_proj = 0
        for i in range(6):
            proj = (all_points[:, 0] - cx) * normals[i, 0] + (all_points[:, 1] - cy) * normals[i, 1]
            max_proj = max(max_proj, proj.max())
        side = 2 * max_proj / sqrt3
        return side, np.array([cx, cy])


def _hex_overlap(c1, a1, c2, a2, tol=1e-9):
    """Check if two unit hexagons overlap using SAT."""
    v1 = _hex_vertices(c1[0], c1[1], 1.0, a1)
    v2 = _hex_vertices(c2[0], c2[1], 1.0, a2)

    def get_axes(verts):
        axes = []
        n = len(verts)
        for i in range(n):
            x1, y1 = verts[i]
            x2, y2 = verts[(i + 1) % n]
            ex = x2 - x1
            ey = y2 - y1
            nx, ny = -ey, ex
            norm = np.sqrt(nx**2 + ny**2)
            if norm > 1e-12:
                axes.append((nx / norm, ny / norm))
        return axes

    axes = get_axes(v1) + get_axes(v2)
    for ax, ay in axes:
        proj1 = v1[:, 0] * ax + v1[:, 1] * ay
        proj2 = v2[:, 0] * ax + v2[:, 1] * ay
        if proj1.max() < proj2.min() - tol or proj2.max() < proj1.min() - tol:
            return False
    return True


def _check_validity(centers, inner_angles, outer_center, outer_side, outer_angle_deg, tol=1e-7):
    """Check containment and non-overlap."""
    sqrt3 = np.sqrt(3)
    outer_normals = _hex_edge_normals(outer_angle_deg)
    inradius = outer_side * sqrt3 / 2

    for i in range(len(centers)):
        verts = _hex_vertices(centers[i][0], centers[i][1], 1.0, inner_angles[i])
        for vx, vy in verts:
            for nx, ny in outer_normals:
                dist = (vx - outer_center[0]) * nx + (vy - outer_center[1]) * ny
                if dist > inradius + tol:
                    return False

    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return False

    return True


def _optimize_outer(centers, inner_angles):
    """Find optimal outer hexagon for fixed inner positions."""
    best_side = float('inf')
    best_center = (0.0, 0.0)
    best_angle = 0.0

    # Sweep angle finely
    for outer_angle in np.linspace(0, 60, 61):
        side, center = _min_enclosing_side_lp(centers, inner_angles, outer_angle)
        if side < best_side:
            best_side = side
            best_center = center
            best_angle = outer_angle

    return best_side, best_center, best_angle


def _try_arrangement(centers, inner_angles=None):
    """Try an arrangement and return (side, result) or (inf, None)."""
    if inner_angles is None:
        inner_angles = [0.0] * N

    # Check overlaps
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_overlap(centers[i], inner_angles[i], centers[j], inner_angles[j]):
                return float('inf'), None

    # Find optimal outer hexagon
    side, center, angle = _optimize_outer(centers, inner_angles)

    # Verify validity
    if _check_validity(centers, inner_angles, center, side, angle):
        return side, (centers, inner_angles, center, side, angle)

    return float('inf'), None


def _get_valid_fallback():
    """Return a known valid, non-overlapping configuration."""
    centers = []
    angles = [0.0] * N
    # Linear arrangement
    for i in range(N):
        centers.append((float(i) * 4.0, 0.0))

    side, center, angle = _optimize_outer(centers, angles)
    return centers, angles, center, side, angle


def _force_directed_relax(centers, angles, max_iter=200, tol=1e-8):
    """
    Force-directed relaxation: push overlapping hexagons apart.
    """
    centers = [list(c) for c in centers]
    angles = list(angles)

    for it in range(max_iter):
        moved = False
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(centers[i], angles[i], centers[j], angles[j]):
                    dx = centers[j][0] - centers[i][0]
                    dy = centers[j][1] - centers[i][1]
                    dist = math.sqrt(dx*dx + dy*dy)
                    if dist < 1e-12:
                        dx, dy = 1.0, 0.0
                        dist = 1.0
                    else:
                        dx /= dist
                        dy /= dist

                    # Push apart
                    push = 0.1
                    centers[i][0] -= push * dx
                    centers[i][1] -= push * dy
                    centers[j][0] += push * dx
                    centers[j][1] += push * dy
                    moved = True

        if not moved:
            break

    return centers, angles


def _make_hex_lattice(n):
    """Create a hexagonal lattice of n unit hexagons."""
    positions = []
    r = 0
    while True:
        count = 1 + 3 * r * (r + 1) if r > 0 else 1
        if count >= n:
            break
        r += 1

    positions = []
    for q in range(-r, r + 1):
        for q2 in range(max(-r, -q - r), min(r, -q + r) + 1):
            x = 2.0 * q
            y = math.sqrt(3) * (q2 + q / 2.0)
            positions.append((x, y))

    positions = positions[:n]
    return positions


def construct():
    start_time = time.time()
    deadline = start_time + DEADLINE

    # Initial configuration: hexagonal lattice
    pos = _make_hex_lattice(N)
    inner = [[x, y, 0.0] for x, y in pos]

    # Initial outer hexagon
    all_verts = []
    for x, y, ang in inner:
        verts = _hex_vertices(x, y, 1.0, ang)
        all_verts.extend(verts)

    if not all_verts:
        return inner, (0.0, 0.0), 40.0, 0.0

    xs = [v[0] for v in all_verts]
    ys = [v[1] for v in all_verts]
    cx = (min(xs) + max(xs)) / 2
    cy = (min(ys) + max(ys)) / 2

    max_dist = max(math.sqrt((x - cx)**2 + (y - cy)**2) for x, y in all_verts)
    outer_side = max_dist * 1.2 + 2.0
    outer_angle = 0.0

    # Verify initial solution
    valid = _check_validity(
        [(h[0], h[1]) for h in inner],
        [h[2] for h in inner],
        (cx, cy), outer_side, outer_angle
    )

    best_inner = [list(h) for h in inner]
    best_cx, best_cy = cx, cy
    best_side = outer_side
    best_angle = outer_angle
    best_cost = outer_side if valid else float('inf')

    if not valid:
        outer_side *= 1.5
        valid = _check_validity(
            [(h[0], h[1]) for h in inner],
            [h[2] for h in inner],
            (cx, cy), outer_side, outer_angle
        )
        if valid:
            best_side = outer_side
            best_cost = outer_side
        else:
            # Use fallback
            centers, angles, center, side, angle = _get_valid_fallback()
            best_inner = [[c[0], c[1], a] for c, a in zip(centers, angles)]
            best_cx, best_cy = center[0], center[1]
            best_side = side
            best_angle = angle
            best_cost = side
            return best_inner, (best_cx, best_cy), best_side, best_angle

    # Simulated annealing with LP-based outer optimization
    current_inner = [list(h) for h in inner]
    temperature = 0.5
    cooling = 0.997
    step_size = 0.3
    no_improve_count = 0

    while time.time() < deadline and temperature > 0.01:
        # Random move
        idx = random.randint(0, N - 1)
        move_type = random.random()

        if move_type < 0.5:
            dx = random.gauss(0, step_size)
            dy = random.gauss(0, step_size)
            current_inner[idx][0] += dx
            current_inner[idx][1] += dy
        elif move_type < 0.8:
            dangle = random.gauss(0, step_size * 10)
            current_inner[idx][2] += dangle
        else:
            # Nudge all slightly
            for k in range(N):
                current_inner[k][0] += random.gauss(0, step_size * 0.1)
                current_inner[k][1] += random.gauss(0, step_size * 0.1)

        # Check validity and compute cost
        centers = [(h[0], h[1]) for h in current_inner]
        angles = [h[2] for h in current_inner]

        # Quick overlap check
        overlap = False
        for i in range(N):
            for j in range(i + 1, N):
                if _hex_overlap(centers[i], angles[i], centers[j], angles[j]):
                    overlap = True
                    break
            if overlap:
                break

        if not overlap:
            side, center, angle = _optimize_outer(centers, angles)
            if side < best_cost:
                best_inner = [list(h) for h in current_inner]
                best_cx, best_cy = center[0], center[1]
                best_side = side
                best_angle = angle
                best_cost = side
                no_improve_count = 0
            else:
                # Metropolis acceptance
                delta = side - best_cost
                if delta < 0 or random.random() < math.exp(-delta / max(temperature, 0.001)):
                    no_improve_count += 1
                else:
                    # Revert
                    if move_type < 0.5:
                        current_inner[idx][0] -= dx
                        current_inner[idx][1] -= dy
                    elif move_type < 0.8:
                        current_inner[idx][2] -= dangle
                    else:
                        for k in range(N):
                            current_inner[k][0] -= random.gauss(0, step_size * 0.1)
                            current_inner[k][1] -= random.gauss(0, step_size * 0.1)
                    no_improve_count += 1
        else:
            # Revert
            if move_type < 0.5:
                current_inner[idx][0] -= dx
                current_inner[idx][1] -= dy
            elif move_type < 0.8:
                current_inner[idx][2] -= dangle
            else:
                for k in range(N):
                    current_inner[k][0] -= random.gauss(0, step_size * 0.1)
                    current_inner[k][1] -= random.gauss(0, step_size * 0.1)
            no_improve_count += 1

        temperature *= cooling

        if no_improve_count > 80:
            no_improve_count = 0
            step_size *= 0.85

    # Final verification
    centers = [(h[0], h[1]) for h in best_inner]
    angles = [h[2] for h in best_inner]
    if not _check_validity(centers, angles, (best_cx, best_cy), best_side, best_angle):
        # Try to fix with force-directed relaxation
        relaxed_centers, relaxed_angles = _force_directed_relax(centers, angles)
        side, center, angle = _optimize_outer(relaxed_centers, relaxed_angles)
        if _check_validity(relaxed_centers, relaxed_angles, center, side, angle):
            best_inner = [[c[0], c[1], a] for c, a in zip(relaxed_centers, relaxed_angles)]
            best_cx, best_cy = center[0], center[1]
            best_side = side
            best_angle = angle

    return best_inner, (best_cx, best_cy), best_side, best_angle
