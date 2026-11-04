import math
import time
import random
import numpy as np
from scipy.optimize import minimize

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 140.0

    def get_hex_vertices(cx, cy, side, angle_deg):
        angle = math.radians(angle_deg)
        vertices = []
        for i in range(6):
            theta = angle + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def polygons_disjoint(v1, v2, eps=1e-9):
        """Check if two convex polygons are disjoint using SAT."""
        n1 = len(v1)
        n2 = len(v2)
        for i in range(n1):
            x1, y1 = v1[i]
            x2, y2 = v1[(i + 1) % n1]
            nx = -(y2 - y1)
            ny = x2 - x1
            # Project all vertices of both polygons onto this normal
            min_p1 = min((px * nx + py * ny) for px, py in v1)
            max_p1 = max((px * nx + py * ny) for px, py in v1)
            min_p2 = min((px * nx + py * ny) for px, py in v2)
            max_p2 = max((px * nx + py * ny) for px, py in v2)
            if max_p1 < min_p2 - eps or max_p2 < min_p1 - eps:
                return True
        for i in range(n2):
            x1, y1 = v2[i]
            x2, y2 = v2[(i + 1) % n2]
            nx = -(y2 - y1)
            ny = x2 - x1
            min_p1 = min((px * nx + py * ny) for px, py in v1)
            max_p1 = max((px * nx + py * ny) for px, py in v1)
            min_p2 = min((px * nx + py * ny) for px, py in v2)
            max_p2 = max((px * nx + py * ny) for px, py in v2)
            if max_p1 < min_p2 - eps or max_p2 < min_p1 - eps:
                return True
        return False

    def check_disjoint(centers, angles, eps=1e-9):
        """Check if all hexagons are disjoint using SAT."""
        n = len(centers)
        verts = []
        for i in range(n):
            verts.append(get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
        for i in range(n):
            for j in range(i + 1, n):
                if not polygons_disjoint(verts[i], verts[j], eps):
                    return False
        return True

    def compute_outer_side(centers, angles, ocx, ocy, otheta_deg):
        """Compute minimum outer hexagon side for given configuration."""
        otheta = math.radians(otheta_deg)
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = math.radians(angles[i])
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        return 2.0 * max_proj / math.sqrt(3)

    def full_objective(params, centers_init, angles_init):
        """Full objective: outer side length with penalty for overlap."""
        n = len(centers_init)
        centers = []
        angles = []
        idx = 0
        for i in range(n):
            centers.append([params[idx], params[idx + 1]])
            angles.append(params[idx + 2])
            idx += 3
        ocx = params[idx]
        ocy = params[idx + 1]
        otheta = params[idx + 2]

        # Check for overlaps
        verts = []
        for i in range(n):
            verts.append(get_hex_vertices(centers[i][0], centers[i][1], 1.0, angles[i]))
        overlap_penalty = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                # Compute minimum separation using SAT
                min_sep = float('inf')
                for vi in range(6):
                    x1, y1 = verts[i][vi]
                    x2, y2 = verts[i][(vi + 1) % 6]
                    nx = -(y2 - y1)
                    ny = x2 - x1
                    norm = math.sqrt(nx * nx + ny * ny)
                    if norm > 1e-12:
                        nx /= norm
                        ny /= norm
                        min_p1 = min((px * nx + py * ny) for px, py in verts[i])
                        max_p1 = max((px * nx + py * ny) for px, py in verts[i])
                        min_p2 = min((px * nx + py * ny) for px, py in verts[j])
                        max_p2 = max((px * nx + py * ny) for px, py in verts[j])
                        sep = min(max_p1 - max_p2, max_p2 - max_p1)
                        if sep < min_sep:
                            min_sep = sep
                for vi in range(6):
                    x1, y1 = verts[j][vi]
                    x2, y2 = verts[j][(vi + 1) % 6]
                    nx = -(y2 - y1)
                    ny = x2 - x1
                    norm = math.sqrt(nx * nx + ny * ny)
                    if norm > 1e-12:
                        nx /= norm
                        ny /= norm
                        min_p1 = min((px * nx + py * ny) for px, py in verts[i])
                        max_p1 = max((px * nx + py * ny) for px, py in verts[i])
                        min_p2 = min((px * nx + py * ny) for px, py in verts[j])
                        max_p2 = max((px * nx + py * ny) for px, py in verts[j])
                        sep = min(max_p1 - max_p2, max_p2 - max_p1)
                        if sep < min_sep:
                            min_sep = sep
                if min_sep < 0:
                    overlap_penalty += min_sep * min_sep * 10000

        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        return side + overlap_penalty

    # Lattice vectors for hexagonal packing
    eps = 0.005
    d = math.sqrt(3) + eps

    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    # Ring 0
    centers = [np.array([0.0, 0.0])]

    # Ring 1
    ring1_vectors = [
        u,
        v,
        v - u,
        -u,
        -v,
        u - v
    ]
    for vec in ring1_vectors:
        centers.append(vec.copy())

    # Ring 2 Candidates
    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]

    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]), atol=0.01)
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j, atol=0.01):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc, atol=0.01):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2.copy())

    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))

    for i in range(5):
        centers.append(ring2_candidates[i].copy())

    assert len(centers) == 12

    # Convert to list format
    centers_list = [[float(c[0]), float(c[1])] for c in centers]
    angles_list = [30.0] * 12

    # Build initial parameter vector: x, y, angle for each hex + outer center and angle
    init_params = []
    for c in centers_list:
        init_params.append(c[0])
        init_params.append(c[1])
        init_params.append(30.0)
    init_params.extend([0.0, 0.0, 0.0])  # outer center and angle

    # Use scipy.optimize.minimize with COBYLA (no gradient needed)
    if time.time() < deadline - 40:
        result = minimize(
            full_objective,
            init_params,
            args=(centers_list, angles_list),
            method='COBYLA',
            options={'maxiter': 500, 'rhobeg': 0.1}
        )
        if result.fun < float('inf'):
            best_params = result.x
        else:
            best_params = init_params
    else:
        best_params = init_params

    # Extract best configuration
    n = len(centers_list)
    best_centers = []
    best_angles = []
    idx = 0
    for i in range(n):
        best_centers.append([best_params[idx], best_params[idx + 1]])
        best_angles.append(best_params[idx + 2])
        idx += 3
    best_ocx = best_params[idx]
    best_ocy = best_params[idx + 1]
    best_otheta = best_params[idx + 2]

    # Verify feasibility
    if not check_disjoint(best_centers, best_angles):
        # Fall back to initial lattice configuration
        best_centers = centers_list
        best_angles = angles_list
        best_ocx = 0.0
        best_ocy = 0.0
        best_otheta = 0.0

    # Final high-precision polish: fine grid search over outer angle
    best_outer_side = float('inf')
    best_ocx_f, best_ocy_f, best_otheta_f = best_ocx, best_ocy, best_otheta

    if time.time() < deadline - 20:
        for otheta_deg in np.linspace(best_otheta - 5, best_otheta + 5, 41):
            for ocx in np.linspace(best_ocx - 0.3, best_ocx + 0.3, 13):
                for ocy in np.linspace(best_ocy - 0.3, best_ocy + 0.3, 13):
                    side = compute_outer_side(best_centers, best_angles, float(ocx), float(ocy), float(otheta_deg))
                    if side < best_outer_side:
                        best_outer_side = side
                        best_ocx_f, best_ocy_f, best_otheta_f = float(ocx), float(ocy), float(otheta_deg)

    inner = [[float(c[0]), float(c[1]), float(a)] for c, a in zip(best_centers, best_angles)]
    outer_center = (best_ocx_f, best_ocy_f)
    outer_side = best_outer_side
    outer_angle = best_otheta_f

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
