import math
import time
import numpy as np
from scipy.optimize import minimize, differential_evolution

N = 12

# EVOLVE-BLOCK-START
def construct():
    deadline = time.time() + 145.0

    def get_hex_vertices(cx, cy, side, angle_rad):
        """Get vertices of a regular hexagon."""
        vertices = []
        for i in range(6):
            theta = angle_rad + i * math.pi / 3
            x = cx + side * math.cos(theta)
            y = cy + side * math.sin(theta)
            vertices.append((x, y))
        return vertices

    def compute_outer_side(centers, angles, ocx, ocy, otheta):
        """Compute minimum outer hexagon side for given configuration.
        centers: list of [x, y]
        angles: list of angles in radians
        Returns the side length of the outer hexagon.
        """
        max_proj = 0.0
        for k in range(6):
            normal_angle = otheta + math.pi / 6 + k * math.pi / 3
            nx = math.cos(normal_angle)
            ny = math.sin(normal_angle)
            for i in range(len(centers)):
                cx, cy = centers[i]
                angle = angles[i]
                for vi in range(6):
                    vert_angle = angle + vi * math.pi / 3
                    vx = cx + math.cos(vert_angle)
                    vy = cy + math.sin(vert_angle)
                    proj = (vx - ocx) * nx + (vy - ocy) * ny
                    if proj > max_proj:
                        max_proj = proj
        # The side length s of a regular hexagon with inradius r is s = r * 2/sqrt(3)
        # The inradius is max_proj (distance from center to the supporting line)
        return 2.0 * max_proj / math.sqrt(3)

    def check_disjoint(centers, angles, min_dist=math.sqrt(3) - 0.0001):
        """Check if all hexagons are disjoint."""
        n = len(centers)
        for i in range(n):
            for j in range(i + 1, n):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    return False
        return True

    def objective(params):
        """Objective function for optimization.
        params: [cx1, cy1, a1, cx2, cy2, a2, ..., ocx, ocy, otheta]
        """
        if time.time() > deadline - 5:
            return 1e6
        centers = []
        angles = []
        for i in range(N):
            cx = params[3 * i]
            cy = params[3 * i + 1]
            a = params[3 * i + 2]
            centers.append([cx, cy])
            angles.append(a)
        ocx = params[3 * N]
        ocy = params[3 * N + 1]
        otheta = params[3 * N + 2]

        # Check disjointness with a penalty
        min_dist = math.sqrt(3)
        penalty = 0.0
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx * dx + dy * dy)
                if dist < min_dist:
                    penalty += (min_dist - dist) ** 2

        side = compute_outer_side(centers, angles, ocx, ocy, otheta)
        return side + 100.0 * penalty

    # Initial configuration: honeycomb lattice
    d = math.sqrt(3)
    u = np.array([d, 0.0])
    v = np.array([d / 2.0, d * math.sqrt(3) / 2.0])

    centers = [np.array([0.0, 0.0])]
    ring1_vectors = [u, v, v - u, -u, -v, u - v]
    for vec in ring1_vectors:
        centers.append(vec.copy())

    ring2_candidates = []
    basis = [u, v, v - u, -u, -v, u - v]
    for i in range(6):
        p1 = ring1_vectors[i]
        for b in basis:
            p2 = p1 + b
            is_r0 = np.allclose(p2, np.array([0.0, 0.0]))
            is_r1 = False
            for p1j in ring1_vectors:
                if np.allclose(p2, p1j):
                    is_r1 = True
                    break
            if not is_r0 and not is_r1:
                duplicate = False
                for pc in ring2_candidates:
                    if np.allclose(p2, pc):
                        duplicate = True
                        break
                if not duplicate:
                    ring2_candidates.append(p2)

    ring2_candidates.sort(key=lambda p: np.linalg.norm(p))
    for i in range(5):
        centers.append(ring2_candidates[i].copy())

    assert len(centers) == 12

    # Convert to parameter vector
    angles_init = [0.0] * N
    params_init = []
    for i in range(N):
        params_init.extend([centers[i][0], centers[i][1], angles_init[i]])
    params_init.extend([0.0, 0.0, 0.0])  # outer center and angle

    # Bounds for differential evolution
    bounds = []
    for i in range(N):
        bounds.extend([(-8, 8), (-8, 8), (-math.pi, math.pi)])
    bounds.extend([(-4, 4), (-4, 4), (-math.pi / 6, math.pi / 6)])

    # Run differential evolution for global search
    best_result = None
    best_side = 1e6

    if time.time() < deadline - 30:
        result = differential_evolution(
            objective, bounds,
            maxiter=200,
            popsize=15,
            tol=1e-6,
            seed=42,
            polish=False,
            init='latinhypercube',
            updating='deferred'
        )
        if result.fun < best_side:
            best_side = result.fun
            best_result = result.x

    # Local refinement with Nelder-Mead
    if best_result is not None and time.time() < deadline - 10:
        result2 = minimize(
            objective, best_result,
            method='Nelder-Mead',
            options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        if result2.fun < best_side:
            best_side = result2.fun
            best_result = result2.x

    if best_result is None:
        best_result = np.array(params_init)

    # Extract final configuration
    final_centers = []
    final_angles = []
    for i in range(N):
        cx = best_result[3 * i]
        cy = best_result[3 * i + 1]
        a = best_result[3 * i + 2]
        final_centers.append([cx, cy])
        final_angles.append(a)
    ocx = best_result[3 * N]
    ocy = best_result[3 * N + 1]
    otheta = best_result[3 * N + 2]

    # Final polish: fine grid search for outer hexagon parameters
    best_outer_side = compute_outer_side(final_centers, final_angles, ocx, ocy, otheta)
    best_ocx, best_ocy, best_otheta = ocx, ocy, otheta

    for otheta_deg in np.linspace(math.degrees(otheta) - 10, math.degrees(otheta) + 10, 21):
        otheta_r = math.radians(otheta_deg)
        for ocx_try in np.linspace(ocx - 0.3, ocx + 0.3, 7):
            for ocy_try in np.linspace(ocy - 0.3, ocy + 0.3, 7):
                side = compute_outer_side(final_centers, final_angles, ocx_try, ocy_try, otheta_r)
                if side < best_outer_side - 1e-12:
                    best_outer_side = side
                    best_ocx, best_ocy, best_otheta = ocx_try, ocy_try, otheta_r

    inner = [[c[0], c[1], math.degrees(a)] for c, a in zip(final_centers, final_angles)]
    outer_center = (best_ocx, best_ocy)
    outer_side = best_outer_side
    outer_angle = math.degrees(best_otheta)

    return inner, outer_center, outer_side, outer_angle
# EVOLVE-BLOCK-END
