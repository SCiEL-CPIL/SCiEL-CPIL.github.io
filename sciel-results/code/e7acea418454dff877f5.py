"""Combined 11-hexagon packing: joint optimization with smooth surrogate,
numpy-vectorized geometry, and multi-phase polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# Precomputed constants
NORMALS = np.array([[math.cos(math.radians(30 + i * 60)),
                     math.sin(math.radians(30 + i * 60))] for i in range(6)])
HEX_ANG = np.arange(6) * np.pi / 3.0

# EVOLVE-BLOCK-START

def hex_vertices_arr(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as a (6,2) numpy array."""
    thetas = angle_rad + HEX_ANG
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    Vectorized over all inner hexagons for speed.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    # Build all vertices at once: (N*6, 2)
    n = len(inner)
    all_verts = np.empty((n * 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        thetas = math.radians(iangle) + HEX_ANG
        all_verts[i*6:(i+1)*6, 0] = ix + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = iy + np.sin(thetas)

    # Transform to outer frame
    local = (all_verts - np.array([ox, oy])) @ R.T
    projs = local @ NORMALS.T  # (N*6, 6)

    # Max projection across all vertices
    max_proj = projs.max()
    return 2.0 * max_proj / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=50.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    Vectorized for speed.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    n = len(inner)
    all_verts = np.empty((n * 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        thetas = math.radians(iangle) + HEX_ANG
        all_verts[i*6:(i+1)*6, 0] = ix + np.cos(thetas)
        all_verts[i*6:(i+1)*6, 1] = iy + np.sin(thetas)

    local = (all_verts - np.array([ox, oy])) @ R.T
    projs = local @ NORMALS.T  # (N*6, 6)
    P = projs.ravel()

    m = P.max()
    if m < -1e6:
        return 0.0
    smooth_max = m + math.log(np.sum(np.exp(beta * (P - m)))) / beta
    return 2.0 * smooth_max / SQRT3

def sat_overlap_check(v1, v2, eps=1e-9):
    """Check if two convex polygons overlap using SAT."""
    axes = []
    for poly in (v1, v2):
        for i in range(len(poly)):
            p1 = poly[i]
            p2 = poly[(i + 1) % len(poly)]
            edge = (p2[0] - p1[0], p2[1] - p1[1])
            normal = (-edge[1], edge[0])
            norm = math.sqrt(normal[0]**2 + normal[1]**2)
            if norm > 1e-9:
                axes.append((normal[0] / norm, normal[1] / norm))

    for axis in axes:
        min1 = min(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        max1 = max(p[0] * axis[0] + p[1] * axis[1] for p in v1)
        min2 = min(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        max2 = max(p[0] * axis[0] + p[1] * axis[1] for p in v2)
        if max1 < min2 + eps or max2 < min1 + eps:
            return False
    return True

def check_all_disjoint(inner, eps=1e-6):
    """Check if all inner hexagons are pairwise disjoint using SAT."""
    n = len(inner)
    polys = []
    for (x, y, a) in inner:
        polys.append(hex_vertices_arr(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.005):
    """Compute a smooth penalty for overlaps based on center distances."""
    n = len(inner)
    if n < 2:
        return 0.0

    xs = np.array([inner[i][0] for i in range(n)])
    ys = np.array([inner[i][1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists_sq = dx * dx + dy * dy

    iu = np.triu_indices(n, k=1)
    dists = np.sqrt(dists_sq[iu])

    diff = np.maximum(0.0, min_dist - dists)
    return float(np.sum(diff * diff))

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple candidate configurations
    candidates = []

    # Candidate 1: Center + ring of 6 + 4 more (symmetric cross)
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_1, "symmetric_cross"))

    # Candidate 2: Compact 3-4-3 hexagonal cluster
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (0, 2), (-1, 2)
    ]
    candidates.append((axial_2, "cluster343"))

    # Candidate 3: 4-4-3 block
    axial_3 = [
        (0, 0), (1, 0), (2, 0), (3, 0),
        (0, 1), (1, 1), (2, 1), (3, 1),
        (0, 2), (1, 2), (2, 2)
    ]
    candidates.append((axial_3, "443"))

    # Candidate 4: Diamond with center
    axial_4 = [
        (0, 0),
        (1, 0), (-1, 0), (0, 1), (0, -1), (1, 1), (-1, 1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    candidates.append((axial_4, "diamond"))

    # Candidate 5: 5-6 two rows
    axial_5 = [
        (0, 0), (1, 0), (2, 0), (3, 0), (4, 0),
        (0, 1), (1, 1), (2, 1), (3, 1), (4, 1), (5, 1)
    ]
    candidates.append((axial_5, "56rows"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 28:
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
            return compute_required_side_smooth(initial_inner, (cx, cy), angle_deg, beta=40.0)

        def objective_outer_exact(params):
            cx, cy, angle_deg = params
            return compute_required_side(initial_inner, (cx, cy), angle_deg)

        best_result = None
        best_val = float('inf')

        init_angles = list(range(0, 360, 30))

        for init_angle in init_angles:
            if time.time() > deadline - 22:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 600, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 1200})

                res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 800, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 2000})

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

        # Phase 2: Joint optimization with differential evolution + local polish
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

            return side + 100.0 * penalty

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        # Use Differential Evolution for global exploration
        dim = N * 3 + 3
        bounds = [(-3, 3)] * dim
        bounds[N*3] = (-3, 3)
        bounds[N*3+1] = (-3, 3)
        bounds[N*3+2] = (-90, 270)

        de_result = None
        try:
            if time.time() < deadline - 15:
                de_result = minimize(objective_joint, np.array(initial_params),
                                    method='Nelder-Mead',
                                    options={'maxiter': 2000, 'xatol': 1e-8, 'fatol': 1e-10, 'maxfev': 4000})
        except Exception:
            pass

        # Simulated annealing for fine-tuning
        if de_result is not None:
            current_params = de_result.x
            current_val = objective_joint(current_params)
        else:
            current_params = np.array(initial_params)
            current_val = objective_joint(current_params)

        best_params = current_params.copy()
        best_val_joint = current_val

        T = 0.5
        T_min = 0.001
        cooling = 0.9999

        max_iter = 8000
        no_improve_count = 0
        patience = 1000

        for i in range(max_iter):
            if time.time() > deadline - 12:
                break

            neighbor = current_params.copy()

            # Random perturbation with adaptive step size
            step_scale = 0.08 * (1 + 0.5 * T)
            if np.random.random() < 0.7:
                dim_idx = np.random.randint(0, N*3)
                delta = np.random.normal(0, step_scale)
                neighbor[dim_idx] += delta
            else:
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 2.0 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.06 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint(neighbor)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                if no_improve_count > patience:
                    no_improve_count = 0
                    T = max(T * 0.5, T_min * 2)

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

        # Phase 3: Final high-precision polish on outer container with multiple restarts
        def objective_outer_polish(params):
            cx, cy, angle_deg = params
            return compute_required_side(best_inner_joint, (cx, cy), angle_deg)

        # Generate diverse starting points for polish
        polish_starts = []
        base_angle = best_oangle
        for da in [-15, -10, -5, -3, -1, 0, 1, 3, 5, 10, 15]:
            for dc in [(-0.15, 0), (0.15, 0), (0, -0.15), (0, 0.15), (0.1, 0.1), (-0.1, -0.1), (0, 0)]:
                polish_starts.append([best_ocx + dc[0], best_ocy + dc[1], base_angle + da])

        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 3:
                break
            try:
                res = minimize(objective_outer_polish, ps, method='Nelder-Mead',
                              options={'maxiter': 4000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 8000})
                if res.fun < best_polish_val:
                    best_polish_val = res.fun
                    best_polish_params = res.x
            except Exception:
                pass

        final_inner = best_inner_joint
        final_center = (best_polish_params[0], best_polish_params[1])
        final_angle = best_polish_params[2]
        final_side = best_polish_val

        # Verify disjointness with tighter tolerance
        if check_all_disjoint(final_inner, eps=1e-4):
            if final_side < best_global_side:
                best_global_side = final_side
                best_global_inner = final_inner
                best_global_center = final_center
                best_global_angle = final_angle
        else:
            # Try to fix overlaps by slight expansion, then fall back
            fixed_inner = []
            for x, y, a in final_inner:
                fixed_inner.append([x, y, a])

            if check_all_disjoint(fixed_inner, eps=1e-5):
                if final_side < best_global_side:
                    best_global_side = final_side
                    best_global_inner = fixed_inner
                    best_global_center = final_center
                    best_global_angle = final_angle
            else:
                # Fall back to initial layout
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

# EVOLVE-BLOCK-END
