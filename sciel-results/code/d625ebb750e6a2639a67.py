"""Improved 11-hexagon packing: tighter initial layouts, adaptive smooth surrogate, enhanced polish."""
import math
import time
import numpy as np
from scipy.optimize import minimize

N = 11
SQRT3 = math.sqrt(3)
APOTHEM = math.sqrt(3) / 2.0

# EVOLVE-BLOCK-START

def get_hex_vertices(cx, cy, side, angle_rad):
    """Return 6 vertices of a regular hexagon as numpy array (6,2)."""
    thetas = angle_rad + np.arange(6) * math.pi / 3.0
    return np.column_stack([cx + side * np.cos(thetas), cy + side * np.sin(thetas)])

def compute_required_side(inner, outer_center, outer_angle_deg):
    """
    Compute the minimum outer side length to contain all inner hexagons.
    inner: list of (x, y, angle_deg)
    outer_center: (x, y)
    outer_angle_deg: rotation of the outer hexagon
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    # Container normals: directions perpendicular to each side of the outer hexagon
    # For a regular hexagon with center at origin and rotation angle, the support
    # in direction n_i gives the distance from center to the side.
    # The apothem (inradius) of a regular hexagon with side s is s*sqrt(3)/2.
    # So side = 2 * max_projection / sqrt(3) where max_projection is the max
    # of dot(p - center, n_i) over all vertices and all 6 normal directions.
    
    # Precompute container normals for the given rotation
    base_angles = np.array([30 + i * 60 for i in range(6)]) * math.pi / 180.0
    normals = np.column_stack([np.cos(base_angles), np.sin(base_angles)])
    
    # Rotate normals by outer_angle_deg
    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    # Rotation matrix applied to normals
    normals = normals @ np.array([[cos_a, sin_a], [-sin_a, cos_a]]).T

    max_projection = -np.inf
    for (ix, iy, iangle) in inner:
        verts = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
        local = verts - np.array([ox, oy])
        projs = local @ normals.T
        cur = projs.max()
        if cur > max_projection:
            max_projection = cur

    if max_projection < 0:
        return 0.0
    return 2.0 * max_projection / SQRT3

def compute_required_side_smooth(inner, outer_center, outer_angle_deg, beta=80.0):
    """
    Smooth surrogate for compute_required_side using log-sum-exp soft-max.
    Higher beta gives sharper approximation to the max.
    """
    if not inner:
        return 0.0

    ox, oy = outer_center
    normals = np.array([[math.cos(math.radians(30 + i * 60)),
                         math.sin(math.radians(30 + i * 60))] for i in range(6)])

    cos_a = math.cos(math.radians(outer_angle_deg))
    sin_a = math.sin(math.radians(outer_angle_deg))
    R = np.array([[cos_a, sin_a], [-sin_a, cos_a]])

    # Vectorized: compute all vertices at once
    n_inner = len(inner)
    # Shape: (n_inner, 6, 2)
    all_verts = np.zeros((n_inner, 6, 2))
    for i, (ix, iy, iangle) in enumerate(inner):
        all_verts[i] = get_hex_vertices(ix, iy, 1.0, math.radians(iangle))
    
    # local: (n_inner, 6, 2)
    local = (all_verts - np.array([ox, oy])) @ R.T
    # projs: (n_inner, 6, 6)
    projs = local @ normals.T
    
    P = projs.ravel()
    if P.size == 0:
        return 0.0

    m = np.max(P)
    if m < -1e6:
        return 0.0
    # Numerically stable log-sum-exp
    shifted = beta * (P - m)
    # Clip to avoid overflow in exp
    shifted = np.clip(shifted, -50, 50)
    max_proj = float(m + np.log(np.sum(np.exp(shifted))) / beta)

    if max_proj < 0:
        return 0.0
    return 2.0 * max_proj / SQRT3

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
        polys.append(get_hex_vertices(x, y, 1.0, math.radians(a)))
    for i in range(n):
        for j in range(i + 1, n):
            if sat_overlap_check(polys[i], polys[j], eps):
                return False
    return True

def compute_overlap_penalty(inner, min_dist=SQRT3 - 0.004):
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

def compute_force_directed_penalty(params, adjacency, k_rep=100.0, k_spring=10.0, min_dist=SQRT3 - 0.002):
    """
    Force-directed penalty: repulsive for non-adjacent, spring for adjacent.
    """
    n = N
    xs = np.array([params[i*3] for i in range(n)])
    ys = np.array([params[i*3+1] for i in range(n)])

    dx = xs[:, None] - xs[None, :]
    dy = ys[:, None] - ys[None, :]
    dists = np.sqrt(dx * dx + dy * dy + 1e-12)

    penalty = 0.0

    iu = np.triu_indices(n, k=1)
    pair_dists = dists[iu]

    diff = np.maximum(0.0, min_dist - pair_dists)
    penalty += k_rep * np.sum(diff * diff)

    for (i, j) in adjacency:
        if i < n and j < n:
            d = dists[i, j]
            target = SQRT3 * 0.99
            penalty += k_spring * (d - target) ** 2

    return float(penalty)

def construct():
    start_time = time.time()
    deadline = start_time + 145.0

    # Multiple candidate configurations - focus on symmetric, compact layouts
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

    # Candidate 6: Compact 2-3-3-2-1 (centered column)
    axial_6 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1),
        (2, 0), (1, 1), (0, 2),
        (-1, 2), (0, 3),
        (0, 4)
    ]
    candidates.append((axial_6, "column"))

    # Candidate 7: Asymmetric ring
    axial_7 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (2, -1), (1, 1), (0, 2)
    ]
    candidates.append((axial_7, "asym_ring"))

    # Candidate 8: Tight 3x3 grid with offset
    axial_8 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (1, 3), (1, -1)
    ]
    candidates.append((axial_8, "grid_offset"))

    # Candidate 9: Spectral-inspired compact layout (minimize variance)
    axial_9 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (0, 0), (0, 0), (0, 0), (0, 0)
    ]
    candidates.append((axial_9, "spectral_init"))

    # Candidate 10: Tighter hexagonal arrangement - center + 6 + 4 in compact positions
    axial_10 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (1, 1), (2, -1), (-1, 2)
    ]
    candidates.append((axial_10, "tight_hex"))

    best_global_side = float('inf')
    best_global_inner = None
    best_global_center = None
    best_global_angle = 0.0

    for idx, (axial_coords, name) in enumerate(candidates):
        if time.time() > deadline - 30:
            break

        if len(axial_coords) > N:
            axial_coords = axial_coords[:N]
        elif len(axial_coords) < N:
            # Pad with a point far away or duplicate, optimizer will fix
            axial_coords = list(axial_coords) + [(0, 0)] * (N - len(axial_coords))

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

        # More diverse starting angles: 0, 15, 30, 45, 60, 75, 90 (exploit hex symmetry: 60-degree period)
        init_angles = [0, 15, 30, 45, 60, 75, 90]

        for init_angle in init_angles:
            if time.time() > deadline - 25:
                break

            init_params = [cx_avg, cy_avg, float(init_angle)]

            try:
                # First pass: smooth surrogate for good basin finding
                res = minimize(objective_outer_smooth, init_params, method='Nelder-Mead',
                              options={'maxiter': 1200, 'xatol': 1e-9, 'fatol': 1e-11, 'maxfev': 2000,
                                       'adaptive': True})

                # Second pass: exact objective for precision
                res2 = minimize(objective_outer_exact, res.x, method='Nelder-Mead',
                               options={'maxiter': 2000, 'xatol': 1e-12, 'fatol': 1e-14, 'maxfev': 5000,
                                        'adaptive': True})

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

        # Phase 2: Joint optimization with simulated annealing
        # Build adjacency for force-directed penalty
        adjacency = set()
        for i in range(N):
            for j in range(i + 1, N):
                dx = centers[i][0] - centers[j][0]
                dy = centers[i][1] - centers[j][1]
                dist = math.sqrt(dx*dx + dy*dy)
                if dist < SQRT3 * 1.1:
                    adjacency.add((i, j))

        def objective_joint(params, beta=40.0):
            inner_list = []
            for i in range(N):
                x = params[i*3]
                y = params[i*3+1]
                a = params[i*3+2]
                inner_list.append([x, y, a])

            ocx = params[N*3]
            ocy = params[N*3+1]
            oangle = params[N*3+2]

            # Use smooth surrogate for better gradient info during SA
            side = compute_required_side_smooth(inner_list, (ocx, ocy), oangle, beta=beta)
            penalty = compute_overlap_penalty(inner_list, min_dist=SQRT3 - 0.003)
            fd_penalty = compute_force_directed_penalty(params, adjacency, k_rep=120.0, k_spring=2.0)

            return side + 200.0 * penalty + 0.1 * fd_penalty

        initial_params = []
        for x, y, a in initial_inner:
            initial_params.extend([x, y, a])
        initial_params.extend([outer_center[0], outer_center[1], outer_angle])

        current_params = np.array(initial_params)
        current_val = objective_joint(current_params, beta=40.0)
        best_params = current_params.copy()
        best_val_joint = current_val

        # Adaptive simulated annealing with gradually increasing beta
        T = 0.8
        T_min = 0.001
        cooling = 0.99988

        max_iter = 12000
        no_improve_count = 0
        patience = 2000

        for i in range(max_iter):
            if time.time() > deadline - 18:
                break

            # Gradually increase beta for sharper approximation
            progress = 1.0 - T / 0.8
            beta_sa = 30.0 + 30.0 * min(progress, 1.0)
            beta_sa = min(beta_sa, 60.0)

            neighbor = current_params.copy()

            # Random perturbation - mix of single-dim and multi-dim moves
            r = np.random.random()
            if r < 0.5:
                # Single inner hexagon move
                hex_idx = np.random.randint(0, N)
                dim_idx = hex_idx * 3 + np.random.randint(0, 3)
                if dim_idx % 3 == 2:
                    # Angle perturbation
                    delta = np.random.normal(0, 0.15 * (1 + 0.3 * T))
                else:
                    # Position perturbation
                    delta = np.random.normal(0, 0.10 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta
            elif r < 0.75:
                # Two inner hexagons move (helps resolve local overlaps)
                hex_i = np.random.randint(0, N)
                hex_j = np.random.randint(0, N)
                if hex_i != hex_j:
                    for h in [hex_i, hex_j]:
                        dim_idx = h * 3 + np.random.randint(0, 2)  # x or y
                        neighbor[dim_idx] += np.random.normal(0, 0.08 * (1 + 0.4 * T))
            else:
                # Outer container move
                dim_idx = np.random.randint(N*3, N*3+3)
                if dim_idx == N*3+2:
                    delta = np.random.normal(0, 1.5 * (1 + 0.3 * T))
                else:
                    delta = np.random.normal(0, 0.06 * (1 + 0.5 * T))
                neighbor[dim_idx] += delta

            neighbor_val = objective_joint(neighbor, beta=beta_sa)

            if neighbor_val < current_val or np.random.random() < math.exp(-(neighbor_val - current_val) / max(T, T_min)):
                current_params = neighbor
                current_val = neighbor_val

                if current_val < best_val_joint:
                    best_params = current_params.copy()
                    best_val_joint = current_val
                    no_improve_count = 0
                else:
                    no_improve_count += 1

                # Restart if stuck
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
        # Focus on angles near the current best with fine granularity
        for da in [-8, -5, -3, -1, 0, 1, 3, 5, 8]:
            for dc in [(-0.1, 0), (0.1, 0), (0, -0.1), (0, 0.1), (0.07, 0.07), (-0.07, -0.07), (0, 0)]:
                polish_starts.append([best_ocx + dc[0], best_ocy + dc[1], base_angle + da])

        best_polish_val = compute_required_side(best_inner_joint, (best_ocx, best_ocy), best_oangle)
        best_polish_params = [best_ocx, best_ocy, best_oangle]

        for ps in polish_starts:
            if time.time() > deadline - 2:
                break
            try:
                # Two-stage: smooth then exact
                res_smooth = minimize(lambda p: compute_required_side_smooth(best_inner_joint, (p[0], p[1]), p[2], beta=80.0),
                                     ps, method='Nelder-Mead',
                                     options={'maxiter': 2000, 'xatol': 1e-10, 'fatol': 1e-12, 'maxfev': 3000,
                                              'adaptive': True})
                res = minimize(objective_outer_polish, res_smooth.x, method='Nelder-Mead',
                              options={'maxiter': 8000, 'xatol': 1e-14, 'fatol': 1e-16, 'maxfev': 15000,
                                       'adaptive': True})
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
                # Try a quick local fix: nudge overlapping hexagons apart
                fixed_inner2 = [list(h) for h in final_inner]
                for attempt in range(50):
                    overlap_found = False
                    for i in range(N):
                        vi = get_hex_vertices(fixed_inner2[i][0], fixed_inner2[i][1], 1.0, math.radians(fixed_inner2[i][2]))
                        for j in range(i + 1, N):
                            vj = get_hex_vertices(fixed_inner2[j][0], fixed_inner2[j][1], 1.0, math.radians(fixed_inner2[j][2]))
                            if sat_overlap_check(vi, vj, eps=1e-6):
                                overlap_found = True
                                # Nudge apart
                                dx = fixed_inner2[j][0] - fixed_inner2[i][0]
                                dy = fixed_inner2[j][1] - fixed_inner2[i][1]
                                d = math.sqrt(dx*dx + dy*dy) + 1e-9
                                nx = dx / d
                                ny = dy / d
                                fixed_inner2[i][0] -= 0.01 * nx
                                fixed_inner2[i][1] -= 0.01 * ny
                                fixed_inner2[j][0] += 0.01 * nx
                                fixed_inner2[j][1] += 0.01 * ny
                    if not overlap_found:
                        break

                if check_all_disjoint(fixed_inner2, eps=1e-5):
                    side_fixed2 = compute_required_side(fixed_inner2, final_center, final_angle)
                    if side_fixed2 < best_global_side:
                        best_global_side = side_fixed2
                        best_global_inner = fixed_inner2
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

    # Final safety: ensure we return valid types
    return (best_global_inner, best_global_center, 
            float(best_global_side), float(best_global_angle))

# EVOLVE-BLOCK-END
