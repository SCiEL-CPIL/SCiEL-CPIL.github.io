import math
import numpy as np
import time
from scipy.optimize import linprog, minimize, minimize_scalar

N = 11
APOTHEM = math.sqrt(3) / 2.0
START_TIME = time.time()
TIME_LIMIT = 150.0
SAFETY_MARGIN = 10.0
EPS = 1e-9

def _hex_dist_sq(p1, p2):
    """Squared distance between two hexagon centers (same orientation)."""
    dx = p1[0] - p2[0]
    dy = p1[1] - p2[1]
    return dx * dx + dy * dy

def _check_non_overlap(pts, tol=1e-8):
    """
    Check if all hexagons are non-overlapping.
    For two unit regular hexagons with the same orientation,
    they don't overlap iff the center-to-center distance >= 2*apothem = sqrt(3).
    """
    min_dist_sq = 3.0 - tol  # (2*APOTHEM)^2 = 3
    for i in range(N):
        for j in range(i + 1, N):
            if _hex_dist_sq(pts[i], pts[j]) < min_dist_sq:
                return False
    return True

def _compute_S(pts, cx, cy, beta):
    """Compute the minimal outer hexagon side length."""
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    centered = pts - np.array([cx, cy])
    projs = np.abs(centered @ normals.T)
    max_proj = np.max(projs)
    return (max_proj + APOTHEM) / APOTHEM

def _smooth_S(pts, cx, cy, beta, temperature=0.1):
    """Smooth surrogate for S using log-sum-exp."""
    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
    normals = np.column_stack([np.cos(angles), np.sin(angles)])
    centered = pts - np.array([cx, cy])
    projs = np.abs(centered @ normals.T)
    all_projs = projs.ravel()
    max_val = np.max(all_projs)
    shifted = (all_projs - max_val) / temperature
    shifted = np.clip(shifted, -50, 50)
    log_sum = np.log(np.sum(np.exp(shifted))) + max_val / temperature
    D = temperature * log_sum
    return (D + APOTHEM) / APOTHEM

def _overlap_penalty(pts, strength=100.0):
    """Compute overlap penalty based on minimum pairwise distance."""
    min_dist_sq = 3.0
    penalty = 0.0
    for i in range(N):
        for j in range(i + 1, N):
            d_sq = _hex_dist_sq(pts[i], pts[j])
            if d_sq < min_dist_sq:
                # Quadratic penalty for being too close
                deficit = min_dist_sq - d_sq
                penalty += strength * deficit * deficit
    return penalty

def construct():
    if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
        return [[0.0, 0.0, 0.0]], [0.0, 0.0], 1.0, 0.0

    # Generate diverse initial candidates
    candidates = []

    # 1. Axial lattice candidates
    axial_1 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (1, 1), (-1, 2), (-2, 1), (2, -1)
    ]
    axial_2 = [
        (0, 0),
        (1, 0), (0, 1), (-1, 1), (-1, 0), (0, -1), (1, -1),
        (2, 0), (-2, 0), (0, 2), (0, -2)
    ]
    axial_3 = [
        (0, 0), (1, 0), (2, 0),
        (0, 1), (1, 1), (2, 1),
        (0, 2), (1, 2), (2, 2),
        (-1, 1), (3, 1)
    ]

    def axial_to_cartesian(axial_list):
        pts = []
        for q, r in axial_list:
            x = math.sqrt(3) * (q + r / 2.0)
            y = 1.5 * r
            pts.append((x, y))
        return np.array(pts)

    candidates.append(axial_to_cartesian(axial_1))
    candidates.append(axial_to_cartesian(axial_2))
    candidates.append(axial_to_cartesian(axial_3))

    # 2. Force-directed candidates
    def _force_directed_layout(n_hex, max_iter=500, dt=0.01, repulse_strength=5.0, attract_strength=0.5, seed=None):
        if seed is not None:
            np.random.seed(seed)
        positions = np.zeros((n_hex, 2))
        idx = 0
        for r in range(-3, 4):
            for q in range(-3, 4):
                if idx >= n_hex:
                    break
                x = math.sqrt(3) * (q + r / 2.0)
                y = 1.5 * r
                positions[idx] = [x, y]
                idx += 1
            if idx >= n_hex:
                break
        positions += np.random.uniform(-0.2, 0.2, size=(n_hex, 2))
        for iteration in range(max_iter):
            forces = np.zeros((n_hex, 2))
            for i in range(n_hex):
                for j in range(i + 1, n_hex):
                    diff = positions[i] - positions[j]
                    dist = np.linalg.norm(diff)
                    if dist < 1e-9:
                        diff = np.random.uniform(-0.01, 0.01, size=2)
                        dist = np.linalg.norm(diff)
                    if dist < 2.5:
                        force_mag = repulse_strength / (dist ** 2)
                        forces[i] += force_mag * diff / dist
                        forces[j] -= force_mag * diff / dist
            centroid = np.mean(positions, axis=0)
            for i in range(n_hex):
                forces[i] += -attract_strength * (positions[i] - centroid)
            positions += dt * forces
            dt *= 0.995
        return positions

    for trial in range(15):
        seed = 100 + trial
        repulse = 3.0 + (trial % 5) * 0.5
        attract = 0.2 + (trial % 3) * 0.1
        max_iter = 300 + (trial % 4) * 100
        pos = _force_directed_layout(N, max_iter=max_iter, dt=0.02,
                                      repulse_strength=repulse,
                                      attract_strength=attract, seed=seed)
        pos -= np.mean(pos, axis=0)
        candidates.append(pos)

    def bbox_heuristic(pts):
        extent_x = np.max(pts[:, 0]) - np.min(pts[:, 0])
        extent_y = np.max(pts[:, 1]) - np.min(pts[:, 1])
        return extent_x * extent_y

    sorted_candidates = sorted(candidates, key=bbox_heuristic)

    best_global_S = float('inf')
    best_global_pts = None
    best_global_beta = 0.0
    best_global_center = None

    for pts in sorted_candidates:
        if time.time() - START_TIME > TIME_LIMIT - SAFETY_MARGIN:
            break

        # Center the points
        pts_centered = pts - np.mean(pts, axis=0)

        # Optimize rotation beta and center
        best_S_local = float('inf')
        best_beta_local = 0.0
        best_center_local = None

        # Coarse grid search over beta
        for i in range(61):
            beta = i * (math.pi / 6) / 60
            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            projs = np.abs(pts_centered @ normals.T)
            max_proj = np.max(projs)
            S = (max_proj + APOTHEM) / APOTHEM
            if S < best_S_local:
                best_S_local = S
                best_beta_local = beta
                best_center_local = (0.0, 0.0)

        # Refine with Nelder-Mead on (cx, cy, beta)
        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                def objective(params):
                    cx, cy, beta = params
                    beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
                    angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
                    normals = np.column_stack([np.cos(angles), np.sin(angles)])
                    centered = pts_centered - np.array([cx, cy])
                    projs = np.abs(centered @ normals.T)
                    max_proj = np.max(projs)
                    return (max_proj + APOTHEM) / APOTHEM

                res = minimize(objective, [0.0, 0.0, best_beta_local], method='Nelder-Mead',
                             options={'maxiter': 1000, 'xatol': 1e-12, 'fatol': 1e-12})
                if res.fun < best_S_local:
                    best_S_local = res.fun
                    best_center_local = (res.x[0], res.x[1])
                    best_beta_local = ((res.x[2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)
        except Exception:
            pass

        if best_S_local < best_global_S:
            best_global_S = best_S_local
            best_global_pts = pts_centered
            best_global_beta = best_beta_local
            best_global_center = best_center_local

    if best_global_center is None or best_global_S == float('inf'):
        best_global_pts = sorted_candidates[0]
        best_global_center = (float(np.mean(best_global_pts[:, 0])), float(np.mean(best_global_pts[:, 1])))
        best_global_beta = 0.0
        best_global_S = _compute_S(best_global_pts, best_global_center[0], best_global_center[1], best_global_beta)

    # Phase 2: Joint optimization of positions, center, and rotation
    # Key insight: all inner hexagons share the same orientation as the outer
    # This reduces the problem from 2N+3 to N+1 effective variables
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:

        def joint_objective(params):
            n_pts = N
            pts = params[:2 * n_pts].reshape(n_pts, 2)
            cx = params[2 * n_pts]
            cy = params[2 * n_pts + 1]
            beta = params[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            centered = pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            max_proj = np.max(projs)
            S = (max_proj + APOTHEM) / APOTHEM

            # Overlap penalty: for same-orientation hexagons, min distance must be >= sqrt(3)
            penalty = _overlap_penalty(pts, strength=50.0)

            return S + penalty

        init_params = list(best_global_pts.flatten()) + [best_global_center[0], best_global_center[1], best_global_beta]

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(joint_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 5000, 'xatol': 1e-11, 'fatol': 1e-11})

                if res.fun < best_global_S + 1.0:  # Allow some penalty in objective
                    new_pts = res.x[:2 * N].reshape(N, 2)
                    new_cx = res.x[2 * N]
                    new_cy = res.x[2 * N + 1]
                    new_beta = ((res.x[2 * N + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

                    # Verify no overlap
                    if _check_non_overlap(new_pts, tol=1e-6):
                        new_S = _compute_S(new_pts, new_cx, new_cy, new_beta)
                        if new_S < best_global_S:
                            best_global_S = new_S
                            best_global_pts = new_pts
                            best_global_center = (new_cx, new_cy)
                            best_global_beta = new_beta
        except Exception:
            pass

    # Phase 3: Smooth surrogate optimization for better convergence
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:

        def smooth_joint_objective(params):
            n_pts = N
            pts = params[:2 * n_pts].reshape(n_pts, 2)
            cx = params[2 * n_pts]
            cy = params[2 * n_pts + 1]
            beta = params[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

            S_smooth = _smooth_S(pts, cx, cy, beta, temperature=0.05)
            penalty = _overlap_penalty(pts, strength=50.0)

            return S_smooth + penalty

        init_params = list(best_global_pts.flatten()) + [best_global_center[0], best_global_center[1], best_global_beta]

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(smooth_joint_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 3000, 'xatol': 1e-11, 'fatol': 1e-11})

                new_pts = res.x[:2 * N].reshape(N, 2)
                new_cx = res.x[2 * N]
                new_cy = res.x[2 * N + 1]
                new_beta = ((res.x[2 * N + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

                if _check_non_overlap(new_pts, tol=1e-6):
                    new_S = _compute_S(new_pts, new_cx, new_cy, new_beta)
                    if new_S < best_global_S:
                        best_global_S = new_S
                        best_global_pts = new_pts
                        best_global_center = (new_cx, new_cy)
                        best_global_beta = new_beta
        except Exception:
            pass

    # Phase 4: High-precision polish
    if best_global_S < float('inf') and time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:

        def polish_objective(params):
            n_pts = N
            pts = params[:2 * n_pts].reshape(n_pts, 2)
            cx = params[2 * n_pts]
            cy = params[2 * n_pts + 1]
            beta = params[2 * n_pts + 2]
            beta = ((beta % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

            angles = beta + math.pi / 6.0 + np.arange(6) * math.pi / 3.0
            normals = np.column_stack([np.cos(angles), np.sin(angles)])
            centered = pts - np.array([cx, cy])
            projs = np.abs(centered @ normals.T)
            max_proj = np.max(projs)
            S = (max_proj + APOTHEM) / APOTHEM

            # Very light overlap penalty for final polish
            penalty = _overlap_penalty(pts, strength=10.0)

            return S + penalty

        init_params = list(best_global_pts.flatten()) + [best_global_center[0], best_global_center[1], best_global_beta]

        try:
            if time.time() - START_TIME < TIME_LIMIT - SAFETY_MARGIN:
                res = minimize(polish_objective, init_params, method='Nelder-Mead',
                             options={'maxiter': 10000, 'xatol': 1e-13, 'fatol': 1e-13})

                new_pts = res.x[:2 * N].reshape(N, 2)
                new_cx = res.x[2 * N]
                new_cy = res.x[2 * N + 1]
                new_beta = ((res.x[2 * N + 2] % (math.pi / 6)) + math.pi / 6) % (math.pi / 6)

                if _check_non_overlap(new_pts, tol=1e-6):
                    new_S = _compute_S(new_pts, new_cx, new_cy, new_beta)
                    if new_S < best_global_S:
                        best_global_S = new_S
                        best_global_pts = new_pts
                        best_global_center = (new_cx, new_cy)
                        best_global_beta = new_beta
        except Exception:
            pass

    # Final output
    cx, cy = best_global_center
    angle_deg = math.degrees(best_global_beta)

    # Safety: verify and adjust if needed
    if not _check_non_overlap(best_global_pts, tol=1e-5):
        # Fallback to a known valid configuration
        fallback_pts = axial_to_cartesian(axial_1)
        fallback_pts -= np.mean(fallback_pts, axis=0)
        best_global_pts = fallback_pts
        best_global_center = (0.0, 0.0)
        best_global_beta = 0.0
        best_global_S = _compute_S(fallback_pts, 0.0, 0.0, 0.0)
        cx, cy = best_global_center
        angle_deg = math.degrees(best_global_beta)

    inner = []
    for x, y in best_global_pts:
        inner.append([float(x), float(y), float(angle_deg)])

    center = [float(cx), float(cy)]
    side = float(best_global_S)
    rotation = float(angle_deg)

    return inner, center, side, rotation
