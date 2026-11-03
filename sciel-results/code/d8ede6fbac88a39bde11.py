"""Circle packing in unit square: maximize sum of radii for 26 disjoint circles.
Uses Differential Evolution for global search and high-precision L-BFGS-B for local polishing."""
import numpy as np
import time
from scipy.optimize import differential_evolution, minimize
from scipy.spatial import Delaunay

N = 26

_start_time = time.time()
_TIME_LIMIT = 145  # Safety margin before 150s kill

def _elapsed():
    return time.time() - _start_time

def _remaining():
    return _TIME_LIMIT - _elapsed()

def _softplus(z, eps=1e-4):
    # Stable softplus
    return np.where(z > 0, z + np.log1p(np.exp(-z/eps)) * eps, np.log1p(np.exp(z/eps)) * eps)

def _validate(circles, tol=1e-7):
    x, y, r = circles[:, 0], circles[:, 1], circles[:, 2]
    if np.any(r < -tol) or np.any(x < -tol) or np.any(x > 1 + tol) or np.any(y < -tol) or np.any(y > 1 + tol):
        return False
    if np.any(x - r < -tol) or np.any(x + r > 1 + tol):
        return False
    if np.any(y - r < -tol) or np.any(y + r > 1 + tol):
        return False
    n = len(circles)
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            dist = np.sqrt(dx * dx + dy * dy)
            if dist < r[i] + r[j] - tol:
                return False
    return True

def _fix_boundaries(circles):
    """Aggressive repair: shrink radii to remove overlaps and ensure boundary constraints."""
    circles = circles.copy()
    for i in range(N):
        r = max(0.0, circles[i, 2])
        r = min(r, 0.5)
        circles[i, 2] = r
        circles[i, 0] = np.clip(circles[i, 0], r, 1.0 - r)
        circles[i, 1] = np.clip(circles[i, 1], r, 1.0 - r)
    
    for _iter in range(50):
        changed = False
        for i in range(N):
            for j in range(i + 1, N):
                dx = circles[i, 0] - circles[j, 0]
                dy = circles[i, 1] - circles[j, 1]
                dist = np.sqrt(dx * dx + dy * dy)
                min_dist = circles[i, 2] + circles[j, 2]
                if dist < min_dist:
                    overlap = min_dist - dist
                    shrink = overlap / 2.0 + 1e-5
                    circles[i, 2] = max(0.0, circles[i, 2] - shrink)
                    circles[j, 2] = max(0.0, circles[j, 2] - shrink)
                    circles[i, 0] = np.clip(circles[i, 0], circles[i, 2], 1.0 - circles[i, 2])
                    circles[i, 1] = np.clip(circles[i, 1], circles[i, 2], 1.0 - circles[i, 2])
                    circles[j, 0] = np.clip(circles[j, 0], circles[j, 2], 1.0 - circles[j, 2])
                    circles[j, 1] = np.clip(circles[j, 1], circles[j, 2], 1.0 - circles[j, 2])
                    changed = True
        if not changed:
            break
    return circles

def _objective_and_grad(params, penalty_weight, smooth_eps=1e-4):
    """
    Objective: -sum(r) + penalty_weight * (penalties)
    Penalties:
    1. Boundary violations (soft)
    2. Overlap violations (soft)
    """
    x = params[0::3]
    y = params[1::3]
    r = params[2::3]
    
    val = -np.sum(r)
    grad = np.zeros_like(params)
    grad[2::3] = -1.0  # derivative of -sum(r) w.r.t r

    # Softplus and Sigmoid for gradients
    def _sigmoid(z, eps):
        # Stable sigmoid
        z_scaled = z / eps
        return np.where(z_scaled > 0, 1.0 / (1.0 + np.exp(-z_scaled)), np.exp(z_scaled) / (1.0 + np.exp(z_scaled)))

    # Boundary Penalties
    # Left: x - r < 0  =>  r - x > 0
    lx = r - x
    sp_lx = _softplus(lx, smooth_eps)
    sg_lx = _sigmoid(lx, smooth_eps)
    penalty_lx = np.sum(sp_lx ** 2)
    val += penalty_weight * penalty_lx
    grad[0::3] += -2.0 * penalty_weight * sp_lx * sg_lx
    grad[2::3] += 2.0 * penalty_weight * sp_lx * sg_lx

    # Right: x + r - 1 > 0
    ux = x + r - 1.0
    sp_ux = _softplus(ux, smooth_eps)
    sg_ux = _sigmoid(ux, smooth_eps)
    penalty_ux = np.sum(sp_ux ** 2)
    val += penalty_weight * penalty_ux
    grad[0::3] += 2.0 * penalty_weight * sp_ux * sg_ux
    grad[2::3] += 2.0 * penalty_weight * sp_ux * sg_ux

    # Bottom: y - r < 0 => r - y > 0
    ly = r - y
    sp_ly = _softplus(ly, smooth_eps)
    sg_ly = _sigmoid(ly, smooth_eps)
    penalty_ly = np.sum(sp_ly ** 2)
    val += penalty_weight * penalty_ly
    grad[1::3] += -2.0 * penalty_weight * sp_ly * sg_ly
    grad[2::3] += 2.0 * penalty_weight * sp_ly * sg_ly

    # Top: y + r - 1 > 0
    uy = y + r - 1.0
    sp_uy = _softplus(uy, smooth_eps)
    sg_uy = _sigmoid(uy, smooth_eps)
    penalty_uy = np.sum(sp_uy ** 2)
    val += penalty_weight * penalty_uy
    grad[1::3] += 2.0 * penalty_weight * sp_uy * sg_uy
    grad[2::3] += 2.0 * penalty_weight * sp_uy * sg_uy

    # Negative Radius Penalty
    nr = -r
    sp_nr = _softplus(nr, smooth_eps)
    sg_nr = _sigmoid(nr, smooth_eps)
    penalty_nr = np.sum(sp_nr ** 2)
    val += penalty_weight * penalty_nr
    grad[2::3] += -2.0 * penalty_weight * sp_nr * sg_nr

    # Overlap Penalties
    n = len(r)
    idx_i, idx_j = np.triu_indices(n, k=1)
    dx = x[idx_i] - x[idx_j]
    dy = y[idx_i] - y[idx_j]
    dist_sq = dx * dx + dy * dy
    dist = np.sqrt(np.maximum(dist_sq, 1e-12))
    overlap = r[idx_i] + r[idx_j] - dist
    sp_ov = _softplus(overlap, smooth_eps)
    sg_ov = _sigmoid(overlap, smooth_eps)
    
    factor = 2.0 * penalty_weight * sp_ov * sg_ov
    inv_dist = 1.0 / dist
    
    penalty_ov = np.sum(sp_ov ** 2)
    val += penalty_weight * penalty_ov
    
    # Gradients for overlaps
    # d(overlap)/dx_i = -dx/dist, d(overlap)/dx_j = dx/dist
    np.add.at(grad[0::3], idx_i, -factor * dx * inv_dist)
    np.add.at(grad[0::3], idx_j, factor * dx * inv_dist)
    np.add.at(grad[1::3], idx_i, -factor * dy * inv_dist)
    np.add.at(grad[1::3], idx_j, factor * dy * inv_dist)
    # d(overlap)/dr_i = 1, d(overlap)/dr_j = 1
    np.add.at(grad[2::3], idx_i, factor)
    np.add.at(grad[2::3], idx_j, factor)

    return val, grad

def _local_polish(xyz0, deadline, max_iter_stage1=300, max_iter_stage2=500):
    """
    Two-stage local optimization:
    1. Moderate penalty to fix gross violations.
    2. High penalty for precise polishing.
    """
    params = xyz0.flatten().copy()
    bounds = [(0, 1), (0, 1), (0, 0.5)] * N
    
    # Stage 1: Moderate penalty
    penalty_w1 = 50.0
    eps1 = 1e-3
    if time.time() < deadline - 10:
        try:
            res1 = minimize(
                lambda p: _objective_and_grad(p, penalty_w1, eps1)[0],
                params,
                jac=lambda p: _objective_and_grad(p, penalty_w1, eps1)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_stage1, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            if res1.fun < _objective_and_grad(params, penalty_w1, eps1)[0]:
                params = res1.x
        except Exception:
            pass

    # Stage 2: High penalty, tight tolerance
    penalty_w2 = 500.0
    eps2 = 1e-5
    if time.time() < deadline - 5:
        try:
            res2 = minimize(
                lambda p: _objective_and_grad(p, penalty_w2, eps2)[0],
                params,
                jac=lambda p: _objective_and_grad(p, penalty_w2, eps2)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': max_iter_stage2, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            if res2.fun < _objective_and_grad(params, penalty_w2, eps2)[0]:
                params = res2.x
        except Exception:
            pass

    # Stage 3: Very high penalty, extremely tight tolerance (Polish)
    penalty_w3 = 2000.0
    eps3 = 1e-6
    if time.time() < deadline - 2:
        try:
            res3 = minimize(
                lambda p: _objective_and_grad(p, penalty_w3, eps3)[0],
                params,
                jac=lambda p: _objective_and_grad(p, penalty_w3, eps3)[1],
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 300, 'ftol': 1e-15, 'gtol': 1e-13}
            )
            if res3.fun < _objective_and_grad(params, penalty_w3, eps3)[0]:
                params = res3.x
        except Exception:
            pass

    circles = np.column_stack([params[0::3], params[1::3], params[2::3]])
    # Final validation and repair
    if not _validate(circles):
        circles = _fix_boundaries(circles)
    
    return circles

def _initial_hex():
    cols = 5
    rows = 6
    positions = []
    r = 0.09
    for row in range(rows):
        for col in range(cols):
            if row % 2 == 1:
                x = (col + 0.5) / cols + 0.5 / cols
            else:
                x = (col + 0.5) / cols
            y = (row + 0.5) / rows
            if x > 1 or y > 1:
                continue
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_grid():
    side = int(np.ceil(np.sqrt(N)))
    positions = []
    r = 0.5 / side * 0.9
    for i in range(side):
        for j in range(side):
            x = (i + 0.5) / side
            y = (j + 0.5) / side
            positions.append([x, y, r])
            if len(positions) >= N:
                break
        if len(positions) >= N:
            break
    while len(positions) < N:
        positions.append([0.5, 0.5, r * 0.5])
    return np.array(positions[:N])

def _initial_corner_weighted(seed=0):
    rng = np.random.default_rng(seed)
    positions = []
    corners = [[0.08, 0.08], [0.92, 0.08], [0.08, 0.92], [0.92, 0.92],
               [0.5, 0.08], [0.5, 0.92], [0.08, 0.5], [0.92, 0.5]]
    for cx, cy in corners:
        positions.append([cx, cy, 0.08])
    for _ in range(N - len(positions)):
        x = rng.uniform(0.05, 0.95)
        y = rng.uniform(0.05, 0.95)
        positions.append([x, y, rng.uniform(0.06, 0.10)])
    return np.array(positions[:N])

def _initial_voronoi(seed=0):
    rng = np.random.default_rng(seed)
    points = rng.uniform(0.05, 0.95, size=(N, 2))
    radii = np.zeros(N)
    for i in range(N):
        dists = np.linalg.norm(points - points[i], axis=1)
        dists[i] = np.inf
        min_dist_neighbor = np.min(dists)
        wall_dist = np.min([points[i, 0], 1 - points[i, 0], points[i, 1], 1 - points[i, 1]])
        radii[i] = 0.5 * np.min([min_dist_neighbor, wall_dist]) * 0.95
    return np.column_stack([points[:, 0], points[:, 1], radii])

def construct():
    start_time = time.time()
    deadline = _start_time + _TIME_LIMIT

    # 1. Global Search using Differential Evolution
    # DE is robust for finding diverse configurations
    # We use a penalty objective for DE as well
    de_penalty = 100.0
    de_eps = 1e-2
    
    bounds_de = [(0, 1), (0, 1), (0, 0.5)] * N
    
    de_result_params = None
    
    # Check if we have enough time for DE
    if _remaining() > 40:
        try:
            de_res = differential_evolution(
                lambda p: _objective_and_grad(p, de_penalty, de_eps)[0],
                bounds=bounds_de,
                maxiter=30, # Keep it short
                popsize=10,
                tol=1e-4,
                seed=42,
                polish=False # We will do our own polishing
            )
            if de_res.success or de_res.fun < 1000: # Sanity check
                de_result_params = de_res.x
        except Exception:
            pass

    # 2. Local Polishing on multiple starts
    starts = []
    
    # Add heuristic starts
    starts.append(_initial_hex())
    starts.append(_initial_grid())
    starts.append(_initial_corner_weighted(seed=42))
    starts.append(_initial_voronoi(seed=123))
    
    # Add DE result if available
    if de_result_params is not None:
        de_circles = np.column_stack([de_result_params[0::3], de_result_params[1::3], de_result_params[2::3]])
        # Quick fix to ensure validity before polishing
        de_circles = _fix_boundaries(de_circles)
        starts.append(de_circles)

    best_circles = None
    best_score = -np.inf

    # Allocate time for local polishing
    n_starts = len(starts)
    time_per_start = max(5, min(20, (_remaining() - 5) / n_starts))

    for i, xyz0 in enumerate(starts):
        if _remaining() < 10:
            break
        
        local_deadline = time.time() + time_per_start
        
        # If we have plenty of time, allow more iterations for later starts
        if i == len(starts) - 1 and _remaining() > 30:
             local_deadline = deadline - 5 # Use almost all remaining time for the last (likely DE) start
            
        try:
            polished = _local_polish(xyz0, local_deadline)
            score = np.sum(polished[:, 2])
            
            if _validate(polished):
                if score > best_score:
                    best_score = score
                    best_circles = polished
        except Exception:
            continue

    # 3. Final Polish on the best candidate
    if best_circles is not None and _remaining() > 10:
        try:
            final_polished = _local_polish(best_circles, deadline - 2, max_iter_stage1=100, max_iter_stage2=200)
            if _validate(final_polished):
                final_score = np.sum(final_polished[:, 2])
                if final_score > best_score:
                    best_circles = final_polished
                    best_score = final_score
        except Exception:
            pass

    # 4. Fallback
    if best_circles is None:
        # Return a valid grid packing
        result = np.zeros((N, 3))
        side = 5
        for idx in range(N):
            i = idx % side
            j = idx // side
            result[idx, 0] = (i + 0.5) / side
            result[idx, 1] = (j + 0.5) / side
            result[idx, 2] = 0.5 / side * 0.8
        best_circles = result

    return best_circles
