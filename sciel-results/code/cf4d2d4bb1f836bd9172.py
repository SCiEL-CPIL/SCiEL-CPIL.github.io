import numpy as np
import time
import math

N = 50
START_TIME = time.time()
TIME_LIMIT = 140.0  # Leave 10s buffer for safety


def _calc_score(points):
    """Calculate the minimum radius for a given configuration."""
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dist = np.sqrt(np.sum(diff**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    min_pair_dist = np.min(dist)
    
    radii = np.hypot(points[:, 0], points[:, 1])
    min_boundary_dist = np.min(1.0 - radii)
    
    return min(min_pair_dist / 2.0, min_boundary_dist)


def _ring_init(r_est=0.125):
    """Generate ring-based initial configuration matching known optimal structure.
    
    Structure: 1 center + 5 ring1 + 13 ring2 + 31 ring3 = 50
    """
    centers = np.zeros((N, 2))
    
    # Center point
    centers[0] = [0.0, 0.0]
    
    # Ring 1: 5 points
    R1 = r_est / np.sin(np.pi / 5)
    for i in range(5):
        angle = 2 * np.pi * i / 5 + np.pi / 5
        centers[1 + i] = [R1 * np.cos(angle), R1 * np.sin(angle)]
    
    # Ring 2: 13 points
    R2 = R1 + 1.5 * r_est
    for i in range(13):
        angle = 2 * np.pi * i / 13
        centers[6 + i] = [R2 * np.cos(angle), R2 * np.sin(angle)]
    
    # Ring 3: 31 points
    R3 = r_est / np.sin(np.pi / 31)
    R3 = min(R3, 1.0 - r_est)
    for i in range(31):
        angle = 2 * np.pi * i / 31 + np.pi / 31
        centers[19 + i] = [R3 * np.cos(angle), R3 * np.sin(angle)]
    
    return centers


def _hex_init(perturb=0.0):
    """Generate hexagonal lattice configuration as fallback."""
    best_points = None
    best_score = -1
    
    for scale in np.linspace(0.22, 0.28, 8):
        k = int(1.0 / scale) + 2
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * scale
                y = j * scale * np.sqrt(3.0) / 2.0
                if np.hypot(x, y) <= 1.0:
                    rows.append([x, y])
        points = np.array(rows)
        if len(points) < N:
            continue
        
        radii = np.hypot(points[:, 0], points[:, 1])
        indices = np.argsort(radii)
        points = points[indices[:N]]
        
        if perturb > 0:
            points = points + np.random.randn(N, 2) * perturb
        
        score = _calc_score(points)
        if score > best_score:
            best_score = score
            best_points = points.copy()
            
    return best_points


def _soft_min_energy(points, beta=50.0):
    """Compute a smooth surrogate for the minimum radius using log-sum-exp.
    
    Returns (-r_smooth, gradient) where r_smooth approximates min(min_pair/2, min_boundary).
    We minimize -r_smooth (equivalently maximize r_smooth).
    """
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    sq_dist = np.sum(diff**2, axis=2)
    np.fill_diagonal(sq_dist, 1e10)
    
    # Pair distances / 2
    pair_vals = 0.5 * np.sqrt(sq_dist)
    # Boundary distances
    radii = np.hypot(points[:, 0], points[:, 1])
    boundary_vals = 1.0 - radii
    
    # Use log-sum-exp to approximate min
    # For soft min: min(x) ~ -log(sum(exp(-beta*x)))/beta
    # We'll combine pair and boundary terms
    
    # Flatten all constraints
    # Pair constraints: i<j to avoid double counting
    pair_min_vals = []
    for i in range(N):
        for j in range(i+1, N):
            pair_min_vals.append(0.5 * np.sqrt(sq_dist[i, j]))
    pair_min_vals = np.array(pair_min_vals)
    boundary_min_vals = boundary_vals
    
    all_vals = np.concatenate([pair_min_vals, boundary_min_vals])
    
    # Soft min
    max_val = np.max(all_vals)
    exp_vals = np.exp(-beta * (all_vals - max_val))
    log_sum = np.log(np.sum(exp_vals)) + beta * max_val
    r_smooth = max_val - log_sum / beta
    
    # Gradient with respect to points
    # d(r_smooth)/d(point_k) = sum over constraints involving k of d(r_smooth)/d(val_c) * d(val_c)/d(point_k)
    # d(r_smooth)/d(val_c) = -exp(-beta*(val_c - max_val)) / sum(exp(...))
    weights = -exp_vals / np.sum(exp_vals)
    
    grad = np.zeros_like(points)
    
    # Pair constraints
    idx = 0
    for i in range(N):
        for j in range(i+1, N):
            w = weights[idx]
            d = np.sqrt(sq_dist[i, j])
            if d > 1e-9:
                vec = (points[i] - points[j]) / d
                grad[i] += 0.5 * w * vec
                grad[j] -= 0.5 * w * vec
            idx += 1
    
    # Boundary constraints
    for i in range(N):
        w = weights[N*(N-1)//2 + i]
        r = radii[i]
        if r > 1e-9:
            grad[i] -= w * points[i] / r
        else:
            grad[i] -= w * np.array([1.0, 0.0])
    
    return r_smooth, grad


def _smooth_optimize(points, max_iter=200, beta=30.0, lr=0.005):
    """Optimize using the soft-min surrogate with gradient ascent."""
    best_points = points.copy()
    best_score = _calc_score(best_points)
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        r_smooth, grad = _soft_min_energy(points, beta)
        
        # Gradient ascent on r_smooth
        points_new = points + lr * grad
        
        # Project back inside unit circle
        radii = np.hypot(points_new[:, 0], points_new[:, 1])
        for i in range(N):
            if radii[i] > 1.0 - 0.01:
                scale = (1.0 - 0.01) / radii[i]
                points_new[i] *= scale
        
        score_new = _calc_score(points_new)
        if score_new > best_score:
            best_score = score_new
            best_points = points_new.copy()
        
        points = points_new
        lr *= 0.98
        beta *= 1.05  # Increase sharpness over time
    
    return best_points


def _polish(points, max_iter=500, tol=1e-8):
    """High-precision polish pass using small steps on the exact objective."""
    best_points = points.copy()
    best_score = _calc_score(best_points)
    step = 0.002
    
    for it in range(max_iter):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        if step < tol:
            break
        
        improved = False
        for _ in range(N * 2):
            if time.time() - START_TIME > TIME_LIMIT:
                break
            
            diff = best_points[:, np.newaxis, :] - best_points[np.newaxis, :, :]
            dist = np.sqrt(np.sum(diff**2, axis=2))
            np.fill_diagonal(dist, np.inf)
            min_pair_dist = np.min(dist)
            
            radii = np.hypot(best_points[:, 0], best_points[:, 1])
            min_boundary_dist = np.min(1.0 - radii)
            
            current_score = min(min_pair_dist / 2.0, min_boundary_dist)
            
            if min_pair_dist / 2.0 < min_boundary_dist:
                idx_i, idx_j = np.unravel_index(np.argmin(dist), dist.shape)
                if idx_i > idx_j:
                    idx_i, idx_j = idx_j, idx_i
                
                vec = best_points[idx_i] - best_points[idx_j]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec /= norm
                else:
                    vec = np.array([1.0, 0.0])
                
                candidate = best_points.copy()
                candidate[idx_i] += step * vec
                candidate[idx_j] -= step * vec
            else:
                idx_i = np.argmin(min_boundary_dist)
                vec = -best_points[idx_i]
                norm = np.linalg.norm(vec)
                if norm > 1e-9:
                    vec /= norm
                else:
                    vec = np.array([1.0, 0.0])
                
                candidate = best_points.copy()
                candidate[idx_i] += step * vec
            
            score_candidate = _calc_score(candidate)
            if score_candidate > best_score + 1e-12:
                best_score = score_candidate
                best_points = candidate
                improved = True
        
        if not improved:
            step *= 0.8
        else:
            step *= 0.95
    
    return best_points


def _optimize(points, max_steps=3000, initial_step=0.015):
    """Gradient ascent on the exact min objective."""
    step = initial_step
    best_score = _calc_score(points)
    best_points = points.copy()
    no_improve_count = 0
    
    for s in range(max_steps):
        if time.time() - START_TIME > TIME_LIMIT:
            break
        
        diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
        dist = np.sqrt(np.sum(diff**2, axis=2))
        np.fill_diagonal(dist, np.inf)
        
        radii = np.hypot(points[:, 0], points[:, 1])
        boundary_dist = 1.0 - radii
        
        min_pair_dist = np.min(dist)
        min_boundary_dist = np.min(boundary_dist)
        
        current_score = min(min_pair_dist / 2.0, min_boundary_dist)
        
        if current_score > best_score + 1e-10:
            best_score = current_score
            best_points = points.copy()
            no_improve_count = 0
        else:
            no_improve_count += 1
            if no_improve_count > 300:
                break
        
        if min_pair_dist / 2.0 < min_boundary_dist:
            idx_i, idx_j = np.unravel_index(np.argmin(dist), dist.shape)
            if idx_i > idx_j:
                idx_i, idx_j = idx_j, idx_i
            
            vec = points[idx_i] - points[idx_j]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
            points[idx_j] -= step * vec
        else:
            idx_i = np.argmin(boundary_dist)
            vec = -points[idx_i]
            norm = np.linalg.norm(vec)
            if norm > 1e-9:
                vec /= norm
            else:
                vec = np.array([1.0, 0.0])
            
            points[idx_i] += step * vec
        
        if s % 50 == 0 and s > 0:
            points += np.random.randn(N, 2) * step * 0.1
        
        step *= 0.995
    
    return best_points


def construct():
    best_points = None
    best_score = -1
    
    # Strategy 1: Ring-based initialization with smooth optimization
    for r_est in [0.125, 0.126, 0.124, 0.127, 0.123]:
        if time.time() - START_TIME > TIME_LIMIT - 30:
            break
        
        points = _ring_init(r_est)
        score_init = _calc_score(points)
        
        if score_init > best_score:
            best_score = score_init
            best_points = points.copy()
        
        # Smooth optimization phase
        if time.time() - START_TIME < TIME_LIMIT - 20:
            smoothed = _smooth_optimize(points.copy(), max_iter=150, beta=25.0, lr=0.004)
            score_smooth = _calc_score(smoothed)
            if score_smooth > best_score:
                best_score = score_smooth
                best_points = smoothed.copy()
        
        # Exact gradient ascent
        if time.time() - START_TIME < TIME_LIMIT - 15:
            optimized = _optimize(smoothed.copy(), max_steps=2000, initial_step=0.01)
            score_opt = _calc_score(optimized)
            if score_opt > best_score:
                best_score = score_opt
                best_points = optimized.copy()
    
    # Strategy 2: Hex lattice with perturbation
    for restart in range(2):
        if time.time() - START_TIME > TIME_LIMIT - 25:
            break
        
        points = _hex_init(perturb=0.004 * (restart + 1))
        if points is None:
            continue
        
        smoothed = _smooth_optimize(points.copy(), max_iter=100, beta=30.0, lr=0.003)
        score_smooth = _calc_score(smoothed)
        if score_smooth > best_score:
            best_score = score_smooth
            best_points = smoothed.copy()
        
        if time.time() - START_TIME < TIME_LIMIT - 15:
            optimized = _optimize(smoothed.copy(), max_steps=1500, initial_step=0.008)
            score = _calc_score(optimized)
            if score > best_score:
                best_score = score
                best_points = optimized.copy()
    
    # Final high-precision polish
    if best_points is not None and time.time() - START_TIME < TIME_LIMIT - 5:
        polished = _polish(best_points.copy(), max_iter=400, tol=1e-8)
        score = _calc_score(polished)
        if score > best_score:
            best_points = polished
    
    # Ensure validity
    if best_points is not None:
        radii = np.hypot(best_points[:, 0], best_points[:, 1])
        for i in range(N):
            if radii[i] >= 1.0:
                scale = 0.999 / radii[i] if radii[i] > 0 else 1.0
                best_points[i] *= scale
    
    return best_points
