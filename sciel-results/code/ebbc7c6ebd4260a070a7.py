"""Improved packing: hexagonal lattice init + gradient-free local optimization with deadline guard."""
N = 300

# EVOLVE-BLOCK-START
import numpy as np
import time


def _score(points):
    """Compute the packing radius for N points in unit disk."""
    # Boundary constraint
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    # Pairwise distances
    diff = points[:, np.newaxis, :] - points[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Mask out diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists) / 2.0
    return min(min_pair, np.min(boundary))


def _score_fast(points):
    """Faster score computation using chunked pairwise distances."""
    n = len(points)
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    min_boundary = np.min(boundary)
    
    # Compute pairwise distances in chunks to save memory
    min_pair = np.inf
    chunk = 100
    for i in range(0, n, chunk):
        j_end = min(i + chunk, n)
        diff = points[i:j_end, np.newaxis, :] - points[np.newaxis, :, :]
        dists = np.sqrt(np.sum(diff * diff, axis=2))
        np.fill_diagonal(dists, np.inf)
        min_pair = min(min_pair, np.min(dists)) / 2.0
    return min(min_pair, min_boundary)


def _init_hexagonal():
    """Generate hexagonal lattice points inside unit disk, select N closest to center."""
    spacing = 2.0 / (np.sqrt(N) * 1.1 + 1.0)
    k = int(1.0 / spacing) + 3
    rows = []
    for i in range(-k, k + 1):
        for j in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            rows.append((x, y))
    points = np.array(rows)
    radius = np.hypot(points[:, 0], points[:, 1])
    # Keep points that fit with margin
    margin = spacing / 2.0
    mask = radius <= 1.0 - margin
    points = points[mask]
    radius = radius[mask]
    
    if len(points) >= N:
        # Sort by distance from center, take closest N
        idx = np.argsort(radius, kind="stable")[:N]
        return points[idx]
    else:
        # Fallback: generate more densely
        spacing *= 0.9
        k = int(1.0 / spacing) + 5
        rows = []
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                x = (i + 0.5 * (j % 2)) * spacing
                y = j * spacing * np.sqrt(3.0) / 2.0
                rows.append((x, y))
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        radius = radius[mask]
        if len(points) >= N:
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx]
        # Last resort: random
        return np.random.randn(N, 2) * 0.3


def _local_optimize(points, time_limit):
    """Iterative local optimization: move each point to maximize minimum distance."""
    start_time = time.time()
    n = len(points)
    points = points.copy()
    
    # Precompute initial score
    current_score = _score_fast(points)
    
    # Multi-restart with decreasing step sizes
    step_sizes = [0.02, 0.01, 0.005, 0.002, 0.001, 0.0005]
    
    for step in step_sizes:
        if time.time() - start_time > time_limit - 10:
            break
        
        improved = True
        while improved:
            if time.time() - start_time > time_limit - 10:
                break
            improved = False
            for i in range(n):
                if time.time() - start_time > time_limit - 10:
                    break
                
                orig = points[i].copy()
                # Try moving in 8 directions
                best_pos = orig.copy()
                best_score = current_score
                
                for dx in [-step, 0, step]:
                    for dy in [-step, 0, step]:
                        if dx == 0 and dy == 0:
                            continue
                        points[i] = orig + np.array([dx, dy])
                        s = _score_fast(points)
                        if s > best_score:
                            best_score = s
                            best_pos = points[i].copy()
                
                # Also try radial adjustments
                r = np.linalg.norm(orig)
                if r > 1e-10:
                    for dr in [-step, step]:
                        new_r = r + dr
                        if new_r < 0:
                            continue
                        points[i] = orig * (new_r / r)
                        s = _score_fast(points)
                        if s > best_score:
                            best_score = s
                            best_pos = points[i].copy()
                
                if np.any(best_pos != orig):
                    points[i] = best_pos
                    current_score = best_score
                    improved = True
    
    # Final polish with very small steps
    if time.time() - start_time < time_limit - 5:
        for step in [0.0002, 0.0001]:
            for i in range(n):
                if time.time() - start_time > time_limit - 3:
                    break
                orig = points[i].copy()
                best_pos = orig.copy()
                best_score = _score_fast(points)
                
                for dx in [-step, step]:
                    for dy in [-step, step]:
                        points[i] = orig + np.array([dx, dy])
                        s = _score_fast(points)
                        if s > best_score:
                            best_score = s
                            best_pos = points[i].copy()
                
                if np.any(best_pos != orig):
                    points[i] = best_pos
    
    return points, _score_fast(points)


def _multi_restart_optimize(time_limit):
    """Multiple restarts with different initial configurations."""
    start_time = time.time()
    best_points = None
    best_score = -np.inf
    
    # Initial hexagonal
    points = _init_hexagonal()
    pts, sc = _local_optimize(points, time_limit - 5)
    best_points = pts
    best_score = sc
    
    elapsed = time.time() - start_time
    # If we have time, try a few more restarts with perturbed initial configs
    restarts = 3
    for r in range(restarts):
        if time.time() - start_time > time_limit - 15:
            break
        # Perturb the best solution
        noise_scale = 0.01 / (r + 1)
        perturbed = best_points + np.random.randn(N, 2) * noise_scale
        # Clip to disk
        norms = np.linalg.norm(perturbed, axis=1)
        mask = norms > 0.95
        perturbed[mask] *= 0.95 / norms[mask]
        
        pts, sc = _local_optimize(perturbed, time_limit - (time.time() - start_time) - 5)
        if sc > best_score:
            best_points = pts
            best_score = sc
    
    return best_points, best_score


def construct():
    """Construct N=300 circle centers in unit disk maximizing packing radius."""
    start_time = time.time()
    total_time_limit = 110  # Safety margin before 120s evaluator timeout
    
    # Phase 1: Initial optimization
    points, score = _multi_restart_optimize(total_time_limit)
    
    elapsed = time.time() - start_time
    
    # Phase 2: If time remains, do a final fine-tuning pass
    if elapsed < total_time_limit - 5:
        remaining = total_time_limit - elapsed - 3
        if remaining > 10:
            points, score = _local_optimize(points, remaining)
    
    # Ensure all points are strictly inside the unit circle
    norms = np.linalg.norm(points, axis=1)
    mask = norms >= 1.0
    if np.any(mask):
        points[mask] *= (1.0 - 1e-6) / norms[mask]
    
    return points
# EVOLVE-BLOCK-END
