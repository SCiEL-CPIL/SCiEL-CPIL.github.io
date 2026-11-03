"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
N = 50

# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.optimize import minimize


def _hexagonal_initial():
    """Generate a valid hexagonal lattice initial configuration."""
    # Try different spacings to get N points inside unit circle
    for attempt in range(50):
        spacing = 0.25 - attempt * 0.003
        if spacing <= 0:
            spacing = 0.1
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        mask = radius <= 1.0 - spacing / 2.0
        points = points[mask]
        if len(points) >= N:
            # Sort by distance from origin and take N closest
            idx = np.argsort(radius, kind="stable")[:N]
            return points[idx].copy()
    # Fallback: random points
    return np.random.randn(N, 2) * 0.5


def _compute_score(points_2d):
    """Compute the packing score: min of all pair distances/2 and boundary distances."""
    if points_2d.shape != (N, 2) or not np.all(np.isfinite(points_2d)):
        return 0.0
    # Boundary check
    radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
    if np.any(radii > 1.0 + 1e-9):
        return 0.0
    min_boundary = np.min(1.0 - radii)
    # Pairwise distances
    diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    return min(min_pair, min_boundary)


def _soft_min_objective(points_flat, beta=80.0):
    """Smooth surrogate for the negative of the minimum distance constraint."""
    points_2d = points_flat.reshape(N, 2)
    # Pairwise distances
    diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair_dists = dists[iu]
    # Boundary distances
    radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
    boundary_dists = 1.0 - radii
    # Combine all distances
    all_dists = np.concatenate([min_pair_dists, boundary_dists])
    scaled = all_dists / 2.0
    # Soft minimum via log-sum-exp
    neg_beta_scaled = -beta * scaled
    max_val = np.max(neg_beta_scaled)
    log_sum = max_val + np.log(np.sum(np.exp(neg_beta_scaled - max_val)))
    soft_min = -log_sum / beta
    return -soft_min


def _anneal_step(points, temperature, rng):
    """Perform one annealing step: move a random particle."""
    idx = rng.integers(0, N)
    new_points = points.copy()
    # Random displacement
    displacement = rng.normal(0, temperature, size=2)
    new_points[idx] += displacement
    # Keep inside container (project if outside)
    r = np.hypot(new_points[idx, 0], new_points[idx, 1])
    if r > 0.99:
        new_points[idx] *= 0.99 / r
    return new_points, idx


def _score_fast(points_2d):
    """Fast score computation for annealing."""
    radii = np.hypot(points_2d[:, 0], points_2d[:, 1])
    if np.any(radii > 1.0):
        return -1.0  # Invalid
    min_boundary = np.min(1.0 - radii)
    # Only compute pairwise if boundary is not the bottleneck
    diff = points_2d[:, np.newaxis, :] - points_2d[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff**2, axis=2))
    iu = np.triu_indices(N, k=1)
    min_pair = np.min(dists[iu]) / 2.0
    return min(min_pair, min_boundary)


def construct():
    t_start = time.time()
    deadline = 145.0  # 5s safety margin before 150s kill
    
    rng = np.random.default_rng(42)
    
    # Initial configuration
    best_points = _hexagonal_initial()
    best_score = _score_fast(best_points)
    
    # Phase 1: Simulated Annealing
    t0 = time.time()
    initial_temp = 0.05
    final_temp = 0.001
    cooling_rate = 0.9995
    
    temp = initial_temp
    patience = 0
    max_patience = 2000
    
    while time.time() < deadline - 30.0:  # Leave 30s for polishing
        new_points, idx = _anneal_step(best_points, temp, rng)
        new_score = _score_fast(new_points)
        
        if new_score > best_score:
            best_points = new_points
            best_score = new_score
            patience = 0
        else:
            # Metropolis acceptance
            delta = best_score - new_score
            if delta < 0 or rng.random() < np.exp(-delta / (temp + 1e-15)):
                best_points = new_points
                best_score = new_score
            patience += 1
        
        temp *= cooling_rate
        if temp < final_temp:
            temp = final_temp
        
        if patience > max_patience:
            break
    
    # Phase 2: Smooth optimization polish
    if time.time() < deadline - 5.0:
        x0 = best_points.flatten()
        try:
            result = minimize(
                _soft_min_objective,
                x0,
                method='L-BFGS-B',
                options={'maxiter': 500, 'ftol': 1e-14, 'gtol': 1e-12}
            )
            optimized_points = result.x.reshape(N, 2)
            opt_score = _score_fast(optimized_points)
            if opt_score > best_score:
                best_points = optimized_points
                best_score = opt_score
        except Exception:
            pass
    
    # Phase 3: Fine-tune with smaller annealing steps
    if time.time() < deadline - 2.0:
        temp = 0.005
        for _ in range(2000):
            if time.time() > deadline - 1.0:
                break
            new_points, idx = _anneal_step(best_points, temp, rng)
            new_score = _score_fast(new_points)
            if new_score > best_score:
                best_points = new_points
                best_score = new_score
            temp *= 0.999
    
    # Final validation
    if not np.all(np.isfinite(best_points)):
        return _hexagonal_initial()
    radii = np.hypot(best_points[:, 0], best_points[:, 1])
    if np.any(radii > 1.0):
        # Project inside
        mask = radii > 1.0
        best_points[mask] *= 0.999 / radii[mask]
    
    return best_points

# EVOLVE-BLOCK-END
