# EVOLVE-BLOCK-START
import numpy as np
import time
import scipy.spatial.distance as spd
from scipy.optimize import minimize


def calculate_score(points):
    """Calculate R = (d_min / d_max)^2"""
    n = len(points)
    if n < 2:
        return 0.0
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0:
        return 0.0
    ratio = d_min / d_max
    return ratio * ratio


def smooth_objective(x, n, alpha=20.0):
    """
    Smooth objective to maximize: log(d_min) - log(d_max)
    Using log-sum-exp approximation for min and max.
    We want to maximize (d_min/d_max)^2, so maximizing log(d_min) - log(d_max) is equivalent.
    
    x is a flat array of 2*n coordinates.
    """
    points = x.reshape(n, 2)
    
    # Compute all pairwise distances
    # Use scipy for efficiency
    dists = spd.pdist(points)
    if len(dists) == 0:
        return 0.0
    
    # Clip to avoid log(0)
    dists = np.maximum(dists, 1e-12)
    
    # log-sum-exp for min: log(sum(exp(-alpha * d))) -> -1/alpha * log(sum(exp(-alpha * d))) approximates min
    # Actually: min(d) ≈ -1/alpha * log(sum(exp(-alpha * d)))
    # log(min(d)) ≈ log(-1/alpha * log(sum(exp(-alpha * d)))) ... this is getting complex.
    
    # Simpler: just use d_min and d_max directly, but they are non-smooth.
    # However, L-BFGS-B can often handle non-smooth functions if we provide a smooth surrogate.
    
    # Let's try a different smooth surrogate:
    # f(x) = (1/k) * sum(d_i^k) approximates max(d_i) as k -> inf
    # g(x) = (1/k) * sum(1/d_i^k) approximates 1/min(d_i) as k -> inf, so min(d_i) approx 1/g(x)
    
    # Let's use k=10 for smoothness
    k = 10
    max_d = (np.sum(dists**k) / len(dists)) ** (1.0/k)
    # For min, we can use: min_d = 1 / ((1/k) * sum((1/d_i)^k))^(1/k)
    inv_dists = 1.0 / dists
    min_d = 1.0 / ((np.sum(inv_dists**k) / len(inv_dists)) ** (1.0/k))
    
    if max_d <= 0 or min_d <= 0:
        return 0.0
    
    # We want to maximize log(min_d) - log(max_d)
    # But we can also just maximize (min_d / max_d)^2
    # Let's maximize the log ratio for numerical stability
    return np.log(min_d) - np.log(max_d)


def local_search(points, start_time, max_time, seed=0, n_iter=5000):
    """
    Perform local search using L-BFGS-B on the smooth objective.
    """
    n = points.shape[0]
    x0 = points.flatten()
    
    best_x = x0.copy()
    best_val = smooth_objective(x0, n)
    
    # Run L-BFGS-B
    try:
        result = minimize(
            lambda x: -smooth_objective(x, n),  # Negate for minimization
            x0,
            method='L-BFGS-B',
            options={'maxiter': 100, 'ftol': 1e-10, 'gtol': 1e-8}
        )
        if result.success:
            val = smooth_objective(result.x, n)
            if val > best_val:
                best_val = val
                best_x = result.x.copy()
    except Exception:
        pass
    
    # Check time
    if time.time() - start_time > max_time - 10:
        return best_x.reshape(n, 2)
    
    # Additional local refinement with smaller steps
    rng = np.random.default_rng(seed)
    for i in range(n_iter):
        if time.time() - start_time > max_time - 10:
            break
        
        # Random perturbation
        idx = rng.integers(n)
        delta = rng.normal(0, 0.01, 2)
        new_x = best_x.copy()
        new_x[idx*2:idx*2+2] += delta
        
        new_val = smooth_objective(new_x, n)
        if new_val > best_val:
            best_val = new_val
            best_x = new_x.copy()
    
    return best_x.reshape(n, 2)


def min_max_dist_dim2_16() -> np.ndarray:
    """
    Creates 16 points in 2 dimensions in order to maximize the ratio of minimum to maximum distance.

    Returns
        points: np.ndarray of shape (16,2) containing the (x,y) coordinates of the 16 points.

    """
    start_time = time.time()
    max_time = 580.0  # Safety margin before 630s limit
    
    n = 16
    best_points = None
    best_score = -1.0
    
    # Multiple restarts with different initializations
    n_restarts = 10
    
    for restart in range(n_restarts):
        remaining_time = max_time - (time.time() - start_time)
        if remaining_time < 30:
            break
        
        # Choose initialization
        rng = np.random.default_rng(restart * 100 + 42)
        
        if restart % 3 == 0:
            # Hexagonal grid
            rows = 4
            cols = 4
            dx = 1.0
            dy = dx * np.sqrt(3) / 2
            points = []
            for i in range(rows):
                for j in range(cols):
                    x = j * dx + (i % 2) * (dx / 2)
                    y = i * dy
                    points.append([x, y])
            init_points = np.array(points[:n])
            init_points += rng.normal(0, 0.1, init_points.shape)
        elif restart % 3 == 1:
            # Circular
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False) + rng.normal(0, 0.2, n)
            radii = 1.0 + rng.normal(0, 0.2, n)
            init_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        else:
            # Random in unit disk
            angles = rng.uniform(0, 2 * np.pi, n)
            radii = rng.uniform(0.3, 1.0, n)
            init_points = np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])
        
        # Run local search with L-BFGS-B
        time_per_restart = remaining_time / (n_restarts - restart)
        points, score = local_search(
            init_points, start_time, start_time + time_per_restart, seed=restart * 1000
        )
        
        actual_score = calculate_score(points)
        if actual_score > best_score:
            best_score = actual_score
            best_points = points.copy()
    
    # If no valid result, return a default configuration
    if best_points is None:
        best_points = np.zeros((n, 2))
        angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
        best_points[:, 0] = np.cos(angles)
        best_points[:, 1] = np.sin(angles)
    
    # Ensure shape is (16, 2)
    assert best_points.shape == (16, 2)
    
    return best_points


# EVOLVE-BLOCK-END
