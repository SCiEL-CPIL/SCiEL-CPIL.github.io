"""Pack 300 equal circles in unit circle using gradient-based optimization."""
import numpy as np
from scipy.spatial.distance import cdist
import time

N = 300


def _soft_min(x, beta):
    """Soft minimum using log-sum-exp trick."""
    x = np.asarray(x)
    if x.size == 0:
        return 0.0
    m = np.min(x)
    return m + np.log(np.sum(np.exp(-(x - m) * beta)) / len(x)) / beta


def _score(points):
    """Compute the packing radius (min of inter-circle distances/2 and boundary distances)."""
    if len(points) < 2:
        return 0.0
    # Inter-circle distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    min_inter = np.min(d) / 2.0
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    min_boundary = np.min(1.0 - norms)
    return min(min_inter, min_boundary)


def _hex_initialization(n_points):
    """Generate hexagonal lattice points inside unit circle, return first n_points."""
    # Try multiple spacings to find one that gives at least n_points
    for spacing in np.linspace(0.03, 0.12, 30):
        # Generate hex lattice
        dx = spacing
        dy = spacing * np.sqrt(3) / 2
        # Range to cover unit circle
        R = 1.0
        nx = int(R / dx) + 3
        ny = int(R / dy) + 3
        
        points = []
        for j in range(-ny, ny + 1):
            for i in range(-nx, nx + 1):
                x = i * dx + (j % 2) * dx / 2
                y = j * dy
                if x * x + y * y <= (1.0 - spacing / 2) ** 2:
                    points.append([x, y])
        
        points = np.array(points)
        if len(points) >= n_points:
            # Sort by distance from center, take closest n_points
            dists = np.linalg.norm(points, axis=1)
            idx = np.argsort(dists)
            return points[idx[:n_points]]
    
    # Fallback: random
    return np.random.randn(n_points, 2) * 0.5


def _gradient_step(points, lr, beta):
    """
    Compute gradient of soft-min objective and take a step.
    Objective: maximize min(min_inter/2, min_boundary)
    Using soft-min surrogate.
    """
    n = len(points)
    # Compute all pairwise distances
    d = cdist(points, points)
    np.fill_diagonal(d, np.inf)
    
    # Boundary distances
    norms = np.linalg.norm(points, axis=1)
    boundary = 1.0 - norms
    
    # Soft-min of inter-circle distances / 2
    inter = d / 2.0
    # Soft-min of boundary distances
    # Use log-sum-exp for numerical stability
    def softmin_vals(arr, beta):
        if arr.size == 0:
            return 0.0, np.zeros_like(arr)
        m = np.min(arr)
        exp_vals = np.exp(-(arr - m) * beta)
        s = np.sum(exp_vals)
        vals = exp_vals / s
        return m + np.log(s) / beta, vals
    
    soft_inter, w_inter = softmin_vals(inter, beta)
    soft_bound, w_bound = softmin_vals(boundary, beta)
    
    # Combined soft-min: min(soft_inter, soft_bound)
    # Use another soft-min to combine
    combined = np.array([soft_inter, soft_bound])
    m_c = np.min(combined)
    exp_c = np.exp(-(combined - m_c) * beta)
    s_c = np.sum(exp_c)
    w_c = exp_c / s_c
    obj = m_c + np.log(s_c) / beta
    
    # Gradient computation
    # d obj / d points[i]
    grad = np.zeros_like(points)
    
    # From inter-circle term
    # w_inter[i,j] is weight for distance between i and j
    # d (d_ij/2) / d p_i = (p_i - p_j) / (2 * d_ij)
    # But we have soft-min, so gradient is weighted sum
    
    # For each pair (i,j), contribution to grad[i] from w_inter[i,j]:
    # w_inter[i,j] * (p_i - p_j) / (2 * d_ij) * w_c[0]
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if d[i, j] > 1e-10:
                dir_vec = (points[i] - points[j]) / d[i, j]
                grad[i] += w_inter[i, j] * dir_vec * 0.5 * w_c[0]
    
    # From boundary term
    # d (1 - |p_i|) / d p_i = -p_i / |p_i|
    for i in range(n):
        if norms[i] > 1e-10:
            grad[i] += w_bound[i] * (-points[i] / norms[i]) * w_c[1]
    
    # Take gradient ascent step (maximizing objective)
    new_points = points + lr * grad
    
    # Project back: ensure points stay inside circle
    new_norms = np.linalg.norm(new_points, axis=1)
    mask = new_norms > 0.999
    new_points[mask] *= 0.999 / new_norms[mask]
    
    return new_points, obj


def construct():
    start_time = time.time()
    deadline = 145.0  # Safety margin
    
    # Initialize with hexagonal lattice
    points = _hex_initialization(N)
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Multi-stage optimization with annealing
    # Stage 1: Coarse optimization with large learning rate
    # Stage 2: Fine optimization with smaller learning rate
    # Stage 3: Polish with very small steps and high beta
    
    stages = [
        # (num_steps, lr, beta)
        (200, 0.01, 100),
        (300, 0.005, 500),
        (500, 0.002, 2000),
        (800, 0.001, 10000),
        (1000, 0.0005, 50000),
    ]
    
    for step_count, lr, beta in stages:
        for step in range(step_count):
            if time.time() > deadline - 10:  # Check deadline with margin
                break
            
            # Occasionally try random restarts from current best
            if step % 100 == 50 and np.random.rand() < 0.1:
                # Perturb current points
                perturb = np.random.randn(*points.shape) * 0.01
                trial = points + perturb
                trial_norms = np.linalg.norm(trial, axis=1)
                mask = trial_norms > 0.999
                trial[mask] *= 0.999 / trial_norms[mask]
                trial_score = _score(trial)
                if trial_score > best_score:
                    points = trial
                    best_points = trial.copy()
                    best_score = trial_score
            
            points, obj = _gradient_step(points, lr, beta)
            
            # Check if improved
            curr_score = _score(points)
            if curr_score > best_score:
                best_score = curr_score
                best_points = points.copy()
        
        if time.time() > deadline - 10:
            break
    
    # Final polish: use exact local optimization with very small steps
    if time.time() < deadline - 5:
        for _ in range(2000):
            if time.time() > deadline - 2:
                break
            # Try small random perturbations and keep if better
            for _ in range(5):
                trial = best_points + np.random.randn(*best_points.shape) * 0.001
                trial_norms = np.linalg.norm(trial, axis=1)
                mask = trial_norms > 0.999
                trial[mask] *= 0.999 / trial_norms[mask]
                trial_score = _score(trial)
                if trial_score > best_score:
                    best_score = trial_score
                    best_points = trial
    
    return best_points
