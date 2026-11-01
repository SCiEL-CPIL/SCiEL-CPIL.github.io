"""
Packing 50 equal circles in a unit circle.
Uses a combination of:
1. Good initializations (hex, ring, random)
2. Simulated Annealing for global search
3. Gradient-based polishing using a smooth soft-min surrogate (log-sum-exp)
4. Final high-precision local optimization

The key insight is to use the smooth surrogate to get a good gradient signal
and then switch to exact evaluation for final polishing.
"""
import numpy as np
import time
import math
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit


def _compute_score(centers):
    """
    Compute the packing score for given centers.
    Score = min(min_pair_dist / 2, min_bound_dist)
    """
    n = len(centers)
    if n < 2:
        return 0.0
    
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    
    # Mask diagonal
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    
    # Boundary distances
    norms = np.linalg.norm(centers, axis=1)
    min_bound = np.min(1.0 - norms)
    
    return min(min_pair / 2.0, min_bound)


def _soft_min_objective(centers, beta):
    """
    Smooth surrogate: minimize -soft_min of (pair_dists/2, bound_dists).
    Returns (loss, gradient) where gradient is w.r.t. centers.
    Uses log-sum-exp for numerical stability.
    """
    n = len(centers)
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dists, np.inf)
    
    # Collect all values
    iu = np.triu_indices(n, k=1)
    pair_vals = dists[iu] / 2.0
    norms = np.linalg.norm(centers, axis=1)
    bound_vals = 1.0 - norms
    
    all_vals = np.concatenate([pair_vals, bound_vals])
    
    # For numerical stability
    m = -beta * all_vals
    m_max = np.max(m)
    m_shifted = m - m_max
    exp_m = np.exp(m_shifted)
    log_sum = m_max + np.log(np.sum(exp_m))
    loss = log_sum / beta
    
    weights = exp_m / np.sum(exp_m)
    pair_weights = weights[:len(pair_vals)]
    bound_weights = weights[len(pair_vals):]
    
    grad = np.zeros_like(centers)
    
    # Pair contributions
    for k in range(len(pair_vals)):
        i, j = iu[0][k], iu[1][k]
        w = pair_weights[k]
        d = dists[i, j]
        if d > 1e-12:
            diff_vec = (centers[i] - centers[j]) / d
            grad[i] += w * diff_vec
            grad[j] -= w * diff_vec
    
    # Boundary contributions
    for i in range(n):
        w = bound_weights[i]
        norm_i = norms[i]
        if norm_i > 1e-12:
            grad[i] -= w * centers[i] / norm_i
    
    return loss, grad


def _get_hex_initialization(n):
    """Generate a hexagonal lattice initialization."""
    r_est = 0.12
    s = 2 * r_est
    
    points = []
    max_range = int(2.0 / s) + 3
    for j in range(-max_range, max_range + 1):
        for i in range(-max_range, max_range + 1):
            x = i * s + (j % 2) * s / 2.0
            y = j * s * np.sqrt(3) / 2.0
            if np.hypot(x, y) <= 0.95:
                points.append([x, y])
                
    if len(points) == 0:
        return _get_random_initialization(n, seed=42)
        
    points = np.array(points)
    
    if len(points) < n:
        return _get_random_initialization(n, seed=42)
    
    dists = np.linalg.norm(points, axis=1)
    sorted_idx = np.argsort(dists)
    return points[sorted_idx[:n]]


def _get_random_initialization(n, seed=42):
    """Generate random initialization inside the unit circle."""
    rng = np.random.default_rng(seed)
    centers = np.zeros((n, 2))
    for i in range(n):
        while True:
            x = rng.uniform(-1, 1)
            y = rng.uniform(-1, 1)
            if x*x + y*y <= 0.95**2:
                centers[i] = [x, y]
                break
    return centers


def _get_ring_initialization(n):
    """Generate a ring-based initialization matching known optimal structure."""
    centers = []
    centers.append([0.0, 0.0])
    
    # 1 center + 6 + 12 + 12 + 19 = 50
    rings = [
        (6, 0.25),
        (12, 0.48),
        (12, 0.70),
        (19, 0.88)
    ]
    
    count = 1
    for count_in_ring, radius in rings:
        angles = np.linspace(0, 2 * np.pi, count_in_ring, endpoint=False)
        for angle in angles:
            if count >= n:
                break
            centers.append([radius * np.cos(angle), radius * np.sin(angle)])
            count += 1
        if count >= n:
            break
            
    return np.array(centers[:n])


def _simulated_annealing(centers, start_time, time_budget, seed=12345):
    """Simulated Annealing to maximize the packing score."""
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    T = 0.02
    alpha = 0.9997
    max_delta = 0.06
    
    rng = np.random.default_rng(seed)
    
    while time.time() - start_time < time_budget:
        i = rng.integers(0, N)
        
        angle = rng.uniform(0, 2 * np.pi)
        magnitude = rng.uniform(0, max_delta)
        
        dx = magnitude * np.cos(angle)
        dy = magnitude * np.sin(angle)
        
        new_point = current[i] + np.array([dx, dy])
        
        if np.linalg.norm(new_point) > 0.999:
            continue
            
        old_point = current[i].copy()
        current[i] = new_point
        
        new_score = _compute_score(current)
        
        if new_score > current_score or (rng.random() < math.exp((new_score - current_score) / T)):
            current_score = new_score
            if new_score > best_score:
                best_score = new_score
                best = current.copy()
        else:
            current[i] = old_point
            
        T *= alpha
        
        if T < 0.001:
            T = 0.005
            max_delta = 0.02
            
    return best, best_score


def _gradient_polish(centers, start_time, time_budget):
    """
    Gradient-based polishing using the soft-min surrogate.
    Uses L-BFGS-B with the smooth objective, then switches to exact.
    """
    current = centers.copy()
    best = current.copy()
    best_score = _compute_score(current)
    
    # Phase 1: Use smooth surrogate with increasing beta
    betas = [30, 80, 200, 500, 1200]
    
    for beta in betas:
        if time.time() - start_time > time_budget - 2:
            break
        
        def obj_and_grad(flat):
            c = flat.reshape(-1, 2)
            loss, grad = _soft_min_objective(c, beta)
            return loss, grad.flatten()
        
        x0 = current.flatten()
        
        try:
            result = minimize(
                obj_and_grad, x0,
                jac=True,
                method='L-BFGS-B',
                options={
                    'maxiter': 300,
                    'ftol': 1e-12,
                    'gtol': 1e-8,
                    'maxls': 50
                }
            )
            
            new_centers = result.x.reshape(-1, 2)
            # Project into unit circle
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.9999
            if np.any(mask):
                new_centers = new_centers.copy()
                new_centers[mask] *= 0.9999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best = new_centers.copy()
            current = new_centers
            
        except Exception:
            pass
    
    # Phase 2: Fine-tune with exact score using Nelder-Mead
    if time.time() - start_time < time_budget - 1:
        def exact_obj(flat):
            c = flat.reshape(-1, 2)
            # Add small penalty for being outside
            norms = np.linalg.norm(c, axis=1)
            penalty = np.sum(np.maximum(0, norms - 1.0)) * 100
            return -_compute_score(c) + penalty
        
        x0 = current.flatten()
        try:
            result = minimize(
                exact_obj, x0,
                method='Nelder-Mead',
                options={
                    'xatol': 1e-8,
                    'fatol': 1e-10,
                    'maxiter': 2000,
                    'maxfev': 5000
                }
            )
            
            new_centers = result.x.reshape(-1, 2)
            norms = np.linalg.norm(new_centers, axis=1)
            mask = norms > 0.9999
            if np.any(mask):
                new_centers = new_centers.copy()
                new_centers[mask] *= 0.9999 / norms[mask]
            
            new_score = _compute_score(new_centers)
            if new_score > best_score:
                best_score = new_score
                best = new_centers.copy()
                
        except Exception:
            pass
    
    return best, best_score


def _local_polish(centers, start_time, time_budget):
    """
    Coordinate descent style local polish for final precision.
    """
    current = centers.copy()
    current_score = _compute_score(current)
    best = current.copy()
    best_score = current_score
    
    step_size = 0.0005
    rng = np.random.default_rng(54321)
    
    while time.time() - start_time < time_budget:
        improved = False
        
        for i in range(N):
            if time.time() - start_time > time_budget:
                break
                
            # Try several directions
            for _ in range(8):
                angle = rng.uniform(0, 2 * np.pi)
                d = np.array([step_size * np.cos(angle), step_size * np.sin(angle)])
                
                new_point = current[i] + d
                if np.linalg.norm(new_point) > 0.9999:
                    continue
                    
                old_point = current[i].copy()
                current[i] = new_point
                new_score = _compute_score(current)
                
                if new_score > current_score:
                    current_score = new_score
                    if new_score > best_score:
                        best_score = new_score
                        best = current.copy()
                    improved = True
                else:
                    current[i] = old_point
        
        if not improved:
            step_size *= 0.8
            if step_size < 1e-6:
                break
                
    return best, best_score


def construct():
    start_time = time.time()
    best_centers = None
    best_score = 0.0
    
    # Diverse initializations
    initializations = [
        _get_ring_initialization(N),
        _get_hex_initialization(N),
        _get_random_initialization(N, seed=1),
        _get_random_initialization(N, seed=42),
        _get_random_initialization(N, seed=99),
        _get_random_initialization(N, seed=200),
    ]
    
    # Process each initialization
    n_init = len(initializations)
    # Allocate time: SA gets ~60% of per-init budget, gradient polish ~35%, local ~5%
    
    for i, init in enumerate(initializations):
        elapsed = time.time() - start_time
        remaining = DEADLINE - elapsed - 5  # 5s reserve for final polish
        
        if remaining <= 0:
            break
        
        # Per-init budget
        per_init_budget = remaining / max(1, n_init - i)
        
        # Pre-process: ensure inside
        norms = np.linalg.norm(init, axis=1)
        mask = norms > 0.95
        if np.any(mask):
            init = init.copy()
            init[mask] *= 0.95 / norms[mask]
        
        # SA Phase
        sa_budget = per_init_budget * 0.55
        if sa_budget > 0.5:
            sa_centers, sa_score = _simulated_annealing(init, start_time, sa_budget, seed=12345 + i)
        else:
            sa_centers, sa_score = init, _compute_score(init)
        
        # Gradient Polish Phase
        grad_budget = per_init_budget * 0.38
        if grad_budget > 1.0:
            grad_centers, grad_score = _gradient_polish(sa_centers, start_time, grad_budget)
        else:
            grad_centers, grad_score = sa_centers, sa_score
        
        # Local Polish Phase
        local_budget = per_init_budget * 0.07
        if local_budget > 0.3:
            final_centers, final_score = _local_polish(grad_centers, start_time, local_budget)
        else:
            final_centers, final_score = grad_centers, grad_score
        
        if final_score > best_score:
            best_score = final_score
            best_centers = final_centers.copy()
    
    # Final high-precision polish on the best solution
    if best_centers is not None:
        elapsed = time.time() - start_time
        final_budget = DEADLINE - elapsed - 2
        if final_budget > 2.0:
            # Do a final gradient + local polish
            fc, fs = _gradient_polish(best_centers, start_time, final_budget * 0.7)
            if time.time() - start_time < DEADLINE - 1:
                fc, fs = _local_polish(fc, start_time, final_budget * 0.3)
            if fs > best_score:
                best_score = fs
                best_centers = fc
    
    if best_centers is None:
        best_centers = initializations[0]
        
    # Final validation: ensure all centers are strictly inside
    norms = np.linalg.norm(best_centers, axis=1)
    mask = norms > 0.99999
    if np.any(mask):
        best_centers = best_centers.copy()
        best_centers[mask] *= 0.99999 / norms[mask]
    
    return best_centers
