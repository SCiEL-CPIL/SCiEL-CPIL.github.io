"""Rewritten program for packing 50 equal circles in a unit circle."""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
DEADLINE = 145.0  # Safety margin for 150s limit


def _compute_score(centers):
    """Compute the packing score for given centers."""
    n = len(centers)
    if n < 2:
        return 0.0
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    min_bound = np.min(1.0 - norms)
    return min(min_pair / 2.0, min_bound)


def _soft_min(values, beta):
    """Compute the soft minimum of an array of values."""
    v = np.asarray(values, dtype=np.float64)
    m = np.min(v)
    exp_terms = np.exp(-beta * (v - m))
    return m - (1.0 / beta) * np.log(np.sum(exp_terms))


def _objective_and_gradient(params, n, beta):
    """
    Objective to minimize: -soft_min(constraints)
    params: (n, 2) array of centers
    """
    centers = params.reshape(n, 2)
    
    # Calculate pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    i_idx, j_idx = np.triu_indices(n, k=1)
    pair_dists = dists[i_idx, j_idx]
    
    # Constraints values
    norms = np.sqrt(np.sum(centers * centers, axis=1))
    vals = np.concatenate([pair_dists / 2.0, 1.0 - norms])
    
    # Soft min
    m = np.min(vals)
    exp_vals = np.exp(-beta * (vals - m))
    S = np.sum(exp_vals)
    soft_min_val = m - (1.0 / beta) * np.log(S)
    obj = -soft_min_val
    
    # Gradient
    weights = exp_vals / S
    grad = np.zeros_like(centers)
    
    # Pair contributions
    pair_weights = weights[:n*(n-1)//2]
    for k in range(len(pair_weights)):
        w = pair_weights[k]
        i, j = i_idx[k], j_idx[k]
        d = dists[i, j]
        if d > 1e-10:
            vec = (centers[i] - centers[j]) / d
            grad[i] += w * vec / 2.0
            grad[j] -= w * vec / 2.0
    
    # Bound contributions
    bound_weights = weights[n*(n-1)//2:]
    for i in range(n):
        w = bound_weights[i]
        if norms[i] > 1e-10:
            grad[i] -= w * centers[i] / norms[i]
    
    return obj, -grad


def _get_hex_lattice(n):
    """Generate a hexagonal lattice and pick n points inside unit circle."""
    r_est = 0.125
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
        rng = np.random.default_rng(42)
        points = []
        for _ in range(n * 10):
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points.append([x, y])
    
    points = np.array(points)
    
    if len(points) < n:
        rng = np.random.default_rng(42)
        while len(points) < n:
            angle = rng.uniform(0, 2 * np.pi)
            radius = rng.uniform(0, 0.9)
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            points = np.vstack([points, [[x, y]]])
    
    dists = np.linalg.norm(points, axis=1)
    indices = np.argsort(dists)
    return points[indices[:n]]


def _get_ring_initialization(n, seed=0):
    """Generate initial points on concentric rings."""
    rng = np.random.default_rng(seed)
    points = []
    
    # Center point
    points.append([0.0, 0.0])
    
    # Ring 1: 6 points
    for i in range(6):
        angle = 2 * np.pi * i / 6
        r = 0.25
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 2: 12 points
    for i in range(12):
        angle = 2 * np.pi * i / 12 + np.pi / 12
        r = 0.5
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 3: 18 points
    for i in range(18):
        angle = 2 * np.pi * i / 18
        r = 0.75
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    # Ring 4: remaining points
    remaining = n - len(points)
    for i in range(remaining):
        angle = 2 * np.pi * i / max(remaining, 1) + np.pi / max(remaining, 1)
        r = 0.9
        points.append([r * np.cos(angle), r * np.sin(angle)])
    
    return np.array(points[:n])


def _get_random_initialization(n, seed=0):
    """Generate random initial points."""
    rng = np.random.default_rng(seed)
    points = []
    for _ in range(n):
        angle = rng.uniform(0, 2 * np.pi)
        radius = rng.uniform(0, 0.95)
        x = radius * np.cos(angle)
        y = radius * np.sin(angle)
        points.append([x, y])
    return np.array(points)


def _get_golden_angle_initialization(n, seed=0):
    """Generate points using golden angle spiral."""
    rng = np.random.default_rng(seed)
    points = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(n):
        r = np.sqrt((i + 0.5) / n) * 0.95
        theta = i * golden_angle + rng.uniform(-0.05, 0.05)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y])
    return np.array(points)


def _optimize_from_start(centers, start_time, deadline, beta_schedule=None):
    """Optimize from a given starting point with a schedule of betas."""
    if beta_schedule is None:
        beta_schedule = [20.0, 50.0, 100.0, 200.0]
    
    best_score = _compute_score(centers)
    best_centers = centers.copy()
    
    for beta in beta_schedule:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective_and_gradient,
                best_centers.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 200, 'gtol': 1e-6}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if np.any(~np.isfinite(new_centers)):
                continue
            
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            continue
    
    return best_centers, best_score


def _local_search(centers, start_time, deadline, n_iterations=5000):
    """Perform local search by randomly perturbing pairs of points."""
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(12345)
    
    for _ in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        # Pick two random points to perturb
        i, j = rng.choice(N, size=2, replace=False)
        
        # Random perturbation
        delta = rng.normal(0, 0.01, size=2)
        
        trial = best_centers.copy()
        trial[i] += delta
        trial[j] -= delta
        
        # Ensure points stay inside the circle
        norms = np.linalg.norm(trial, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            trial[mask] *= 0.99 / norms[mask]
        
        trial_score = _compute_score(trial)
        
        if trial_score > best_score:
            best_score = trial_score
            best_centers = trial.copy()
    
    return best_centers, best_score


def _final_high_precision_polish(centers, start_time, deadline):
    """Final polish with very high beta and many iterations."""
    if time.time() > deadline - 10:
        return centers, _compute_score(centers)
    
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    
    # Try several high beta values
    for beta in [200.0, 500.0, 1000.0]:
        if time.time() > deadline - 5:
            break
        
        try:
            res = minimize(
                _objective_and_gradient,
                best_centers.flatten(),
                args=(N, beta),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 500, 'gtol': 1e-8}
            )
            
            new_centers = res.x.reshape(N, 2)
            
            if np.any(~np.isfinite(new_centers)):
                continue
            
            new_score = _compute_score(new_centers)
            
            if new_score > best_score:
                best_score = new_score
                best_centers = new_centers.copy()
        except Exception:
            continue
    
    return best_centers, best_score


def _nelder_mead_polish(centers, start_time, deadline):
    """Nelder-Mead polish for final refinement."""
    if time.time() > deadline - 10:
        return centers, _compute_score(centers)
    
    def neg_score(params):
        c = params.reshape(N, 2)
        if np.any(~np.isfinite(c)):
            return 1e10
        norms = np.linalg.norm(c, axis=1)
        if np.any(norms > 1.0):
            return 1e10
        return -_compute_score(c)
    
    try:
        res = minimize(
            neg_score,
            centers.flatten(),
            method='Nelder-Mead',
            options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-10}
        )
        
        new_centers = res.x.reshape(N, 2)
        
        if np.any(~np.isfinite(new_centers)):
            return centers, _compute_score(centers)
        
        new_score = _compute_score(new_centers)
        old_score = _compute_score(centers)
        
        if new_score > old_score:
            return new_centers, new_score
        else:
            return centers, old_score
    except Exception:
        return centers, _compute_score(centers)


def _swap_optimization(centers, start_time, deadline, n_iterations=1000):
    """Try swapping pairs of points to improve packing."""
    best_centers = centers.copy()
    best_score = _compute_score(best_centers)
    rng = np.random.default_rng(99999)
    
    for _ in range(n_iterations):
        if time.time() > deadline - 5:
            break
        
        i, j = rng.choice(N, size=2, replace=False)
        
        trial = best_centers.copy()
        trial[i], trial[j] = trial[j].copy(), trial[i].copy()
        
        # Small perturbation after swap
        delta = rng.normal(0, 0.005, size=2)
        trial[i] += delta
        trial[j] -= delta
        
        norms = np.linalg.norm(trial, axis=1)
        mask = norms > 0.99
        if np.any(mask):
            trial[mask] *= 0.99 / norms[mask]
        
        trial_score = _compute_score(trial)
        
        if trial_score > best_score:
            best_score = trial_score
            best_centers = trial.copy()
    
    return best_centers, best_score


def construct():
    start_time = time.time()
    
    # Generate multiple initializations
    initializations = [
        _get_hex_lattice(N),
        _get_ring_initialization(N, seed=0),
        _get_random_initialization(N, seed=42),
        _get_golden_angle_initialization(N, seed=0),
        _get_random_initialization(N, seed=123),
        _get_ring_initialization(N, seed=1),
    ]
    
    best_centers = None
    best_score = 0.0
    
    # Optimize from each initialization
    for init in initializations:
        if time.time() > DEADLINE - 30:
            break
        
        centers, score = _optimize_from_start(init, start_time, DEADLINE)
        
        if score > best_score:
            best_score = score
            best_centers = centers.copy()
    
    if best_centers is None:
        return initializations[0]
    
    # Local search
    if time.time() < DEADLINE - 20:
        best_centers, best_score = _local_search(best_centers, start_time, DEADLINE)
    
    # High precision polish
    if time.time() < DEADLINE - 15:
        best_centers, best_score = _final_high_precision_polish(best_centers, start_time, DEADLINE)
    
    # Nelder-Mead polish
    if time.time() < DEADLINE - 10:
        best_centers, best_score = _nelder_mead_polish(best_centers, start_time, DEADLINE)
    
    # Swap optimization
    if time.time() < DEADLINE - 5:
        best_centers, best_score = _swap_optimization(best_centers, start_time, DEADLINE)
    
    # Final check
    final_score = _compute_score(best_centers)
    if final_score > best_score:
        best_score = final_score
    
    return best_centers
