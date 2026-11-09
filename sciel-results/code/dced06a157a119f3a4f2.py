# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

def _score(points):
    """Compute (d_min / d_max)^2 for a set of points."""
    n = len(points)
    if n < 2:
        return 0.0
    dists = pdist(points)
    if len(dists) == 0:
        return 0.0
    d_min = np.min(dists)
    d_max = np.max(dists)
    if d_max == 0 or not np.isfinite(d_min) or not np.isfinite(d_max):
        return 0.0
    return (d_min / d_max) ** 2

def _smooth_objective(x, n=14, d=3, beta=20.0, alpha=20.0):
    """
    Smooth surrogate for the objective.
    Minimizes - (log(softmin(d)) - log(softmax(d)))
    which is equivalent to maximizing log(softmin(d)/softmax(d))
    """
    points = x.reshape(n, d)
    dists = pdist(points)
    
    # Handle zero distances
    if np.any(dists < 1e-12):
        return 1e6  # Large penalty
    
    # Stable log-sum-exp
    neg_beta_d = -beta * dists
    max_neg_beta_d = np.max(neg_beta_d)
    log_sum_exp_neg = max_neg_beta_d + np.log(np.sum(np.exp(neg_beta_d - max_neg_beta_d)))
    log_softmin = -1.0 / beta * log_sum_exp_neg
    
    alpha_d = alpha * dists
    max_alpha_d = np.max(alpha_d)
    log_sum_exp_pos = max_alpha_d + np.log(np.exp(alpha_d - max_alpha_d).sum())
    log_softmax = 1.0 / alpha * log_sum_exp_pos
    
    objective_value = -(log_softmin - log_softmax)
    
    if not np.isfinite(objective_value):
        return 1e6
        
    return objective_value

def _make_icosahedron_plus_2():
    """Icosahedron (12 verts) + 2 points at poles of a face."""
    phi = (1 + np.sqrt(5)) / 2
    verts = []
    for s1 in [1, -1]:
        for s2 in [1, -1]:
            verts.append([0, s1, s2 * phi])
            verts.append([s1, s2 * phi, 0])
            verts.append([s2 * phi, 0, s1])
    points = np.array(verts, dtype=float)
    # Normalize to unit sphere
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    # Add 2 points: place them at the centers of two opposite faces
    p1 = np.array([0.0, 0.0, 1.0])
    p2 = np.array([0.0, 0.0, -1.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_cuboctahedron_plus_2():
    """Cuboctahedron (12 verts) + 2 points."""
    verts = []
    for x in [1, -1]:
        for y in [1, -1]:
            verts.append([x, y, 0])
            verts.append([x, 0, y])
            verts.append([0, x, y])
    points = np.array(verts, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    points = points / norms
    
    p1 = np.array([1.0, 0.0, 0.0])
    p2 = np.array([-1.0, 0.0, 0.0])
    points = np.vstack([points, p1, p2])
    return points

def _make_fibonacci_sphere(n=14, d=3, seed=42):
    """Generate points using Fibonacci sphere for good initial distribution."""
    rng = np.random.RandomState(seed)
    golden_ratio = (1 + np.sqrt(5)) / 2
    points = []
    for i in range(n):
        z = 1 - 2 * (i + 0.5) / n
        r = np.sqrt(1 - z**2)
        theta = 2 * np.pi * i / golden_ratio + rng.randn() * 0.01
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        points.append([x, y, z])
    points = np.array(points, dtype=float)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _make_random_unit(n=14, d=3, seed=None):
    if seed is not None:
        rng = np.random.RandomState(seed)
    else:
        rng = np.random.RandomState()
    points = rng.randn(n, d)
    norms = np.linalg.norm(points, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    points = points / norms
    return points

def _final_polish(points, deadline, max_iter=10000):
    """
    High-precision final polish using Nelder-Mead on the true objective.
    """
    n, d = points.shape
    best_points = points.copy()
    best_score = _score(best_points)
    
    # Use multiple restarts with small perturbations
    rng = np.random.RandomState(12345)
    
    for restart in range(5):
        if time.time() > deadline - 5:
            break
        
        # Start from best or perturbed best
        if restart == 0:
            x_start = best_points.flatten()
        else:
            x_start = best_points.flatten().copy()
            # Add small random perturbation
            perturbation = rng.randn(len(x_start)) * 0.001
            x_start += perturbation
        
        try:
            res = minimize(
                lambda x: -_score(x.reshape(n, d)),
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 5000,
                    'xatol': 1e-12,
                    'fatol': 1e-14,
                    'adaptive': True,
                    'initial_simplex': None
                }
            )
            
            if not np.any(np.isnan(res.x)):
                points_try = res.x.reshape(n, d)
                score_try = _score(points_try)
                if score_try > best_score:
                    best_score = score_try
                    best_points = points_try
        except Exception:
            pass
    
    return best_points

def _adaptive_smooth_optimize(x0, n, d, deadline, max_time_per_beta=15):
    """
    Run smooth optimization with adaptive beta values.
    """
    best_points = x0.reshape(n, d).copy()
    best_score = _score(best_points)
    
    # Try different beta values
    beta_values = [5.0, 10.0, 20.0, 40.0]
    
    for beta in beta_values:
        if time.time() > deadline - 10:
            break
        
        start_time = time.time()
        try:
            res = minimize(
                lambda x: _smooth_objective(x, n, d, beta=beta, alpha=beta),
                best_points.flatten(),
                method='L-BFGS-B',
                options={
                    'maxiter': 2000,
                    'ftol': 1e-12,
                    'gtol': 1e-10,
                    'maxls': 100
                }
            )
            
            if not np.any(np.isnan(res.x)):
                points_smooth = res.x.reshape(n, d)
                score_smooth = _score(points_smooth)
                if score_smooth > best_score:
                    best_score = score_smooth
                    best_points = points_smooth
        except Exception:
            pass
        
        elapsed = time.time() - start_time
        if elapsed > max_time_per_beta:
            break
    
    return best_points

def min_max_dist_dim3_14() -> np.ndarray:
    """
    Creates 14 points in 3 dimensions in order to maximize the ratio of minimum to maximum distance.
    """
    n = 14
    d = 3
    start_time = time.time()
    deadline = start_time + 580  # 630s limit, keep 50s margin
    
    best_score = -1.0
    best_points = None
    
    # Generate diverse initial guesses
    guesses = []
    guesses.append(_make_icosahedron_plus_2())
    guesses.append(_make_cuboctahedron_plus_2())
    guesses.append(_make_fibonacci_sphere(n, d, seed=42))
    guesses.append(_make_fibonacci_sphere(n, d, seed=123))
    
    # Add random unit sphere points with various seeds
    for i in range(8):
        guesses.append(_make_random_unit(n, d, seed=1000 + i * 137))
    
    # Optimize each guess
    for i, guess in enumerate(guesses):
        if time.time() > deadline - 30:
            break
            
        # Check initial score
        current_score = _score(guess)
        if current_score > best_score:
            best_score = current_score
            best_points = guess.copy()
        
        # Phase 1: Adaptive smooth optimization
        if time.time() > deadline - 20:
            continue
            
        try:
            points_smooth = _adaptive_smooth_optimize(guess, n, d, deadline - 20)
            score_smooth = _score(points_smooth)
            if score_smooth > best_score:
                best_score = score_smooth
                best_points = points_smooth
        except Exception:
            pass
        
        # Phase 2: Quick Nelder-Mead polish
        if time.time() > deadline - 15:
            continue
            
        try:
            x_start = best_points.flatten() if best_points is not None else guess.flatten()
            res_polish = minimize(
                lambda x: -_score(x.reshape(n, d)),
                x_start,
                method='Nelder-Mead',
                options={
                    'maxiter': 3000,
                    'xatol': 1e-10,
                    'fatol': 1e-12,
                    'adaptive': True
                }
            )
            
            if not np.any(np.isnan(res_polish.x)):
                points_polish = res_polish.x.reshape(n, d)
                score_polish = _score(points_polish)
                if score_polish > best_score:
                    best_score = score_polish
                    best_points = points_polish
        except Exception:
            pass
    
    # Final high-precision polish from the best point
    if best_points is not None and time.time() < deadline - 10:
        try:
            best_points = _final_polish(best_points, deadline)
            final_score = _score(best_points)
            if final_score > best_score:
                best_score = final_score
        except Exception:
            pass
    
    # Fallback
    if best_points is None:
        best_points = _make_icosahedron_plus_2()
    
    # Ensure finite and correct shape
    if not np.all(np.isfinite(best_points)):
        best_points = _make_icosahedron_plus_2()
    
    return best_points

# EVOLVE-BLOCK-END
