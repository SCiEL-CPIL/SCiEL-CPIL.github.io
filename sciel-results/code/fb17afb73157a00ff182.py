"""Optimize 14 points in 3D to minimize (max_dist/min_dist)^2."""
import time
import itertools
import numpy as np
from scipy.optimize import minimize

N = 14
D = 3
TIME_LIMIT = 110.0  # safety margin before 120s kill

def _pairwise_indices(n):
    """Return list of (i, j) pairs for i < j."""
    return list(itertools.combinations(range(n), 2))

PAIRS = _pairwise_indices(N)
N_PAIRS = len(PAIRS)

def _flatten(x):
    return x.reshape(-1)

def _unflatten(flat):
    return flat.reshape(N, D)

def _pairwise_sq_dists(flat):
    """Compute squared pairwise distances from flattened (N*D,) array."""
    pts = _unflatten(flat)
    # Use broadcasting for efficiency
    diff = pts[:, np.newaxis, :] - pts[np.newaxis, :, :]
    sq_dists = np.sum(diff ** 2, axis=2)
    # Extract upper triangle
    idx = np.triu_indices(N, 1)
    return sq_dists[idx]

def _objective_ratio(flat):
    """Objective: (max_sq / min_sq)^2, i.e., ratio^2 of max/min distance."""
    sq = _pairwise_sq_dists(flat)
    dmin = np.min(sq)
    dmax = np.max(sq)
    if dmin < 1e-15:
        return 1e10
    ratio = dmax / dmin
    return ratio ** 2

def _objective_log(flat):
    """Objective: log(max_sq) - log(min_sq), which is proportional to log(ratio).
    Minimizing this is equivalent to minimizing ratio."""
    sq = _pairwise_sq_dists(flat)
    dmin = np.min(sq)
    dmax = np.max(sq)
    if dmin < 1e-15:
        return 1e10
    return np.log(dmax) - np.log(dmin)

def _objective_penalty(flat, penalty=100.0):
    """Penalty-based objective: maximize min_sq - penalty * max_sq.
    This encourages all distances to be close."""
    sq = _pairwise_sq_dists(flat)
    dmin = np.min(sq)
    dmax = np.max(sq)
    return -(dmin) + penalty * dmax

def _make_initial_guesses():
    """Generate diverse initial guesses."""
    guesses = []
    
    # 1. Grid-based
    side = int(np.ceil(N ** (1.0 / D)))
    grid = list(itertools.product(range(side), repeat=D))[:N]
    guesses.append(np.array(grid, dtype=float))
    
    # 2. Fibonacci sphere (projected to 3D, scaled)
    n = N
    golden = (1 + np.sqrt(5)) / 2
    ys = 1 - 2 * (np.arange(n) + 0.5) / n
    rs = np.sqrt(1 - ys ** 2)
    th = 2 * np.pi * np.arange(n) / golden
    pts = np.column_stack([rs * np.cos(th), rs * np.sin(th), ys])
    guesses.append(pts)
    
    # 3. Icosahedron vertices + extra points
    t = (1 + np.sqrt(5)) / 2
    ico = np.array([
        [0, 1, t], [0, -1, t], [0, 1, -t], [0, -1, -t],
        [1, t, 0], [-1, t, 0], [1, -t, 0], [-1, -t, 0],
        [t, 0, 1], [-t, 0, 1], [t, 0, -1], [-t, 0, -1]
    ], dtype=float)
    # Add 2 more points: center and one offset
    extra = np.array([[0, 0, 0], [0.5, 0.5, 0.5]])
    guesses.append(np.vstack([ico, extra]))
    
    # 4. Random points
    np.random.seed(42)
    for seed in range(5):
        np.random.seed(seed * 12345 + 67890)
        guesses.append(np.random.randn(N, D))
    
    # 5. Ring configurations
    # Two rings of 7
    angles1 = 2 * np.pi * np.arange(7) / 7
    ring1 = np.column_stack([np.cos(angles1), np.sin(angles1), np.ones(7) * 1.0])
    angles2 = 2 * np.pi * (np.arange(7) + 0.5) / 7
    ring2 = np.column_stack([np.cos(angles2), np.sin(angles2), np.ones(7) * -1.0])
    guesses.append(np.vstack([ring1, ring2]))
    
    # 6. Cubic lattice with perturbation
    np.random.seed(999)
    side = 2
    grid2 = list(itertools.product(range(side), repeat=D))[:N]
    perturbed = np.array(grid2, dtype=float) + np.random.randn(N, D) * 0.1
    guesses.append(perturbed)
    
    return guesses

def _optimize_from_guess(x0, obj_func, time_budget):
    """Run L-BFGS-B optimization from a given starting point."""
    x0_flat = _flatten(x0)
    try:
        result = minimize(
            obj_func,
            x0_flat,
            method='L-BFGS-B',
            options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-10}
        )
        return result.x, obj_func(result.x)
    except Exception:
        return x0_flat, obj_func(x0_flat)

def construct():
    start_time = time.time()
    
    guesses = _make_initial_guesses()
    
    best_flat = None
    best_score = np.inf
    
    # Stage 1: Coarse optimization with penalty objective
    for i, guess in enumerate(guesses):
        if time.time() - start_time > TIME_LIMIT - 30:
            break
        x_opt, score = _optimize_from_guess(guess, _objective_penalty, 10.0)
        ratio_score = _objective_ratio(x_opt)
        if ratio_score < best_score:
            best_score = ratio_score
            best_flat = x_opt
    
    # Stage 2: Fine-tune with log objective
    if best_flat is not None and time.time() - start_time < TIME_LIMIT - 20:
        x_opt, score = _optimize_from_guess(_unflatten(best_flat), _objective_log, 20.0)
        ratio_score = _objective_ratio(x_opt)
        if ratio_score < best_score:
            best_score = ratio_score
            best_flat = x_opt
    
    # Stage 3: Final polish with ratio objective directly
    if best_flat is not None and time.time() - start_time < TIME_LIMIT - 10:
        x_opt, score = _optimize_from_guess(_unflatten(best_flat), _objective_ratio, 10.0)
        ratio_score = _objective_ratio(x_opt)
        if ratio_score < best_score:
            best_score = ratio_score
            best_flat = x_opt
    
    # Ensure we have a valid result
    if best_flat is None:
        # Fallback: use the first guess
        best_flat = _flatten(guesses[0])
    
    return _unflatten(best_flat)
