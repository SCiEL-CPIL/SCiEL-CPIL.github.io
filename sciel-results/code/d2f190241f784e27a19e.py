"""
Packing of 50 equal circles in a unit square.
Structural change: direct soft-min objective optimization (the objective value IS the radius),
combined with simulated annealing for global search and L-BFGS polishing.
"""
import numpy as np
import time
from scipy.optimize import minimize

N = 50
HALF = 0.5

def _compute_radius(centers):
    """Compute the true packing radius for a set of centers."""
    if not np.all(np.isfinite(centers)):
        return 0.0
    # Distance to each wall
    dx = HALF - np.abs(centers[:, 0])
    dy = HALF - np.abs(centers[:, 1])
    wall_min = np.min(np.minimum(dx, dy))
    if wall_min < -1e-12:
        return 0.0
    wall_min = max(wall_min, 0.0)
    # Pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=2))
    np.fill_diagonal(dists, np.inf)
    pair_min = np.min(dists) / 2.0
    return min(pair_min, wall_min)

def _soft_min_value_and_grad(centers, beta):
    """
    Soft-min of the 2N + N(N-1)/2 clearance terms.
    Returns (value, gradient w.r.t. centers).
    
    Clearance terms:
      - For each circle i and each wall: d = HALF - |c_ik| (4 per circle)
      - For each pair (i,j): d = |c_i - c_j| / 2
    
    soft_min = -(1/beta) * log(sum(exp(-beta * d_k)))
    """
    n = len(centers)
    
    # Wall clearances: shape (n, 4)
    # For point (x,y): clearances are HALF-x, -HALF-x, HALF-y, -HALF-y
    cx = centers[:, 0]
    cy = centers[:, 1]
    wall_clear = np.stack([HALF - cx, -HALF - cx, HALF - cy, -HALF - cy], axis=1)  # (n, 4)
    
    # Pair clearances: for each pair (i,j), clearance = dist/2
    # We'll compute all pairwise distances
    diff = centers[:, np.newaxis, :] - centers[np.newaxis, :, :]  # (n, n, 2)
    dists = np.sqrt(np.sum(diff ** 2, axis=2))  # (n, n)
    np.fill_diagonal(dists, np.inf)
    pair_clear = dists / 2.0  # (n, n), symmetric
    
    # Collect all clearance values
    # Wall: n*4 values
    # Pair: n*(n-1)/2 values (upper triangle)
    wall_flat = wall_clear.ravel()  # (4n,)
    iu = np.triu_indices(n, k=1)
    pair_flat = pair_clear[iu]  # (n*(n-1)/2,)
    
    all_clear = np.concatenate([wall_flat, pair_flat])
    
    # Soft min: -1/beta * log(sum(exp(-beta * d)))
    # For numerical stability
    max_clear = np.max(all_clear)
    shifted = all_clear - max_clear
    exp_vals = np.exp(-beta * shifted)
    log_sum = np.log(np.sum(exp_vals))
    value = max_clear - (1.0 / beta) * log_sum
    
    # Gradient of soft_min w.r.t. each clearance d_k:
    # d(soft_min)/d(d_k) = exp(-beta*d_k) / sum(exp(-beta*d_j))
    # (normalized weights)
    weights = exp_vals / np.sum(exp_vals)
    
    # Now back-propagate to centers
    grad = np.zeros_like(centers)
    
    # Wall contributions
    # wall_flat[k] corresponds to circle i = k // 4, wall_type = k % 4
    wall_weights = weights[:4 * n].reshape(n, 4)
    # For circle i:
    #   wall_clear[i,0] = HALF - x  -> d/dx = -1
    #   wall_clear[i,1] = -HALF - x -> d/dx = -1
    #   wall_clear[i,2] = HALF - y  -> d/dy = -1
    #   wall_clear[i,3] = -HALF - y -> d/dy = -1
    grad[:, 0] -= wall_weights[:, 0] + wall_weights[:, 1]
    grad[:, 1] -= wall_weights[:, 2] + wall_weights[:, 3]
    
    # Pair contributions
    pair_weights = weights[4 * n:]  # (n*(n-1)/2,)
    # For pair (i,j) with i<j:
    #   d_clear/d(c_i) = -(c_i - c_j)/dist = (c_j - c_i)/dist
    #   d_clear/d(c_j) = -(c_j - c_i)/dist = (c_i - c_j)/dist
    #   grad contribution: weight * d_clear/d(center)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            w = pair_weights[idx]
            d = dists[i, j]
            if d < 1e-15:
                idx += 1
                continue
            direction = (centers[j] - centers[i]) / d  # unit vector from i to j
            # d(pair_clear)/d(c_i) = -direction  (since pair_clear = dist/2, d(dist)/d(c_i) = -(c_i-c_j)/dist = direction... wait)
            # pair_clear = |c_i - c_j| / 2
            # d(pair_clear)/d(c_i) = (c_i - c_j) / (2 * |c_i - c_j|) = -(c_j - c_i)/(2*dist) = -direction/2
            # d(pair_clear)/d(c_j) = (c_j - c_i) / (2 * |c_j - c_i|) = direction/2
            grad[i] += w * (-direction / 2.0)
            grad[j] += w * (direction / 2.0)
            idx += 1
    
    return value, grad

def _soft_min_obj(centers_flat, beta):
    """Objective for scipy: negative soft-min (we maximize soft-min, so minimize negative)."""
    centers = centers_flat.reshape(-1, 2)
    val, _ = _soft_min_value_and_grad(centers, beta)
    return -val

def _soft_min_grad(centers_flat, beta):
    """Gradient for scipy."""
    centers = centers_flat.reshape(-1, 2)
    _, grad = _soft_min_value_and_grad(centers, beta)
    return -grad.ravel()

def _hex_initial(n_cols=None):
    """Hexagonal lattice initial guess, clipped to square."""
    if n_cols is None:
        n_cols = 8
    rows = int(np.ceil(N / n_cols))
    dx = 1.0 / (n_cols + 0.5)
    dy = dx * np.sqrt(3) / 2
    
    centers = []
    for row in range(rows):
        for col in range(n_cols):
            if len(centers) >= N:
                break
            x = -0.5 + dx * (col + 0.5) + (dx / 2 if row % 2 else 0)
            y = -0.5 + dy * (row + 0.5)
            x = np.clip(x, -0.45, 0.45)
            y = np.clip(y, -0.45, 0.45)
            centers.append([x, y])
    centers = np.array(centers[:N])
    # Perturb slightly
    rng = np.random.default_rng(42)
    centers += rng.normal(0, 0.005, size=centers.shape)
    return centers

def _grid_initial():
    """Grid initial guess."""
    m = int(np.ceil(np.sqrt(N)))
    ticks = (np.arange(m) + 0.5) / m - 0.5
    grid = np.array([(x, y) for y in ticks for x in ticks])
    centers = grid[:N].copy()
    rng = np.random.default_rng(7)
    centers += rng.normal(0, 0.003, size=centers.shape)
    return centers

def _random_initial(seed):
    """Random initial guess."""
    rng = np.random.default_rng(seed)
    return rng.uniform(-0.42, 0.42, size=(N, 2))

def _simulated_annealing(centers, beta, max_iter=8000, T_start=0.05, T_end=0.001, 
                          deadline=None, rng_seed=0):
    """
    Simulated annealing to maximize the soft-min radius.
    Moves are random perturbations; accept if soft-min improves or with probability exp(delta/T).
    """
    rng = np.random.default_rng(rng_seed)
    current = centers.copy()
    current_val, _ = _soft_min_value_and_grad(current, beta)
    best = current.copy()
    best_val = current_val
    
    n_iter = max_iter
    for it in range(n_iter):
        if deadline is not None and time.time() > deadline:
            break
        
        # Adaptive temperature
        frac = it / max(1, n_iter - 1)
        T = T_start * (T_end / T_start) ** frac
        
        # Generate a random move: perturb 1-3 random circles
        n_move = rng.integers(1, 4)
        idx = rng.choice(N, size=n_move, replace=False)
        move = rng.normal(0, 0.02, size=(n_move, 2))
        
        new_centers = current.copy()
        new_centers[idx] += move
        
        new_val, _ = _soft_min_value_and_grad(new_centers, beta)
        delta = new_val - current_val
        
        if delta > 0 or (T > 0 and rng.random() < np.exp(delta / T)):
            current = new_centers
            current_val = new_val
            if current_val > best_val:
                best_val = current_val
                best = current.copy()
    
    return best, best_val

def _lbfgs_polish(centers, beta, deadline=None):
    """Polish using L-BFGS-B with the soft-min objective."""
    bounds = [(-0.5, 0.5)] * (2 * N)
    x0 = centers.flatten()
    
    try:
        res = minimize(
            _soft_min_obj,
            x0,
            args=(beta,),
            jac=_soft_min_grad,
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 300, 'ftol': 1e-14, 'gtol': 1e-10, 'maxfun': 2000}
        )
        if res.fun < -_soft_min_value_and_grad(centers, beta)[0]:
            # Check time
            if deadline is not None and time.time() > deadline:
                return centers
            return res.x.reshape(-1, 2)
    except Exception:
        pass
    return centers

def _high_beta_polish(centers, deadline=None):
    """Final polish with very high beta (nearly exact min) for precision."""
    for beta in [200.0, 500.0, 1000.0]:
        if deadline is not None and time.time() > deadline - 2:
            break
        centers = _lbfgs_polish(centers, beta, deadline)
    return centers

def construct():
    """
    Main entry point. Returns centers of shape (N, 2).
    """
    start_time = time.time()
    DEADLINE = start_time + 135.0  # 15s safety margin
    
    best_centers = None
    best_radius = 0.0
    
    # Phase 1: Simulated annealing with moderate beta, multiple restarts
    # Use beta that's high enough to approximate min but not so high that SA gets stuck
    beta_sa = 50.0
    
    inits = [
        ('hex', _hex_initial()),
        ('grid', _grid_initial()),
        ('rand0', _random_initial(0)),
        ('rand1', _random_initial(1)),
        ('rand2', _random_initial(2)),
    ]
    
    sa_results = []
    
    for name, init in inits:
        if time.time() > DEADLINE - 30:
            break
        
        # SA phase
        sa_deadline = min(DEADLINE - 20, time.time() + 20)
        sa_centers, sa_val = _simulated_annealing(
            init, beta_sa, max_iter=6000, 
            T_start=0.04, T_end=0.0005,
            deadline=sa_deadline,
            rng_seed=hash(name) % 10000
        )
        
        # L-BFGS polish with increasing beta
        if time.time() < DEADLINE - 10:
            polished = _lbfgs_polish(sa_centers, beta=100.0, deadline=DEADLINE - 8)
        else:
            polished = sa_centers
        
        r = _compute_radius(polished)
        if r > best_radius:
            best_radius = r
            best_centers = polished.copy()
        
        sa_results.append(polished)
    
    # Phase 2: Take the best result and do high-precision polishing
    if best_centers is not None:
        if time.time() < DEADLINE - 5:
            best_centers = _high_beta_polish(best_centers, deadline=DEADLINE - 2)
            best_radius = _compute_radius(best_centers)
    
    # Phase 3: Try a few more targeted restarts from the best solution with small perturbations
    if best_centers is not None and time.time() < DEADLINE - 15:
        rng = np.random.default_rng(99)
        for attempt in range(3):
            if time.time() > DEADLINE - 5:
                break
            perturbed = best_centers + rng.normal(0, 0.01, size=(N, 2))
            # Quick SA
            sa_deadline = min(DEADLINE - 5, time.time() + 10)
            sa_c, _ = _simulated_annealing(
                perturbed, beta=80.0, max_iter=3000,
                T_start=0.02, T_end=0.001,
                deadline=sa_deadline,
                rng_seed=100 + attempt
            )
            if time.time() < DEADLINE - 3:
                sa_c = _lbfgs_polish(sa_c, beta=200.0, deadline=DEADLINE - 2)
            r = _compute_radius(sa_c)
            if r > best_radius:
                best_radius = r
                best_centers = sa_c.copy()
    
    # Final validation
    if best_centers is None or not np.all(np.isfinite(best_centers)):
        # Fallback: simple grid
        best_centers = _grid_initial()
    
    # Ensure centers are within bounds
    best_centers = np.clip(best_centers, -0.5, 0.5)
    
    return best_centers
