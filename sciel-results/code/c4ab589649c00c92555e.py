import numpy as np
from scipy.optimize import minimize
import time

N = 50

def _calc_score(x):
    """Calculate the actual packing score (min radius). Vectorized."""
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    # Boundary distances
    radii = np.sqrt(np.sum(x * x, axis=1))
    b_dists = 1.0 - radii
    # Min pairwise distance (excluding diagonal)
    np.fill_diagonal(dists, np.inf)
    min_pair = np.min(dists)
    min_bound = np.min(b_dists)
    return min(min_pair, min_bound) / 2.0

def _hex_lattice_init():
    """Generate initial hexagonal lattice points."""
    spacing = 2.0 / (np.sqrt(N) + 1.0)
    while True:
        k = int(1.0 / spacing) + 2
        rows = [((i + 0.5 * (j % 2)) * spacing, j * spacing * np.sqrt(3.0) / 2.0)
                for i in range(-k, k + 1) for j in range(-k, k + 1)]
        points = np.array(rows)
        radius = np.hypot(points[:, 0], points[:, 1])
        points, radius = points[radius <= 1.0 - spacing / 2.0], radius[radius <= 1.0 - spacing / 2.0]
        if len(points) >= N:
            return points[np.argsort(radius, kind="stable")[:N]]
        spacing *= 0.98

def _random_init(rng, n=N):
    """Random initialization inside unit circle."""
    angles = rng.uniform(0, 2 * np.pi, size=n)
    rads = np.sqrt(rng.uniform(0, 1, size=n))
    return np.column_stack([rads * np.cos(angles), rads * np.sin(angles)])

def _ring_init(n=N, n_rings=5):
    """Ring-based initialization: distribute points on concentric rings."""
    points = []
    # Center point
    points.append([0.0, 0.0])
    remaining = n - 1
    for ring in range(n_rings):
        r = (ring + 1) / n_rings * 0.9
        count = max(1, int(2 * np.pi * r / (2 * 0.125)))
        if ring == n_rings - 1:
            count = remaining
        angles = np.linspace(0, 2 * np.pi, count, endpoint=False) + ring * 0.3
        ring_pts = np.column_stack([r * np.cos(angles), r * np.sin(angles)])
        points.append(ring_pts)
        remaining -= count
        if remaining <= 0:
            break
    result = np.vstack(points)[:n]
    return result

def _soft_min_energy_and_grad(x, alpha):
    """
    Compute the soft-min energy and gradient.
    x: (N, 2) array of centers.
    alpha: temperature parameter for soft-min.
    Returns: (energy, gradient)
    """
    x = x.reshape(N, 2)
    # Pairwise distances
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    
    # Avoid division by zero in gradient calculation for i=j
    np.fill_diagonal(dists, np.inf)
    
    # Boundary distances: 1 - |x_i|
    radii = np.sqrt(np.sum(x * x, axis=1))
    b_dists = 1.0 - radii
    
    p_dists = dists[np.triu_indices(N, k=1)]
    all_dists = np.concatenate([p_dists, b_dists])
    
    exponents = -alpha * all_dists
    max_exp = np.max(exponents)
    if max_exp > 0:
        exponents -= max_exp
        shift = max_exp
    else:
        shift = 0.0
        
    weights = np.exp(exponents)
    sum_weights = np.sum(weights)
    
    if sum_weights == 0:
        return 0.0, np.zeros_like(x)
        
    energy = (1.0 / alpha) * (np.log(sum_weights) + shift)
    
    grad = np.zeros_like(x)
    
    i_idx, j_idx = np.triu_indices(N, k=1)
    w_pairs = weights[:len(p_dists)]
    d_pairs = p_dists
    
    diff_vec = x[i_idx] - x[j_idx]
    mask = d_pairs > 1e-10
    grad_unit = np.zeros_like(diff_vec)
    grad_unit[mask] = diff_vec[mask] / d_pairs[mask, np.newaxis]
    
    weighted_grad = w_pairs[:, np.newaxis] * grad_unit
    
    np.add.at(grad, i_idx, -weighted_grad / sum_weights)
    np.add.at(grad, j_idx, weighted_grad / sum_weights)
    
    w_bounds = weights[len(p_dists):]
    radii_safe = np.where(radii < 1e-10, 1e-10, radii)
    grad_bounds = -x / radii_safe[:, np.newaxis]
    
    weighted_bounds = w_bounds[:, np.newaxis] * grad_bounds
    grad += -weighted_bounds / sum_weights
    
    return energy, grad

def _optimize_from(x0, deadline, alphas=None, max_iter=2000):
    """Run multi-scale soft-min optimization from x0. Returns (best_x, best_score)."""
    if alphas is None:
        alphas = [5.0, 10.0, 20.0, 50.0, 100.0]
    
    current_x = x0.copy()
    best_x = x0.copy()
    best_score = _calc_score(best_x)
    
    for alpha in alphas:
        if time.time() > deadline:
            break
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                current_x,
                args=(alpha,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': max_iter, 'ftol': 1e-12, 'gtol': 1e-10}
            )
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
                current_x = x_new.copy()
        except Exception:
            pass
    
    return best_x, best_score

def construct():
    start_time = time.time()
    deadline = 140.0  # Safety margin (150s limit)
    rng = np.random.default_rng(42)
    
    # Generate diverse initializations
    inits = []
    inits.append(_hex_lattice_init())
    inits.append(_ring_init())
    inits.append(_random_init(rng))
    inits.append(_random_init(rng))
    
    # Optimize each initialization
    best_x = None
    best_score = -np.inf
    
    for i, x0 in enumerate(inits):
        if time.time() > deadline - 30:
            break
        x_opt, score_opt = _optimize_from(x0, deadline - 30, alphas=[5.0, 10.0, 20.0, 50.0])
        if score_opt > best_score:
            best_score = score_opt
            best_x = x_opt.copy()
    
    if best_x is None:
        best_x = inits[0]
        best_score = _calc_score(best_x)
    
    # Jitter-and-reoptimize to escape local minima
    n_jitter_rounds = 8
    for r in range(n_jitter_rounds):
        if time.time() > deadline - 15:
            break
        
        # Jitter scale decreases over time
        jitter_scale = 0.03 * (1.0 - r / n_jitter_rounds) + 0.005
        x_perturbed = best_x + rng.normal(0, jitter_scale, size=best_x.shape)
        
        x_opt, score_opt = _optimize_from(
            x_perturbed, deadline - 15,
            alphas=[20.0, 50.0, 100.0],
            max_iter=1500
        )
        if score_opt > best_score:
            best_score = score_opt
            best_x = x_opt.copy()
    
    # Final polish with very high alpha
    if time.time() < deadline - 5:
        try:
            res = minimize(
                _soft_min_energy_and_grad,
                best_x,
                args=(200.0,),
                method='L-BFGS-B',
                jac=True,
                options={'maxiter': 5000, 'ftol': 1e-15, 'gtol': 1e-12}
            )
            x_new = res.x.reshape(N, 2)
            score_new = _calc_score(x_new)
            if score_new > best_score:
                best_score = score_new
                best_x = x_new.copy()
        except Exception:
            pass
    
    return best_x
