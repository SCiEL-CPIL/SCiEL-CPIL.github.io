"""Packing 50 equal circles in a unit circle - soft-max L-BFGS-B with bisection on radius."""
import time
import numpy as np
from scipy.optimize import minimize

N = 50
_START = time.time()
_DEADLINE = _START + 130.0  # safety margin before 150s kill


def _score(pts):
    """Compute the valid packing radius (or 0 if invalid)."""
    if len(pts) != N:
        return 0.0
    if pts.shape != (N, 2):
        return 0.0
    if not np.all(np.isfinite(pts)):
        return 0.0
    dists = np.hypot(pts[:, 0], pts[:, 1])
    if np.any(dists > 1.0):
        return 0.0
    r_cont = 1.0 - np.max(dists)
    if r_cont < 0:
        return 0.0
    if N < 2:
        return r_cont
    dx = pts[:, 0:1] - pts[0:1, 0]
    dy = pts[:, 1:2] - pts[0:1, 1]
    dmat = np.sqrt(dx ** 2 + dy ** 2)
    np.fill_diagonal(dmat, np.inf)
    min_pair = np.min(dmat)
    r_pair = min_pair / 2.0
    return min(r_cont, r_pair)


def _hex_lattice_init(n, r_guess, jitter=0.0, seed=None):
    """Generate initial positions from hexagonal lattice, trimmed to n points."""
    rng = np.random.default_rng(seed)
    spacing = 2.0 * r_guess
    k = int(1.0 / spacing) + 3
    rows = []
    for j in range(-k, k + 1):
        for i in range(-k, k + 1):
            x = (i + 0.5 * (j % 2)) * spacing
            y = j * spacing * np.sqrt(3.0) / 2.0
            if x * x + y * y <= (1.0 - r_guess) ** 2:
                rows.append((x, y))
    pts = np.array(rows)
    if len(pts) >= n:
        dists = np.hypot(pts[:, 0], pts[:, 1])
        idx = np.argsort(dists, kind='stable')[:n]
        pts = pts[idx]
        if jitter > 0:
            pts = pts + rng.normal(0, jitter, size=(n, 2))
        r = np.hypot(pts[:, 0], pts[:, 1])
        mask = r > 1.0
        if np.any(mask):
            pts[mask] /= r[mask, None] * 1.001
        return pts
    # Fallback: random
    theta = rng.uniform(0, 2 * np.pi, n)
    rho = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])


def _ring_init(n, ring_sizes, ring_radii, angle_offsets=None, seed=None):
    """Generate points from concentric rings."""
    rng = np.random.default_rng(seed)
    pts = []
    if angle_offsets is None:
        angle_offsets = [0.0] * len(ring_sizes)
    for ring_idx, (size, radius) in enumerate(zip(ring_sizes, ring_radii)):
        offset = angle_offsets[ring_idx] if ring_idx < len(angle_offsets) else 0.0
        for j in range(size):
            angle = 2 * np.pi * j / size + offset
            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            if x * x + y * y <= 1.0:
                pts.append((x, y))
    pts = np.array(pts[:n])
    if len(pts) < n:
        extra = n - len(pts)
        theta = rng.uniform(0, 2 * np.pi, extra)
        rho = np.sqrt(rng.uniform(0, 1, extra))
        extra_pts = np.column_stack([rho * np.cos(theta), rho * np.sin(theta)])
        pts = np.vstack([pts, extra_pts])
    return pts


def _random_init(n, seed=None):
    """Random initialization inside unit circle."""
    rng = np.random.default_rng(seed)
    angles = rng.uniform(0, 2 * np.pi, n)
    radii = np.sqrt(rng.uniform(0, 1, n))
    return np.column_stack([radii * np.cos(angles), radii * np.sin(angles)])


def _soft_max_objective(pts, r_target, beta):
    """
    Soft-max objective: approximates max of all constraint violations.
    Violations: boundary (|c_i| - (1-r)) and pairwise (2r - |c_i-c_j|).
    Returns (objective_value, gradient).
    """
    n = len(pts)
    norms = np.sqrt(np.sum(pts ** 2, axis=1))
    cont_viol = norms - (1.0 - r_target)
    
    iu = np.triu_indices(n, k=1)
    diff = pts[iu[0]] - pts[iu[1]]
    dists = np.sqrt(np.sum(diff ** 2, axis=1))
    pair_viol = 2.0 * r_target - dists
    
    all_viol = np.concatenate([cont_viol, pair_viol])
    
    # Soft-max approximation of max
    max_v = np.max(all_viol)
    if max_v > 50.0 / beta:
        return max_v, np.zeros_like(pts)
    
    exp_v = np.exp(beta * (all_viol - max_v))
    sum_exp = np.sum(exp_v)
    soft_max_val = max_v + np.log(sum_exp) / beta
    weights = exp_v / sum_exp
    
    grad = np.zeros_like(pts)
    
    # Container gradient: d/dx_i (|x_i|) = x_i / |x_i|
    safe_norms = np.where(norms > 1e-10, norms, 1.0)
    unit_vecs = pts / safe_norms[:, np.newaxis]
    grad += weights[:n, np.newaxis] * unit_vecs
    
    # Pairwise gradient
    iw = weights[n:]
    valid = dists > 1e-10
    if np.any(valid):
        iw_valid = iw[valid]
        iu_valid = (iu[0][valid], iu[1][valid])
        d_valid = dists[valid]
        diff_valid = pts[iu_valid[0]] - pts[iu_valid[1]]
        unit_diff = diff_valid / d_valid[:, np.newaxis]
        
        np.add.at(grad, iu_valid[0], -iw_valid[:, np.newaxis] * unit_diff)
        np.add.at(grad, iu_valid[1], iw_valid[:, np.newaxis] * unit_diff)
    
    return soft_max_val, grad


def _soft_max_optimize(pts_init, r_target, start_beta=50.0, end_beta=100000.0,
                       n_stages=6, max_iter_per_stage=400, tol=1e-12):
    """
    Optimize using soft-max objective with beta annealing.
    Returns optimized positions.
    """
    current_pts = pts_init.copy()
    
    for stage in range(n_stages):
        if time.time() > _DEADLINE - 5.0:
            break
        
        frac = stage / max(n_stages - 1, 1)
        beta = start_beta * (end_beta / start_beta) ** frac
        
        def obj_grad(x):
            pts = x.reshape(N, 2)
            val, grad = _soft_max_objective(pts, r_target, beta)
            return val, grad.reshape(-1)
        
        x0 = current_pts.reshape(-1)
        try:
            result = minimize(
                obj_grad,
                x0,
                jac=True,
                method='L-BFGS-B',
                options={'maxiter': max_iter_per_stage, 'ftol': tol, 'gtol': 1e-10, 'maxls': 50}
            )
            current_pts = result.x.reshape(N, 2)
        except Exception:
            pass
    
    return current_pts


def _check_feasible(pts, r_target, tol=1e-6):
    """Check if configuration is feasible for given radius."""
    val, _ = _soft_max_objective(pts, r_target, beta=100000.0)
    return val < tol


def _local_refine(pts, r_target, max_iters=2000):
    """Local refinement using small random perturbations."""
    pts = pts.copy().astype(np.float64)
    n = len(pts)
    best_score = _score(pts)
    best_pts = pts.copy()
    
    for it in range(max_iters):
        if time.time() > _DEADLINE - 10.0:
            break
        
        k = np.random.randint(1, 6)
        idx = np.random.choice(n, k, replace=False)
        
        perturbation = np.random.normal(0, 0.003, size=(k, 2))
        new_pts = pts.copy()
        new_pts[idx] += perturbation
        
        nr = np.hypot(new_pts[:, 0], new_pts[:, 1])
        mask = nr > 1.0
        if np.any(mask):
            new_pts[mask] /= nr[mask, None] * (1.0 + 1e-6)
        
        new_score = _score(new_pts)
        
        if new_score > best_score + 1e-14:
            pts = new_pts
            best_score = new_score
            best_pts = pts.copy()
    
    return best_pts, best_score


def construct():
    """Construct 50 circle centers in unit circle maximizing common radius."""
    best_pts = None
    best_score = 0.0
    
    # Multiple diverse initializations
    inits = []
    
    # Hex lattice with different spacing guesses
    for r_guess in [0.10, 0.11, 0.12, 0.13]:
        for seed in range(3):
            inits.append(_hex_lattice_init(N, r_guess, jitter=0.0, seed=seed))
    
    # Ring initializations
    inits.append(_ring_init(N, [1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95], seed=0))
    inits.append(_ring_init(N, [1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95],
                            angle_offsets=[0.0, np.pi/6, np.pi/12, np.pi/8, np.pi/10], seed=1))
    inits.append(_ring_init(N, [1, 6, 12, 18, 13], [0.0, 0.25, 0.50, 0.75, 0.95],
                            angle_offsets=[0.0, np.pi/5, np.pi/7, np.pi/9, np.pi/11], seed=2))
    
    # Random initializations
    for seed in range(5):
        inits.append(_random_init(N, seed=seed))
    
    # Jittered hex lattice
    for seed in range(5):
        inits.append(_hex_lattice_init(N, 0.12, jitter=0.03, seed=seed))
    
    # Shuffle
    np.random.shuffle(inits)
    
    for init_idx, init_pos in enumerate(inits):
        if time.time() > _DEADLINE - 25.0:
            break
        
        # Get initial score
        init_score = _score(init_pos)
        
        # Bisection search for optimal radius using soft-max optimizer
        lo, hi = 0.08, 0.14
        best_r_for_init = init_score
        best_pos_for_init = init_pos.copy()
        
        for bisect_iter in range(10):
            if time.time() > _DEADLINE - 10.0:
                break
            
            mid_r = (lo + hi) / 2.0
            
            # Try to optimize for this radius
            optimized_pos = _soft_max_optimize(
                init_pos, mid_r,
                start_beta=50.0,
                end_beta=100000.0,
                n_stages=5,
                max_iter_per_stage=300
            )
            
            # Check feasibility
            if _check_feasible(optimized_pos, mid_r, tol=1e-5):
                actual_score = _score(optimized_pos)
                if actual_score > best_score:
                    best_score = actual_score
                    best_pts = optimized_pos.copy()
                if actual_score > best_r_for_init:
                    best_r_for_init = actual_score
                    best_pos_for_init = optimized_pos.copy()
                lo = mid_r
            else:
                hi = mid_r
        
        # Refine around best radius found for this initialization
        if best_r_for_init > 0 and time.time() < _DEADLINE - 15.0:
            for refine_iter in range(3):
                if time.time() > _DEADLINE - 10.0:
                    break
                test_r = best_r_for_init * 1.005
                optimized_pos = _soft_max_optimize(
                    best_pos_for_init, test_r,
                    start_beta=100.0,
                    end_beta=100000.0,
                    n_stages=4,
                    max_iter_per_stage=200
                )
                actual_score = _score(optimized_pos)
                if actual_score > best_score:
                    best_score = actual_score
                    best_pts = optimized_pos.copy()
                    best_r_for_init = actual_score
                    best_pos_for_init = optimized_pos.copy()
    
    # Final local refinement on best found
    if best_pts is not None and time.time() < _DEADLINE - 5.0:
        refined_pts, refined_score = _local_refine(best_pts, best_score, max_iters=1500)
        if refined_score > best_score:
            best_score = refined_score
            best_pts = refined_pts.copy()
    
    if best_pts is None:
        best_pts = _hex_lattice_init(N, 0.12, jitter=0.0, seed=0)
        best_score = _score(best_pts)
    
    # Final validation and cleanup
    r = np.hypot(best_pts[:, 0], best_pts[:, 1])
    mask = r > 1.0
    if np.any(mask):
        best_pts[mask] /= r[mask, None] * (1.0 + 1e-6)
    
    return best_pts
