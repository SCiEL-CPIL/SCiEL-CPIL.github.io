"""Different strategy: low-dimensional modal parameterization + Nelder-Mead on exact objective."""
import time
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import pdist
from scipy.optimize import minimize

N = 200

_START_TIME = time.time()
_TIME_LIMIT = 145.0  # 5s safety margin before 150s kill

def _time_left():
    return _TIME_LIMIT - (time.time() - _START_TIME)

def _compute_radius(centers):
    """Compute the maximum valid radius for given centers (exact, non-smooth)."""
    if centers.shape[0] != N:
        return 0.0
    if not np.all(np.isfinite(centers)):
        return 0.0
    if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
        return 0.0
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    min_wall = np.min(wall_dists)
    tree = cKDTree(centers)
    dists, _ = tree.query(centers, k=2)
    min_pair = np.min(dists[:, 1])
    r = min(min_wall, min_pair / 2.0)
    return max(r, 0.0)

def _eval_radius(centers):
    """Safe wrapper that returns 0.0 on any invalidity."""
    try:
        if centers.shape != (N, 2):
            return 0.0
        if not np.all(np.isfinite(centers)):
            return 0.0
        if np.any(centers < -0.5 - 1e-12) or np.any(centers > 0.5 + 1e-12):
            return 0.0
        wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
        min_wall = np.min(wall_dists)
        tree = cKDTree(centers)
        dists, _ = tree.query(centers, k=2)
        min_pair = np.min(dists[:, 1])
        r = min(min_wall, min_pair / 2.0)
        return max(float(r), 0.0)
    except Exception:
        return 0.0

def _build_hex_base(n_rows):
    """Build a hexagonal lattice base of N points, centered, scaled to fit in [-0.5,0.5]^2.
    Returns (base_array, scale) where base_array is the raw lattice and scale is the fit factor."""
    n_cols = int(np.ceil(N / n_rows))
    raw = []
    for i in range(n_rows):
        for j in range(n_cols):
            if len(raw) >= N:
                break
            x = j + (i % 2) * 0.5
            y = i * np.sqrt(3) / 2
            raw.append([x, y])
        if len(raw) >= N:
            break
    raw = np.array(raw[:N], dtype=float)
    # Center
    raw -= raw.mean(axis=0)
    # Scale so that max |coord| + r fits; we want the bounding box of centers to leave room.
    # We'll return raw and let the caller choose scale so that the lattice fits.
    return raw

def _fit_scale(raw, r_target):
    """Given raw lattice (centered) and a target radius, find scale so that scaled lattice
    with radius r_target fits in [-0.5, 0.5]^2. Returns scale or None if infeasible."""
    # After scaling by s, centers are raw*s, radius is r_target.
    # Constraint: |raw*s| + r_target <= 0.5 in each coordinate.
    # => s <= (0.5 - r_target) / max(|raw|)
    max_abs = np.max(np.abs(raw))
    if max_abs < 1e-12:
        return 1.0
    s = (0.5 - r_target) / max_abs
    return s

def _hex_initial(r_target, n_rows, seed=0):
    """Produce N centers for a hex lattice with the given target radius, centered and scaled to fit."""
    raw = _build_hex_base(n_rows)
    s = _fit_scale(raw, r_target)
    if s is None or s <= 0:
        return None
    return raw * s

def _soft_min(x, beta):
    """Soft minimum, numerically stable."""
    x_min = np.min(x)
    return x_min - np.log(np.sum(np.exp(-(x - x_min) * beta))) / beta

def _softmax_neg(x, beta):
    """Soft max of negatives = -soft_min(-x)."""
    return -_soft_min(-x, beta)

def _epigraph_objective_and_grad(centers, t, beta):
    """
    Objective: minimize  max( -min_pair_margin, -min_wall_margin )  where
      pair_margin = min_{i<j} (|c_i-c_j| - 2t)
      wall_margin = min_i (wall_dist_i - t)
    We use soft-min to make it smooth. Returns (value, grad).
    """
    # Pairwise distances
    d = pdist(centers)  # shape (M,)
    pair_margin = d - 2.0 * t
    # Wall distances
    wall_dists = np.min(np.minimum(0.5 - centers, centers + 0.5), axis=1)
    wall_margin = wall_dists - t

    # Soft min of pair_margin
    pm_min = np.min(pair_margin)
    exp_pm = np.exp(-(pair_margin - pm_min) * beta)
    sum_exp_pm = np.sum(exp_pm)
    pair_soft_min = pm_min - np.log(sum_exp_pm) / beta

    # Soft min of wall_margin
    wm_min = np.min(wall_margin)
    exp_wm = np.exp(-(wall_margin - wm_min) * beta)
    sum_exp_wm = np.sum(exp_wm)
    wall_soft_min = wm_min - np.log(sum_exp_wm) / beta

    # We want to maximize min(pair_soft_min, wall_soft_min)
    # => minimize -min(...)  => use soft max of negatives
    # value = -soft_min(min(pair_soft_min, wall_soft_min))
    # Actually: value = soft_max(-pair_soft_min, -wall_soft_min)
    vals = np.array([-pair_soft_min, -wall_soft_min])
    v_max = np.max(vals)
    exp_v = np.exp(-(vals - v_max) * beta)
    sum_exp_v = np.sum(exp_v)
    value = v_max - np.log(sum_exp_v) / beta

    # Gradients
    # d value / d centers
    # value = soft_max(-pair_sm, -wall_sm)
    # Let a = -pair_sm, b = -wall_sm
    # value = max(a,b) - log(exp(-a+max)+exp(-b+max))/beta  ... let's just do chain rule numerically-ish
    # Simpler: compute weights
    w_a = np.exp(-(vals[0] - v_max) * beta) / sum_exp_v
    w_b = np.exp(-(vals[1] - v_max) * beta) / sum_exp_v

    # d a / d centers = -d pair_sm / d centers
    # pair_sm = pm_min - log(sum_exp_pm)/beta
    # d pair_sm / d centers[i] = sum_j (d pair_margin_j / d centers[i]) * (-exp_pm_j / (beta * sum_exp_pm))
    # d pair_margin_j / d centers: for pair (p,q), d/d centers[p] = 2*(c_p - c_q)/d_pq, d/d centers[q] = 2*(c_q - c_p)/d_pq
    # This is expensive to build full gradient. Let's use a simpler approach:
    # Use the fact that we can compute gradient of soft_min via weights.
    # grad_pair_sm = - (1/beta) * sum_j exp_pm_j/sum_exp_pm * grad(pair_margin_j)
    # But pair_margin_j = d_j - 2t, grad wrt centers is grad of d_j.
    # For 200 points, 19900 pairs. We can vectorize.

    # Let's compute gradient numerically via a trick: use the subgradient of the soft min.
    # Actually, let's just use finite-difference-free analytic for the wall part and a
    # sampled pair gradient for the pair part.

    # Wall gradient (exact, cheap):
    # wall_sm = wm_min - log(sum_exp_wm)/beta
    # d wall_sm / d centers[i,k] = - (exp_wm_i / (beta * sum_exp_wm)) * d wall_margin_i / d centers[i,k]
    # d wall_margin_i / d centers[i,k]:
    #   wall_dist_i = min(0.5 - c_ik, 0.5 + c_ik)
    #   if 0.5 - c_ik <= 0.5 + c_ik (i.e., c_ik >= 0): d/d c_ik = -1
    #   else: d/d c_ik = +1
    wall_grad = np.zeros_like(centers)
    wm_weights = exp_wm / (beta * sum_exp_wm)
    for i in range(N):
        for k in range(2):
            if centers[i, k] >= 0:
                wall_grad[i, k] = -wm_weights[i]
            else:
                wall_grad[i, k] = wm_weights[i]
    # d b / d centers = -d wall_sm / d centers
    grad_from_wall = -wall_grad  # this is d b / d centers

    # Pair gradient:
    # a = -pair_sm, d a / d centers = -d pair_sm / d centers
    # d pair_sm / d centers[i] = - (1/beta) * sum_j (exp_pm_j / sum_exp_pm) * d d_j / d centers[i]
    # d d_j / d centers[i]: for pair j connecting (p,q):
    #   if i==p: 2*(c_p - c_q)/d_pq
    #   if i==q: 2*(c_q - c_p)/d_pq
    # We need the pair index mapping.
    # Build pair indices
    ii, jj = np.triu_indices(N, k=1)
    # d_j for each pair
    diff = centers[ii] - centers[jj]  # shape (M, 2)
    d_vals = np.sqrt(np.sum(diff**2, axis=1))
    d_vals = np.maximum(d_vals, 1e-12)
    # unit vectors
    unit = diff / d_vals[:, None]
    pm_weights = exp_pm / (beta * sum_exp_pm)
    # d pair_sm / d centers[i] = - sum_j pm_weights[j] * (d d_j / d centers[i])
    # d d_j / d centers[p] = 2 * unit_j
    # d d_j / d centers[q] = -2 * unit_j
    # So: d pair_sm / d centers = - sum_j pm_weights[j] * (2*unit_j for p, -2*unit_j for q)
    # => d a / d centers = -d pair_sm / d centers = sum_j pm_weights[j] * (2*unit_j for p, -2*unit_j for q)
    # Let's accumulate
    pair_grad_a = np.zeros_like(centers)
    # For each pair j, add to centers[ii[j]]: +2*pm_weights[j]*unit[j]
    # and to centers[jj[j]]: -2*pm_weights[j]*unit[j]
    np.add.at(pair_grad_a, ii, 2.0 * pm_weights[:, None] * unit)
    np.add.at(pair_grad_a, jj, -2.0 * pm_weights[:, None] * unit)

    # Total gradient of value wrt centers:
    # value = w_a * a + w_b * b  (soft max approximation, but actually it's a smooth function)
    # d value / d centers = w_a * d a / d centers + w_b * d b / d centers
    grad = w_a * pair_grad_a + w_b * grad_from_wall

    return value, grad

def _final_polish(centers, beta_start=100.0, beta_end=10000.0, max_iter=3000):
    """Epigraph-based polish: jointly optimize centers and t to maximize t."""
    centers = centers.copy()
    t = _compute_radius(centers)
    best_t = t
    best_centers = centers.copy()

    # Use projected gradient ascent on t, gradient descent on the objective
    # We'll do: at each step, compute grad of objective wrt centers (for current t),
    # move centers to reduce objective, then increase t.

    current_beta = beta_start
    lr = 0.002
    no_improve = 0

    for it in range(max_iter):
        if _time_left() < 10.0:
            break

        # Animate beta
        progress = it / max_iter
        current_beta = beta_start * (beta_end / beta_start) ** progress

        # Compute objective and gradient
        try:
            val, grad = _epigraph_objective_and_grad(centers, t, current_beta)
        except Exception:
            break

        # Move centers to reduce objective (gradient descent on val)
        centers -= lr * grad

        # Project to square
        centers[:, 0] = np.clip(centers[:, 0], -0.5 + 1e-10, 0.5 - 1e-10)
        centers[:, 1] = np.clip(centers[:, 1], -0.5 + 1e-10, 0.5 - 1e-10)

        # Update t: set t to the current achievable radius (or a fraction)
        t_new = _compute_radius(centers)
        # Blend: t = 0.9*t + 0.1*t_new, but don't decrease below current best
        t = 0.85 * t + 0.15 * t_new
        t = max(t, best_t * 0.99)

        if t_new > best_t + 1e-12:
            best_t = t_new
            best_centers = centers.copy()
            no_improve = 0
        else:
            no_improve += 1
            if no_improve > 200:
                lr *= 0.9
                no_improve = 0
                if lr < 1e-7:
                    break

    # Final: set t to the true radius of best_centers
    final_r = _compute_radius(best_centers)
    return best_centers, final_r

def _nelder_mead_on_modes(base_raw, n_rows, seed=0, max_iter=2000):
    """
    Parameterize centers as: base_raw * scale + shape_modes + jitter
    Optimize scale and a few shape mode amplitudes via Nelder-Mead on the exact objective.
    """
    rng = np.random.default_rng(seed)
    n_points = base_raw.shape[0]

    # Shape modes: low-frequency Fourier-like perturbations
    # We'll use a few sinusoidal modes in x and y
    n_modes = 6  # 3 in x, 3 in y (low frequency)
    # Precompute mode basis
    idx = np.arange(n_points)
    # Use the raw lattice indices to define spatial frequency
    x_base = base_raw[:, 0]
    y_base = base_raw[:, 1]
    x_range = np.max(x_base) - np.min(x_base) + 1e-12
    y_range = np.max(y_base) - np.min(y_base) + 1e-12
    x_norm = (x_base - np.min(x_base)) / x_range
    y_norm = (y_base - np.min(y_base)) / y_range

    # Modes: sin/cos of low frequencies
    mode_x = np.column_stack([
        np.sin(2 * np.pi * x_norm),
        np.cos(2 * np.pi * x_norm),
        np.sin(4 * np.pi * x_norm),
    ])
    mode_y = np.column_stack([
        np.sin(2 * np.pi * y_norm),
        np.cos(2 * np.pi * y_norm),
        np.sin(4 * np.pi * y_norm),
    ])

    def decode(params):
        """params: [scale, ax1, ax2, ax3, ay1, ay2, ay3, jitter_scale]"""
        scale = params[0]
        ax = params[1:4]
        ay = params[4:7]
        jitter_scale = params[7]
        centers = base_raw * scale
        centers[:, 0] += mode_x @ ax
        centers[:, 1] += mode_y @ ay
        # Jitter
        if jitter_scale > 1e-12:
            jitter = rng.standard_normal((n_points, 2)) * jitter_scale
            centers += jitter
        # Center
        centers -= centers.mean(axis=0)
        return centers

    def objective(params):
        if _time_left() < 5.0:
            return 1e6
        centers = decode(params)
        r = _eval_radius(centers)
        return -r  # minimize negative radius

    # Initial parameters
    # Choose initial scale so that a reasonable radius fits
    # Try to find a scale that gives r ~ 0.036
    # For hex lattice, nearest neighbor distance = scale * (lattice spacing)
    # Lattice spacing in raw: 1.0 (horizontal)
    # So nearest neighbor ~ scale * 1.0, radius ~ scale * 0.5
    # We want radius ~ 0.036, so scale ~ 0.072
    init_scale = 0.072
    init_params = np.array([init_scale, 0, 0, 0, 0, 0, 0, 0.001])

    result = minimize(objective, init_params, method='Nelder-Mead',
                      options={'maxiter': max_iter, 'xatol': 1e-8, 'fatol': 1e-10,
                               'adaptive': True})
    best_params = result.x
    best_centers = decode(best_params)
    best_r = _eval_radius(best_centers)
    return best_centers, best_r

def _generate_hex_grid():
    """Generate the best hexagonal lattice initial configuration."""
    best_centers = None
    best_r = 0.0

    for n_rows in range(12, 22):
        raw = _build_hex_base(n_rows)
        if raw.shape[0] != N:
            continue
        # Find scale for a target radius
        # We want the lattice to fit. Let's try a few target radii.
        for r_target in [0.035, 0.036, 0.0366, 0.037, 0.038]:
            s = _fit_scale(raw, r_target)
            if s is None or s <= 0:
                continue
            centers = raw * s
            r_actual = _compute_radius(centers)
            if r_actual > best_r:
                best_r = r_actual
                best_centers = centers.copy()

    if best_centers is None:
        # Fallback square grid
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        grid = np.array([(x, y) for y in ticks for x in ticks])
        return grid[:N]

    return best_centers

def construct():
    """Construct 200 circle centers in [-0.5, 0.5]^2 maximizing the minimum radius."""
    # Phase 1: Generate hex lattice initial
    centers = _generate_hex_grid()
    r = _compute_radius(centers)

    # Phase 2: Nelder-Mead on modal parameters (different representation)
    if _time_left() > 60.0:
        # Use the best hex lattice as base
        # Reconstruct base_raw from the current centers
        # Actually, let's pick the n_rows that gave the best result
        best_n_rows = None
        best_r_hex = 0.0
        for n_rows in range(12, 22):
            raw = _build_hex_base(n_rows)
            if raw.shape[0] != N:
                continue
            for r_target in [0.035, 0.036, 0.0366, 0.037, 0.038]:
                s = _fit_scale(raw, r_target)
                if s is None or s <= 0:
                    continue
                c = raw * s
                rr = _compute_radius(c)
                if rr > best_r_hex:
                    best_r_hex = rr
                    best_n_rows = n_rows
                    best_raw = raw.copy()
                    best_s = s

        if best_n_rows is not None:
            # Run Nelder-Mead with a few different seeds
            for seed in [0, 42, 123]:
                if _time_left() < 30.0:
                    break
                c_nm, r_nm = _nelder_mead_on_modes(best_raw, best_n_rows, seed=seed, max_iter=800)
                if r_nm > r:
                    r = r_nm
                    centers = c_nm.copy()

    # Phase 3: High-beta epigraph polish
    if _time_left() > 30.0:
        centers, r = _final_polish(centers, beta_start=100.0, beta_end=10000.0, max_iter=2500)

    # Final verification and clipping
    centers = np.clip(centers, -0.5, 0.5)
    # Ensure finite
    if not np.all(np.isfinite(centers)):
        # Fallback
        m = int(np.ceil(np.sqrt(N)))
        ticks = (np.arange(m) + 0.5) / m - 0.5
        centers = np.array([(x, y) for y in ticks for x in ticks])[:N]

    return centers
