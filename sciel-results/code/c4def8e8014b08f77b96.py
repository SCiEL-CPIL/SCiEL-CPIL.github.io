# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.optimize import minimize

_N = 10
_VALS = np.arange(_N, dtype=np.float64)
_YS = np.repeat(_VALS, _N)
_XS = np.tile(_VALS, _N)

# Precompute bin indices for each projection
_IDX_D = (_XS - _YS).astype(np.int64) + 9
_IDX_H0 = _XS.astype(np.int64)
_IDX_H1 = (_XS + _YS).astype(np.int64)
_IDX_H2 = (_XS + 2 * _YS).astype(np.int64)

_N_BINS_D = 19
_N_BINS_H0 = 10
_N_BINS_H1 = 19
_N_BINS_H2 = 28


def _entropy_from_counts(counts):
    """Compute Shannon entropy in bits from a count vector, ignoring zeros."""
    s = counts.sum()
    if s <= 1e-30:
        return 0.0
    p = counts / s
    p = p[p > 0.0]
    if len(p) == 0:
        return 0.0
    return float(-(p * np.log2(p)).sum())


def _compute_value(P):
    """Compute D/M ratio for a given 10x10 distribution P."""
    p = P.ravel()
    total = p.sum()
    if total <= 1e-30:
        return 0.0
    p = p / total

    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m = max(h0, h1, h2)
    if m <= 1e-9:
        return 0.0
    return hd / m


def _soft_max(values, temperature=0.1):
    """Smooth approximation of max using log-sum-exp."""
    values = np.asarray(values, dtype=np.float64)
    if len(values) == 0:
        return 0.0
    if len(values) == 1:
        return float(values[0])
    scaled = values / temperature
    max_val = np.max(scaled)
    exp_scaled = np.exp(scaled - max_val)
    log_sum_exp = max_val + np.log(exp_scaled.sum())
    return float(log_sum_exp * temperature)


def _theta_to_P(theta):
    """Convert log-probabilities to a normalized 10x10 probability matrix."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    P = np.clip(P, 0.0, None)
    s = P.sum()
    if s > 1e-9:
        P = P / s
    return P


def _negative_value_smooth(theta, temperature=0.1):
    """Objective to minimize: -D/M_smooth. Uses soft-max for M."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()

    d_counts = np.zeros(_N_BINS_D)
    np.add.at(d_counts, _IDX_D, p)
    hd = _entropy_from_counts(d_counts)

    h0_counts = np.zeros(_N_BINS_H0)
    np.add.at(h0_counts, _IDX_H0, p)
    h0 = _entropy_from_counts(h0_counts)

    h1_counts = np.zeros(_N_BINS_H1)
    np.add.at(h1_counts, _IDX_H1, p)
    h1 = _entropy_from_counts(h1_counts)

    h2_counts = np.zeros(_N_BINS_H2)
    np.add.at(h2_counts, _IDX_H2, p)
    h2 = _entropy_from_counts(h2_counts)

    m_smooth = _soft_max(np.array([h0, h1, h2]), temperature)
    if m_smooth <= 1e-9:
        return 1e6
    return -(hd / m_smooth)


def _negative_value(theta):
    """Objective to minimize: -D/M. theta is a 100-dim vector of log-probabilities."""
    theta = np.asarray(theta, dtype=np.float64)
    theta_shifted = theta - theta.max()
    exp_theta = np.exp(theta_shifted)
    p = exp_theta / exp_theta.sum()
    P = p.reshape(_N, _N)
    val = _compute_value(P)
    if val <= 1e-9:
        return 1e6
    return -val


def _support_optimize(active_mask, w_init=None, maxiter=500):
    """
    Optimize weights on a given support set.
    active_mask: boolean array of shape (10, 10)
    w_init: initial weights on active cells (optional)
    Returns: best P found and its score.
    """
    active_indices = np.where(active_mask)
    n_active = len(active_indices[0])
    if n_active < 2:
        return None, 0.0

    if w_init is None:
        w_init = np.ones(n_active) / n_active

    def obj(theta_a):
        theta_a = np.asarray(theta_a, dtype=np.float64)
        theta_shifted = theta_a - theta_a.max()
        exp_theta = np.exp(theta_shifted)
        p_a = exp_theta / exp_theta.sum()

        P_full = np.zeros((_N, _N))
        P_full[active_indices] = p_a
        val = _compute_value(P_full)
        if val <= 1e-9:
            return 1e6
        return -val

    theta_init = np.log(np.maximum(w_init, 1e-18))
    theta_init = theta_init - theta_init.max()

    try:
        result = minimize(
            obj,
            theta_init,
            method='L-BFGS-B',
            options={'maxiter': maxiter, 'gtol': 1e-10, 'ftol': 1e-12}
        )
        theta_opt = result.x
        theta_shifted = theta_opt - theta_opt.max()
        exp_theta = np.exp(theta_shifted)
        p_opt = exp_theta / exp_theta.sum()

        P_opt = np.zeros((_N, _N))
        P_opt[active_indices] = p_opt
        score = _compute_value(P_opt)
        return P_opt, score
    except Exception:
        return None, 0.0


def _random_support(n_points, rng):
    """Generate a random support set of n_points distinct cells."""
    indices = rng.choice(_N * _N, size=n_points, replace=False)
    mask = np.zeros((_N, _N), dtype=bool)
    mask.flat[indices] = True
    return mask


def _line_support(slope_type, intercept, rng):
    """
    Generate a support set along a line.
    slope_type: 'diag' (x-y=c), 'anti' (x+y=c), 'row' (y=c), 'col' (x=c),
                'slope2' (x+2y=c)
    """
    mask = np.zeros((_N, _N), dtype=bool)

    if slope_type == 'diag':
        for y in range(_N):
            x = y + intercept
            if 0 <= x < _N:
                mask[y, x] = True
    elif slope_type == 'anti':
        for y in range(_N):
            x = intercept - y
            if 0 <= x < _N:
                mask[y, x] = True
    elif slope_type == 'row':
        if 0 <= intercept < _N:
            mask[intercept, :] = True
    elif slope_type == 'col':
        if 0 <= intercept < _N:
            mask[:, intercept] = True
    elif slope_type == 'slope2':
        for y in range(_N):
            x = intercept - 2 * y
            if 0 <= x < _N:
                mask[y, x] = True

    # Add some random noise points
    n_extra = rng.randint(0, 5)
    if n_extra > 0:
        extra_indices = rng.choice(_N * _N, size=n_extra, replace=False)
        for idx in extra_indices:
            if not mask.flat[idx]:
                mask.flat[idx] = True

    if mask.sum() < 2:
        # Ensure at least 2 points
        mask[0, 0] = True
        mask[0, 1] = True

    return mask


def _two_line_support(line1_type, line1_int, line2_type, line2_int, rng):
    """Generate a support set as union of two lines."""
    mask1 = _line_support(line1_type, line1_int, rng)
    mask2 = _line_support(line2_type, line2_int, rng)
    mask = mask1 | mask2

    if mask.sum() < 3:
        mask[0, 0] = True
        mask[1, 1] = True
        mask[2, 2] = True

    return mask


def _grid_support(step, offset=0, rng=None):
    """Generate a support set as a regular grid pattern."""
    mask = np.zeros((_N, _N), dtype=bool)
    for y in range(0, _N, step):
        for x in range(0, _N, step):
            mask[y + offset, x + offset] = True

    if mask.sum() < 2:
        mask[0, 0] = True
        mask[1, 1] = True

    return mask


def _diagonal_band_support(center, width, rng):
    """Generate support as a band around the main diagonal."""
    mask = np.zeros((_N, _N), dtype=bool)
    for y in range(_N):
        for x in range(_N):
            if abs((x - y) - center) <= width:
                mask[y, x] = True

    if mask.sum() < 2:
        mask[0, 0] = True
        mask[1, 1] = True

    return mask


def _random_structured_support(rng):
    """Generate a random structured support set."""
    choice = rng.randint(0, 6)

    if choice == 0:
        # Single diagonal
        intercept = rng.randint(-9, 10)
        return _line_support('diag', intercept, rng)
    elif choice == 1:
        # Single anti-diagonal
        intercept = rng.randint(0, 19)
        return _line_support('anti', intercept, rng)
    elif choice == 2:
        # Two lines
        types = ['diag', 'anti', 'row', 'col', 'slope2']
        t1 = types[rng.randint(0, 5)]
        t2 = types[rng.randint(0, 5)]
        i1 = rng.randint(-9, 19)
        i2 = rng.randint(-9, 19)
        return _two_line_support(t1, i1, t2, i2, rng)
    elif choice == 3:
        # Grid pattern
        step = rng.randint(2, 5)
        offset = rng.randint(0, step)
        return _grid_support(step, offset, rng)
    elif choice == 4:
        # Diagonal band
        center = rng.randint(-4, 5)
        width = rng.randint(0, 3)
        return _diagonal_band_support(center, width, rng)
    else:
        # Random sparse
        n = rng.randint(5, 20)
        return _random_support(n, rng)


def arithmetic_kakeya_distribution() -> np.ndarray:
    """
    Arithmetic Kakeya conjecture (entropy formulation). Construct a joint probability
    distribution P[y, x] of two discrete random variables X, Y supported on an n x n grid
    (here n = 10) so as to maximize the entropy projection ratio

        H(pi_{r_inf}) / max_i H(pi_{r_i}),

    where pi_r(x, y) = x + r * y projects the grid along slope r, H is the Shannon entropy
    (in bits) of the resulting projected distribution, the finite slopes are
    r_1, ..., r_k = 0.0, 1.0, 2.0 and the distinguished slope is r_inf = -1.0.

    Returns:
        P: np.ndarray of shape (10, 10), non-negative and summing to 1, where P[y, x] is the
            joint probability that X = x and Y = y.
    """
    start_time = time.time()
    deadline = start_time + 610.0  # safety margin before 630s

    # Initialize with uniform
    best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)
    best_score = _compute_value(best_P)

    rng = np.random.RandomState(42)

    # Phase 1: Support-based optimization with diverse structures
    n_support_trials = 300
    for i in range(n_support_trials):
        if time.time() > deadline - 120.0:
            break

        # Generate a random structured support
        if i < 50:
            # Focus on lines early
            choice = rng.randint(0, 3)
            if choice == 0:
                intercept = rng.randint(-9, 10)
                mask = _line_support('diag', intercept, rng)
            elif choice == 1:
                intercept = rng.randint(0, 19)
                mask = _line_support('anti', intercept, rng)
            else:
                types = ['diag', 'anti', 'row', 'col', 'slope2']
                t1 = types[rng.randint(0, 5)]
                t2 = types[rng.randint(0, 5)]
                i1 = rng.randint(-9, 19)
                i2 = rng.randint(-9, 19)
                mask = _two_line_support(t1, i1, t2, i2, rng)
        elif i < 150:
            # Mix of lines and other structures
            mask = _random_structured_support(rng)
        else:
            # More random and diverse
            if rng.random() < 0.5:
                mask = _random_structured_support(rng)
            else:
                n = rng.randint(3, 25)
                mask = _random_support(n, rng)

        # Optimize weights on this support
        P_candidate, score = _support_optimize(mask, maxiter=300)

        if P_candidate is not None and score > best_score:
            best_score = score
            best_P = P_candidate.copy()

    # Phase 2: Full-grid optimization from best solution
    if time.time() < deadline - 60.0:
        p_flat = best_P.ravel()
        p_safe = np.maximum(p_flat, 1e-15)
        theta_best = np.log(p_safe / p_safe.sum())
        theta_best = theta_best - theta_best.max()

        try:
            result = minimize(
                _negative_value,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 1000, 'gtol': 1e-12, 'ftol': 1e-14}
            )
            P = _theta_to_P(result.x)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass

    # Phase 3: Support refinement - try removing points from best solution
    if time.time() < deadline - 40.0:
        active = best_P > 1e-10
        n_active = active.sum()

        # Try removing each active point and re-optimizing
        for iter_remove in range(min(50, n_active)):
            if time.time() > deadline - 40.0:
                break

            # Find the smallest active cell
            masked = np.where(active, best_P, np.inf)
            min_idx = np.unravel_index(np.argmin(masked), best_P.shape)

            if best_P[min_idx] < 1e-12:
                break

            # Zero out this cell
            active_new = active.copy()
            active_new[min_idx] = False

            if active_new.sum() < 3:
                break

            # Re-optimize on the reduced support
            P_candidate, score = _support_optimize(active_new, maxiter=200)

            if P_candidate is not None and score > best_score:
                best_score = score
                best_P = P_candidate.copy()
                active = active_new.copy()
            else:
                # Restore
                active = active | (active_new == False)
                break

    # Phase 4: Final high-precision polish
    if time.time() < deadline - 15.0:
        p_flat = best_P.ravel()
        p_safe = np.maximum(p_flat, 1e-18)
        theta_best = np.log(p_safe / p_safe.sum())
        theta_best = theta_best - theta_best.max()

        try:
            result = minimize(
                _negative_value,
                theta_best,
                method='L-BFGS-B',
                options={'maxiter': 2000, 'gtol': 1e-14, 'ftol': 1e-16}
            )
            P = _theta_to_P(result.x)
            score = _compute_value(P)
            if score > best_score:
                best_score = score
                best_P = P.copy()
        except Exception:
            pass

    # Final cleanup: ensure validity
    best_P = np.clip(best_P, 0.0, None)
    s = best_P.sum()
    if s > 1e-9:
        best_P = best_P / s
    else:
        best_P = np.ones((_N, _N), dtype=np.float64) / (_N * _N)

    return best_P


# EVOLVE-BLOCK-END
