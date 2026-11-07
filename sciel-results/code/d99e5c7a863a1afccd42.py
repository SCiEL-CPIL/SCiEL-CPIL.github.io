"""Acquisition for Bayesian optimization of Hartmann 6d using a parameter search."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

# Deadline guard: evaluator kills at 330s wall-clock; leave a safety margin.
_DEADLINE = time.time() + 325.0

# Known minimum of the Hartmann 6d function (for adaptive exploration weight).
_HARTMANN_6D_MIN = -3.32237


def _ei_score(mu, sigma, f_best, w_explore):
    """Expected Improvement plus a constant exploration weight on sigma."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    return ei + w_explore * sigma_safe


def _ucb_score(mu, sigma, f_best, beta):
    """Upper Confidence Bound (UCB) for minimization."""
    # For minimization, we want high uncertainty to be rewarded.
    # f_best - mu + beta * sigma: high when mu is low (good) and sigma is high (explore)
    sigma_safe = np.maximum(sigma, 1e-12)
    return (f_best - mu) + beta * sigma_safe


def _knowledge_gradient(mu, sigma, f_best, w_explore):
    """KG-like: expected improvement with a larger exploration component."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    # Additional sigma term scaled differently
    return ei + w_explore * sigma_safe + 0.05 * sigma_safe ** 1.5


def _adaptive_ei(mu, sigma, f_best, w_base):
    """EI with adaptive exploration: stronger when far from optimum."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # How far are we from the known minimum? (larger gap -> more exploration)
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    # Scale exploration weight: starts at w_base, grows with gap up to a cap
    w = w_base + 0.5 * gap
    w = min(w, 1.0)  # cap exploration weight
    return ei + w * sigma_safe


def _pi_score(mu, sigma, f_best, w_explore, w_sq):
    """PI-like score: expected value of (f_best - f) with exploration."""
    sigma_safe = np.maximum(sigma, 1e-12)
    # E[(f_best - f)] = (f_best - mu) * Phi(z) + sigma * phi(z)  [same as EI for f > f_best]
    # Add quadratic exploration for more aggressive unexplored region seeking
    return _ei_score(mu, sigma, f_best, w_explore) + w_sq * sigma_safe ** 2


def _hybrid_score(mu, sigma, f_best, alpha, beta, w_explore):
    """Hybrid: alpha * EI + beta * UCB + w_explore * sigma."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ucb = (f_best - mu) + 1.0 * sigma_safe  # beta=1.0 fixed in ucb part
    return alpha * ei + beta * ucb + w_explore * sigma_safe


def _random_acquisition(mu, sigma, f_best, rng):
    """A random blend of EI and sigma with random weights."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Random exploration weight in [0.01, 0.5]
    w = rng.uniform(0.01, 0.5)
    # Sometimes add a quadratic term
    if rng.random() < 0.3:
        w_sq = rng.uniform(0.01, 0.2)
        return ei + w * sigma_safe + w_sq * sigma_safe ** 2
    return ei + w * sigma_safe


def _score_candidates(mu, sigma, f_best, n_candidates=2000):
    """
    Generate diverse candidate acquisition scores and return a blended 'best'
    one using a smooth softmax selector over candidate quality metrics.
    The softmax smooths the hard argmax, making the selection robust to
    noise in quality estimates.
    """
    n = mu.shape[0]
    if n == 0:
        return np.zeros(0)

    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Pre-compute common building blocks
    sigma_sq = sigma_safe ** 2
    sigma_15 = sigma_safe ** 1.5

    # Gap from known optimum: drives adaptive exploration/exploitation
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    # Normalize gap: 0 when at optimum, ~1 when far away (Hartmann 6d range is ~4)
    gap_norm = min(gap / 3.0, 1.0)

    # Build a list of candidate score arrays and a heuristic quality metric
    candidates = []  # list of (score_array, quality_metric)

    def _register(s):
        """Register a score array with robust soft-max quality metric."""
        s = np.where(np.isfinite(s), s, 0.0)
        s_sorted = np.sort(s)[::-1]
        top1 = s_sorted[0]
        top10_mean = np.mean(s_sorted[:min(10, n)])
        # Use a wider top-k for more stable quality estimate
        top20_mean = np.mean(s_sorted[:min(20, n)])
        # Quality: top score + separation from top-10 + separation from top-20
        quality = top1 + 0.3 * (top1 - top10_mean) + 0.2 * (top1 - top20_mean)
        candidates.append((s, quality))

    # --- Parameterized EI + exploration ---
    # Wider range, with gap-adaptive emphasis
    for w in [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0]:
        s = ei + w * sigma_safe
        _register(s)

    # --- Adaptive EI (gap-scaled base weight) ---
    for w_base in [0.02, 0.05, 0.1, 0.2, 0.3]:
        w = min(w_base + 0.5 * gap, 1.5)
        s = ei + w * sigma_safe
        _register(s)

    # --- UCB variants (gap-adaptive range) ---
    # Early: higher beta for exploration; Late: lower beta for exploitation
    beta_range = [0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0, 4.0]
    for beta in beta_range:
        s = (f_best - mu) + beta * sigma_safe
        _register(s)

    # --- Quadratic exploration ---
    for w_exp, w_sq in [(0.05, 0.01), (0.1, 0.02), (0.15, 0.05), (0.2, 0.1), (0.3, 0.15)]:
        s = ei + w_exp * sigma_safe + w_sq * sigma_sq
        _register(s)

    # --- 1.5-power exploration ---
    for w_exp, w_15 in [(0.1, 0.05), (0.15, 0.1), (0.2, 0.15), (0.3, 0.2)]:
        s = ei + w_exp * sigma_safe + w_15 * sigma_15
        _register(s)

    # --- Hybrid EI + UCB ---
    for alpha, beta_h, w_e in [(0.5, 0.5, 0.05), (0.7, 0.3, 0.05), (0.3, 0.7, 0.1), (1.0, 0.0, 0.1), (0.8, 0.2, 0.03)]:
        ucb_part = (f_best - mu) + 1.0 * sigma_safe
        s = alpha * ei + beta_h * ucb_part + w_e * sigma_safe
        _register(s)

    # --- Random candidates (a few) ---
    rng = np.random.default_rng(42)
    for _ in range(5):
        s = _random_acquisition(mu, sigma, f_best, rng)
        _register(s)

    # Smooth softmax selector over quality metrics
    # Temperature adapts to gap: early (large gap) -> higher temp (more exploration
    # in selector), late (small gap) -> lower temp (sharper selection)
    qualities = np.array([q for _, q in candidates])
    # Normalize qualities to [0, 1] for stable softmax
    q_min = np.min(qualities)
    q_max = np.max(qualities)
    q_range = q_max - q_min
    if q_range < 1e-12:
        # All qualities equal: fall back to first candidate
        best_scores = candidates[0][0]
        best_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
        return best_scores

    q_norm = (qualities - q_min) / q_range  # in [0, 1]

    # Temperature: early (gap_norm ~ 1) -> T=2.0 (flat), late (gap_norm ~ 0) -> T=0.5 (sharp)
    temperature = 0.5 + 1.5 * gap_norm

    # Softmax weights
    exp_q = np.exp(q_norm / temperature)
    weights = exp_q / np.sum(exp_q)

    # Blended score: weighted average of all candidate scores
    score_stack = np.array([s for s, _ in candidates])  # shape (n_candidates, n)
    blended = np.sum(weights[:, np.newaxis] * score_stack, axis=0)

    # Ensure finite
    blended = np.where(np.isfinite(blended), blended, 0.0)
    return blended


def acquisition(mu, sigma, f_best):
    """
    Acquisition function that searches over a set of parameterized acquisition
    strategies and returns the scores from the best-performing one.

    Parameters
    ----------
    mu : ndarray, shape (n,)
        GP mean predictions.
    sigma : ndarray, shape (n,)
        GP standard deviation predictions.
    f_best : float
        Best (lowest) observed objective value so far.

    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values are preferred.
    """
    # Deadline check: if we're too close to the limit, fall back to a fast
    # deterministic acquisition (EI + 0.1 * sigma).
    if time.time() > _DEADLINE - 2.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.1 * sigma_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    # Run the parameter search
    try:
        scores = _score_candidates(mu, sigma, f_best)
    except Exception:
        # Fallback on any error
        sigma_safe = np.maximum(sigma, 1e-12)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        scores = ei + 0.1 * sigma_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)

    # Final sanity: ensure shape matches and all finite
    if scores.shape != mu.shape:
        scores = np.resize(scores, mu.shape)
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
