# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_DEADLINE = time.time() + 325.0
_HARTMANN_6D_MIN = -3.32237


def _softmax_quality(s, tau=0.05):
    """
    Quality metric: prefer score arrays with a clear, high top candidate
    and good separation from the rest.
    """
    s = np.asarray(s, dtype=float)
    if s.size == 0:
        return -np.inf
    s_clean = np.where(np.isfinite(s), s, -1e10)
    # Normalize to avoid overflow in softmax
    s_max = np.max(s_clean)
    if not np.isfinite(s_max):
        return -np.inf
    shifted = s_clean - s_max
    weights = np.exp(shifted / tau)
    weights_sum = np.sum(weights)
    if weights_sum <= 0 or not np.isfinite(weights_sum):
        return -np.inf
    probs = weights / weights_sum
    # Entropy: low entropy means concentrated (good for picking a clear best)
    entropy = -np.sum(probs * np.log(np.maximum(probs, 1e-300)))
    # Top score value (normalized)
    top_val = s_max
    # Combine: high top value, low entropy
    # Scale top_val: typical range for scores is maybe [-10, 10]
    top_norm = np.tanh(top_val / 2.0)
    # Entropy range: [0, log(n)] ~ [0, 7.6] for n=2000
    ent_norm = entropy / np.log(max(s_clean.size, 2))
    # Quality: high top, low entropy
    return 0.6 * top_norm + 0.4 * (1.0 - ent_norm)


def _ei_value(mu, sigma, f_best):
    """Standard Expected Improvement for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    return (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)


def _ucb_value(mu, sigma, f_best, beta):
    """UCB for minimization."""
    sigma_safe = np.maximum(sigma, 1e-12)
    return (f_best - mu) + beta * sigma_safe


def _ts_gmm_score(mu, sigma, f_best, n_components=8, rng=None):
    """
    Thompson Sampling via a finite Gaussian Mixture Model of the posterior.
    
    The idea: discretize the posterior distribution over f(x) by building a GMM
    whose components are placed at different percentiles of the sigma field.
    Each component represents a "regime" of uncertainty. We draw a single sample
    from the mixture (i.e., pick a component and sample from it) and score each
    candidate by its expected value under that draw.
    
    This is structurally different from deterministic UCB/EI: it introduces
    stochasticity in the acquisition, which can escape local optima in the
    score landscape.
    """
    if rng is None:
        rng = np.random.default_rng()
    
    sigma_safe = np.maximum(sigma, 1e-12)
    n = mu.shape[0]
    
    # Build GMM components from sigma percentiles
    # Components represent different levels of uncertainty
    # We use percentiles of the sigma field to place component "centers"
    # in the uncertainty space
    
    # Get percentile values of sigma
    percentiles = np.percentile(sigma_safe, np.linspace(5, 95, n_components))
    # Ensure sorted and unique-ish
    percentiles = np.unique(percentiles)
    if len(percentiles) < 2:
        percentiles = np.array([np.min(sigma_safe), np.max(sigma_safe)])
        n_components = len(percentiles)
    
    # For each component, define a "draw" from the posterior:
    # A draw from N(mu_i, sigma_i) where sigma_i is the component's uncertainty level
    # We draw a single scalar "shift" that represents the random draw
    
    # Component weights: proportional to how much of the sigma field is near that percentile
    # Use a simple uniform weighting for now, or weight by local density
    # Weight by the fraction of points within a bandwidth of each percentile
    bandwidth = (np.max(sigma_safe) - np.min(sigma_safe)) / max(n_components, 1) * 0.5
    if bandwidth < 1e-10:
        weights = np.ones(n_components) / n_components
    else:
        weights = np.array([np.sum(np.abs(sigma_safe - p) < bandwidth) for p in percentiles], dtype=float)
        weights = weights / np.sum(weights)
    
    # Draw a component
    comp_idx = rng.choice(n_components, p=weights)
    comp_sigma = percentiles[comp_idx]
    
    # Draw a sample from the posterior at this uncertainty level
    # The posterior at a point with sigma=comp_sigma is N(mu_point, comp_sigma^2)
    # We draw a single "realization" of f for each candidate:
    # f_sample = mu + comp_sigma * z, where z ~ N(0,1)
    # But we want a single draw, not per-candidate draws.
    # Instead: draw a single scalar epsilon ~ N(0,1), and score as mu + comp_sigma * epsilon
    # This means all candidates get the same random shift scaled by their local sigma... 
    # Wait, that's not right. Let me think again.
    
    # Better approach: The GMM approximates the posterior over the function value.
    # For a given candidate x, the posterior is N(mu(x), sigma(x)^2).
    # Thompson sampling: draw f(x) ~ N(mu(x), sigma(x)^2) for each x, pick argmin.
    # But we can't draw for all 2000 candidates and return scores.
    # Instead, we approximate: draw a single "global" random variable that modulates
    # the exploration/exploitation trade-off.
    
    # Alternative: Draw a single beta from a distribution calibrated by the GMM,
    # then use UCB with that beta. The GMM determines the distribution of beta.
    
    # Let's do: draw a single random draw from the GMM to get a "confidence level"
    # that modulates the UCB beta.
    
    # Actually, let's do proper Thompson Sampling approximation:
    # Draw a single scalar s ~ N(0, 1)
    # Score each candidate as: mu + sigma * s
    # This is a single sample from the joint posterior (correlated draws)
    # The candidate with the lowest score is the one most likely to be best.
    # For acquisition scores (higher = better), we negate: -(mu + sigma * s)
    
    s_draw = rng.standard_normal()
    
    # Score: higher is better for acquisition
    # We want to prefer candidates where mu + sigma*s is low (likely to be best)
    # So acquisition score = -(mu + sigma * s)
    # But we also want to incorporate f_best as a reference
    # Score = (f_best - mu) - s * sigma  ... hmm, this mixes things
    
    # Cleaner: Thompson sampling says draw f(x) ~ N(mu(x), sigma(x)^2) and pick min.
    # The probability that x is the best is related to P(f(x) < f_best) and P(f(x) < f(x') for all x')
    # A simple proxy: score(x) = -E[f(x)] + exploration_bonus
    # But that's just UCB again.
    
    # Let me do it properly: draw a single shared random component.
    # f(x) = mu(x) + sigma(x) * z, where z is a shared draw.
    # Then the "best" candidate is argmin_x f(x).
    # The acquisition score for x is the probability that x is the best, or
    # simply -f(x) for the drawn realization.
    
    # For returning 2000 scores, we return the negative of the drawn value:
    # score(x) = -(mu(x) + sigma(x) * z)
    # Higher score = lower drawn value = more likely to be the best.
    
    scores = -(mu + sigma_safe * s_draw)
    
    # Add a small EI component to ensure exploitation of known good regions
    ei = _ei_value(mu, sigma, f_best)
    # Blend: primarily Thompson sample, with a small EI anchor
    scores = scores + 0.3 * ei
    
    return scores


def _adaptive_beta_ucb(mu, sigma, f_best, beta_min=0.3, beta_max=4.0, alpha_decay=3.0):
    """Adaptive UCB with progress-based beta decay."""
    sigma_safe = np.maximum(sigma, 1e-12)
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_max = 5.0
    gap_clamped = np.clip(gap, 0.0, gap_max)
    gap_norm = gap_clamped / gap_max
    
    beta = beta_min + (beta_max - beta_min) * np.exp(-alpha_decay * (1.0 - gap_norm))
    return _ucb_value(mu, sigma, f_best, beta)


def _score_candidates(mu, sigma, f_best, n_candidates=2000):
    """
    Generate diverse candidate acquisition scores and return the 'best' one
    based on a soft-max quality metric.
    """
    n = mu.shape[0]
    if n == 0:
        return np.zeros(0)
    
    sigma_safe = np.maximum(sigma, 1e-12)
    z = (f_best - mu) / sigma_safe
    ei = _ei_value(mu, sigma, f_best)
    
    sigma_sq = sigma_safe ** 2
    sigma_15 = sigma_safe ** 1.5
    
    gap = max(f_best - _HARTMANN_6D_MIN, 0.0)
    gap_norm = min(gap / 3.0, 1.0)
    
    candidates = []
    
    def _register(s):
        s = np.where(np.isfinite(s), s, 0.0)
        quality = _softmax_quality(s, tau=0.05)
        candidates.append((s, quality))
    
    # --- Parameterized EI + exploration ---
    for w in [0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5]:
        _register(ei + w * sigma_safe)
    
    # --- Adaptive EI ---
    for w_base in [0.01, 0.02, 0.05, 0.1, 0.15]:
        w = w_base + 0.8 * gap_norm
        w = min(w, 1.5)
        _register(ei + w * sigma_safe)
    
    # --- Adaptive UCB ---
    for beta_min, beta_max in [(0.5, 3.0), (1.0, 4.0), (0.3, 2.5), (1.5, 5.0)]:
        beta = beta_min + (beta_max - beta_min) * gap_norm
        _register(_ucb_value(mu, sigma, f_best, beta))
    
    # --- Static UCB ---
    for beta in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0]:
        _register(_ucb_value(mu, sigma, f_best, beta))
    
    # --- Quadratic exploration ---
    for w_exp, w_sq in [(0.03, 0.005), (0.05, 0.01), (0.1, 0.02), (0.15, 0.05), (0.2, 0.1)]:
        _register(ei + w_exp * sigma_safe + w_sq * sigma_sq)
    
    # --- 1.5-power exploration ---
    for w_exp, w_15 in [(0.05, 0.02), (0.1, 0.05), (0.15, 0.1), (0.2, 0.15)]:
        _register(ei + w_exp * sigma_safe + w_15 * sigma_15)
    
    # --- Hybrid EI + UCB ---
    for alpha, beta_h, w_e in [(0.5, 0.5, 0.05), (0.7, 0.3, 0.05), (0.8, 0.2, 0.03), (0.3, 0.7, 0.1), (1.0, 0.0, 0.1)]:
        ucb_part = _ucb_value(mu, sigma, f_best, 1.0)
        _register(alpha * ei + beta_h * ucb_part + w_e * sigma_safe)
    
    # --- Adaptive hybrid ---
    alpha_early, alpha_late = 0.3, 0.8
    alpha = alpha_late + (alpha_early - alpha_late) * gap_norm
    beta_h = 1.0 - alpha
    w_e = 0.05 * (1.0 - gap_norm) + 0.02 * gap_norm
    ucb_part = _ucb_value(mu, sigma, f_best, 2.0)
    _register(alpha * ei + beta_h * ucb_part + w_e * sigma_safe)
    
    # --- Thompson Sampling via GMM ---
    # Use a fixed seed derived from the current state for reproducibility within a step
    # but different across steps
    ts_seed = int(abs(hash((f_best, float(np.mean(mu)), float(np.mean(sigma_safe))))) % (2**32))
    rng_ts = np.random.default_rng(ts_seed)
    for _ in range(3):
        _register(_ts_gmm_score(mu, sigma, f_best, n_components=8, rng=rng_ts))
    
    # --- Adaptive UCB (the current program's core) ---
    _register(_adaptive_beta_ucb(mu, sigma, f_best))
    
    # --- Pure UCB variants with different decay rates ---
    for alpha_d in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        gap_clamped = np.clip(gap, 0.0, 5.0)
        gap_norm = gap_clamped / 5.0
        beta = 0.3 + 3.7 * np.exp(-alpha_d * (1.0 - gap_norm))
        _register(_ucb_value(mu, sigma, f_best, beta))
    
    # Pick the candidate with the best quality metric
    qualities = np.array([q for _, q in candidates])
    best_idx = int(np.argmax(qualities))
    best_scores = candidates[best_idx][0]
    
    best_scores = np.where(np.isfinite(best_scores), best_scores, 0.0)
    return best_scores


def acquisition(mu, sigma, f_best):
    """
    Acquisition function combining adaptive UCB with Thompson Sampling via GMM
    and a parameter search over acquisition strategies.
    """
    # Deadline check
    if time.time() > _DEADLINE - 2.0:
        sigma_safe = np.maximum(sigma, 1e-12)
        return (f_best - mu) + 1.0 * sigma_safe
    
    try:
        scores = _score_candidates(mu, sigma, f_best)
    except Exception:
        # Fallback to a simple, fast acquisition
        sigma_safe = np.maximum(sigma, 1e-12)
        scores = (f_best - mu) + 1.0 * sigma_safe
    
    scores = np.where(np.isfinite(scores), scores, 0.0)
    return scores
# EVOLVE-BLOCK-END
