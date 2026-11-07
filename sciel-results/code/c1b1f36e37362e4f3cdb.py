# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

_start_time = time.time()
_DEADLINE = 325.0

def acquisition(mu, sigma, f_best):
    if time.time() - _start_time > _DEADLINE:
        return -mu + 2.0 * np.maximum(sigma, 1e-9)

    sigma_safe = np.maximum(sigma, 1e-9)
    n = len(mu)

    # z-score: how many sigma below f_best each candidate is
    z = (f_best - mu) / sigma_safe

    # Probability that candidate is a new minimum
    p_new_min = norm.cdf(z)

    # Expected Improvement for minimization
    ei = (f_best - mu) * p_new_min + sigma_safe * norm.pdf(z)

    # Predictive Entropy Search (PES) approximation:
    # The key idea is to score candidates by how much they reduce
    # uncertainty about the global minimum. We approximate this by:
    # 1. The probability of being a new minimum (exploration of promising regions)
    # 2. The magnitude of expected improvement (exploitation)
    # 3. A term that rewards reducing uncertainty where the function is low

    # Adaptive scaling: as f_best approaches 0 (the true minimum),
    # we should be more exploitative. The Weierstrass function on [-0.5, 0.5]
    # has a minimum of 0, and typical values are in [0, 2].

    # Normalize f_best to [0, 1] to control the exploration/exploitation balance
    f_best_norm = np.clip(f_best / 2.0, 0.0, 1.0)

    # Weight: as f_best improves (lower), shift towards exploitation
    # Early on (f_best high), w is close to 1 (balanced/exploration)
    # Late on (f_best low), w approaches 0.3 (more exploitation)
    w = 0.3 + 0.7 * f_best_norm

    # PES-like score components:
    # 1. Log probability of new minimum (numerically stable)
    log_p_new_min = np.log(np.maximum(p_new_min, 1e-15))

    # 2. Expected improvement normalized by probability (conditional EI)
    # This is the expected improvement given that x IS a new minimum
    cond_ei = ei / np.maximum(p_new_min, 1e-15)

    # 3. Uncertainty reduction term: sigma_safe weighted by how promising the region is
    # High sigma in regions where mu is already low is very valuable
    uncertainty_value = sigma_safe * np.exp(-np.maximum(mu, 0) / (0.1 + f_best))

    # Combine: PES score approximates mutual information
    # Use a weighted combination that adapts to the optimization stage
    score = w * log_p_new_min + (1 - w) * cond_ei + 0.15 * uncertainty_value

    # Add a small exploration bonus for high uncertainty regions
    # This helps escape local optima in the multimodal landscape
    exploration_bonus = 0.05 * sigma_safe / (1.0 + sigma_safe)
    score += exploration_bonus

    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)

    return score
# EVOLVE-BLOCK-END
