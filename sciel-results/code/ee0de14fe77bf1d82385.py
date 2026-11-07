# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Track elapsed time for deadline guard
_start_time = time.time()
_DEADLINE = 315.0  # Safety margin before 330s kill

_call_count = 0
_last_f_best = None

def acquisition(mu, sigma, f_best):
    global _call_count, _last_f_best
    
    # Check deadline
    if time.time() - _start_time > _DEADLINE:
        # Fallback: return UCB-like scores quickly
        sigma_safe = np.maximum(sigma, 1e-9)
        return -mu + 2.0 * sigma_safe
    
    # Reset logic for new seed/run
    if _last_f_best is not None and f_best > _last_f_best:
        _call_count = 0
    _last_f_best = f_best
    _call_count += 1
    
    n = len(mu)
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Predictive Entropy Search (PES) variant
    # Score each candidate by expected reduction in global minimum's posterior entropy
    # Approximated by mutual information between function value at candidate and location of current best
    
    # 1. Compute probability that each point is the new minimum
    # For minimization, P(x is new min) ~ P(f(x) < f_best)
    z = (f_best - mu) / sigma_safe
    prob_new_min = norm.cdf(z)
    
    # 2. Compute expected information gain about the global minimum
    # If x is likely to be the new minimum, evaluating it gives high info gain
    # Info gain is higher when uncertainty (sigma) is also high
    # Use log(1 + sigma) as a proxy for information content
    
    # 3. PES score: balance probability of being new min with information gain
    # Higher score = more likely to reduce uncertainty about global minimum
    info_gain = np.log1p(sigma_safe)
    
    # Combine: points that are both likely to be new minima AND have high uncertainty
    # are most valuable for reducing global minimum entropy
    pes_score = prob_new_min * (1.0 + 2.0 * info_gain)
    
    # 4. Add a secondary component: expected improvement weighted by exploration
    # This helps when PES alone might be too conservative
    ei = (f_best - mu) * prob_new_min + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # 5. UCB component for robustness
    mean_sigma = np.mean(sigma_safe)
    std_sigma = np.std(sigma_safe)
    if mean_sigma > 0:
        beta = 1.0 + 0.8 * (std_sigma / (mean_sigma + 1e-9))
        beta = np.clip(beta, 0.5, 2.5)
    else:
        beta = 1.0
    ucb = -mu + beta * sigma_safe
    
    # 6. Normalize all components to [0, 1]
    def normalize(arr):
        mn, mx = np.min(arr), np.max(arr)
        if mx - mn < 1e-12:
            return np.zeros_like(arr)
        return (arr - mn) / (mx - mn)
    
    pes_norm = normalize(pes_score)
    ei_norm = normalize(ei)
    ucb_norm = normalize(ucb)
    
    # 7. Adaptive weighting: shift from exploration (PES/UCB) to exploitation (EI)
    # as convergence progresses
    frac_better = np.mean(mu < f_best)
    x = _call_count - 1
    
    # Early: more PES (global structure), later: more EI (local refinement)
    w_pes = 0.5 * (1.0 - frac_better) + 0.1
    w_ei = 0.3 * frac_better + 0.1
    w_ucb = 1.0 - w_pes - w_ei
    
    # Ensure weights are positive and sum to 1
    w_pes = np.clip(w_pes, 0.1, 0.8)
    w_ei = np.clip(w_ei, 0.05, 0.6)
    w_ucb = np.clip(w_ucb, 0.05, 0.5)
    total_w = w_pes + w_ei + w_ucb
    w_pes /= total_w
    w_ei /= total_w
    w_ucb /= total_w
    
    score = w_pes * pes_norm + w_ei * ei_norm + w_ucb * ucb_norm
    
    # 8. Add small exploration bonus for high-uncertainty regions
    sigma_rank = np.argsort(np.argsort(sigma_safe)) / n
    score += 0.02 * sigma_rank
    
    # Ensure finite values
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
