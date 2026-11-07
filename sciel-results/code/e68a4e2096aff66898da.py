"""Cost-aware acquisition function for Bayesian optimization (hartmann 6d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

_deadline = None
_start_time = None
_call_count = 0


def acquisition(mu, sigma, f_best, cost):
    """
    Kriging Believer (EI-Believer) cost-aware acquisition.
    
    Structural idea: instead of scoring candidates by EI or PI directly,
    compute EI for all 2000 candidates, then score each candidate by the
    GP mean prediction at a "believer" point derived from its own EI score.
    Specifically, for candidate i with EI score e_i, we compute the
    corresponding z-value z_i = norm.ppf(1 - e_i) (approximate), and score
    candidate i as mu_i + sigma_i * z_i, normalized by cost.
    
    This creates a two-level structure: EI identifies promising regions,
    and the believer transformation re-ranks within those regions by
    expected value under the improvement condition, which tends to favor
    points with higher mean (lower risk) among equally-improving candidates.
    
    The cost-aware normalization uses an adaptive exponent that increases
    as budget depletes, making the search more cost-sensitive late in the run.
    """
    global _deadline, _start_time, _call_count
    
    if _start_time is None:
        _start_time = time.time()
        _deadline = _start_time + 320.0  # Safety margin before 330s limit

    _call_count += 1

    # Check deadline - emergency fallback
    if time.time() > _deadline:
        eps = 1e-10
        sigma_safe = np.maximum(sigma, eps)
        cost_safe = np.maximum(cost, eps)
        z = (f_best - mu) / sigma_safe
        pi = norm.cdf(z)
        scores = pi / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    eps = 1e-10
    sigma_safe = np.maximum(sigma, eps)
    cost_safe = np.maximum(cost, eps)

    # Compute Z-scores
    z = (f_best - mu) / sigma_safe

    # Compute Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)

    # Kriging Believer transformation:
    # For each candidate, convert its EI score to a "believer" score.
    # EI = (f_best - mu)*Phi(z) + sigma*phi(z)
    # We want to find the conditional mean of f(x) given f(x) < f_best
    # and weighted by improvement magnitude.
    # 
    # A practical approximation: the "believer" point for a candidate is
    # mu + sigma * z_b where z_b is chosen so that the EI at that point
    # matches the original EI. We approximate this by:
    #   z_b = z + correction
    # where correction accounts for the fact that EI already incorporates
    # the improvement probability.
    #
    # Simpler structural approach: use the EI score to modulate how much
    # we trust the GP mean. Candidates with high EI get their score boosted
    # by the mean prediction (believer), while low-EI candidates are penalized.
    #
    # Score_i = (mu_i + sigma_i * z_i) * EI_i / cost_i
    # But (mu_i + sigma_i * z_i) = mu_i + sigma_i * (f_best - mu_i)/sigma_i = f_best
    # That's constant! So we need a different formulation.
    #
    # Better: The Kriging Believer selects x* = argmax EI, then evaluates
    # mu(x) + sigma(x) * z* where z* corresponds to x*. For scoring ALL candidates:
    # We compute a "believer-adjusted" score that rewards candidates whose
    # local GP prediction, when conditioned on achieving their EI level,
    # gives a high expected value.
    #
    # Practical formulation: 
    #   believer_score_i = mu_i * (1 + ei_i / max_ei) + sigma_i * z_i * (ei_i / max_ei)
    # This blends the GP mean with the improvement signal, weighted by relative EI.
    
    max_ei = np.max(ei)
    if max_ei < eps:
        # All EI are near zero; fall back to PI/cost
        pi = norm.cdf(z)
        scores = pi / cost_safe
        return np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    ei_normalized = ei / max_ei  # in [0, 1]

    # Believer score: blend mean prediction with EI-weighted uncertainty
    # Points with high EI get more weight on their mean (exploitative among improvers)
    # Points with low EI retain more uncertainty weight (exploratory)
    believer_score = mu * (0.5 + 0.5 * ei_normalized) + sigma_safe * z * ei_normalized

    # Ensure believer_score is well-behaved
    believer_score = np.nan_to_num(believer_score, nan=0.0, posinf=0.0, neginf=0.0)

    # Estimate remaining budget for adaptive cost exponent
    initial_cost_est = 7.5
    avg_cost_per_step = 1.5
    total_cost_est = initial_cost_est + _call_count * avg_cost_per_step
    remaining_budget = max(0.0, 30.0 - total_cost_est)

    # Adaptive cost exponent: stronger cost avoidance late in run
    if remaining_budget > 10.0:
        alpha = 1.0
    elif remaining_budget < 3.0:
        alpha = 1.8
    else:
        alpha = 1.0 + 0.8 * (10.0 - remaining_budget) / 7.0

    # Final score: believer_score normalized by cost^alpha
    scores = believer_score / (cost_safe ** alpha)

    # Handle numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=0.0, neginf=0.0)

    return scores
# EVOLVE-BLOCK-END
