import numpy as np
from scipy.stats import norm
import time

_START_TIME = time.time()

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization (hartmann 3d).
    Combines a dynamically-tuned UCB-style score with normalized EI/PI blending.
    
    mu, sigma, f_best, cost: arrays of shape (2000,)
    Returns: 2000 finite scores (higher = evaluate next)
    """
    # Deadline guard
    if time.time() - _START_TIME > 325:
        # Fallback: simple EI / cost
        sigma_safe = np.maximum(sigma, 1e-10)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        cost_safe = np.maximum(cost, 1e-10)
        scores = ei / cost_safe
        return np.where(np.isfinite(scores), scores, 0.0)

    epsilon = 1e-9
    
    # Ensure inputs are properly shaped and finite
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    f_best = float(f_best)
    
    # Replace non-finite values
    mu = np.where(np.isfinite(mu), mu, f_best)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-9)
    cost = np.where(np.isfinite(cost), cost, 1.0)
    
    # Ensure sigma is positive to avoid division issues
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)
    
    # Compute z-score
    z = (f_best - mu) / sigma_safe
    
    # Expected Improvement (stable computation)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    ei = np.maximum(ei, 0.0)
    
    # Probability of Improvement
    pi = norm.cdf(z)
    
    # --- Dynamic UCB kappa based on remaining budget ---
    # The total budget is 30 units of cost.
    # cost is the cost of evaluating each candidate.
    # f_best is the current best value.
    # We estimate progress by the number of evaluations done so far.
    # However, we don't have the count directly, but we can infer it from the GP state or just use a heuristic.
    # A simpler heuristic: as the search progresses, we want to shift from exploration to exploitation.
    # Since we don't have the exact iteration count, let's look at the spread of sigma.
    # Early on, sigma is high everywhere. Later, it's low near the optimum.
    # Let's use the median sigma as a proxy for "remaining uncertainty" or progress.
    # High median sigma -> early -> high kappa.
    # Low median sigma -> late -> low kappa.
    
    med_sigma = np.median(sigma_safe)
    
    # Hartmann 3d range is roughly [-3.8, 3.0]. Standardized targets might scale this.
    # Let's assume sigma is in objective units.
    # If med_sigma is large, we are uncertain.
    # Map med_sigma to kappa.
    # If med_sigma > 1.0, kappa ~ 2.0 (high exploration)
    # If med_sigma < 0.1, kappa ~ 0.5 (low exploration)
    
    # A smooth transition
    # kappa = 0.5 + 1.5 * exp(-med_sigma)
    # Or linearly scale based on normalized sigma
    # Let's try: kappa = 1.0 + np.exp(-med_sigma)
    
    kappa = 1.0 + np.exp(-med_sigma)
    
    # UCB score for minimization:
    # We want to minimize. So we want low mu and high sigma (uncertainty).
    # Standard UCB for minimization: mu - kappa * sigma? No, that's for finding low values.
    # Wait, UCB for maximization is mu + kappa * sigma.
    # For minimization, we want to find the minimum.
    # The "upper confidence bound" of the minimum?
    # Actually, usually in BO for minimization, we use:
    # score = -mu + kappa * sigma (if we want to maximize the score to find the min)
    # OR score = mu - kappa * sigma (if we want to minimize the score)
    # The task says: "Return 2000 finite scores (higher = evaluate next)"
    # So we need a score where higher is better.
    # If we are minimizing f(x), we want to pick x where f(x) is likely small.
    # A good candidate has low mu (predicted low value) and high sigma (potential for even lower value).
    # So, score = -mu + kappa * sigma?
    # Let's check the inspiration 3:
    # "ucb = mu + kappa * sigma_safe"
    # "ucb_norm = 1.0 - (ucb - ucb_min) / ucb_range" # invert so higher = better
    # This means they treated ucb as a value to minimize, then inverted it.
    # Let's stick to the prompt's strategy: "score = (mu + kappa sigma) / cost"
    # Wait, if we minimize, mu + kappa*sigma is an upper bound on the value? No, it's a confidence interval.
    # If we want to find the minimum, we should look for points where the LOWER bound is low, or where the expected value is low.
    # Actually, the prompt says: "Replace the standard Expected Improvement with a cost-aware Upper Confidence Bound (UCB) formulation, specifically score = (mu + kappa sigma) / cost"
    # This formula (mu + kappa sigma) is typically for MAXIMIZATION.
    # If we are MINIMIZING, we should use (mu - kappa sigma) or (-mu + kappa sigma).
    # Let's re-read carefully: "minimizes hartmann 3d".
    # If I use mu + kappa sigma, I am maximizing the upper bound. This encourages exploring high uncertainty areas, but also high mean areas.
    # For minimization, high mean is BAD.
    # So, for minimization, the UCB score (higher is better) should be:
    # score = -mu + kappa * sigma
    # This rewards low mu (good) and high sigma (exploration).
    
    ucb_score = -mu + kappa * sigma_safe
    
    # Normalize ucb_score
    ucb_min = np.min(ucb_score)
    ucb_max = np.max(ucb_score)
    ucb_range = ucb_max - ucb_min + 1e-9
    ucb_norm = (ucb_score - ucb_min) / ucb_range
    
    # Normalize EI and PI to [0, 1] range for blending
    ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
    pi_max = np.max(pi) if np.max(pi) > 0 else 1.0
    
    ei_norm = ei / ei_max
    pi_norm = pi / pi_max
    
    # Blend
    # We want to combine the UCB exploration with the EI exploitation.
    # EI is already a strong exploitation signal.
    # UCB is a strong exploration signal.
    # Let's weight them.
    # Early on, UCB should dominate. Late on, EI should dominate.
    # We can use the same 'progress' heuristic (med_sigma) to blend.
    # If med_sigma is high (early), weight UCB more.
    # If med_sigma is low (late), weight EI more.
    
    # w_ucb = 0.5 + 0.5 * exp(-med_sigma)  # 1.0 when early, 0.5 when late
    # w_ei  = 1.0 - 0.5 * w_ucb           # 0.5 when early, 0.5 when late? No.
    
    # Let's try:
    # alpha = exp(-med_sigma)  # 1.0 early, 0.0 late
    # blended = (1 - alpha) * ei_norm + alpha * ucb_norm
    # This shifts from EI (late) to UCB (early).
    
    alpha = np.exp(-med_sigma)
    # Clamp alpha to [0, 1]
    alpha = np.clip(alpha, 0.0, 1.0)
    
    # Also include PI?
    # PI is a good compromise.
    # Let's blend EI and PI first, then blend that with UCB.
    ei_pi_blend = 0.5 * ei_norm + 0.5 * pi_norm
    
    blended = (1.0 - alpha) * ei_pi_blend + alpha * ucb_norm
    
    # Cost-aware: divide by cost
    scores = blended / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
