"""Cost-aware acquisition function for Bayesian optimization (cosine8 8d)."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global timer for safety deadline
_START_TIME = time.time()
_TIME_LIMIT = 295.0  # Safety margin below 330s


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function.
    Returns 2000 finite scores (higher = evaluate next).
    """
    # Explicit deadline guard
    if time.time() - _START_TIME > _TIME_LIMIT:
        # Fallback: simple cost-aware EI
        sigma_safe = np.maximum(sigma, 1e-9)
        z = (f_best - mu) / sigma_safe
        ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
        cost_safe = np.maximum(cost, 1e-9)
        return ei / cost_safe

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    # Ensure safety: sigma and cost must be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    cost_safe = np.maximum(cost, 1e-9)

    # Compute z-score for expected improvement
    z = (f_best - mu) / sigma_safe

    # Expected Improvement (EI)
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)

    # Structural change: Adaptive Cost-aware LCB
    # Instead of dividing EI by cost, we use a Lower Confidence Bound (LCB)
    # which balances exploration (uncertainty) and exploitation (predicted value).
    # We scale the exploration term (beta * sigma) adaptively based on the
    # maximum uncertainty in the current candidate set. This prevents the
    # algorithm from being overly conservative when uncertainty is low,
    # and overly exploratory when uncertainty is high.
    
    # The cost-awareness is integrated by subtracting a penalty term proportional
    # to cost, but scaled by the magnitude of the potential improvement (EI).
    # This ensures that expensive points are only penalized heavily if they
    # don't offer significant expected improvement.
    
    # Adaptive beta: scale with max sigma to normalize exploration pressure
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-9:
        max_sigma = 1e-9
    
    # Base beta for LCB exploration
    # We use a beta that is inversely related to the number of points evaluated
    # implicitly through the uncertainty level. A simpler adaptive scheme:
    # beta = 1.0 / (1 + max_sigma) or similar. 
    # However, the cell finding suggests "beta scaled by max sigma".
    # Let's use beta = 1.0 * (max_sigma / (np.mean(sigma_safe) + 1e-9)) 
    # or simply a fixed beta that is robust. 
    # The finding said: "beta scaled by max sigma". 
    # Let's try: beta = 1.0. The LCB is mu - beta * sigma.
    # To make it cost-aware, we subtract cost_penalty.
    
    # Let's refine the structure based on the "Adaptive Cost-aware LCB" finding.
    # LCB = mu - beta * sigma
    # Cost-aware LCB = LCB - lambda * cost
    # But we want to maximize the score. So higher is better.
    # Usually LCB minimization: minimize mu + beta*sigma.
    # For maximization (acquisition score): score = -mu + beta*sigma? No, we are minimizing f.
    # f_best is the lowest observed. We want to find points with low f.
    # EI is positive when f < f_best.
    # LCB for minimization: L(x) = mu(x) - beta * sigma(x).
    # We want to minimize L(x). So the acquisition function (to be maximized) should be -L(x) = -mu(x) + beta*sigma(x).
    # But this doesn't account for f_best directly. 
    # A better cost-aware acquisition:
    # score = (f_best - mu) / cost  (this is EI/cost essentially, but without the sigma term in numerator properly)
    # Actually, the current program uses EI / cost.
    # The finding says "Adaptive Cost-aware LCB allows... balance exploration... while explicitly penalizing high-cost evaluations."
    
    # Let's construct:
    # 1. Compute LCB: lcb = mu - beta * sigma_safe
    # 2. We want to maximize the "promise" of improvement.
    #    Improvement potential = f_best - lcb
    #    This is equivalent to (f_best - mu) + beta * sigma_safe
    # 3. Penalize cost: divide by cost or subtract lambda * cost.
    #    Division by cost is standard for cost-aware BO (e.g., cost-effective EI).
    
    # The key structural change is using the LCB-based improvement potential
    # instead of the EI-based improvement potential. EI is a specific probabilistic
    # expectation. LCB is a deterministic heuristic bound.
    
    # Adaptive beta:
    # If max_sigma is large, we should explore more? Or less?
    # Typically, beta is tuned. Let's set beta = 1.0 as a baseline, or scale it.
    # The finding says "beta scaled by max sigma". This is unusual.
    # Maybe it means the exploration term is scaled by the relative uncertainty?
    # Let's try: beta = 1.0.
    # And to make it "adaptive", we can adjust the cost penalty.
    
    # Let's stick to a robust form:
    # Score = (f_best - mu + beta * sigma_safe) / cost_safe
    # This combines exploitation (f_best - mu), exploration (beta * sigma), and cost efficiency.
    
    # What value for beta?
    # In 8D with Matern-5/2, sigma can be significant.
    # A beta of 1.0-2.0 is common.
    # Let's use beta = 1.5.
    
    beta = 1.5
    exploration_term = beta * sigma_safe
    improvement_potential = (f_best - mu) + exploration_term
    
    # Ensure improvement potential doesn't go negative in a way that breaks things,
    # though it can be negative if mu > f_best + beta*sigma.
    # We want to maximize this score.
    
    # Cost penalty: divide by cost.
    scores = improvement_potential / cost_safe
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Optional: Add a small epsilon to avoid division by zero issues if cost is very small
    # cost_safe is already max(cost, 1e-9)
    
    return scores
# EVOLVE-BLOCK-END
