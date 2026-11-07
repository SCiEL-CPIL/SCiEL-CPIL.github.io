# EVOLVE-BLOCK-START
import numpy as np
import time


def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function using a Bayesian model-evidence-inspired
    score that balances exploitation of low means against the penalty for
    high uncertainty, with adaptive exploration strength based on progress.
    
    Args:
        mu: Array of shape (N,) with GP mean predictions.
        sigma: Array of shape (N,) with GP standard deviations.
        f_best: Scalar, the best observed function value so far.
        cost: Array of shape (N,) with evaluation costs for each candidate.
        
    Returns:
        Array of shape (N,) with scores. Higher score is better.
    """
    start_time = time.time()
    deadline = start_time + 290.0

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    f_best_val = float(f_best)
    N = len(mu)

    # Numerical safety
    sigma_safe = np.maximum(sigma, 1e-10)
    cost_safe = np.maximum(cost, 1e-10)

    # Progress measure: how close are we to the optimum?
    # Griewank 2D values are in [0, 1], with minimum at 0.
    progress = np.clip(1.0 - f_best_val, 0.0, 1.0)
    
    # Adaptive beta for exploration:
    # - Early on (low progress), use larger beta for more exploration
    # - Later (high progress), use smaller beta for more exploitation
    beta = 0.5 + 2.0 * (1.0 - progress)
    
    # Core evidence-inspired score:
    # The expected log-likelihood of observing improvement can be approximated.
    # For a Gaussian posterior, the "Bayesian model evidence" style score
    # rewards low mean (exploitation) and penalizes high uncertainty (over-exploration).
    # 
    # Score = -mu/sigma - log(sigma) * beta
    # This is equivalent to: exploit low mu (via -mu/sigma) while
    # penalizing high sigma (via -beta * log(sigma)).
    # 
    # However, -mu/sigma can be unstable when sigma is very small.
    # We use a stabilized version:
    #   score = -mu / (sigma + eps) - beta * log(sigma + eps)
    
    eps = 1e-10
    denom = sigma_safe + eps
    
    # Exploitation term: lower mu is better, normalized by uncertainty
    exploit_term = -mu / denom
    
    # Exploration penalty: higher sigma is penalized logarithmically
    # log(sigma) is negative for sigma < 1, so -beta * log(sigma) rewards small sigma
    # But we want to encourage exploration when sigma is large, so we use:
    #   explore_term = -beta * log(1 + sigma)
    # This is smooth, always negative, and grows slowly with sigma.
    explore_term = -beta * np.log1p(sigma_safe)
    
    # Alternative: use the direct evidence approximation
    # -(f_best - mu)/sigma - log(sigma)
    # This is more directly tied to the likelihood of improvement.
    # Let's blend both approaches.
    
    # Evidence-based score: expected log-likelihood of observing f(x) < f_best
    # Under N(mu, sigma^2), this is related to:
    #   E[log P(f < f_best | f)] ≈ -(f_best - mu)/sigma - 0.5*log(2*pi*sigma^2)
    # Dropping constants: -(f_best - mu)/sigma - log(sigma)
    
    evidence_score = -(f_best_val - mu) / denom - np.log(denom)
    
    # LCB-style score: mu - beta*sigma, we negate for "higher is better"
    lcb_score = -mu + beta * sigma_safe
    
    # Blend evidence and LCB:
    # When progress is low, LCB is better for exploration.
    # When progress is high, evidence is better for exploitation.
    w_evidence = 0.3 + 0.7 * progress
    w_lcb = 1.0 - w_evidence
    
    combined = w_evidence * evidence_score + w_lcb * lcb_score
    
    # Cost adjustment: divide by cost to penalize expensive evaluations
    scores = combined / cost_safe
    
    # Deadline guard
    if time.time() > deadline:
        # Fallback: simple cost-normalized exploitation
        scores = -mu / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
# EVOLVE-BLOCK-END
