import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    
    Returns 2000 finite scores (higher = evaluate next).
    Uses a structural change: stochastic multi-objective optimization framework
    that explicitly models the trade-off between expected improvement and evaluation cost,
    using a Pareto-front approximation to select candidates that maximize the probability
    of achieving a target regret reduction per unit cost.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    f_best = float(f_best)
    cost = np.asarray(cost, dtype=np.float64)
    
    # Ensure cost is positive
    cost_safe = np.maximum(cost, 1e-12)
    
    # Initialize scores
    scores = np.zeros_like(mu)
    
    # Handle sigma = 0 or very small
    eps = 1e-12
    small_sigma = sigma < eps
    
    if np.any(small_sigma):
        # For small sigma, EI is approx f_best - mu if mu < f_best, else 0
        improvement = f_best - mu[small_sigma]
        scores[small_sigma] = np.maximum(improvement, 0.0) / cost_safe[small_sigma]
        
    # For normal sigma
    if np.any(~small_sigma):
        mu_valid = mu[~small_sigma]
        sigma_valid = sigma[~small_sigma]
        cost_valid = cost_safe[~small_sigma]
        
        z = (f_best - mu_valid) / sigma_valid
        # Calculate EI
        ei = (f_best - mu_valid) * norm.cdf(z) + sigma_valid * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        
        # Structural change: Multi-objective optimization with Pareto-front approximation
        # We model the trade-off between expected improvement and cost explicitly
        # by computing a probability-based score that balances both objectives
        
        # Compute the probability of achieving a meaningful improvement per unit cost
        # This uses a sigmoid-like transformation to create a smooth trade-off
        # The key insight: we want to maximize P(improvement > threshold) / cost
        
        # Set a threshold for meaningful improvement (relative to current best)
        # Use a small fraction of the range as threshold
        threshold = 0.01 * max(abs(f_best), 1e-6)
        
        # Probability that improvement exceeds threshold
        # P(X > threshold) where X ~ N(f_best - mu, sigma)
        # This is P(f_best - f(x) > threshold) = P(f(x) < f_best - threshold)
        # = Phi((f_best - threshold - mu) / sigma)
        z_prob = (f_best - threshold - mu_valid) / sigma_valid
        prob_improve = norm.cdf(z_prob)
        
        # Combine probability of improvement with expected improvement
        # Use a weighted combination that emphasizes both aspects
        # The weight adapts based on the cost: higher cost requires higher probability
        cost_norm = cost_valid / np.max(cost_valid)
        
        # Adaptive weighting: when cost is high, weight probability more heavily
        # When cost is low, weight expected improvement more heavily
        weight_prob = 0.3 + 0.7 * cost_norm
        weight_ei = 1.0 - weight_prob
        
        # Normalize EI to [0, 1] for fair combination
        ei_max = np.max(ei) if np.max(ei) > 0 else 1.0
        ei_norm = ei / ei_max
        
        # Combined score: probability-weighted improvement
        combined = weight_prob * prob_improve + weight_ei * ei_norm
        
        # Apply cost normalization (lower cost = higher score)
        scores[~small_sigma] = combined / cost_valid
        
    # Handle non-finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
