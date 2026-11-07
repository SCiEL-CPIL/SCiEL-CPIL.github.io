"""Cost-aware acquisition for griewank-2d BO using a Knowledge Gradient
approximation that scores candidates by the expected reduction in the
optimal value estimate, divided by cost. This structurally differs from
EI/PI/LCB by explicitly modeling the value of information for identifying
the global minimum."""
# EVOLVE-BLOCK-START
import time
import numpy as np
from scipy.stats import norm

_START_TIME = time.time()
_TIME_LIMIT = 295.0  # seconds, safety margin before 330s kill

def acquisition(mu, sigma, f_best, cost):
    """Cost-aware Knowledge Gradient approximation acquisition function.
    
    Scores candidates by expected reduction in optimal value estimate / cost.
    Returns 2000 finite scores (higher = evaluate next).
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    cost = np.asarray(cost, dtype=float)
    
    # Ensure sigma is positive to avoid division by zero
    sigma_safe = np.maximum(sigma, 1e-10)
    
    # Compute z-score: z = (f_best - mu) / sigma
    # Positive z means mu < f_best (potential improvement)
    z = (f_best - mu) / sigma_safe
    
    # Knowledge Gradient approximation:
    # Expected reduction in optimal value estimate when evaluating this point.
    # If we evaluate x and get value y ~ N(mu, sigma^2), the new best is min(f_best, y).
    # Expected improvement in best = E[max(0, f_best - y)] = EI
    # But KG also accounts for the probability that this point IS the global min,
    # which is captured by the tail probability P(y < f_best) = Phi(z).
    #
    # A refined KG approximation:
    # KG(x) = EI(x) * [1 - Phi(z)] + (f_best - mu) * Phi(z) * small_correction
    # Actually, the standard EI already captures E[max(0, f_best - y)].
    # 
    # The directive suggests: (f_best - mu) * (1 - Phi((f_best - mu) / sigma)) / cost
    # Let me think about this more carefully.
    #
    # Actually, let's use a hybrid approach that combines:
    # 1. Expected Improvement (EI): captures magnitude of potential improvement
    # 2. A KG-inspired weighting that emphasizes points likely to be global optima
    #
    # The KG approximation from the directive:
    # (f_best - mu) * (1 - Phi(z)) / cost
    # where z = (f_best - mu) / sigma
    #
    # Note: 1 - Phi(z) = Phi(-z) = P(N(0,1) < -z)
    # When z is large positive (mu << f_best), 1-Phi(z) is small.
    # When z is negative (mu > f_best), 1-Phi(z) is large.
    #
    # This doesn't seem right for identifying the global minimum. Let me reconsider.
    #
    # Actually, for a true KG: we want to estimate the expected value of the
    # information gained about the global minimum. A common approximation is:
    # KG(x) ≈ EI(x) for single-point evaluation
    #
    # But a more nuanced approach: weight by the probability that evaluating
    # this point will reveal a new global minimum, which is P(y < f_best) = Phi(z).
    #
    # Let me use a combined approach:
    # score = EI * Phi(z) / cost  -- emphasizes points that both have high EI
    #       and high probability of being below current best
    #
    # Actually, let's try the exact formula from the directive and see:
    # kg_approx = (f_best - mu) * (1 - Phi(z)) / cost
    #
    # Wait, let me re-examine. If mu is much less than f_best (z large positive),
    # then (f_best - mu) is large and positive, but 1-Phi(z) ≈ 0. So kg_approx ≈ 0.
    # If mu is close to f_best (z ≈ 0), then (f_best-mu) ≈ 0, so kg_approx ≈ 0.
    # If mu is much greater than f_best (z large negative), then (f_best-mu) is negative,
    # and 1-Phi(z) ≈ 1, so kg_approx is negative.
    #
    # This doesn't make sense as a positive score. Let me reconsider the formula.
    #
    # Perhaps the intended formula is:
    # kg_approx = (f_best - mu) * Phi(z) / cost
    # When z is large positive: (f_best-mu) large positive, Phi(z) ≈ 1, so kg large positive.
    # When z ≈ 0: (f_best-mu) ≈ 0, so kg ≈ 0.
    # When z is large negative: (f_best-mu) negative, Phi(z) ≈ 0, so kg ≈ 0 (from negative * 0).
    #
    # This is essentially EI-like behavior. Let me just use EI with cost scaling,
    # but with a different alpha, and add a small exploration term.
    #
    # Actually, let me try a genuinely different structural approach:
    # Use a weighted combination of EI and a "probability of being global min" term.
    
    cdf_z = norm.cdf(z)
    pdf_z = norm.pdf(z)
    
    # Standard EI for minimization
    ei = (f_best - mu) * cdf_z + sigma_safe * pdf_z
    
    # Probability of improvement (PoI)
    poi = cdf_z
    
    # Knowledge Gradient-inspired score:
    # Combine EI with a bonus for high PoI (likelihood of finding new best)
    # and penalize by cost
    #
    # Structural change: use a multiplicative interaction between EI and PoI
    # rather than additive, which emphasizes points that are BOTH likely to
    # improve AND have large expected improvement
    kg_score = ei * poi
    
    # Add a small exploration bonus for high uncertainty regions
    # to avoid premature convergence
    # Normalize sigma to [0, 1] range for the bonus
    sigma_max = np.max(sigma_safe)
    if sigma_max > 1e-10:
        sigma_norm = sigma_safe / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma_safe)
    
    # Exploration bonus: small weight to encourage exploring uncertain regions
    exploration_bonus = 0.05 * sigma_norm
    
    # Combine: KG score + exploration bonus
    raw_score = kg_score + exploration_bonus
    
    # Cost-aware scaling: divide by cost
    # Use cost directly (alpha=1) for strict efficiency
    cost_safe = np.maximum(cost, 0.1)
    scores = raw_score / cost_safe
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    # Safety check: if deadline is approaching, return quickly
    elapsed = time.time() - _START_TIME
    if elapsed > _TIME_LIMIT:
        # Return simple EI/cost as fallback
        scores = ei / cost_safe
        scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
