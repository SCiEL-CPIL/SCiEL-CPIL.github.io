# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit


def acquisition(mu, sigma, f_best, cost):
    """
    Knowledge Gradient (KG) acquisition with sublinear cost discount.
    
    Structural change: Instead of EI or UCB, we use the Knowledge Gradient,
    which estimates the expected improvement in the *global optimum* (not just
    the current best observation). This is a different modeling family that
    better captures the value of information from each evaluation.
    
    The KG is approximated as:
      KG(x) = E[max(f_best, min(f(x), f_best))] - f_best
    which simplifies to the probability-weighted improvement.
    
    We use a sublinear cost discount (1/sqrt(cost)) to balance exploration
    of high-cost, high-uncertainty regions against budget efficiency.
    """
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        # Fallback: simple EI-based score
        z = (f_best - mu) / sigma
        scores = (f_best - mu) / np.sqrt(cost) + 0.5 * sigma / np.sqrt(cost)
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)

    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)

    # Sublinear cost discount: 1/sqrt(cost) allows exploration of high-cost regions
    cost_discount = 1.0 / np.sqrt(cost)

    # Knowledge Gradient approximation:
    # The KG estimates the expected reduction in the global minimum.
    # For a single candidate x with posterior N(mu, sigma^2):
    #   KG(x) ≈ E[max(0, f_best - f(x))] + E[max(0, E[f_best|x] - f_best)]
    # The first term is standard EI. The second term captures the value of
    # updating our belief about the global minimum.
    
    # Standard EI component (improvement over current best)
    z = (f_best - mu) / sigma
    ei = sigma * (z * norm.cdf(z) + norm.pdf(z))
    ei = np.maximum(ei, 0.0)

    # Value of information component:
    # If evaluating x gives a new minimum, we learn the true minimum is lower.
    # The expected value of this learning is proportional to:
    #   P(f(x) < f_best) * E[f_best - f(x) | f(x) < f_best]
    # which is related to the "expected minimum" improvement.
    
    # A simpler but effective KG approximation:
    # KG ≈ EI + sigma * P(f(x) < f_best) * (1 - P(f(x) < f_best))
    # The second term peaks at P=0.5, rewarding points where we're uncertain
    # whether they beat the current best.
    
    p_improve = norm.cdf(z)
    p_improve = np.clip(p_improve, 0.0, 1.0)
    
    # Value of information: uncertainty about whether x is the new minimum
    voi = sigma * p_improve * (1.0 - p_improve)

    # Combine: EI provides exploitation, VOI provides exploration
    # Both scaled by sublinear cost discount
    w_ei = 1.0
    w_voi = 0.6
    scores = (w_ei * ei + w_voi * voi) * cost_discount

    # Ensure finite
    scores = np.where(np.isfinite(scores), scores, 0.0)

    return scores
# EVOLVE-BLOCK-END
