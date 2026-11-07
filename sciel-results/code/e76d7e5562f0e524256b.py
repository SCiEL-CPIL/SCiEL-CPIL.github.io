"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = None

def _ensure_start():
    global _START_TIME
    if _START_TIME is None:
        _START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    _ensure_start()
    
    # Safety margin: return EI if we're getting close to deadline
    elapsed = time.time() - _START_TIME
    if elapsed > 280:
        z = (f_best - mu) / (sigma + 1e-9)
        return (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    n = len(mu)
    
    # Knowledge Gradient approximation
    # The KG is the expected reduction in the minimum value after observing a new point.
    # Approximation: E[min(f_best, X)] where X ~ N(mu, sigma^2)
    # This equals: min(f_best, mu) - sigma * phi((f_best - mu)/sigma) * (1 - Phi((f_best - mu)/sigma))
    # Wait, that's not quite right. Let's derive it properly.
    
    # Let Y = min(f_best, X) where X ~ N(mu, sigma^2)
    # E[Y] = E[Y | X < f_best] * P(X < f_best) + E[Y | X >= f_best] * P(X >= f_best)
    #      = E[X | X < f_best] * Phi(z) + f_best * (1 - Phi(z))
    # where z = (f_best - mu) / sigma
    # E[X | X < f_best] = mu - sigma * phi(z) / Phi(z)
    # So E[Y] = (mu - sigma * phi(z) / Phi(z)) * Phi(z) + f_best * (1 - Phi(z))
    #         = mu * Phi(z) - sigma * phi(z) + f_best * (1 - Phi(z))
    
    # The KG is f_best - E[Y]
    # KG = f_best - mu * Phi(z) + sigma * phi(z) - f_best * (1 - Phi(z))
    #    = f_best * Phi(z) - mu * Phi(z) + sigma * phi(z)
    #    = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    # This is exactly the same as EI! So the simple KG approximation equals EI.
    # We need a better approximation that captures the joint structure.
    
    # A better KG approximation considers that observing one point gives information about others.
    # For 1D GP, we can use the fact that the posterior at other points changes when we observe x.
    
    # Practical KG approximation: 
    # KG(x) ≈ E[max(0, f_best - f(x))] + beta * E[reduction in expected minimum due to updated posterior]
    
    # The second term is hard to compute exactly. Let's use a simpler heuristic:
    # Combine EI with a term that rewards high uncertainty in regions that could contain the global minimum.
    
    # Alternative: Use the "expected improvement of the minimum" which is:
    # E[min(f_best, min_{x in D} f(x))] where D is the candidate set
    # This is approximately: f_best - E[max(0, f_best - min_{x in D} f(x))]
    
    # Let's use a hybrid approach: EI + weighted UCB where the weight adapts to the problem.
    
    z = (f_best - mu) / (sigma + 1e-9)
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # UCB term
    beta = 2.0
    ucb = -mu + beta * sigma
    
    # Adaptive combination: use more UCB when sigma is large relative to improvement potential
    # Normalize EI and UCB
    ei_max = np.max(ei) if np.max(ei) > 0 else 1e-9
    ucb_max = np.max(ucb) if np.max(ucb) > 0 else 1e-9
    
    # Weight based on uncertainty: higher sigma -> more exploration
    sigma_mean = np.mean(sigma)
    sigma_max = np.max(sigma)
    
    if sigma_max > 1e-9:
        sigma_norm = sigma / sigma_max
    else:
        sigma_norm = np.zeros_like(sigma)
    
    # Blend: use EI when confident, UCB when uncertain
    # The blend factor depends on how much improvement is expected
    improvement_potential = f_best - np.min(mu)
    if improvement_potential < 1e-9:
        # Near optimum, exploit
        weights = 1.0
    else:
        # Explore more when improvement potential is high
        weights = 0.5 + 0.5 * np.exp(-sigma_norm * 2)
    
    score = weights * ei + (1 - weights) * ucb
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
# EVOLVE-BLOCK-END
