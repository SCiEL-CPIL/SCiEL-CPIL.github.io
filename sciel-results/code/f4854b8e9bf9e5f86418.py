# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm
import time

# Global deadline guard
_START_TIME = time.time()
_DEADLINE = _START_TIME + 290.0  # Safety margin before 300s limit

def acquisition(mu, sigma, f_best, cost):
    """
    Cost-aware acquisition function for Bayesian optimization.
    Uses a Knowledge Gradient (KG) approximation with cost discounting.
    """
    # Check deadline
    if time.time() > _DEADLINE:
        mu = np.asarray(mu, dtype=np.float64)
        sigma = np.asarray(sigma, dtype=np.float64)
        cost = np.asarray(cost, dtype=np.float64)
        sigma = np.maximum(sigma, 1e-9)
        cost = np.maximum(cost, 1e-9)
        # Fallback: simple EI / cost
        z = (f_best - mu) / sigma
        ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
        ei = np.maximum(ei, 0.0)
        scores = ei / cost
        scores = np.where(np.isfinite(scores), scores, 0.0)
        return scores
    
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    cost = np.asarray(cost, dtype=np.float64)
    
    sigma = np.maximum(sigma, 1e-9)
    cost = np.maximum(cost, 1e-9)
    
    # Knowledge Gradient approximation:
    # KG(x) = E_x[f(x)] - E_x[min(f(x), f_best)]
    # For a single candidate, this simplifies to:
    # KG(x) = (f_best - mu) * Phi(z) + sigma * phi(z) - sigma * (1 - Phi(z)) * 0
    # Actually, the exact KG for one step is:
    # KG(x) = E[f(x)] - E[min(f(x), f_best)]
    # = mu - [mu * Phi(z) + (f_best - sigma) * (1 - Phi(z)) + sigma * phi(z)]  -- Wait, let's derive properly.
    
    # Let Y ~ N(mu, sigma^2). We want E[Y] - E[min(Y, f_best)].
    # E[Y] = mu
    # E[min(Y, f_best)] = ∫_{-inf}^{f_best} y * p(y) dy + f_best * P(Y > f_best)
    # = mu * Phi(z) + sigma * phi(z) + f_best * (1 - Phi(z))
    #   where z = (f_best - mu) / sigma
    #   and using the identity: ∫_{-inf}^{a} y * N(mu, sigma^2) dy = mu * Phi(z) + sigma * phi(z)
    #   Wait, the standard identity is:
    #   ∫_{-inf}^{z} t * phi(t) dt = -phi(z)
    #   Let t = (y - mu)/sigma => y = mu + sigma*t, dy = sigma*dt
    #   ∫_{-inf}^{f_best} y * (1/sigma) * phi((y-mu)/sigma) dy
    #   = ∫_{-inf}^{z} (mu + sigma*t) * phi(t) dt
    #   = mu * Phi(z) + sigma * ∫_{-inf}^{z} t * phi(t) dt
    #   = mu * Phi(z) - sigma * phi(z)
    #   So E[min(Y, f_best)] = mu * Phi(z) - sigma * phi(z) + f_best * (1 - Phi(z))
    
    # KG = mu - [mu * Phi(z) - sigma * phi(z) + f_best * (1 - Phi(z))]
    #    = mu * (1 - Phi(z)) + sigma * phi(z) - f_best * (1 - Phi(z))
    #    = (mu - f_best) * (1 - Phi(z)) + sigma * phi(z)
    #    = -(f_best - mu) * (1 - Phi(z)) + sigma * phi(z)
    #    = -(f_best - mu) * Phi(-z) + sigma * phi(z)
    
    # Note: EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # So KG is different. KG rewards points where there's a high chance of being the new minimum.
    
    z = (f_best - mu) / sigma
    # Use stable computation for phi(z) and Phi(z)
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # KG = (mu - f_best) * (1 - Phi_z) + sigma * phi_z
    kg = (mu - f_best) * (1.0 - Phi_z) + sigma * phi_z
    
    # Clip to non-negative (KG should be non-negative theoretically, but numerical errors)
    kg = np.maximum(kg, 0.0)
    
    # Cost-aware: discount by sqrt(cost) to balance exploration
    scores = kg / np.sqrt(cost)
    
    # Add a small exploration bonus based on uncertainty
    exploration_bonus = sigma * 0.05
    
    scores = scores + exploration_bonus
    
    # Ensure finite values
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
