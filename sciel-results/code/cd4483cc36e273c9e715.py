"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time
from scipy.stats import norm

# Global start time for deadline guard
_START_TIME = time.time()

def acquisition(mu, sigma, f_best):
    """
    Computes acquisition scores using a hybrid of Expected Improvement (EI)
    and a dynamic, Pareto-frontier guided exploration term.
    
    The core idea is to combine the robust exploitation of EI with a geometric
    exploration term that prioritizes high-uncertainty regions far from the
    current best prediction. This helps escape local optima in the Ackley function.
    
    For a Gaussian posterior with mean mu and std sigma, and current best f_best:
      z = (f_best - mu) / sigma
      EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    
    The exploration term is based on the distance in prediction space from the
    candidate with the lowest predicted mean (the 'anchor'). This encourages
    exploration in regions that are both uncertain and distinct from the current
    exploitation focus.
    """
    # Deadline guard: 330s limit. Return immediately if close to limit.
    # 10s safety margin.
    if time.time() - _START_TIME > 320:
        return np.zeros_like(mu)
    
    # Clip sigma to avoid division by zero and numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # Compute z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    
    # Compute EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # Use scipy.stats for numerical stability
    Phi_z = norm.cdf(z)
    phi_z = norm.pdf(z)
    
    ei = (f_best - mu) * Phi_z + sigma_safe * phi_z
    
    # Ensure EI is non-negative (it should be, but numerical errors can occur)
    ei = np.maximum(ei, 0.0)
    
    # --- Dynamic Exploration Term ---
    # Identify the 'anchor': the candidate with the lowest predicted mean.
    # This serves as the current exploitation focus.
    anchor_idx = np.argmin(mu)
    anchor_mu = mu[anchor_idx]
    
    # Compute the normalized distance from the anchor in prediction space.
    # We use the difference in predicted means (mu - anchor_mu) as a proxy
    # for distance. Since we are minimizing, points with much higher mu
    # are 'far' from the best prediction. We normalize by the range of mu
    # to make the term scale-invariant.
    mu_range = np.max(mu) - np.min(mu)
    if mu_range < 1e-12:
        mu_range = 1e-12
        
    # Distance from anchor: (mu - anchor_mu) / mu_range
    # This is 0 for the anchor and 1 for the worst prediction.
    dist_from_anchor = (mu - anchor_mu) / mu_range
    
    # The exploration score combines uncertainty (sigma) and distance.
    # High sigma and high distance from the anchor get a high score.
    # We scale this term to be comparable to EI.
    # A simple way is to normalize the exploration term.
    max_sigma = np.max(sigma_safe)
    if max_sigma < 1e-12:
        max_sigma = 1e-12
        
    exploration = sigma_safe * dist_from_anchor / max_sigma
    
    # Combine EI and Exploration.
    # We use a weighted sum. The weight for exploration can be tuned,
    # but a fixed weight of 0.5 is a reasonable starting point.
    # The EI term is typically larger in magnitude, so we might need to
    # scale the exploration term up or down. Let's try a weight of 1.0
    # for the exploration term, as it is already normalized.
    scores = ei + exploration
    
    # Ensure scores are finite
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
# EVOLVE-BLOCK-END
