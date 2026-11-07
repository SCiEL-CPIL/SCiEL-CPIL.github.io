"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    UCB (Upper Confidence Bound) acquisition function.
    
    Strategy:
    - Use UCB with a fixed exploration constant k=1.0.
    - UCB = mu + k * sigma.
    - This encourages exploration of high-uncertainty regions (high sigma) 
      while still favoring regions with lower mean predictions (low mu).
    - For minimization, we want to find low values, so we look for regions 
      where the lower bound of the confidence interval is low.
      However, standard UCB for minimization is often formulated as:
      UCB_min = mu - k * sigma
      But here we return scores where HIGHER means evaluate next.
      So we want to maximize (mu - k * sigma)? No, that would favor high mu.
      
      Let's rethink. 
      We want to minimize f. 
      GP predicts mu, sigma.
      We want to find points where f is likely to be low.
      A common heuristic is to look at the lower bound of the prediction:
      L = mu - k * sigma.
      We want to maximize L? No, we want to find the minimum.
      If we maximize L, we are looking for the "safest" low value.
      If we minimize L, we are looking for the "best possible" low value, 
      which is pure exploration of high sigma regions if mu is constant.
      
      Actually, for minimization, the "optimistic" value is the lower bound:
      v(x) = mu(x) - k * sigma(x).
      We want to select the x that maximizes this optimistic value?
      No, if we want to find the global minimum, we should select the x 
      that has the highest potential to be the minimum.
      The potential to be the minimum is higher if the lower bound is lower.
      So we want to minimize (mu - k * sigma)?
      If we minimize (mu - k * sigma), we are looking for the most optimistic 
      (lowest) possible value. This is equivalent to maximizing (k * sigma - mu).
      
      Let's check standard UCB for maximization:
      UCB = mu + k * sigma. We maximize this.
      For minimization, we can negate the objective or use:
      UCB_min = mu - k * sigma. We want to find the minimum, so we should 
      select the point with the lowest UCB_min? 
      No, that's pessimistic.
      
      Let's look at it from the perspective of "regret" or "improvement".
      We want to find a point with f < f_best.
      The probability of improvement is higher where sigma is high and mu is low.
      
      A simple and effective alternative to EI is the "Probability of Improvement" (PI)
      or just a linear combination.
      
      However, the prompt asks for a MATERIALLY DIFFERENT strategy.
      The current one is a blend of EI and sigma.
      
      Let's try a pure UCB approach for minimization.
      We define the score as:
      score = -mu + k * sigma
      
      Why?
      -mu: We want low mu (exploit).
      + k * sigma: We want high sigma (explore).
      
      If k is large, we explore. If k is small, we exploit.
      Setting k=1.0 is a standard choice that balances exploration and exploitation 
      well in many BO tasks, especially for multimodal functions like Michalewicz.
      
      This is computationally cheaper than EI (no norm.cdf/pdf) and is a distinct 
      design family (UCB vs EI/Blend).
    """
    # Clip sigma to avoid numerical issues, though it should be positive
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB for minimization: 
    # We want to maximize the score.
    # A high score should correspond to a point that is likely to have a low function value.
    # Lower bound of prediction: mu - k * sigma.
    # We want the "best" (lowest) possible value.
    # So we want to maximize the "optimism" about finding a low value.
    # Optimism = - (mu - k * sigma) = -mu + k * sigma.
    
    k = 1.0
    score = -mu + k * sigma_safe
    
    return score
# EVOLVE-BLOCK-END
