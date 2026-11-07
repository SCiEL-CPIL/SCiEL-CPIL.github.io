"""Acquisition function for Bayesian optimization of Weierstrass 1D."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm

# Reference minimum of the Weierstrass function on [-0.5, 0.5] is 0.
F_MIN = 0.0

def acquisition(mu, sigma, f_best):
    """
    Compute acquisition scores using an adaptive blend of Expected Improvement (EI)
    and Upper Confidence Bound (UCB) logic.
    
    The exploration weight is adaptive: it is high when the current best f_best
    is far from the known minimum (0), encouraging exploration of the rugged
    Weierstrass landscape to escape local minima. As f_best approaches 0, the
    weight shifts towards exploitation (EI) to refine the solution.
    
    Parameters:
    -----------
    mu : ndarray, shape (2000,)
        Mean of the GP posterior.
    sigma : ndarray, shape (2000,)
        Standard deviation of the GP posterior.
    f_best : float
        The best (lowest) objective value observed so far.
        
    Returns:
    --------
    scores : ndarray, shape (2000,)
        Acquisition scores. Higher is better.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is positive and finite to avoid numerical issues
    sigma_safe = np.maximum(sigma, 1e-12)
    
    # 1. Compute Expected Improvement (EI) for minimization
    # EI = (f_best - mu) * Phi(z) + sigma * phi(z)
    # where z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma_safe
    # Clip z to avoid numerical underflow/overflow in norm.cdf/pdf
    z_clipped = np.clip(z, -10, 10)
    
    ei = (f_best - mu) * norm.cdf(z_clipped) + sigma_safe * norm.pdf(z_clipped)
    # EI is theoretically non-negative, but numerical errors can make it slightly negative
    ei = np.maximum(ei, 0.0)
    
    # 2. Compute UCB-like score (Exploration)
    # We use a standard deviation multiplier that is part of the adaptive logic.
    # A simple UCB is mu + kappa * sigma. However, since we are minimizing,
    # we want to pick points with low mu and high sigma.
    # Let's define an exploration score that rewards high sigma and low mu.
    # ucb_score = -mu + kappa * sigma
    # We will normalize or scale this to be comparable to EI if needed, 
    # but since we blend them, we can just use the raw values if scales are similar.
    # EI is in objective units. UCB term -mu + k*sigma is also in objective units.
    
    # 3. Adaptive Exploration Weight (Beta)
    # We want beta to be high when f_best is large (far from 0) and low when f_best is small.
    # The Weierstrass function on [-0.5, 0.5] has a maximum value around 1.0 (depending on parameters, but typically < 1).
    # Let's assume a range of [0, 1] for the objective values.
    # beta = exp(-f_best / scale) or similar.
    # If f_best is 1.0, beta should be ~1.0 (mostly explore).
    # If f_best is 0.0, beta should be ~0.0 (mostly exploit).
    # Let's use a logistic-like or exponential decay.
    # scale controls how quickly we switch from exploration to exploitation.
    # A scale of 0.5 means if f_best is 0.5, beta is exp(-1) ~ 0.37.
    # If f_best is 0.1, beta is exp(-0.2) ~ 0.82? No, that's still high exploration.
    # We want to switch faster.
    # Let's try: beta = exp(-f_best / 0.1)
    # f_best=0.5 -> exp(-5) ~ 0.006 (exploit)
    # f_best=0.1 -> exp(-1) ~ 0.37 (balanced)
    # f_best=0.0 -> 1.0 (explore? No, if f_best is 0, we are done. But f_best won't be exactly 0).
    # Actually, if f_best is very small, we want to exploit.
    # So beta should go to 0 as f_best -> 0.
    # beta = exp(-f_best / 0.05)
    # f_best=0.1 -> exp(-2) ~ 0.13
    # f_best=0.01 -> exp(-0.2) ~ 0.82 -> Wait, this is still high exploration when close to optimum.
    # We want beta to be LOW when f_best is LOW.
    # So beta = 1 - exp(-f_best / 0.05)?
    # f_best=0 -> beta=0 (Pure EI)
    # f_best=0.1 -> beta = 1 - exp(-2) = 0.86 (Mostly Explore)
    # f_best=0.5 -> beta = 1 - exp(-10) ~ 1.0 (Pure Explore)
    
    # This looks correct. High f_best -> High Beta (Explore). Low f_best -> Low Beta (Exploit).
    
    beta = 1.0 - np.exp(-f_best / 0.05)
    # Clamp beta to [0, 1]
    beta = np.clip(beta, 0.0, 1.0)
    
    # 4. Compute UCB Score
    # We need to choose a kappa for UCB. 
    # Standard UCB: mu + kappa * sigma.
    # For minimization, we want to minimize this, so we maximize -mu + kappa * sigma.
    # Let's use kappa = 1.0 or 2.0. Given the ruggedness, 2.0 might be better for early stages.
    # But since beta handles the weight, a moderate kappa is fine.
    # Let's use kappa = 1.5.
    kappa = 1.5
    ucb_score = -mu + kappa * sigma_safe
    
    # 5. Combine Scores
    # We need to ensure EI and UCB are on similar scales.
    # EI is typically small when close to optimum. UCB term -mu can be large (negative).
    # This might cause issues if we just add them.
    # However, if beta is high, we rely on UCB. If beta is low, we rely on EI.
    # To make them comparable, we can normalize or just trust the relative ordering.
    # EI is always >= 0. UCB score can be negative.
    # If we mix them, the negative UCB scores will dominate if beta is high and mu is large positive?
    # Wait, mu is the predicted mean. If the function is Weierstrass, values are > 0.
    # So -mu is negative.
    # EI is positive.
    # This sign mismatch is problematic for direct addition.
    
    # Alternative: Use a "Min-Regret" or "PI" (Probability of Improvement) approach?
    # Or, normalize EI to [0,1] and UCB to [0,1]?
    
    # Let's stick to EI but add an exploration bonus that is always positive.
    # Exploration bonus = kappa * sigma.
    # Score = EI + beta * (kappa * sigma)
    # This ensures the score is always positive and higher is better.
    # EI drives improvement. beta * kappa * sigma drives exploration.
    # When beta is high (early), the sigma term dominates, picking high uncertainty points.
    # When beta is low (late), the EI term dominates, picking points likely to improve.
    
    # Let's refine kappa.
    # If sigma is small, the bonus is small.
    # If sigma is large, the bonus is large.
    # This is effectively UCB but added to EI.
    
    scores = ei + beta * (kappa * sigma_safe)
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
