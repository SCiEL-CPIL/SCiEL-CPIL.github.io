"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
import time

# Deadline guard
_START_TIME = time.time()
_DEADLINE = 330.0
_MARGIN = 5.0

def _time_check():
    """Check if we're close to the deadline."""
    return (time.time() - _START_TIME) > (_DEADLINE - _MARGIN)

def acquisition(mu, sigma, f_best):
    """
    Adaptive UCB acquisition function for Bayesian optimization of Ackley 2D.
    
    Uses a pure UCB structure: score = -mu + beta * sigma.
    The exploration parameter beta is adapted based on f_best to balance
    exploration (early, high f_best) and exploitation (late, low f_best).
    
    mu, sigma: shape (2000,), posterior mean and std in objective units.
    f_best: lowest observed value.
    
    Returns 2000 finite scores, higher means evaluate next.
    """
    # Deadline guard: return simple fallback if running out of time
    if _time_check():
        return 1.0 * sigma - mu

    n = len(mu)

    # Ensure sigma is positive and finite
    sigma = np.maximum(sigma, 1e-12)
    sigma = np.where(np.isfinite(sigma), sigma, 1e-12)
    mu = np.where(np.isfinite(mu), mu, 0.0)

    # Adaptive beta strategy:
    # - When f_best is high (early exploration phase), we want a larger beta to
    #   push the search into unexplored regions, preventing premature convergence
    #   to local minima of the Ackley function.
    # - When f_best is low (approaching the global minimum 0), we want a smaller
    #   beta to focus on the best known regions and refine the solution.
    
    # Ackley 2D on [-5,5]^2 has a global min of 0.
    # Typical initial f_best from uniform random starts might be ~1-5.
    # We map f_best to a beta range [0.5, 2.0].
    
    # Use an exponential decay mapping for smooth transition.
    # If f_best is large, exp(-f_best/2) is small -> beta ~ 2.0
    # If f_best is small, exp(-f_best/2) is large -> beta ~ 0.5
    # Let's check the scale: 
    # f_best = 5 -> exp(-2.5) ~ 0.08 -> beta = 0.5 + 1.5*0.08 ~ 0.62 (Too low for start?)
    
    # Let's reverse the intuition or adjust the scale.
    # We want HIGH beta when f_best is HIGH.
    # So we want the decay term to be SMALL when f_best is HIGH? No, that gives low beta.
    # We want beta = beta_min + (beta_max - beta_min) * weight
    # where weight is 1 when f_best is high, 0 when f_best is low.
    
    # Let's use a sigmoid or simple clamp.
    # Scale of f_best: Ackley max is ~20, but typical start is ~1-3.
    # Let's use a linear mapping clipped, or a smoothstep.
    
    # Simple robust approach:
    # beta = 0.5 + 1.5 * clip(f_best / 5.0, 0, 1)
    # If f_best = 5, beta = 2.0
    # If f_best = 0, beta = 0.5
    # If f_best = 2.5, beta = 1.25
    
    # This is simple and effective. Pure UCB with adaptive beta.
    beta = 0.5 + 1.5 * np.clip(f_best / 5.0, 0.0, 1.0)
    
    # Calculate UCB scores
    scores = -mu + beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
# EVOLVE-BLOCK-END
