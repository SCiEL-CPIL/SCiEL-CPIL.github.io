import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Guard against numerical issues
    sigma = np.maximum(sigma, 1e-12)
    
    # z-score for improvement
    z = (f_best - mu) / sigma
    
    # Expected Improvement (EI) - standard formula for minimization
    ei = (f_best - mu) * norm.cdf(z) + sigma * norm.pdf(z)
    
    # Probability of Improvement (POI)
    poi = norm.cdf(z)
    
    # Structural change: Use a "Thompson Sampling"-inspired score combined with EI
    # Instead of a linear combination of normalized terms, we use a score that
    # directly balances the expected improvement with a stochastic exploration term.
    # The exploration term is scaled by the standard deviation and weighted by POI,
    # but we also add a small "optimism" bias to encourage visiting unexplored regions.
    
    # A more aggressive exploration strategy:
    # score = EI + beta * sigma * (1 - poi)
    # Here, (1 - poi) is the probability of NOT improving, which is high in unexplored regions.
    # This encourages exploration of regions where we are uncertain and might find better values.
    
    beta = 0.5
    exploration_term = beta * sigma * (1 - poi)
    
    score = ei + exploration_term
    
    # Ensure scores are finite
    score = np.nan_to_num(score, nan=0.0, posinf=0.0, neginf=0.0)
    
    return score
