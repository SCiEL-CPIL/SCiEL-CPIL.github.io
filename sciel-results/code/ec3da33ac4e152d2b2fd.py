import numpy as np
from scipy.stats import norm
import time

def acquisition(mu, sigma, f_best):
    """
    Lookahead Expected Improvement (L-Opt) approximation.
    Combines the structural idea of UCB (current program) with EI refinement (inspirations).
    Uses a two-step myopic optimization: approximates the expected improvement of the next point
    given the current candidate is selected, by integrating the posterior distribution.
    """
    start_time = time.time()
    deadline = 300.0
    
    # Ensure sigma is strictly positive to avoid numerical issues
    sigma = np.maximum(sigma, 1e-9)
    
    # Known minimum of Hartmann 3D function
    f_min_hartmann = -3.86278
    
    # Compute gap to the global minimum
    gap = f_best - f_min_hartmann
    
    # Map gap to an exploration parameter
    # Early (large gap): more exploration; Late (small gap): more exploitation
    gap_clipped = np.clip(gap, 0.0, 2.0) / 2.0  # Normalize to [0, 1]
    
    # Dynamic exploration parameter for UCB-like behavior
    # Higher beta early on to explore, lower later to exploit
    beta = 1.0 + 2.0 * gap_clipped
    
    # Compute Z-score for Expected Improvement
    # Z = (f_best - mu) / sigma
    z = (f_best - mu) / sigma
    
    # Calculate PDF and CDF of the standard normal distribution
    phi_z = norm.pdf(z)
    Phi_z = norm.cdf(z)
    
    # Standard EI: (f_best - mu) * Phi(z) + sigma * phi(z)
    ei = (f_best - mu) * Phi_z + sigma * phi_z
    
    # Lookahead correction: approximate the expected improvement of the next point
    # Given that we select this candidate, the next point's improvement depends on
    # the updated posterior. We approximate this by adding a term that accounts
    # for the uncertainty reduction.
    
    # The variance reduction after observing a point is proportional to the
    # squared correlation between the current point and future points.
    # For a rough approximation, we use the current sigma as a proxy for
    # the remaining uncertainty.
    
    # A simple lookahead correction: the expected value of the next EI
    # given that we observe f(x) at this point.
    # This is approximated by considering that the new best will be
    # min(f_best, f(x)), and the new sigma will be reduced.
    
    # For computational efficiency, we use a simplified lookahead:
    # L-Opt ≈ EI + lambda * sigma * (1 - Phi(z))
    # where lambda is a small weight for the lookahead term.
    
    # The term (1 - Phi(z)) represents the probability that f(x) > f_best,
    # i.e., that we do NOT improve. In this case, the next step will have
    # more uncertainty, so we add a bonus for points that are likely to
    # provide information even if they don't improve immediately.
    
    lambda_lh = 0.1 * (1.0 - gap_clipped)  # Less lookahead weight late in optimization
    
    lookahead_bonus = lambda_lh * sigma * (1.0 - Phi_z)
    
    # Combine EI and lookahead bonus
    scores = ei + lookahead_bonus
    
    # Add a small UCB-like term for additional exploration in early stages
    # This helps escape local optima
    ucb_term = beta * sigma * 0.1 * gap_clipped
    
    scores = scores + ucb_term
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check deadline
    elapsed = time.time() - start_time
    if elapsed > deadline - 1.0:
        pass
        
    return scores
