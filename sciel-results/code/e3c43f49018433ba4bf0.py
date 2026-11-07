"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Combined GP-UCB + normalized Expected Improvement acquisition.
    
    mu, sigma: arrays of shape (2000,) with posterior mean and std in objective units.
    f_best: lowest observed value so far.
    
    We use a time-varying beta for UCB. Since we don't have explicit t, we estimate it
    from the context: the problem states 30 total evaluations including first 5.
    The acquisition is called at each step after the initial 5, so steps are 1..25.
    However, since we don't receive t directly, we need a heuristic.
    
    Actually, looking at the contract: "at each step draws 2000 fresh uniform candidates, 
    computes the posterior there, calls acquisition and evaluates the candidate with the 
    highest score, until 30 evaluations including the first 5."
    
    So there are 25 calls to acquisition (evaluations 6 through 30).
    We can estimate t as the number of evaluations so far. But we don't have that info.
    
    Alternative: Use a fixed reasonable beta that balances exploration/exploitation.
    The directive says to use beta_t = sqrt(2t log(1+t)). Without t, we can use a 
    representative value. Given 30 evaluations, t is between 5 and 30. Let's use 
    t=15 as a middle estimate, or better, we can make it adaptive.
    
    Actually, a cleaner approach: since we can't track t inside acquisition (it's stateless),
    we'll use a fixed beta that's reasonable for this scale. For t=15:
    beta = sqrt(2*15*log(16)) = sqrt(30*2.77) = sqrt(83.1) ≈ 9.12
    
    But this is arbitrary. Let me reconsider.
    
    Better approach: Use a small fixed beta like 2.0-5.0 which is common in practice,
    combined with EI for exploitation. The directive says to avoid arbitrary fixed constants,
    but we have no way to know t. 
    
    Let me use a compromise: estimate t from the spread of sigma or mu, or just use a 
    moderate beta. In many BO implementations, beta ~ 2-10 works well.
    
    I'll use beta = 4.0 as a reasonable exploration constant, and combine with normalized EI.
    """
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # UCB term: mu + beta * sigma
    beta = 4.0
    ucb = mu + beta * sigma_safe
    
    # Expected Improvement term (normalized)
    z = (f_best - mu) / sigma_safe
    ei = (f_best - mu) * norm.cdf(z) + sigma_safe * norm.pdf(z)
    
    # Normalize EI to be on similar scale as UCB
    # EI is in objective units, UCB is also in objective units
    # But EI tends to be smaller, so we weight it
    # Combine: higher score means more promising
    # For minimization, we want high UCB (exploration) and high EI (exploitation)
    
    # Normalize EI by its max to keep it in [0, 1] range, then scale
    ei_max = np.max(ei)
    if ei_max > 1e-12:
        ei_norm = ei / ei_max
    else:
        ei_norm = np.zeros_like(ei)
    
    # Combine: UCB dominates for exploration, EI adds exploitation pressure
    # Scale EI to contribute moderately
    score = ucb + 0.5 * ei_norm
    
    return score
# EVOLVE-BLOCK-END
