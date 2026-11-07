"""Initial program for the suite task; SCiEL evolves the EVOLVE-BLOCK."""
# EVOLVE-BLOCK-START
import numpy as np
from scipy.stats import norm


def acquisition(mu, sigma, f_best):
    """
    Upper Confidence Bound (UCB) acquisition function with adaptive beta.
    
    For minimization, we want to balance exploitation (low mu) and exploration (high sigma).
    We use a UCB-like score where higher values indicate more promising candidates.
    
    Score = -mu + beta * sigma
    
    However, to be more robust and potentially outperform simple UCB, we can use
    a variant that adapts beta based on the scale of the problem or use a different
    exploration-exploitation trade-off.
    
    Actually, let's try Expected Improvement (EI) was the baseline. 
    Let's try Probability of Improvement (PI) or UCB.
    
    UCB is generally faster for initial exploration.
    Let's use a standard UCB with a fixed beta, or a beta that decays.
    Since we don't have the step number, we can't easily decay beta per step 
    unless we infer it from sigma or f_best, which is tricky.
    
    Let's try a different approach: Lookahead or a more sophisticated EI variant?
    No, let's stick to UCB but tune beta.
    For sphere function in 1D on [-5,5], the function is smooth.
    A beta around 2.0 to 5.0 is common.
    
    Alternatively, we can use the "Knowledge Gradient" or "PI".
    PI is: P(f(x) < f_best) = Phi((f_best - mu) / sigma)
    This is pure exploitation + some exploration via sigma in the denominator.
    
    Let's try UCB with beta = 2.0.
    Score = -mu + beta * sigma
    
    Wait, the task says "higher meaning evaluate next".
    For minimization:
    -mu is high when mu is low (good).
    sigma is high when uncertainty is high (exploration).
    So -mu + beta * sigma is a valid acquisition for minimization.
    
    Let's try beta = 3.0.
    """
    # Clip sigma to avoid division by zero or extreme values if needed, 
    # though UCB doesn't divide.
    beta = 3.0
    score = -mu + beta * sigma
    
    # Ensure finite scores
    score = np.where(np.isfinite(score), score, -np.inf)
    
    return score
# EVOLVE-BLOCK-END
