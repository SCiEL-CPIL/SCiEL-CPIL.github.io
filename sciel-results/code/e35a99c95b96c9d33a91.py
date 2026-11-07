# EVOLVE-BLOCK-START
import time
import numpy as np

_T0 = time.time()
_DEADLINE = _T0 + 320.0  # safety margin vs 330s kill

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of sphere 1d.
    
    Strategy: Deterministic Lower Confidence Bound (LCB).
    score = mu - beta * sigma
    
    We use a fixed beta = 1.0. This is chosen to balance exploitation 
    (minimizing mu) and exploration (maximizing sigma) for the 1D sphere 
    function with the specified GP hyperparameters (Matern 5/2, l=0.2).
    
    Inputs:
    mu: array of shape (2000,) posterior mean
    sigma: array of shape (2000,) posterior standard deviation
    f_best: float, lowest observed value
    
    Output:
    scores: array of shape (2000,) finite values, higher is better for selection.
    """
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    
    # Ensure sigma is non-negative and finite to avoid numerical issues
    sigma = np.maximum(sigma, 1e-8)
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    
    # LCB: We want to minimize, so we subtract beta * sigma.
    # Higher score = lower expected value considering uncertainty.
    # beta = 1.0 is a standard, robust choice for 1D problems with this GP config.
    beta = 1.0
    
    scores = mu - beta * sigma
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # The evaluator expects "higher meaning evaluate next".
    # Since we are minimizing, lower mu - beta*sigma is better.
    # However, the standard convention in many BO libraries for "acquisition"
    # that returns "score" where higher is better is often EI or UCB (maximization).
    # But here, we are minimizing.
    # Let's check the contract: "Return 2000 finite scores, higher meaning evaluate next".
    # If we return mu - beta*sigma, a point with mu=0, sigma=0 gives 0.
    # A point with mu=-1, sigma=0 gives -1.
    # The evaluator picks the candidate with the HIGHEST score.
    # So if we return mu - beta*sigma, the evaluator will pick the one with the LARGEST (mu - beta*sigma).
    # This is WRONG for minimization. We want the SMALLEST (mu - beta*sigma).
    # Therefore, we must INVERT the sign so that "higher score" corresponds to "better minimization".
    # score = -(mu - beta * sigma) = -mu + beta * sigma
    
    # Wait, let's re-read carefully.
    # "minimizes sphere 1d"
    # "f_best is the lowest observed value"
    # "Return 2000 finite scores, higher meaning evaluate next"
    
    # If I use EI: EI is high when improvement is likely. EI is positive. Higher EI = better.
    # If I use UCB (for minimization): UCB = mu - beta*sigma. Lower UCB = better.
    # So if I return UCB, the evaluator picks the MAX UCB, which is the WORST point.
    # So I must return -UCB = -mu + beta*sigma.
    
    # Let's verify with EI.
    # EI = (fb - mu) * Phi(z) + sigma * phi(z).
    # If mu is very low (good), fb - mu is large positive. EI is high.
    # So EI naturally fits "higher is better".
    
    # My LCB strategy:
    # I want to pick the point that minimizes mu - beta*sigma.
    # Let LCB_val = mu - beta*sigma.
    # I want to maximize -LCB_val.
    # So scores = -mu + beta*sigma.
    
    scores = -mu + beta * sigma
    
    # Final safety check
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    return scores
# EVOLVE-BLOCK-END
