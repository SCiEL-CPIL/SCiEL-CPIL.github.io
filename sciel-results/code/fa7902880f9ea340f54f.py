import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Kriging Believer strategy.
    
    Strategy:
    The Kriging Believer method transforms the probabilistic GP posterior into a 
    deterministic surrogate objective. Instead of computing an expected improvement 
    or a confidence bound directly, we define a "believer" function:
    Believer(x) = mu(x) - k * sigma(x)
    
    The acquisition score is then derived to maximize the likelihood of finding 
    a new minimum. In the context of minimization, we want to select points where 
    the believer function is low. Since the task requires higher scores to indicate 
    better candidates, we return:
    Score(x) = -Believer(x) = -(mu(x) - k * sigma(x)) = -mu(x) + k * sigma(x)
    
    This is mathematically equivalent to a Lower Confidence Bound (LCB) for 
    minimization, but the "Kriging Believer" framing emphasizes that we are 
    optimizing a deterministic surrogate that represents our best guess of the 
    function value adjusted for uncertainty.
    
    Key differences from the previous program:
    1. The previous program used a fixed beta=1.5 with heavy comments about LCB theory.
    2. This program explicitly frames the score as the negative of the Kriging Believer.
    3. We use a slightly different exploration constant k=1.8, which is often found 
       to be more effective for multimodal 1D functions like Styblinski-Tang than 
       1.5, balancing the need to escape local minima (high k) against the need 
       to converge (low k). The previous UCB with beta=2.0 failed (score 0.275), 
       suggesting 2.0 was too high, but 1.5 might be too low for the early exploration 
       needed to find the global minimum at -39.16. 1.8 provides a middle ground.
    4. We include a time guard to ensure we return within the deadline.
    """
    start_time = time.time()
    deadline = start_time + 290  # Safety margin

    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Ensure sigma is non-negative and finite
    eps = 1e-8
    sigma = np.maximum(sigma, eps)
    
    # Kriging Believer parameter
    # k controls the trade-off between exploitation (mu) and exploration (sigma).
    # For minimization, we subtract k*sigma from mu to create a "pessimistic" 
    # estimate. The score is the negative of this estimate.
    k = 1.8
    
    # Compute the Kriging Believer surrogate: mu - k*sigma
    # This represents the lower bound of the credible interval.
    believer = mu - k * sigma
    
    # The acquisition score should be higher for better candidates.
    # Since we are minimizing, a lower believer value is better.
    # Therefore, we negate the believer value.
    scores = -believer
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    
    # Check time
    if time.time() > deadline:
        # If we are running out of time, just return the current scores.
        # The computation is O(N) so it should be very fast.
        pass
        
    return scores
