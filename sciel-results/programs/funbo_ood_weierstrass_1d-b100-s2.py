import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition function for Bayesian optimization.
    
    Instead of computing a deterministic score like EI or UCB, we sample a value
    from the posterior distribution N(mu, sigma^2) for each candidate.
    The candidate with the lowest sampled value is selected for the next evaluation.
    
    Parameters:
    -----------
    mu : array-like, shape (n_candidates,)
        Mean of the GP posterior at the candidate points.
    sigma : array-like, shape (n_candidates,)
        Standard deviation of the GP posterior at the candidate points.
    f_best : float
        The best observed value so far (not strictly used in standard TS, 
        but kept for interface compatibility).
        
    Returns:
    --------
    scores : ndarray, shape (n_candidates,)
        The sampled values. Lower values indicate higher priority for evaluation.
        Note: The task contract says "higher meaning evaluate next", but standard 
        Thompson Sampling minimizes the sampled value. 
        To align with the "higher is better" scoring convention of the acquisition 
        function returning scores where the MAX is chosen, we should return 
        -sampled_values or invert the logic. 
        However, looking at the current program, it returns EI scores where 
        higher EI is better. 
        Let's check the evaluator description: "evaluates the candidate with the 
        highest score". 
        So if we want to pick the point with the lowest predicted f(x), we should 
        return -sampled_values.
    """
    sigma = np.maximum(sigma, 1e-9)
    
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # To ensure the "highest score" corresponds to the "lowest expected f(x)",
    # we return the negative of the sampled values.
    # Alternatively, we can just sample and return -samples.
    
    # Use standard normal random numbers
    z = np.random.standard_normal(size=mu.shape)
    samples = mu + sigma * z
    
    # We want the candidate with the minimum sampled value to have the maximum score.
    # Therefore, we return the negative of the samples.
    scores = -samples
    
    return scores
