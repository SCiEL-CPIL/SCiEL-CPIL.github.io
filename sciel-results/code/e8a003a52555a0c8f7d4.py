import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Thompson Sampling for Bayesian optimization.
    
    Thompson Sampling involves sampling a function from the posterior distribution
    and selecting the point that minimizes the sampled function value.
    Since the evaluator selects the candidate with the highest score, we return
    the negative of the sampled value.
    
    Parameters:
    mu (np.ndarray): Mean of the GP posterior, shape (2000,).
    sigma (np.ndarray): Standard deviation of the GP posterior, shape (2000,).
    f_best (float): The lowest observed function value.
    
    Returns:
    np.ndarray: Scores for each candidate, shape (2000,). Higher score is preferred.
    """
    # Generate standard normal random variables
    # Using a fixed seed is not allowed as it would make the behavior deterministic
    # and potentially exploitable in a bad way if the seed is known, but here
    # we just want stochastic exploration. The evaluator handles the loop.
    z = np.random.normal(0, 1, size=mu.shape)
    
    # Sample from the posterior: y = mu + sigma * z
    # Note: sigma is the standard deviation
    samples = mu + sigma * z
    
    # We want to minimize the objective function.
    # The evaluator picks the candidate with the HIGHEST score.
    # Therefore, we return the negative of the sampled value.
    scores = -samples
    
    # Handle numerical edge cases
    scores = np.nan_to_num(scores, nan=-1e10, posinf=1e10, neginf=-1e10)
    
    return scores
