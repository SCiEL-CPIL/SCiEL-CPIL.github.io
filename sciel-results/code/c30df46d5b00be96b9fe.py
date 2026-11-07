import numpy as np
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    # Structural change: Use Thompson Sampling with a pseudo-count adaptation
    # Thompson Sampling samples from the posterior predictive distribution:
    # sample ~ N(mu, sigma^2)
    # We pick the point with the lowest sampled value (since we minimize).
    # This naturally balances exploration and exploitation.
    # To adapt to the phase (early vs late), we can adjust the variance or use a blend.
    # However, pure Thompson Sampling is known to perform well.
    
    # But we need to return a score for each of the 2000 candidates.
    # In standard Thompson Sampling, we draw one sample per candidate and pick the argmin.
    # Here, the evaluator picks the argmax of the returned scores.
    # So if we return the negative of the sampled value, the argmax of score corresponds to argmin of sample.
    
    # sample = mu + sigma * z, where z ~ N(0, 1)
    # score = -sample = -mu - sigma * z
    
    # To make it deterministic or reproducible? No, the function is called each step.
    # The randomness comes from the sampling.
    
    # One issue: The function signature doesn't allow passing a random state.
    # We can use a global random state or just use np.random.
    # Since the evaluator runs multiple seeds, using np.random is fine.
    
    z = np.random.standard_normal(mu.shape)
    sample = mu + sigma * z
    score = -sample
    
    # Ensure finite
    score = np.where(np.isfinite(score), score, 0.0)
    
    return score
