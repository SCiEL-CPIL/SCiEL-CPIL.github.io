import numpy as np
import time

def acquisition(mu, sigma, f_best):
    """
    Acquisition function for Bayesian optimization of Hartmann 6D.
    Uses Thompson Sampling: samples from the posterior N(mu, sigma^2) 
    and selects the candidate with the lowest sampled value.
    
    Since the evaluator selects the candidate with the HIGHEST score, 
    we return the negative of the sampled value.
    
    Input:
    mu: array of shape (2000,) - posterior mean
    sigma: array of shape (2000,) - posterior standard deviation
    f_best: float - lowest observed value (since we minimize, lower is better)
    
    Output:
    scores: array of shape (2000,) - higher scores indicate better candidates
    """
    start_time = time.time()
    deadline = start_time + 290.0  # Safety margin for 300s limit
    
    # Ensure sigma is strictly positive to avoid issues with std=0
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Generate a random seed based on time to ensure different samples each call
    # This is crucial for Thompson Sampling to work as a stochastic policy
    # We use a simple integer derived from time to seed the RNG
    # Note: In a stateless function, we cannot easily maintain a persistent RNG state
    # across calls unless we use global state or derive it from inputs.
    # However, the prompt implies a stateless function signature.
    # Standard practice in these BO loops is that the function is called fresh.
    # Using np.random directly uses the global state, which is fine.
    
    # Sample from the posterior: x ~ N(mu, sigma^2)
    # We want to minimize the function, so we look for low values.
    # Thompson Sampling picks argmin(sample).
    # The evaluator picks argmax(score).
    # So we should return score = -sample.
    
    # To make the sampling deterministic for a given call but random across calls,
    # we rely on the global numpy random state.
    # However, to be safe and potentially faster/more controlled, we can generate
    # standard normal random numbers and scale them.
    
    eps = np.random.normal(0.0, 1.0, size=mu.shape)
    samples = mu + eps * sigma_safe
    
    # We want to minimize the function value.
    # The "best" candidate has the lowest sample.
    # The acquisition function should return high scores for good candidates.
    # Therefore, we return the negative of the sample.
    scores = -samples
    
    # Check deadline
    if time.time() > deadline:
        pass
    
    # Ensure finiteness
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
