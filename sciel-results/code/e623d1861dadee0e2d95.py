import numpy as np
import time

_start_time = time.time()

def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling Acquisition Function for Minimization.
    Combines the robustness of the current program's sanitization with the
    adaptive exploration of Thompson Sampling, which outperformed UCB in
    population history (score 2.34 vs 1.95).
    """
    # Deadline guard
    elapsed = time.time() - _start_time
    if elapsed > 310:
        # Fallback to greedy (maximize -mu) to ensure fast return
        return -mu
    
    # Sanitize inputs
    # Replace NaNs and Infs with safe values
    mu = np.nan_to_num(mu, nan=f_best, posinf=1e9, neginf=-1e9)
    sigma = np.nan_to_num(sigma, nan=1e-8, posinf=1e-8, neginf=1e-8)
    
    # Ensure sigma is non-negative and has a minimum floor to avoid division by zero
    # or degenerate distributions if the GP is overconfident
    sigma = np.maximum(sigma, 1e-8)
    
    # Thompson Sampling:
    # We model the objective function value f(x) as a Gaussian distribution
    # with mean mu(x) and standard deviation sigma(x).
    # To find the next point to evaluate, we sample from this distribution
    # and pick the point with the lowest sampled value.
    # Since the acquisition function must return scores where HIGHER is better,
    # we negate the sampled values.
    
    # Use a local RNG to allow stochasticity across calls.
    # This is crucial for exploration in Bayesian Optimization.
    rng = np.random.default_rng()
    
    # Draw one sample for each candidate
    samples = rng.normal(loc=mu, scale=sigma)
    
    # The score for each candidate is the negative of the sampled value.
    # Minimizing f(x) is equivalent to maximizing -f(x).
    scores = -samples
    
    # Final safety check for finite outputs
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e9, neginf=-1e9)
    
    return scores
