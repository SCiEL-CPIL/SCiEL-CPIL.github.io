import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using a Thompson Sampling approximation.
    
    Since the cross-covariances between the 2000 candidates are not available,
    we assume independence for the sampling step. This is equivalent to
    selecting the candidate with the lowest value from a set of independent
    posterior samples. This acts as an optimistic exploration strategy,
    similar to UCB, but derived from the sampling distribution.
    """
    # Ensure sigma is positive to avoid division by zero or NaNs
    sigma_safe = np.maximum(sigma, 1e-8)
    
    # Number of samples to draw for each candidate
    # S=100 corresponds to an effective UCB beta of approx 3.0
    n_samples = 100
    
    # Generate random normal samples
    # Shape: (n_samples, num_candidates)
    z = np.random.standard_normal((n_samples, len(mu)))
    
    # Compute the sampled function values
    # Y_s(x_i) = mu(x_i) + sigma(x_i) * z_s
    samples = mu[None, :] + sigma_safe[None, :] * z
    
    # For minimization, the "best" sampled value for a candidate is its minimum.
    # We want to pick the candidate that has the lowest possible outcome.
    min_samples = np.min(samples, axis=0)
    
    # Higher scores are better. Since we are minimizing, we invert the sign.
    scores = -min_samples
    
    # Handle any numerical issues
    scores = np.nan_to_num(scores, nan=0.0, posinf=1e6, neginf=-1e6)
    
    return scores
