import numpy as np

def acquisition(mu, sigma, f_best):
    """
    Acquisition function using Monte Carlo estimation of the probability
    that a candidate point is the minimum among all candidates.
    
    This approach samples from the GP posterior for each candidate and
    determines which candidate is most frequently the lowest value across
    multiple samples. This effectively implements a stochastic exploration
    strategy that balances exploitation (low mu) and exploration (high sigma)
    by simulating the outcome of evaluating each point.
    
    This is a structural change from the deterministic Expected Improvement
    formula, treating the selection as a competitive process among the
    2000 candidates.
    """
    # Ensure inputs are numpy arrays of float64
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Sanitize sigma: ensure positive and finite
    # If sigma is 0, the point is deterministic.
    # If sigma is NaN or Inf, treat as 0 (deterministic) or small?
    # Usually, NaN sigma indicates a problem, but we must return valid scores.
    # If sigma is Inf, the point is infinitely uncertain.
    # Let's clip sigma to a reasonable range to avoid numerical issues in MC.
    sigma = np.where(np.isfinite(sigma), sigma, 0.0)
    sigma = np.maximum(sigma, 0.0)
    
    # Sanitize mu
    # If mu is NaN, treat as f_best (neutral).
    # If mu is Inf, treat as very bad (high value).
    # If mu is -Inf, treat as very good (low value).
    mu = np.where(np.isfinite(mu), mu, f_best)
    
    # Number of Monte Carlo samples
    # 100 samples provides a good balance between accuracy and speed.
    # 2000 candidates * 100 samples = 200,000 operations, which is trivial.
    n_mc = 100
    
    # Use a fixed seed for the internal RNG to ensure the function is deterministic
    # for a given (mu, sigma) pair. This helps in debugging and ensures consistent
    # behavior in the evaluation loop.
    rng = np.random.RandomState(42)
    
    # Generate standard normal samples Z of shape (n_mc, len(mu))
    Z = rng.standard_normal((n_mc, len(mu)))
    
    # Compute the sampled function values Y
    # Y[i, j] = mu[j] + sigma[j] * Z[i, j]
    # Broadcasting: mu and sigma are (N,), Z is (M, N)
    # We need mu[None, :] and sigma[None, :] to broadcast correctly.
    Y = mu[None, :] + sigma[None, :] * Z
    
    # For each MC sample (row), find the index of the candidate with the minimum value
    min_indices = np.argmin(Y, axis=1)
    
    # Count the frequency of each candidate being the minimum
    counts = np.bincount(min_indices, minlength=len(mu))
    
    # Convert counts to probabilities (scores)
    # Higher score means higher probability of being the best candidate
    scores = counts.astype(np.float64) / n_mc
    
    # Ensure all scores are finite
    scores = np.where(np.isfinite(scores), scores, 0.0)
    
    return scores
