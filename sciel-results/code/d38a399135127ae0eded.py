import numpy as np
import time
from scipy.stats import norm

def acquisition(mu, sigma, f_best):
    """
    Acquisition function: Thompson Sampling with adaptive exploitation bias.
    
    Combines stochastic exploration (Thompson Sampling) with a deterministic
    exploitation term that adapts based on the posterior uncertainty.
    
    Parameters
    ----------
    mu : array_like, shape (n,)
        Posterior mean predictions at candidate points.
    sigma : array_like, shape (n,)
        Posterior standard deviation predictions at candidate points.
    f_best : float
        Best (lowest) observed value so far.
    
    Returns
    -------
    scores : ndarray, shape (n,)
        Acquisition scores; higher values indicate more promising candidates.
    """
    mu = np.asarray(mu, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    
    # Guard against zero or negative sigma
    sigma_safe = np.maximum(sigma, 1e-9)
    
    # Thompson Sampling: Sample from posterior predictive distribution
    # y ~ N(mu, sigma^2)
    z = np.random.normal(0, 1, size=mu.shape)
    y_samples = mu + sigma_safe * z
    
    # Base score: higher is better. Since we minimize y, score is -y.
    base_score = -y_samples
    
    # Adaptive exploitation bias: stronger when uncertainty is low
    # This ensures that when the GP is confident, we exploit; when uncertain, we explore more.
    # The bias is scaled by the ratio of typical uncertainty to current uncertainty.
    # When sigma is small (confident), the bias is strong; when sigma is large (uncertain), it's weak.
    
    # Compute a reference scale for uncertainty (median of sigma across candidates)
    sigma_median = np.median(sigma_safe)
    
    # Adaptive bias strength: higher when sigma is small relative to median
    # This creates a natural exploration-exploitation balance
    adaptive_strength = 0.5 * np.exp(-sigma_safe / max(sigma_median, 1e-9))
    
    exploitation_bias = adaptive_strength * (f_best - mu)
    
    scores = base_score + exploitation_bias
    
    # Ensure finiteness
    scores = np.where(np.isfinite(scores), scores, -1e10)
    
    return scores
