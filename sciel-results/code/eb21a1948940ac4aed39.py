"""Thompson Sampling acquisition for Bayesian optimization of Michalewicz 2D."""
# EVOLVE-BLOCK-START
import numpy as np


def acquisition(mu, sigma, f_best):
    """
    Thompson Sampling acquisition: draw one sample from the posterior at each
    candidate and return the negative of that sample so that higher scores
    correspond to lower (better) sampled function values.

    Parameters:
    -----------
    mu : np.ndarray, shape (2000,)
        Posterior mean at each candidate point.
    sigma : np.ndarray, shape (2000,)
        Posterior standard deviation at each candidate point.
    f_best : float
        Best (lowest) observed function value so far.

    Returns:
    --------
    scores : np.ndarray, shape (2000,)
        Finite scores; higher means evaluate next.
    """
    # Draw a single sample from N(mu, sigma^2) at each candidate point.
    # The candidate with the lowest sampled value is most promising for
    # minimization, so we return -sample (higher score = lower sample).
    samples = mu + sigma * np.random.standard_normal(mu.shape)
    return -samples
# EVOLVE-BLOCK-END
